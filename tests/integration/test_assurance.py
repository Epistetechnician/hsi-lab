import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hsi.assurance import assess, bundle, render
from hsi.assurance.reproduce import compare
from hsi.cli import main, template
from hsi.errors import PreconditionError

REPO = Path(__file__).resolve().parents[2]


def test_validator_is_stdlib_only() -> None:
    mods = set()
    for n in ast.walk(ast.parse((REPO / "validator" / "recompute.py").read_text())):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
    assert mods <= {"hashlib", "json", "math", "sys"}


def receipt() -> dict[str, object]:
    return json.loads((REPO / "docs/experiments/E1-receipt.json").read_text())


def test_compare_ignores_timing_but_catches_aggregates() -> None:
    a = receipt()
    b = json.loads(json.dumps(a))
    b["budget_used"]["wall_seconds"] += 5  # type: ignore[index]
    b["budget_used"]["peak_mem_mb"] += 5  # type: ignore[index]
    b["receipt_digest"] = "x"
    ok, total, bad = compare(a, b)
    assert ok == total and not bad
    b["family_results"][0]["successes"] += 1  # type: ignore[index]
    ok, total, bad = compare(a, b)
    assert bad == ["family_results[0].successes"] and ok == total - 1


def test_reproduce_refuses_dirty_or_missing_lock(tmp_path: Path) -> None:
    from hsi.assurance import reproduce

    r = receipt()
    p = tmp_path / "r.json"
    p.write_text(json.dumps({**r, "source_commit": "abc+dirty"}))
    with pytest.raises(PreconditionError):
        reproduce(p, tmp_path, tmp_path, None)
    p.write_text(json.dumps(r))
    with pytest.raises(PreconditionError):
        reproduce(p, tmp_path, tmp_path, None)


def test_bundle_validates_in_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HSI_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HSI_CUSTODY_ROOT", str(tmp_path / "custody"))
    t = template("E1")
    t["params"] = {"families": ["rule_induction"], "trace_len": 40, "pilot_tasks": 4, "min_n": 8,
                   "max_n": 8}
    (tmp_path / "p.json").write_text(json.dumps(t))
    main(["packet", "freeze", str(tmp_path / "p.json")])
    f = str(tmp_path / "p.frozen.json")
    for cmd in ("pilot", "lock", "run"):
        assert main([cmd, f]) == 0
    rd = next((tmp_path / "runs").iterdir())
    assert main(["eval", str(rd)]) == 0
    rc = rd / "eval" / "receipt.json"
    out = bundle(rc, tmp_path / "bundle")
    assert sorted(x.name for x in out.iterdir()) == ["README.txt", "packet.frozen.json",
                                                      "receipt.json", "recompute.py"]
    cmd = [sys.executable, "-I", "recompute.py", "receipt.json", "packet.frozen.json"]
    assert subprocess.run(cmd, cwd=out, capture_output=True, check=False).returncode == 0
    d = json.loads((out / "receipt.json").read_text())
    d["comparisons"][0]["ci_high"] += 0.2
    (out / "receipt.json").write_text(json.dumps(d))
    assert subprocess.run(cmd, cwd=out, capture_output=True, check=False).returncode == 1


def test_assess_reports_all_seven() -> None:
    rows = assess()
    assert [r["experiment"] for r in rows] == ["E1", "E2", "E3", "E4", "E4-llm", "E5", "E6", "E6-llm", "E7"]
    assert all(r["level"] in ("A2", "A3") for r in rows)
    assert "of 7" in render(rows)
