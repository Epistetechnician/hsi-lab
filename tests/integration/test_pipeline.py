import json
import subprocess
import sys
from pathlib import Path

import pytest

from hsi.cli import main, template

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HSI_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HSI_CUSTODY_ROOT", str(tmp_path / "custody"))
    t = template("E1")
    t["params"] = {"families": ["rule_induction", "compositional_transfer"], "trace_len": 40,
                   "pilot_tasks": 6, "min_n": 12, "max_n": 12}
    (tmp_path / "p.json").write_text(json.dumps(t))
    assert main(["packet", "freeze", str(tmp_path / "p.json")]) == 0
    return tmp_path


def rd(env: Path) -> Path:
    return next((env / "runs").iterdir())


def full(env: Path) -> Path:
    f = str(env / "p.frozen.json")
    for cmd in ("pilot", "lock", "run"):
        assert main([cmd, f]) == 0
    assert main(["eval", str(rd(env))]) == 0
    return rd(env) / "eval" / "receipt.json"


def test_end_to_end_and_golden_determinism(env: Path) -> None:
    rc = full(env)
    assert main(["verify-receipt", str(rc)]) == 0
    assert main(["decide", str(rc), "PROMOTE"]) == 0
    r = json.loads(rc.read_text())
    assert r["decision"] == "PROMOTE" and r["guardrails_held"]


def test_decide_requires_verification(env: Path) -> None:
    rc = full(env)
    assert main(["decide", str(rc), "PROMOTE"]) == 8


def test_lock_requires_pilot_and_run_requires_lock(env: Path) -> None:
    f = str(env / "p.frozen.json")
    assert main(["lock", f]) == 9
    assert main(["run", f]) == 9


def test_tampered_receipt_fails_recompute(env: Path) -> None:
    rc = full(env)
    r = json.loads(rc.read_text())
    r["comparisons"][0]["ci_low"] += 0.1
    rc.write_text(json.dumps(r))
    assert main(["verify-receipt", str(rc)]) == 10


def test_tampered_receipt_with_fixed_digest_still_caught(env: Path) -> None:
    rc = full(env)
    r = json.loads(rc.read_text())
    r["comparisons"][0]["pairs"][0]["only_treatment"] += 1
    r["comparisons"][0]["pairs"][0]["neither"] -= 1
    body = {k: v for k, v in r.items() if k != "receipt_digest"}
    from hsi.contracts.packet import canonical_digest
    r["receipt_digest"] = canonical_digest(body)
    rc.write_text(json.dumps(r))
    assert main(["verify-receipt", str(rc)]) == 10


def test_swapped_lock_stops_eval(env: Path) -> None:
    f = str(env / "p.frozen.json")
    for cmd in ("pilot", "lock", "run"):
        main([cmd, f])
    m = rd(env) / "run" / "manifest.json"
    d = json.loads(m.read_text())
    d["lock_digest"] = "0" * 64
    m.write_text(json.dumps(d))
    assert main(["eval", str(rd(env))]) == 7


def test_tampered_packet_stops_run(env: Path) -> None:
    p = env / "p.frozen.json"
    d = json.loads(p.read_text())
    d["packet"]["threshold"] = 0.9
    p.chmod(0o644)
    p.write_text(json.dumps(d))
    assert main(["pilot", str(p)]) == 3


def test_over_budget_arm_stops_run(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    from hsi.evaluation import e1
    f = str(env / "p.frozen.json")
    main(["pilot", f])
    main(["lock", f])
    orig = e1.run_task_e1

    def fat(*a, **k):  # type: ignore[no-untyped-def]
        r = orig(*a, **k)
        return dataclasses.replace(r, stats={**r.stats, "payload_bytes": 9999.0})

    monkeypatch.setattr(e1, "run_task_e1", fat)
    assert main(["run", f]) == 6


def test_custody_is_owner_only_and_run_never_touches_it() -> None:
    src = (REPO / "src/hsi/pipeline.py").read_text()
    run_body = src[src.index("def run("):src.index("def evaluate(")]
    assert "custody_root" not in run_body


def test_validator_runs_isolated_without_args() -> None:
    r = subprocess.run([sys.executable, "-I", str(REPO / "validator/recompute.py")],
                       capture_output=True, text=True, check=False)
    assert r.returncode != 0  # usage error, but imports resolve with no hsi on path
    assert "ModuleNotFoundError" not in r.stderr
