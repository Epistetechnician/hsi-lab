import json
from pathlib import Path

import pytest

import hsi.evaluation.locked as locked
from hsi.cli import main, template
from hsi.evaluation.registry import SPECS


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HSI_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HSI_CUSTODY_ROOT", str(tmp_path / "custody"))
    t = template("E1")
    t["params"] = {"families": ["rule_induction"], "trace_len": 40, "pilot_tasks": 4, "min_n": 8,
                   "max_n": 8}  # fmt: skip
    (tmp_path / "p.json").write_text(json.dumps(t))
    assert main(["packet", "freeze", str(tmp_path / "p.json")]) == 0
    f = str(tmp_path / "p.frozen.json")
    for cmd in ("pilot", "lock", "run"):
        assert main([cmd, f]) == 0
    return next((tmp_path / "runs").iterdir())


def test_crash_mid_eval_keeps_finished_tasks_and_resume_matches_clean_run(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = SPECS["E1"]
    orig = spec.run_task
    calls = {"n": 0}

    def flaky(*a, **k):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 6:
            raise RuntimeError("simulated crash")
        return orig(*a, **k)

    monkeypatch.setattr(spec, "run_task", flaky)
    with pytest.raises(RuntimeError):
        locked.main(run_dir)
    cache = run_dir / "eval" / "results.jsonl"
    assert len(cache.read_text().splitlines()) == 1 + 5  # header + the 5 finished tasks
    assert not (run_dir / "eval" / "receipt.json").exists()

    calls["n"] = 100  # never crash again; count only the tasks that are actually re-run
    resumed_calls = {"n": 0}

    def counting(*a, **k):  # type: ignore[no-untyped-def]
        resumed_calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(spec, "run_task", counting)
    out = locked.main(run_dir)
    total = 2 * 8  # two arms x n=8 held-out tasks
    assert resumed_calls["n"] == total - 5  # the 5 saved tasks were not recomputed
    resumed = json.loads(out.read_text())

    (run_dir / "eval" / "results.jsonl").unlink()  # a clean one-shot evaluation for comparison
    monkeypatch.setattr(spec, "run_task", orig)
    clean = json.loads(locked.main(run_dir).read_text())
    for k in ("data_digest", "family_results", "comparisons", "decision", "guardrail_values"):
        assert resumed[k] == clean[k]


def test_stale_cache_from_another_lock_is_never_reused(run_dir: Path) -> None:
    locked.main(run_dir)
    cache = run_dir / "eval" / "results.jsonl"
    lines = cache.read_text().splitlines()
    head = json.loads(lines[0])
    head["lock_digest"] = "0" * 64
    cache.write_text("\n".join([json.dumps(head), *lines[1:]]) + "\n")
    locked.main(run_dir)
    assert (run_dir / "eval" / "results.stale").exists()
    assert json.loads(cache.read_text().splitlines()[0])["lock_digest"] != "0" * 64
