import json
from pathlib import Path

import pytest

from hsi.cli import main, template


@pytest.mark.parametrize("exp", ["E3", "E5", "E7"])
def test_pipeline(exp: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HSI_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HSI_CUSTODY_ROOT", str(tmp_path / "custody"))
    t = template(exp)
    t["params"] = {**t["params"], "pilot_tasks": 3, "min_n": 6, "max_n": 6}  # type: ignore[dict-item]
    (tmp_path / "p.json").write_text(json.dumps(t))
    assert main(["packet", "freeze", str(tmp_path / "p.json")]) == 0
    f = str(tmp_path / "p.frozen.json")
    for cmd in ("pilot", "lock", "run"):
        assert main([cmd, f]) == 0
    rd = next((tmp_path / "runs").iterdir())
    assert main(["eval", str(rd)]) == 0
    rc = rd / "eval" / "receipt.json"
    assert main(["verify-receipt", str(rc)]) == 0
    r = json.loads(rc.read_text())
    if exp in ("E3", "E5"):
        assert (rd / "run/messages.jsonl").read_text().strip()
        assert r["guardrail_values"]["max_message_bytes_used"] <= (24 if exp == "E3" else 72)
        if exp == "E5":
            assert "link_open" in (rd / "run/decisions.jsonl").read_text()
    else:
        assert (rd / "run/relation_events.jsonl").read_text().strip()
        assert r["guardrail_values"]["duplicate_strength_updates"] == 0
    r["guardrail_values"]["compute_matched_fraction"] = 0.5
    rc.write_text(json.dumps(r))
    assert main(["verify-receipt", str(rc)]) == 10


def test_e3_over_cap_stops_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HSI_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HSI_CUSTODY_ROOT", str(tmp_path / "custody"))
    t = template("E3")
    t["params"] = {**t["params"], "pilot_tasks": 3, "min_n": 6, "max_n": 6}  # type: ignore[dict-item]
    (tmp_path / "p.json").write_text(json.dumps(t))
    main(["packet", "freeze", str(tmp_path / "p.json")])
    f = str(tmp_path / "p.frozen.json")
    main(["pilot", f])
    main(["lock", f])
    import hsi.evaluation.registry as reg
    from hsi.evaluation import e3

    orig = e3.run_task_e3

    def fat(*a, **k):  # type: ignore[no-untyped-def]
        import dataclasses
        r = orig(*a, **k)
        return dataclasses.replace(r, stats={**r.stats, "message_bytes": 999.0})

    monkeypatch.setattr(e3, "run_task_e3", fat)
    assert reg.SPECS["E3"] is not None
    assert main(["run", f]) == 6
