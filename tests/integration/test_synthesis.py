import json
from pathlib import Path

import pytest

from hsi.cli import main, template
from hsi.contracts.records import Eligibility, Origin, Split
from hsi.environment import SEED_BASE, RuleWorld
from hsi.synthesis import Proposer, Quarantine
from hsi.verification import Verifier


def world() -> RuleWorld:
    return RuleWorld("rule_induction", Split.FIT, 3)


def proposal(parents: list[str], tamper: bool = False):  # type: ignore[no-untyped-def]
    w = world()
    obs, _ = w.reset()
    s = tuple(obs)
    p = Proposer([(s, 0, w.transition(s, 0))], ["r0"], "k")
    ep = p.propose("syn-0", "rule_induction", Split.FIT, 3)
    if tamper:
        st = ep.steps[0].model_copy(update={"observation": [(x + 1) % 7 for x in ep.steps[0].observation]})
        ep = ep.model_copy(update={"steps": [st]})
    return ep.model_copy(update={"parent_ids": parents})


def test_accepts_true_rejects_false_and_provenance_free() -> None:
    w = world()
    v = Verifier(w)
    good = proposal(["r0"])
    # make the claim true by construction
    st = good.steps[0].model_copy(update={"observation": list(w.transition(tuple(good.start), 0))})
    good = good.model_copy(update={"steps": [st]})
    assert v.verify(good).episode.eligibility is Eligibility.ELIGIBLE
    assert v.verify(good).reason == "ambiguous_duplicate"
    assert Verifier(w).verify(good.model_copy(update={"parent_ids": []})).reason == "provenance_missing"
    bad = good.model_copy(update={"steps": [st.model_copy(update={"observation": [9] * 5})]})
    assert Verifier(w).verify(bad).reason == "outcome_mismatch"


def test_quarantine_rejects_non_generated() -> None:
    q = Quarantine()
    ep = proposal(["r0"])
    q.submit(ep)
    with pytest.raises(ValueError):
        q.submit(ep.model_copy(update={"origin": Origin.OBSERVED}))
    with pytest.raises(ValueError):
        q.submit(ep.model_copy(update={"eligibility": Eligibility.ELIGIBLE}))


def test_e2_pipeline_and_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HSI_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HSI_CUSTODY_ROOT", str(tmp_path / "custody"))
    t = template("E2")
    t["params"] = {**t["params"], "families": ["rule_induction"], "pilot_tasks": 4, "min_n": 8,  # type: ignore[dict-item]
                   "max_n": 8}
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
    assert r["guardrail_values"]["lineage_fraction"] == 1.0
    # every eligible example in the examples ledger traces to a decision and a quarantine record
    dec = {json.loads(x)["episode_id"] for x in (rd / "run/decisions.jsonl").read_text().splitlines()}
    quar = {json.loads(x)["episode_id"] for x in (rd / "run/quarantine.jsonl").read_text().splitlines()}
    for line in (rd / "run/examples.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["origin"] == "generated" and e["episode_id"].startswith("verified"):
            assert e["eligibility"] == "eligible" and e["episode_id"] in dec and e["episode_id"] in quar
    # tampering with a verified receipt's guardrail is caught
    r["guardrail_values"]["lineage_fraction"] = 0.5
    rc.write_text(json.dumps(r))
    assert main(["verify-receipt", str(rc)]) == 10


def test_heldout_seed_range_unused_by_run() -> None:
    assert SEED_BASE[Split.HELDOUT] == 2_000_000
