import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

import hsi.evaluation.llm_stack as ls
from hsi.agents.learner import HYPOTHESES
from hsi.cli import template
from hsi.contracts.packet import Packet
from hsi.contracts.records import Split
from hsi.evaluation.llm_stack import build_samples, train_chain
from hsi.evaluation.stats import bonferroni_z
from hsi.verification import DecisionLog
from hsi.verification.verifiers import (
    VerifierSpec,
    accepts,
    noise_draw,
    probe_correct,
    probe_states,
    specs_of,
    true_rule,
    world_of,
)


def spec(**kw: Any) -> VerifierSpec:
    base: dict[str, Any] = {"name": "n", "kind": "noisy", "fpr": 0.1, "train_seed_offset": 300}
    return VerifierSpec(**{**base, **kw})


def test_spec_validation() -> None:
    with pytest.raises(ValueError):
        spec(kind="oracle")
    with pytest.raises(ValueError):
        spec(fpr=1.5)
    with pytest.raises(ValueError):
        spec(kind="visible", fpr=0.1)  # noise belongs to the noisy kind only
    with pytest.raises(ValueError):
        spec(kind="confidence", fpr=0.0)  # a confidence verifier needs tau
    dup = {"verifier_chains": [dict(name="a", kind="visible", train_seed_offset=1),
                               dict(name="a", kind="visible", train_seed_offset=2)]}  # fmt: skip
    with pytest.raises(ValueError):
        specs_of(dup)
    same_seed = {"verifier_chains": [dict(name="a", kind="visible", train_seed_offset=1),
                                     dict(name="b", kind="visible", train_seed_offset=1)]}  # fmt: skip
    with pytest.raises(ValueError):
        specs_of(same_seed)


def test_noise_is_a_pure_function_of_its_inputs() -> None:
    s = spec()
    a = noise_draw(s, 0, 1, "k", "swap 0 1")
    assert a == noise_draw(s, 0, 1, "k", "swap 0 1") and 0.0 <= a < 1.0
    assert a != noise_draw(s, 0, 2, "k", "swap 0 1")  # generation
    assert a != noise_draw(s, 0, 1, "k2", "swap 0 1")  # sample
    assert a != noise_draw(s, 0, 1, "k", "swap 0 2")  # text
    assert a != noise_draw(spec(salt="b"), 0, 1, "k", "swap 0 1")  # replicate chain


def test_accept_rule_per_kind() -> None:
    kw: dict[str, Any] = {"parses": True, "consistent": False}
    s = spec(fpr=0.3)
    hits = sum(accepts(s, 0, 1, f"k{i}", "t", **kw) for i in range(6000))
    assert hits / 6000 == pytest.approx(0.3, abs=0.03)  # realised false-accept rate
    assert all(accepts(s, 0, 1, f"k{i}", "t", parses=True, consistent=True) for i in range(50))
    assert not any(accepts(s, 0, 1, f"k{i}", "t", parses=False, consistent=True) for i in range(50))
    assert not any(accepts(spec(fpr=1.0), 0, 1, "k", "t", parses=False, consistent=False)
                   for _ in range(3))  # fmt: skip
    assert all(accepts(spec(fpr=1.0), 0, 1, f"k{i}", "t", **kw) for i in range(50))
    vis = spec(kind="visible", fpr=0.0)
    assert accepts(vis, 0, 1, "k", "t", parses=True, consistent=True)
    assert not accepts(vis, 0, 1, "k", "t", **kw)
    conf = spec(kind="confidence", fpr=0.0, tau=-0.5)
    assert accepts(conf, 0, 1, "k", "t", parses=True, consistent=False, mean_logprob=-0.4)
    assert not accepts(conf, 0, 1, "k", "t", parses=True, consistent=True, mean_logprob=-0.6)
    assert not accepts(conf, 0, 1, "k", "t", parses=False, consistent=True, mean_logprob=-0.1)


def test_world_key_roundtrip_and_probes() -> None:
    smp, _ = build_samples(["compositional_transfer"], Split.TUNE, range(1000, 1004), 8, False)
    assert smp
    for x in smp:
        fam, split, seed, a = world_of(x.key)
        assert (fam, split) == ("compositional_transfer", Split.TUNE) and 0 <= a < 4
        assert seed >= 1_000_000
    truth = true_rule(smp[0].key)
    assert probe_correct(truth, truth, "t") and not probe_correct(None, truth, "t")
    other = next(h for h in HYPOTHESES if len(h) == len(truth) and h != truth)
    assert not probe_correct(other, truth, "t")
    assert probe_states("x") == probe_states("x") != probe_states("y")


def test_probes_separate_evidence_consistent_wrong_rules() -> None:
    """The point of the audit: a wrong rule can fit the visible evidence yet fail on probes."""
    n_consistent_wrong = 0
    smp, _ = build_samples(
        ["rule_induction", "compositional_transfer"], Split.TUNE, range(1000, 1040), 8, False
    )
    for x in smp:
        truth = true_rule(x.key)
        for c in x.cands:
            if not probe_correct(HYPOTHESES[c], truth, x.key):
                n_consistent_wrong += 1
    assert n_consistent_wrong > 0  # ambiguity exists, so a visible-only verifier can be fooled


def test_v12_template_preregisters_the_noisy_verifier_sweep() -> None:
    pk = Packet.model_validate(template("E6", "llm-noisyver"), strict=False)
    q = pk.params
    assert pk.version == 12 and q["eval_mode"] == "probe" and q["depth"] == 2
    specs = specs_of(q)
    assert {s.name for s in specs} >= {"noisy10", "noisy10b", "noisy30", "tested", "conf"}
    assert q["treatment"] == "d2_noisy10" and q["baselines"] == ["d2_poolctl"]
    arms = set(pk.arms)
    for sec in q["secondary"]:  # every contrast refers to evaluated arms
        assert {sec["treatment"], sec["baseline"]} <= arms
    assert q["guard_suffix"] == "noisy10" and {"d1_noisy10", "d2_noisy10"} <= arms
    assert len(q["pinned_adapters"]) == 13 and q["reuse_from"]
    lo, hi = q["verify_pool_base"], q["verify_pool_base"] + q["verify_pool_worlds"]
    a_lo, a_hi = q["audit_base"], q["audit_base"] + q["audit_worlds"]
    assert hi <= a_lo and [a_lo, a_hi] in q["avoid"] and [lo, hi] in q["avoid"]


class FakeLLM:
    def __init__(self) -> None:
        self.trained: list[int] = []

    def set_adapter(self, on: bool) -> None: ...

    def peak_gb(self) -> float:
        return 0.0

    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        return "swap 0 1"

    def mean_logprob(self, prompt: str, text: str) -> float:
        return -0.1

    def reset_lora(self, seed: int) -> None:
        self.seeds.append(seed)

    seeds: list[int] = []

    def train_lora(self, data: list[Any], steps: int, lr: float, seed: int, batch: int = 8,
                   on_step: Any = None) -> float:  # fmt: skip
        self.trained.append(len(data))
        return 1.0

    def save_adapter(self, path: Path) -> str:
        path.write_bytes(b"x")
        return "d" * 64


def small_packet(**over: Any) -> Packet:
    t = template("E6", "llm-noisyver")
    t["params"] = {**t["params"], "n_adapt": 3, "n_replay": 3, "n_replay_tune": 3,
                   "verify_pool_worlds": 12, "adapt_steps": 1, **over}  # fmt: skip
    return Packet.model_validate(t, strict=False)


def test_noisy_chain_audits_precision_and_uses_its_own_training_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ls, "verify_texts", lambda texts, block: [False] * len(texts))
    pk = small_packet()
    sp = spec(name="noisy50", fpr=0.5, train_seed_offset=777)
    llm = FakeLLM()
    llm.seeds = []
    (tmp_path / "learning").mkdir()
    cands: dict[str, Any] = {}
    train_chain(pk, tmp_path, llm, DecisionLog(tmp_path / "learning" / "t.jsonl"), cands,
                name="noisy50", verify=True, spec=sp, gens=2)  # fmt: skip
    assert sorted(cands) == ["d1_noisy50", "d2_noisy50"]  # only the requested depth is trained
    assert llm.seeds == [int(pk.seeds[0]) + 777 + g for g in (1, 2)]
    for c in cands.values():
        # nothing is evidence-consistent, so everything kept is a false accept; swap 0 1 is
        # parseable, so ~half of the proposals pass and almost none is correct
        assert 0.0 <= c["precision"] <= 1.0 and c["verifier"] == "noisy50"
        assert c["validity"] == 0.0 and 0.2 < c["accept_rate"] < 0.9
        assert c["n_synthetic"] + c["n_real"] == llm.trained[0]


def test_confidence_chain_needs_no_world_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ls, "verify_texts", lambda texts, block: [False] * len(texts))
    pk = small_packet()
    sp = VerifierSpec(name="conf", kind="confidence", tau=-0.2, train_seed_offset=900)
    llm = FakeLLM()
    llm.seeds = []
    (tmp_path / "learning").mkdir()
    cands: dict[str, Any] = {}
    train_chain(pk, tmp_path, llm, DecisionLog(tmp_path / "learning" / "t.jsonl"), cands,
                name="conf", verify=True, spec=sp, gens=1)  # fmt: skip
    assert cands["d1_conf"]["shortfall"] == 0  # -0.1 >= tau: every proposal is kept
    sp2 = VerifierSpec(name="conf2", kind="confidence", tau=-0.05, train_seed_offset=901)
    train_chain(pk, tmp_path, llm, DecisionLog(tmp_path / "learning" / "t2.jsonl"), cands,
                name="conf2", verify=True, spec=sp2, gens=1)  # fmt: skip
    assert cands["d1_conf2"]["n_synthetic"] == 0  # nothing is confident enough: real top-up


def test_secondary_contrasts_are_recomputed_with_the_adjusted_multiplier() -> None:
    assert bonferroni_z(1) == pytest.approx(1.959964, abs=1e-5)
    assert bonferroni_z(7) > bonferroni_z(2) > bonferroni_z(1)
    vp = Path(__file__).resolve().parents[2] / "validator" / "recompute.py"
    sp = importlib.util.spec_from_file_location("recompute", vp)
    assert sp and sp.loader
    val = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(val)
    pk = small_packet()
    q = dict(pk.params)
    fams = list(q["families"])
    n = 100
    arms = list(pk.arms)

    def pairs(ot: int, ob: int) -> list[dict[str, Any]]:
        return [{"family": f, "n": n, "both": 20, "only_treatment": ot, "only_baseline": ob,
                 "neither": n - 20 - ot - ob} for f in fams]  # fmt: skip

    succ = {}
    for f in fams:
        for a in arms:
            succ[(f, a)] = 30
    # contrast realised by treatment 20+ot, baseline 20+ob
    t, b = q["treatment"], q["baselines"][0]
    for f in fams:
        succ[(f, t)], succ[(f, b)] = 35, 25
    p1 = pairs(15, 5)
    _, mean, lo, hi = val.stats(p1, [], succ, b, t)
    sec = q["secondary"][0]
    for f in fams:
        succ[(f, sec["treatment"])], succ[(f, sec["baseline"])] = 20 + 12, 20 + 3
    p2 = pairs(12, 3)
    z = bonferroni_z(len(q["secondary"]))
    errs: list[str] = []
    _, m2, l2, h2 = val.stats(p2, errs, succ, sec["baseline"], sec["treatment"], z)
    assert not errs
    _, m2n, l2n, h2n = val.stats(p2, [], succ, sec["baseline"], sec["treatment"])
    assert (h2 - l2) > (h2n - l2n) and m2 == m2n  # adjusted interval is wider, centre unchanged
    assert json.dumps([m2, l2, h2])
    assert mean > 0 and lo < hi


def test_tested_chain_requires_extra_probes_on_top_of_the_visible_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hsi.verification.verifiers as vv

    monkeypatch.setattr(ls, "verify_texts", lambda texts, block: [True] * len(texts))
    pk = small_packet()
    sp = VerifierSpec(name="tested", kind="tested", extra_probes=4, train_seed_offset=600)
    llm = FakeLLM()
    llm.seeds = []
    (tmp_path / "learning").mkdir()
    cands: dict[str, Any] = {}
    # "swap 0 1" is almost never the true rule, so the extra probes reject nearly everything
    train_chain(pk, tmp_path, llm, DecisionLog(tmp_path / "learning" / "t.jsonl"), cands,
                name="tested", verify=True, spec=sp, gens=1)  # fmt: skip
    c = cands["d1_tested"]
    assert c["shortfall"] > 0 and c["n_synthetic"] + c["n_real"] == llm.trained[0]
    monkeypatch.setattr(vv, "probes_agree", lambda *a, **k: True)
    train_chain(pk, tmp_path, llm, DecisionLog(tmp_path / "learning" / "t2.jsonl"), cands,
                name="tested2", verify=True, spec=sp, gens=1)  # fmt: skip
    assert cands["d1_tested2"]["shortfall"] == 0  # passes every probe: the visible check decides
