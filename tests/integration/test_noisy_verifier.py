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
    dup = {"verifier_chains": [{"name": "a", "kind": "visible", "train_seed_offset": 1},
                               {"name": "a", "kind": "visible", "train_seed_offset": 2}]}  # fmt: skip
    with pytest.raises(ValueError):
        specs_of(dup)
    same_seed = {"verifier_chains": [{"name": "a", "kind": "visible", "train_seed_offset": 1},
                                     {"name": "b", "kind": "visible", "train_seed_offset": 1}]}  # fmt: skip
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


# -- E9 dose selector (ADR 0008) ---------------------------------------------------------------


def dose(**kw: Any) -> VerifierSpec:
    base: dict[str, Any] = {
        "name": "dose50", "kind": "dose", "p": 0.5, "k": 40, "train_seed_offset": 800,
    }  # fmt: skip
    return VerifierSpec(**{**base, **kw})


def test_dose_spec_validation() -> None:
    from hsi.verification.verifiers import dose_plan

    for bad in ({"p": None}, {"p": 1.5}, {"k": 0}):
        with pytest.raises(ValueError):
            dose(**bad)
    with pytest.raises(ValueError):
        spec(p=0.5)  # p belongs to dose verifiers only
    with pytest.raises(ValueError):
        spec(k=10)
    assert dose_plan(40, 0.25) == (10, 30)
    assert dose_plan(40, 0.0) == (0, 40) and dose_plan(40, 1.0) == (40, 0)
    assert VerifierSpec.from_dict(
        {"name": "d", "kind": "dose", "p": 0.75, "k": 20, "train_seed_offset": 1}
    ) == dose(name="d", p=0.75, k=20, train_seed_offset=1)


def test_dose_selector_fills_exact_quotas_and_is_order_deterministic() -> None:
    import random

    from hsi.verification.verifiers import DoseSelector

    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        stream = [i % 3 == 0 for i in range(400)]  # a third correct
        random.Random(p).shuffle(stream)
        sel = DoseSelector(dose(p=p, k=40))
        taken = [c for c in stream if not sel.full and sel.offer(c)]
        assert sel.full and sel.size == len(taken) == 40
        assert sel.n_correct == round(40 * p) and sel.realised_p == pytest.approx(p)
        sel2 = DoseSelector(dose(p=p, k=40))
        assert [c for c in stream if not sel2.full and sel2.offer(c)] == taken


def test_dose_selector_reports_shortfall() -> None:
    from hsi.verification.verifiers import DoseSelector

    sel = DoseSelector(dose(p=1.0, k=10))
    for _ in range(50):
        sel.offer(False)  # no correct proposals ever arrive
    for _ in range(4):
        sel.offer(True)
    assert not sel.full and sel.size == 4 and sel.realised_p == 1.0


def test_dose_accepts_only_consistent_parseable() -> None:
    s = dose()
    assert accepts(s, 0, 1, "k", "t", parses=True, consistent=True)
    assert not accepts(s, 0, 1, "k", "t", parses=True, consistent=False)
    assert not accepts(s, 0, 1, "k", "t", parses=False, consistent=True)


def _run_dose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sp: VerifierSpec, correct: Any
) -> Any:
    import hsi.verification.verifiers as vv

    monkeypatch.setattr(ls, "verify_texts", lambda texts, block: [True] * len(texts))
    monkeypatch.setattr(vv, "probe_correct", lambda h, truth, tag: correct(tag))
    pk = small_packet()
    llm = FakeLLM()
    llm.seeds = []
    (tmp_path / "learning").mkdir()
    cands: dict[str, Any] = {}
    log = DecisionLog(tmp_path / "learning" / "t.jsonl")
    train_chain(pk, tmp_path, llm, log, cands, name=sp.name, verify=True, spec=sp, gens=1)
    return cands[f"d1_{sp.name}"], llm


def test_dose_chain_holds_size_and_hits_target_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sp = dose(name="d50", p=0.5, k=6)
    c, llm = _run_dose(tmp_path, monkeypatch, sp, lambda tag: sum(map(ord, tag)) % 2 == 0)
    assert c["n_synthetic"] == 6 and c["shortfall"] == 0 and c["precision"] == 0.5
    assert c["dose_valid"] and c["dose_target"] == 0.5 and c["dose_realised"] == 0.5
    assert llm.trained == [c["n_synthetic"] + c["n_real"]]  # total examples independent of p
    assert llm.seeds == [0 + sp.train_seed_offset + 1]


def test_dose_chain_total_size_is_the_same_for_every_p(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes = set()
    for i, p in enumerate((0.0, 0.5, 1.0)):
        sub = tmp_path / f"p{i}"
        sub.mkdir()
        c, _ = _run_dose(sub, monkeypatch, dose(name=f"d{i}", p=p, k=6),
                         lambda tag: sum(map(ord, tag)) % 2 == 0)  # fmt: skip
        sizes.add(c["n_synthetic"] + c["n_real"])
    assert len(sizes) == 1


def test_dose_chain_flags_a_shortfall_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, _ = _run_dose(tmp_path, monkeypatch, dose(name="short", p=1.0, k=6), lambda tag: False)
    assert c["shortfall"] == 6 and not c["dose_valid"]


def test_v13_template_preregisters_the_dose_response() -> None:
    from hsi.cli import DOSE_K, DOSE_LEVELS

    pk = Packet.model_validate(template("E6", "llm-dose"), strict=False)
    q = pk.params
    assert pk.version == 13 and pk.threshold == 0.10 and q["eval_mode"] == "probe"
    specs = specs_of(q)
    assert len(specs) == 2 * len(DOSE_LEVELS) and {s.k for s in specs} == {DOSE_K}
    assert {s.p for s in specs} == set(DOSE_LEVELS)
    arms = set(pk.arms)
    assert {f"d2_{s.name}" for s in specs} <= arms  # every dose chain is evaluated
    for sec in q["secondary"]:
        assert {sec["treatment"], sec["baseline"]} <= arms
    assert q["treatment"] == "d2_dose100a" and q["baselines"] == ["d2_dose000a"]
    assert q["guard_suffix"] == "dose100a" and {"d1_dose100a", "d2_dose100a"} <= arms
    assert len(q["pinned_adapters"]) == 13


# -- E10 synthetic share (ADR 0009) -------------------------------------------------------------


def test_n_synth_applies_to_visible_only() -> None:
    assert spec(kind="visible", fpr=0.0, n_synth=0).n_synth == 0
    with pytest.raises(ValueError):
        spec(n_synth=5)  # noisy
    with pytest.raises(ValueError):
        spec(kind="visible", fpr=0.0, n_synth=-1)


def test_share_chain_keeps_total_constant_and_reports_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ls, "verify_texts", lambda texts, block: [True] * len(texts))
    totals = set()
    for i, n in enumerate((0, 2, 5)):
        sub = tmp_path / f"s{i}"
        (sub / "learning").mkdir(parents=True)
        llm, cands = FakeLLM(), {}
        sp = spec(name=f"s{i}", kind="visible", fpr=0.0, n_synth=n)
        train_chain(small_packet(), sub, llm, DecisionLog(sub / "learning" / "t.jsonl"), cands,
                    name=sp.name, verify=True, spec=sp, gens=1)  # fmt: skip
        c = cands[f"d1_s{i}"]
        assert c["n_synthetic"] == n and c["synth_valid"] and c["shortfall"] == 0
        totals.add(c["n_synthetic"] + c["n_real"])
    assert len(totals) == 1


def test_v14_template_preregisters_the_share_study() -> None:
    from hsi.cli import SHARE_LEVELS

    pk = Packet.model_validate(template("E6", "llm-share"), strict=False)
    q = pk.params
    assert pk.version == 14 and pk.threshold == 0.05 and q["min_n"] >= 300
    specs = specs_of(q)
    assert len(specs) == 2 * len(SHARE_LEVELS) and {s.n_synth for s in specs} == set(
        SHARE_LEVELS.values()
    )
    assert max(SHARE_LEVELS.values()) <= 350 and SHARE_LEVELS["s050"] == 175
    arms = set(pk.arms)
    assert {f"d2_{s.name}" for s in specs} <= arms
    for sec in q["secondary"]:
        assert {sec["treatment"], sec["baseline"]} <= arms
    assert q["guard_suffix"] == "s050a" and {"d1_s050a", "d2_s050a"} <= arms
    assert q["treatment"] == "d2_s050a" and q["baselines"] == ["d2_s000a"]


# -- E10 on CUDA (v15): batched proposals, lean prepare ------------------------------------------


class BatchLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[int] = []

    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        raise AssertionError("a batched run must not generate one prompt at a time")

    def generate_many(self, prompts: list[str], max_tokens: int = 20) -> list[str]:
        self.blocks.append(len(prompts))
        return ["swap 0 1"] * len(prompts)


def test_gen_block_uses_generate_many_and_respects_the_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ls, "verify_texts", lambda texts, block: [True] * len(texts))
    (tmp_path / "learning").mkdir()
    llm, cands = BatchLLM(), {}
    sp = spec(name="s5", kind="visible", fpr=0.0, n_synth=5)
    train_chain(small_packet(gen_block=16), tmp_path, llm,
                DecisionLog(tmp_path / "learning" / "t.jsonl"), cands, name="s5", verify=True,
                spec=sp, gens=1)  # fmt: skip
    c = cands["d1_s5"]
    assert llm.blocks and max(llm.blocks) <= 16 and c["n_synthetic"] == 5 and c["shortfall"] == 0
    assert c["proposed"] == sum(llm.blocks)  # every generated proposal is counted


def test_v15_template_is_the_lean_torch_variant_of_v14() -> None:
    a = Packet.model_validate(template("E6", "llm-share"), strict=False)
    b = Packet.model_validate(template("E6", "llm-share-torch"), strict=False)
    assert b.version == 15 and b.threshold == a.threshold and b.arms == a.arms
    q = b.params
    assert q["model"] == "qwen2.5-7b-hf" and q["lean_prepare"] and q["gen_block"] == 64
    assert "reuse_from" not in q and "pinned_adapters" not in q
    assert not q["verified_chain"] and not q["control_chain"]
    assert q["verifier_chains"] == a.params["verifier_chains"]  # same ten chains, same quotas
    assert q["treatment"] == a.params["treatment"] and q["secondary"] == a.params["secondary"]
