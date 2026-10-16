import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

import hsi.evaluation.llm_stack as ls
from hsi.agents.learner import HYPOTHESES
from hsi.cli import template
from hsi.contracts.packet import Packet
from hsi.evaluation.llm_stack import _collides, train_verified_chain, verify_texts
from hsi.learning.llm import describe
from hsi.learning.train import Sample
from hsi.verification import DecisionLog


def packet(**over: Any) -> Packet:
    t = template("E6", "llm-verified")
    t["params"] = {**t["params"], **over}
    return Packet.model_validate(t, strict=False)


def test_v8_template_preregisters_verified_vs_replay() -> None:
    pk = packet()
    q = pk.params
    assert pk.version == 8 and q["verified_chain"] and q["guard_suffix"] == "verified"
    assert q["treatment"] == "d2_verified" and q["baselines"] == ["d2_replay"] and q["depth"] == 2
    assert {"d1_verified", "d2_verified", "d3_verified"} <= set(pk.arms) and len(pk.arms) == 11
    assert len(q["pinned_adapters"]) == 7  # only the pre-existing adapters are pinned
    # proposal pool seeds are disjoint from training/replay/pilot/demo ranges and avoided at eval
    lo, hi = q["verify_pool_base"], q["verify_pool_base"] + q["verify_pool_worlds"]
    assert lo >= int(q["n_adapt"]) + 50 + int(q["n_replay_tune"]) + 50 and hi <= q["tune_offset"]
    assert all(_collides(o, pk) for o in (lo, (lo + hi) // 2, hi - 1)) and not _collides(hi, pk)


def test_verify_texts_requires_valid_and_consistent() -> None:
    good, other = 2500, 100
    s_good = Sample("g", (good, 7), good)  # evidence consistent with hypothesis `good`
    s_bad = Sample("b", (other, 8), other)
    texts = [describe(HYPOTHESES[good]), describe(HYPOTHESES[good]), "garbage", ""]
    assert verify_texts(texts, [s_good, s_bad, s_good, s_good]) == [True, False, False, False]


class FakeLLM:
    """Stands in for the MLX model: records calls, never touches a GPU."""

    def __init__(self) -> None:
        self.calls = 0
        self.trained: list[int] = []

    def set_adapter(self, on: bool) -> None: ...

    def peak_gb(self) -> float:
        return 0.0

    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        self.calls += 1
        return "swap 0 1"

    def reset_lora(self, seed: int) -> None: ...

    def train_lora(self, data: list[Any], steps: int, lr: float, seed: int, batch: int = 8,
                   on_step: Any = None) -> float:  # fmt: skip
        self.trained.append(len(data))
        return 1.0

    def save_adapter(self, path: Path) -> str:
        path.write_bytes(b"x")
        return "d" * 64


SMALL = {
    "n_adapt": 3,
    "n_replay": 3,
    "n_replay_tune": 3,
    "verify_pool_worlds": 12,
    "adapt_steps": 1,
}


def run_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accept_every: int
) -> tuple[dict, FakeLLM, int]:  # type: ignore[type-arg]
    pk = packet(**SMALL)
    n = {"i": 0}

    def fake_verify(texts: list[str], samples: list[Any]) -> list[bool]:
        out = []
        for _ in texts:
            n["i"] += 1
            out.append(n["i"] % accept_every == 0)
        return out

    monkeypatch.setattr(ls, "verify_texts", fake_verify)
    llm = FakeLLM()
    (tmp_path / "learning").mkdir()
    cands: dict[str, Any] = {}
    train_verified_chain(pk, tmp_path, llm, DecisionLog(tmp_path / "learning" / "t.jsonl"), cands)
    from hsi.contracts.records import Split
    from hsi.evaluation.llm_stack import build_samples

    total = len(
        build_samples(list(pk.params["pretrain_families"]), Split.TUNE, range(3), 8, False)[0]
    )
    return cands, llm, total


def test_chain_fills_quota_with_matched_example_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cands, llm, total = run_chain(tmp_path, monkeypatch, accept_every=2)
    assert sorted(cands) == ["d1_verified", "d2_verified", "d3_verified"]
    for c in cands.values():
        assert c["n_synthetic"] == total // 2 and c["shortfall"] == 0
        assert c["n_synthetic"] + c["n_real"] == total  # same example count as the replay arms
        assert c["yield"] == pytest.approx(0.5, abs=0.2) and c["validity"] == 1.0
    assert (
        llm.trained == [total] * 3 and (tmp_path / "learning" / "d2_verified.safetensors").exists()
    )


def test_chain_tops_up_with_real_data_when_verification_starves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cands, llm, total = run_chain(tmp_path, monkeypatch, accept_every=10_000)  # nothing verifies
    for c in cands.values():
        assert c["n_synthetic"] == 0 and c["shortfall"] == total // 2
        assert c["n_real"] == total and c["yield"] == 0.0  # counts stay matched via real top-up
    assert llm.trained == [total] * 3


def test_validator_follows_guard_suffix() -> None:
    vp = Path(__file__).resolve().parents[2] / "validator" / "recompute.py"
    sp = importlib.util.spec_from_file_location("recompute", vp)
    assert sp and sp.loader
    val = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(val)
    pk = packet()
    fams = ("rule_induction@tune", "compositional_transfer@tune")
    n = 100
    pairs = [
        {"family": f, "n": n, "both": 20, "only_treatment": 15, "only_baseline": 5, "neither": 60}
        for f in fams
    ]
    fr = []
    for f in fams:
        for arm, k in (("d2_verified", 35), ("d2_replay", 25), ("d1_verified", 36)):
            fr.append({"family": f, "arm": arm, "n": n, "successes": k})
    ok = {(r["family"], r["arm"]): r["successes"] for r in fr}
    _, mean, lo, hi = val.stats(pairs, [], ok, "d2_replay", "d2_verified")
    body = pk.model_dump(mode="json")
    frozen = {"packet": body, "packet_digest": val.digest(body)}
    g = {"min_replay_diversity": 0.4, "diversity_floor": 0.1, "non_collapse_floor": 0.02,
         "rate_d1_verified": 0.36, "rate_d2_verified": 0.35}  # fmt: skip
    rec = {
        "packet_digest": frozen["packet_digest"],
        "threshold": pk.threshold,
        "nonclaims": pk.nonclaims,
        "comparisons": [
            {
                "baseline_arm": "d2_replay",
                "treatment_arm": "d2_verified",
                "pairs": pairs,
                "paired_diff": mean,
                "ci_low": lo,
                "ci_high": hi,
            }
        ],
        "family_results": fr,
        "guardrail_values": g,
        "guardrails_held": True,
        "decision": "INCONCLUSIVE",
    }
    rec["receipt_digest"] = val.digest({k: v for k, v in rec.items() if k != "receipt_digest"})
    assert val.check(rec, frozen) == []
    rec["guardrail_values"] = {**g, "rate_d2_verified": 0.9}
    rec["receipt_digest"] = val.digest({k: v for k, v in rec.items() if k != "receipt_digest"})
    assert any("verified rates differ" in e for e in val.check(rec, frozen))
    assert json.dumps(g)  # keys are plain floats


# ---- v9 (pool-matched unverified control) and v10 (held-out rule types) ----
def test_v9_v10_templates_preregister_the_followups() -> None:
    v9 = Packet.model_validate(template("E6", "llm-poolctl"), strict=False)
    v10 = Packet.model_validate(template("E6", "llm-ood"), strict=False)
    assert v9.version == 9 and v10.version == 10
    assert v9.params["baselines"] == ["d2_poolctl"] and v9.params["control_chain"]
    assert all(f.endswith("@tune") for f in v9.params["families"])
    assert {"d1_poolctl", "d2_poolctl", "d3_poolctl", "d2_verified"} <= set(v9.arms)
    assert v10.params["baselines"] == ["d2_replay"] and v10.params["families"] == [
        "rule_induction", "compositional_transfer"]  # fmt: skip
    for pk in (v9, v10):  # both reuse the v8 adapters: 7 older + 3 verified, pinned
        assert len(pk.params["pinned_adapters"]) == 10 and pk.params["reuse_from"]
        assert pk.params["depth"] == 2 and pk.params["guard_suffix"] == "verified"


def test_control_chain_takes_first_proposals_unfiltered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pk = packet(**SMALL)
    monkeypatch.setattr(
        ls, "verify_texts", lambda texts, samples: [i % 4 == 0 for i in range(len(texts))]
    )
    llm = FakeLLM()
    (tmp_path / "learning").mkdir()
    cands: dict[str, Any] = {}
    ls.train_chain(pk, tmp_path, llm, DecisionLog(tmp_path / "learning" / "t.jsonl"), cands,
                   name="poolctl", verify=False)  # fmt: skip
    assert sorted(cands) == ["d1_poolctl", "d2_poolctl", "d3_poolctl"]
    for c in cands.values():
        assert (
            0 <= c["proposed"] - c["n_synthetic"] < 8
        )  # unfiltered: quota met in the first block(s)
        assert c["n_synthetic"] + c["n_real"] == llm.trained[0]  # same example count as verified
        assert 0.0 < c["validity"] < 1.0 and c["shortfall"] == 0  # most kept are NOT consistent
    assert (tmp_path / "learning" / "d3_poolctl.safetensors").exists()
    log = [json.loads(x) for x in (tmp_path / "learning" / "t.jsonl").read_text().splitlines()]
    assert {r["eligibility"] for r in log if r["origin"] == "generated"} == {"unverified"}


def test_extend_chains_trains_only_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hsi.contracts.packet import canonical_digest

    pk = packet(**SMALL, control_chain=True)
    ld = tmp_path / "learning"
    ld.mkdir()
    man = {
        "packet_digest": "p",
        "candidates": {f"d{g}_verified": {"digest": "v"} for g in (1, 2, 3)},
    }
    man["manifest_digest"] = canonical_digest(man)
    (ld / "manifest.json").write_text(json.dumps(man))
    trained: list[str] = []

    def fake_chain(packet, run_dir, llm, log, cands, name, verify):  # type: ignore[no-untyped-def]
        trained.append(name)
        for g in (1, 2, 3):
            cands[f"d{g}_{name}"] = {"digest": "x"}

    monkeypatch.setattr(ls, "train_chain", fake_chain)
    monkeypatch.setattr(ls, "_ensure_adapter", lambda p: FakeLLM())
    ls._extend_chains(pk, tmp_path)
    assert trained == ["poolctl"]  # the verified chain already exists in the source: not retrained
    new = json.loads((ld / "manifest.json").read_text())
    assert sorted(new["candidates"]) == sorted(
        f"d{g}_{n}" for n in ("poolctl", "verified") for g in (1, 2, 3)
    )
    body = {k: v for k, v in new.items() if k != "manifest_digest"}
    assert new["manifest_digest"] == canonical_digest(body)
