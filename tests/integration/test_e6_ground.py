import json
from pathlib import Path
from typing import Any

import pytest

from hsi.cli import template
from hsi.contracts.packet import Packet, canonical_digest
from hsi.errors import DigestError
from hsi.evaluation.base import TaskResult
from hsi.evaluation.e6 import E6Spec, sig_entropy
from hsi.evaluation.llm_stack import _check_pinned, try_reuse_adapters

ARMS = ["d1_no_replay", "d2_no_replay", "d3_no_replay", "d1_replay", "d2_replay", "d3_replay"]


def packet(**over: Any) -> Packet:
    t = template("E6", "llm-gen3")
    t["params"] = {**t["params"], **over}
    return Packet.model_validate(t, strict=False)


def make_source(root: Path, pk: Packet, digests: dict[str, str]) -> None:
    src = root / "srcrun"
    (src / "learning").mkdir(parents=True)
    old = packet()  # the source recipe == the default recipe
    (src / "packet.frozen.json").write_text(
        json.dumps({"packet": old.model_dump(mode="json"), "packet_digest": "x"})
    )
    for a in digests:
        (src / "learning" / f"{a}.safetensors").write_bytes(b"adapter-" + a.encode())
    (src / "learning" / "training.jsonl").write_text('{"sample_id": "s"}\n')
    man = {
        "packet_digest": "old",
        "runtime": "r",
        "candidates": {a: {"digest": d} for a, d in digests.items()},
    }
    man["manifest_digest"] = canonical_digest(man)
    (src / "learning" / "manifest.json").write_text(json.dumps(man))


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, str]]:
    monkeypatch.setenv("HSI_RUN_ROOT", str(tmp_path))
    digests = {a: f"{i:064x}" for i, a in enumerate(ARMS)}
    make_source(tmp_path, packet(), digests)
    return tmp_path, digests


def test_reuse_copies_and_rebinds_manifest(env: tuple[Path, dict[str, str]]) -> None:
    root, digests = env
    pk = packet(reuse_from="srcrun", pinned_adapters=digests)
    out = root / "newrun"
    assert try_reuse_adapters(pk, out)
    man = json.loads((out / "learning" / "manifest.json").read_text())
    body = {k: v for k, v in man.items() if k != "manifest_digest"}
    assert man["manifest_digest"] == canonical_digest(body)
    assert man["packet_digest"] == canonical_digest(pk.model_dump(mode="json"))
    assert man["reused_from"] == "srcrun"
    assert (out / "learning" / "d3_replay.safetensors").read_bytes() == b"adapter-d3_replay"


def test_reuse_fails_closed_on_pin_mismatch(env: tuple[Path, dict[str, str]]) -> None:
    root, digests = env
    bad = {**digests, "d3_replay": "f" * 64}
    with pytest.raises(DigestError):
        try_reuse_adapters(packet(reuse_from="srcrun", pinned_adapters=bad), root / "x")


def test_reuse_declined_when_recipe_differs_or_source_missing(
    env: tuple[Path, dict[str, str]],
) -> None:
    root, digests = env
    assert not try_reuse_adapters(
        packet(reuse_from="srcrun", pinned_adapters=digests, adapt_steps=7), root / "a"
    )
    assert not try_reuse_adapters(packet(reuse_from="nope", pinned_adapters=digests), root / "b")
    assert not try_reuse_adapters(packet(), root / "c")  # no reuse requested


def test_check_pinned() -> None:
    _check_pinned({"a": {"digest": "1"}}, None)  # nothing pinned: no check
    _check_pinned({"a": {"digest": "1"}}, {"a": "1"})
    with pytest.raises(DigestError):
        _check_pinned({"a": {"digest": "1"}}, {"a": "2"})
    with pytest.raises(DigestError):
        _check_pinned({"a": {"digest": "1"}}, {"a": "1", "b": "3"})


def test_sig_entropy() -> None:
    assert sig_entropy([]) == 0.0 and sig_entropy(["swap"] * 5) == 0.0
    assert 0 < sig_entropy(["swap", "add", "invalid", "swap"]) < 1


def results(rates: dict[str, float], sigs: dict[str, list[str]], n: int = 20) -> list[TaskResult]:
    out = []
    for arm, rate in rates.items():
        for i in range(n):
            ok = i < round(rate * n)
            out.append(
                TaskResult(
                    arm,
                    "rule_induction",
                    i,
                    ok,
                    [],
                    {
                        "gen_seen": 1.0,
                        "gen_valid": 1.0,
                        "gen_consistent": float(ok),
                        "gen_exact": 0.0,
                    },
                    decisions=[{"arm": arm, "sig": sigs[arm][i % len(sigs[arm])]}],
                )
            )
    return out


def test_ground_guardrails_hold_and_fail() -> None:
    pk = packet(eval_mode="ground", diversity_floor_frac=0.5)
    spec = E6Spec()
    spec.manifest = {"candidates": {a: {} for a in ARMS}, "diversity_floor": 0.0}
    diverse = ["swap", "add", "copy", "mix"]
    sigs = dict.fromkeys(ARMS, diverse)
    ok_rates = dict.fromkeys(ARMS, 0.3)
    gv, held = spec.guardrails(results(ok_rates, sigs), [], pk)
    assert held and gv["min_replay_diversity"] >= gv["diversity_floor"] > 0
    assert "eval_diversity_d3_replay" in gv and "eval_exact_d3_replay" in gv
    # replay arm collapses to one signature at depth 3 -> diversity guardrail fails
    sigs_c = {**sigs, "d3_replay": ["invalid"]}
    _, held_c = spec.guardrails(results(ok_rates, sigs_c), [], pk)
    assert not held_c
    # grounded rate drops more than the floor at depth 3 -> non-collapse guardrail fails
    low = {**ok_rates, "d3_replay": 0.1}
    _, held_l = spec.guardrails(results(low, sigs), [], pk)
    assert not held_l


# ---- confirmatory packet (v7): in-distribution worlds drawn from the lock; depth-2 primary ----
def confirm_packet() -> Packet:
    t = template("E6", "llm-confirm")
    return Packet.model_validate(t, strict=False)


def test_confirm_template_preregisters_depth2_in_distribution() -> None:
    pk = confirm_packet()
    q = pk.params
    assert pk.version == 7 and q["depth"] == 2 and q["treatment"] == "d2_replay"
    assert q["baselines"] == ["d2_no_replay"] and q["eval_mode"] == "ground"
    assert all(f.endswith("@tune") for f in q["families"]) and q["model"] == "qwen2.5-7b"
    assert set(q["pinned_adapters"]) == {
        "d1_no_replay",
        "d2_no_replay",
        "d3_no_replay",
        "d1_replay",
        "d2_replay",
        "d3_replay",
        "real_only",
    }


def test_resolve_world_maps_lock_offsets_into_unseen_trained_worlds() -> None:
    from hsi.contracts.records import Split
    from hsi.environment import SEED_BASE
    from hsi.evaluation.llm_stack import _collides, resolve_world

    pk = confirm_packet()
    fam, split, seed = resolve_world("rule_induction@tune", Split.HELDOUT, 2_123_456, pk)
    assert (fam, split, seed) == ("rule_induction", Split.TUNE, 1_123_456)
    # plain families and non-held-out splits are untouched
    assert resolve_world("rule_induction", Split.HELDOUT, 2_123_456, pk)[1:] == (
        Split.HELDOUT,
        2_123_456,
    )
    assert resolve_world("rule_induction@tune", Split.FIT, 777, pk) == (
        "rule_induction",
        Split.FIT,
        777,
    )
    # an offset inside the training range is shifted out of it, deterministically
    for o in (0, 49, 120, 149, 500_000, 500_150, 700_050):
        assert _collides(o, pk)
        f2, s2, seed2 = resolve_world(
            "compositional_transfer@tune", Split.HELDOUT, SEED_BASE[Split.HELDOUT] + o, pk
        )
        assert s2 is Split.TUNE and not _collides(seed2 - SEED_BASE[Split.TUNE], pk)
        assert resolve_world(
            "compositional_transfer@tune", Split.HELDOUT, SEED_BASE[Split.HELDOUT] + o, pk
        ) == (f2, s2, seed2)


def test_depth_param_drives_guardrails_in_spec_and_validator() -> None:
    import importlib.util

    pk = confirm_packet()
    spec = E6Spec()
    spec.manifest = {"candidates": {a: {} for a in ARMS}, "diversity_floor": 0.0}
    sigs = {a: ["swap", "add", "copy", "mix"] for a in ARMS}
    rates = {
        "d1_no_replay": 0.2,
        "d1_replay": 0.3,
        "d2_replay": 0.3,
        "d2_no_replay": 0.2,
        "d3_replay": 0.0,
    }
    rs = results(rates, {**sigs})  # d3_replay collapsed, but depth=2 is the checked depth
    gv, held = spec.guardrails(rs, [], pk)
    assert held and "rate_d2_replay" in gv and "rate_d3_replay" not in gv

    vp = Path(__file__).resolve().parents[2] / "validator" / "recompute.py"
    sp = importlib.util.spec_from_file_location("recompute", vp)
    assert sp and sp.loader
    val = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(val)
    n = 100
    pairs = [
        {"family": f, "n": n, "both": 20, "only_treatment": 15, "only_baseline": 5, "neither": 60}
        for f in ("rule_induction@tune", "compositional_transfer@tune")
    ]
    fr = []
    for f in ("rule_induction@tune", "compositional_transfer@tune"):
        fr += [
            {"family": f, "arm": "d2_replay", "n": n, "successes": 35},
            {"family": f, "arm": "d2_no_replay", "n": n, "successes": 25},
            {"family": f, "arm": "d1_replay", "n": n, "successes": 36},
        ]
    _ds, mean, lo, hi = val.stats(
        pairs,
        [],
        {(r["family"], r["arm"]): r["successes"] for r in fr},
        "d2_no_replay",
        "d2_replay",
    )
    packet_body = pk.model_dump(mode="json")
    frozen = {"packet": packet_body, "packet_digest": val.digest(packet_body)}
    gvals = {
        "min_replay_diversity": 0.4,
        "diversity_floor": 0.1,
        "non_collapse_floor": 0.02,
        "rate_d1_replay": 0.36,
        "rate_d2_replay": 0.35,
    }
    rec = {
        "packet_digest": frozen["packet_digest"],
        "threshold": pk.threshold,
        "nonclaims": pk.nonclaims,
        "comparisons": [
            {
                "baseline_arm": "d2_no_replay",
                "treatment_arm": "d2_replay",
                "pairs": pairs,
                "paired_diff": mean,
                "ci_low": lo,
                "ci_high": hi,
            }
        ],
        "family_results": fr,
        "guardrail_values": gvals,
        "guardrails_held": True,
        "decision": "INCONCLUSIVE",
    }
    rec["receipt_digest"] = val.digest({k: v for k, v in rec.items() if k != "receipt_digest"})
    assert val.check(rec, frozen) == []  # depth=2 respected (a depth-3 lookup would KeyError)
    rec["guardrail_values"] = {**gvals, "rate_d2_replay": 0.5}
    rec["receipt_digest"] = val.digest({k: v for k, v in rec.items() if k != "receipt_digest"})
    assert any("replay rates differ" in e for e in val.check(rec, frozen))
