"""`hsi` command line.

M0: `packet new|freeze|check`. M1: `baselines`. M2/M3: `pilot|lock|run|eval|verify-receipt|decide`
(E1-E7). M6: `reproduce|bundle|assurance`. `watch`: terminal run viewer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from hsi import pipeline
from hsi.contracts.io import load_json
from hsi.contracts.packet import EXPERIMENTS, SCHEMA_VERSION, Packet, canonical_digest
from hsi.errors import HsiError

EXIT_INVALID, EXIT_DIGEST, EXIT_EXISTS = 2, 3, 4


E1_PARAMS = {
    "families": [
        "rule_induction",
        "compositional_transfer",
        "counterevidence",
        "relational_return",
    ],
    "trace_len": 40,
    "pilot_tasks": 30,
    "min_n": 30,
    "max_n": 400,
}


GEN_VARIANTS = {
    "llm-gen", "llm-gen2", "llm-gen3", "llm-ground", "llm-confirm", "llm-verified",
    "llm-poolctl", "llm-ood", "llm-torch", "llm-noisyver", "llm-dose", "llm-share",
}  # fmt: skip
TERMINATED = GEN_VARIANTS - {"llm-gen"}
SEVEN_B = GEN_VARIANTS - {"llm-gen", "llm-gen2"}
VERSIONS = {
    "llm": 2,
    "llm-gen": 3,
    "llm-gen2": 4,
    "llm-gen3": 5,
    "llm-ground": 6,
    "llm-confirm": 7,
    "llm-verified": 8,
    "llm-poolctl": 9,
    "llm-ood": 10,
    "llm-torch": 11,
    "llm-noisyver": 12,
    "llm-dose": 13,
    "llm-share": 14,
}


# ADR 0003 (v12). Verifier chains train to depth 2 on the same 600-world proposal pool as the
# earlier verified and pool-matched chains; each has its own training seed. tau is the 80th
# percentile of mean token log-probability over the audit proposals (docs/experiments/
# E6-noisyver-audit.md), fixed before the packet was frozen.
NOISYVER_CHAINS = [
    {"name": "noisy10", "kind": "noisy", "fpr": 0.10, "salt": "a", "train_seed_offset": 300},
    {"name": "noisy10b", "kind": "noisy", "fpr": 0.10, "salt": "b", "train_seed_offset": 400},
    {"name": "noisy30", "kind": "noisy", "fpr": 0.30, "salt": "a", "train_seed_offset": 500},
    {"name": "tested", "kind": "tested", "extra_probes": 2, "train_seed_offset": 600},
    {"name": "conf", "kind": "confidence", "tau": -0.108, "train_seed_offset": 700},
]
NOISYVER_ARMS = [
    "frozen_base", "real_only", "d2_replay",
    "d1_verified", "d2_verified", "d1_poolctl", "d2_poolctl",
    "d1_noisy10", "d2_noisy10", "d2_noisy10b", "d2_noisy30", "d2_tested", "d2_conf",
]  # fmt: skip
NOISYVER_SECONDARY = [  # exploratory; Bonferroni over 7; never decide the packet
    {"treatment": "d2_verified", "baseline": "d2_poolctl"},
    {"treatment": "d2_noisy30", "baseline": "d2_poolctl"},
    {"treatment": "d2_noisy10b", "baseline": "d2_poolctl"},
    {"treatment": "d2_tested", "baseline": "d2_verified"},
    {"treatment": "d2_conf", "baseline": "d2_poolctl"},
    {"treatment": "d2_verified", "baseline": "d2_noisy10"},
    {"treatment": "d2_noisy10b", "baseline": "d2_noisy10"},
]


# ADR 0008 (v13, E9). Precision dose-response at fixed kept-set size. `dose` chains use ground truth
# to set the probe-correct share p of the K visible-consistent proposals (an intervention, not a
# deployable verifier). K is fixed from docs/experiments/E9-supply-audit.json before the freeze.
DOSE_K = 40
DOSE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
DOSE_CHAINS = [
    {
        "name": f"dose{round(p * 100):03d}{rep}", "kind": "dose", "p": p, "k": DOSE_K,
        "salt": rep, "train_seed_offset": 800 + 100 * i + (50 if rep == "b" else 0),
    }
    for i, p in enumerate(DOSE_LEVELS)
    for rep in ("a", "b")
]  # fmt: skip
DOSE_ARMS = [
    "frozen_base", "real_only", "d2_verified", "d2_poolctl",
    "d1_dose100a", "d2_dose100a",  # d1 is listed for the guardrail on the guarded chain
    *[f"d2_{c['name']}" for c in DOSE_CHAINS if c["name"] != "dose100a"],
]  # fmt: skip
DOSE_SECONDARY = [  # exploratory; Bonferroni over 8; never decide the packet
    {"treatment": "d2_dose100b", "baseline": "d2_dose000b"},  # replication of the primary
    {"treatment": "d2_dose050a", "baseline": "d2_dose000a"},
    {"treatment": "d2_dose050b", "baseline": "d2_dose000b"},
    {"treatment": "d2_dose025a", "baseline": "d2_dose000a"},
    {"treatment": "d2_dose075a", "baseline": "d2_dose000a"},
    {"treatment": "d2_dose100a", "baseline": "d2_dose050a"},
    {"treatment": "d2_dose000a", "baseline": "d2_poolctl"},  # wrong-but-consistent vs unverified
    {"treatment": "d2_dose100a", "baseline": "d2_verified"},  # purely correct vs visible-verified
]


# ADR 0009 (v14, E10). Synthetic share of a fixed 350-example training set, visible verifier only
# (no ground truth anywhere in the arms). Quotas are counts out of 350: 0, 40, 88, 175, 262.
SHARE_LEVELS = {"s000": 0, "s011": 40, "s025": 88, "s050": 175, "s075": 262}
SHARE_CHAINS = [
    {
        "name": f"{lvl}{rep}", "kind": "visible", "n_synth": n, "salt": rep,
        "train_seed_offset": 1000 + 100 * i + (50 if rep == "b" else 0),
    }
    for i, (lvl, n) in enumerate(SHARE_LEVELS.items())
    for rep in ("a", "b")
]  # fmt: skip
SHARE_ARMS = [
    "frozen_base", "real_only", "d1_s050a",  # d1 is listed for the guardrail on the guarded chain
    *[f"d2_{c['name']}" for c in SHARE_CHAINS],
]  # fmt: skip
SHARE_SECONDARY = [  # exploratory; Bonferroni over 8; never decide the packet
    {"treatment": "d2_s050b", "baseline": "d2_s000b"},  # replication of the primary
    {"treatment": "d2_s011a", "baseline": "d2_s000a"},
    {"treatment": "d2_s025a", "baseline": "d2_s000a"},
    {"treatment": "d2_s075a", "baseline": "d2_s000a"},
    {"treatment": "d2_s075a", "baseline": "d2_s050a"},
    {"treatment": "d2_s000a", "baseline": "real_only"},  # retrained s=0 vs the pinned real_only
    {"treatment": "d2_s050a", "baseline": "real_only"},
    {"treatment": "d2_s000b", "baseline": "d2_s000a"},  # chain-to-chain variance at one setting
]


E8_CHAINS = [
    {"name": "none", "tier": "none", "seed_offset": 10},
    {"name": "ex", "tier": "ex", "seed_offset": 20},
    {"name": "base", "tier": "base", "seed_offset": 30},
    {"name": "plus", "tier": "plus", "seed_offset": 40},
    {"name": "plusb", "tier": "plus", "seed_offset": 50},
]
E8_ARMS = [
    "frozen_base",
    *[f"d{g}_{c['name']}" for c in E8_CHAINS for g in (1, 2)],
]
E8_SECONDARY = [  # exploratory; Bonferroni over 6; never decide the packet
    {"treatment": "d2_plus", "baseline": "d2_none"},
    {"treatment": "d2_plus", "baseline": "d2_base"},
    {"treatment": "d2_base", "baseline": "d2_ex"},
    {"treatment": "d2_ex", "baseline": "d2_none"},
    {"treatment": "d2_plus", "baseline": "frozen_base"},
    {"treatment": "d2_plusb", "baseline": "d2_ex"},
]
E8_PARAMS = {
    "model": "qwen2.5-1.5b",
    "families": ["mbpp_a", "mbpp_b", "mbpp_c"],
    "tune_frac": 0.4,
    "holdout_mod": 75,
    "pilot_tasks": 3,
    "min_n": 75,
    "max_n": 75,
    "depth": 2,
    "k": 8,
    "temp": 0.8,
    "per_task": 2,
    "quota": 150,
    "max_tokens": 320,
    "workers": 6,
    "rank": 8,
    "lora_layers": 8,
    "lora_scale": 20.0,
    "lr": 1e-4,
    "steps": 150,
    "batch": 8,
    "non_collapse_floor": 0.02,
    "guard_suffix": "plus",
    "chains": E8_CHAINS,
    "treatment": "d2_plus",
    "baselines": ["d2_ex"],
    "secondary": E8_SECONDARY,
}


E8R_CHAINS = [
    {"name": "none", "tier": "none", "seed_offset": 10, "replicates": 2},
    {"name": "ex", "tier": "ex", "seed_offset": 20, "replicates": 4},
    {"name": "plus", "tier": "plus", "seed_offset": 40, "replicates": 4},
]
E8R_ARMS = ["frozen_base", *[f"d{g}_{c['name']}" for c in E8R_CHAINS for g in (1, 2)]]
E8R_SECONDARY = [  # exploratory; Bonferroni over 4; never decide the packet
    {"treatment": "d2_plus", "baseline": "d2_none"},
    {"treatment": "d2_ex", "baseline": "d2_none"},
    {"treatment": "d2_plus", "baseline": "frozen_base"},
    {"treatment": "d1_plus", "baseline": "d1_ex"},
]
E8R_PARAMS = {
    **E8_PARAMS,
    "tune_frac": 0.3,
    "holdout_mod": 87,
    "min_n": 87,
    "max_n": 87,
    "k": 6,
    "quota": 100,
    "steps": 100,
    "lr": 5e-5,
    "chains": E8R_CHAINS,
    "secondary": E8R_SECONDARY,
}


E8_7B_CHAINS = [
    {"name": "none", "tier": "none", "seed_offset": 10, "replicates": 1},
    {"name": "ex", "tier": "ex", "seed_offset": 20, "replicates": 3},
    {"name": "plus", "tier": "plus", "seed_offset": 40, "replicates": 3},
]
E8_7B_ARMS = ["frozen_base", *[f"d{g}_{c['name']}" for c in E8_7B_CHAINS for g in (1, 2)]]


E8_V2_RUN = "d55611f54cc09b94"  # the e8-v2 run whose 20 adapters v3 evaluates on HumanEval+


def _e8_v2_pins() -> dict[str, str]:
    from hsi.paths import run_root

    man = json.loads((run_root() / E8_V2_RUN / "learning" / "manifest.json").read_text("utf-8"))
    return {a: str(c["digest"]) for a, c in man["candidates"].items()}


def template(exp: str, variant: str = "") -> dict[str, object]:
    t: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": exp,
        "version": 1,
        "hypothesis": "TODO",
        "null_hypothesis": "TODO",
        "arms": ["baseline", "treatment"],
        "primary_metric": "held_out_success_rate",
        "threshold": 0.05,
        "guardrails": ["TODO"],
        "budget": {
            "max_tokens": 0,
            "max_calls": 0,
            "max_wall_seconds": 3600,
            "max_message_bytes": 0,
        },
        "seeds": [0],
        "stop_rules": ["any digest, schema, split, or budget mismatch"],
        "claim_ceiling": "LocalSyntheticExperimentOnly",
        "nonclaims": ["no general intelligence claim", "not a proof"],
        "params": {},
    }
    if exp == "E1":
        t.update(
            hypothesis="The typed-event codec yields higher held-out success than the text codec "
            "at an equal byte budget.",
            null_hypothesis=(
                "Held-out success difference (typed_event - text) is below the threshold."
            ),
            arms=["text", "typed_event"],
            primary_metric="heldout_success_diff_typed_minus_text",
            guardrails=["lossless_roundtrip", "payload_within_byte_budget"],
            params=E1_PARAMS,
        )
        t["budget"] = {
            "max_tokens": 0,
            "max_calls": 10_000_000,
            "max_wall_seconds": 3600,
            "max_message_bytes": 200,
        }
    if exp == "E2":
        t.update(
            hypothesis="Oracle-verified synthetic examples give higher held-out success than "
            "unverified synthetic and real-only examples at equal example count.",
            null_hypothesis="Verified synthetic does not beat both baselines by the threshold.",
            arms=["real_only", "unverified_synthetic", "verified_synthetic"],
            primary_metric="heldout_success_diff_verified_minus_baseline",
            guardrails=[
                "no family regresses vs real_only beyond the floor",
                "equal example count",
                "full lineage",
            ],
            params={
                **E1_PARAMS,
                "trace_len": 12,
                "examples": 12,
                "real_seed": 6,
                "max_proposals": 60,
                "family_regression_floor": 0.02,
                "treatment": "verified_synthetic",
                "baselines": ["real_only", "unverified_synthetic"],
            },
        )
        t["budget"] = {
            "max_tokens": 0,
            "max_calls": 10_000_000,
            "max_wall_seconds": 3600,
            "max_message_bytes": 0,
        }
    if exp == "E3":
        t.update(
            hypothesis="A fixed-graph collective of two agents with complementary action rights "
            "discovers more than one agent or two non-communicating agents at matched "
            "exploration and a message-byte cap.",
            null_hypothesis="Fixed-graph success exceeds neither baseline by the threshold.",
            arms=["single", "independent", "centralized", "fixed_graph"],
            primary_metric="collective_discovery_success_diff_fixed_graph_minus_baseline",
            guardrails=["message bytes within cap", "exploration calls matched"],
            params={
                "families": ["collective_discovery"],
                "explore_total": 12,
                "pilot_tasks": 30,
                "min_n": 30,
                "max_n": 400,
                "treatment": "fixed_graph",
                "baselines": ["single", "independent"],
            },
        )
        t["budget"] = {
            "max_tokens": 0,
            "max_calls": 10_000_000,
            "max_wall_seconds": 3600,
            "max_message_bytes": 24,
        }
    if exp == "E8":
        t.update(
            hypothesis="When a code model trains on its own sampled solutions, filtering them with "
            "the strong EvalPlus tests beats filtering with only the single example assertion "
            "shown in the prompt: at recursion depth 2, greedy pass@1 on held-out MBPP+ tasks "
            "under the strong tests is higher for the plus-filtered chain than for the "
            "example-filtered one (same sampling budget, same example quota).",
            null_hypothesis="d2 plus-filtered does not beat d2 example-filtered by the threshold.",
            arms=E8R_ARMS if variant in ("e8-rep", "e8-he", "e8-7b") else E8_ARMS,
            primary_metric="mbppplus_pass_at_1_diff_d2_plus_minus_d2_ex",
            threshold=0.03,
            guardrails=[
                "d2_plus not below d1_plus beyond the floor",
                "held-out tasks never used by any chain",
            ],
            params=E8R_PARAMS if variant in ("e8-rep", "e8-he", "e8-7b") else E8_PARAMS,
            version={"e8-rep": 2, "e8-he": 3, "e8-7b": 4}.get(variant, 1),
        )
        if variant == "e8-7b":  # v4: ADR 0007, the same question on Qwen2.5-7B-Instruct bf16 (CUDA)
            t["hypothesis"] = (
                "On a competent 7B generator, at recursion depth 2, greedy pass@1 under the "
                "EvalPlus tests on held-out MBPP+ and HumanEval+ tasks is higher for chains that "
                "filter their own solutions with the strong tests than for chains that filter "
                "with only the one example assertion shown in the prompt (same sampling budget and "
                "example quota)."
            )
            t["params"] = {
                **E8R_PARAMS,
                "model": "qwen2.5-7b-hf",
                "families": ["mbpp_a", "mbpp_b", "mbpp_c", "he_a", "he_b"],
                "eval_pools": {"mbpp": ["mbpp_a", "mbpp_b", "mbpp_c"], "he": ["he_a", "he_b"]},
                "tune_families": ["mbpp_a", "mbpp_b", "mbpp_c", "mbpp_a", "mbpp_b"],
                "holdout_mod": 81,
                "min_n": 81,
                "max_n": 81,
                "gen_batch": 64,
                "workers": 8,
                "chains": E8_7B_CHAINS,
            }
            t["arms"] = E8_7B_ARMS
            t["claim_ceiling"] = "LocalSmallModelCodeBenchmarkOnly"
            t["nonclaims"] = [
                "no general intelligence claim",
                "one 7B model, two benchmarks (MBPP+, HumanEval+), possibly seen in pretraining",
                "PyTorch/CUDA runtime: a replication stack, not a digest-level reproduction of MLX",
            ]
        if variant == "e8-he":  # v3: the v2 adapters, scored on a benchmark they never saw
            t["hypothesis"] = (
                "On HumanEval+ (163 tasks no chain trained on or was tuned against), greedy pass@1 "
                "under the EvalPlus tests is higher at depth 2 for the plus-filtered replicate "
                "chains of e8-v2 than for the example-filtered ones."
            )
            t["primary_metric"] = "humanevalplus_pass_at_1_diff_d2_plus_minus_d2_ex"
            t["params"] = {
                **E8R_PARAMS,
                "eval_pool": "humaneval",
                "families": ["he_a", "he_b", "he_c"],
                "tune_families": ["mbpp_a", "mbpp_b", "mbpp_c"],
                "holdout_mod": 54,
                "min_n": 54,
                "max_n": 54,
                "reuse_from": E8_V2_RUN,
                "pinned_adapters": _e8_v2_pins(),
                "secondary": E8R_SECONDARY,
            }
            t["nonclaims"] = [
                "no general intelligence claim",
                "one 1.5B model; adapters trained on MBPP+ only, scored on HumanEval+",
                "HumanEval+ may have been seen in pretraining; the held-out pool is public",
            ]
        t["claim_ceiling"] = "LocalSmallModelCodeBenchmarkOnly"
        t["nonclaims"] = [
            "no general intelligence claim",
            "one 1.5B model, one benchmark (MBPP+), possibly contaminated in pretraining",
            "held-out membership is public by design; the lock fixes order only",
        ]
        t["budget"] = {
            "max_tokens": 0,
            "max_calls": 10_000_000,
            "max_wall_seconds": 3600,
            "max_message_bytes": 0,
        }
    if exp == "E7":
        t.update(
            hypothesis="Adaptive relations (dormant-over-delete, reactivation, counterevidence "
            "retirement) improve success on relational return and counterevidence "
            "streams over no relations and static relations at matched exploration.",
            null_hypothesis="Adaptive relations beat neither baseline by the threshold.",
            arms=["no_relations", "static_relations", "adaptive_relations"],
            primary_metric="relational_return_and_counterevidence_success_diff",
            guardrails=["shared-source evidence counted once", "exploration calls matched"],
            params={
                "families": ["relational_return", "counterevidence"],
                "explore_full": 12,
                "explore_final": 4,
                "relation_cap": 2,
                "probes": 2,
                "pilot_tasks": 30,
                "min_n": 30,
                "max_n": 400,
                "treatment": "adaptive_relations",
                "baselines": ["no_relations", "static_relations"],
            },
        )
        t["budget"] = {
            "max_tokens": 0,
            "max_calls": 10_000_000,
            "max_wall_seconds": 3600,
            "max_message_bytes": 0,
        }
    if exp == "E5":
        t.update(
            hypothesis="With four agents that each own one action, an adaptive graph (links opened "
            "on demand, dropped after use) reaches more successes than a fixed path "
            "graph with delta-sync gossip under the same per-task byte cap.",
            null_hypothesis="Adaptive-graph success does not exceed fixed-graph success by the "
            "threshold at the byte cap.",
            arms=["fixed_graph", "adaptive_graph"],
            primary_metric="success_within_byte_cap_diff_adaptive_minus_fixed",
            guardrails=[
                "success per KB of adaptive >= fixed",
                "message bytes within cap",
                "exploration matched",
                "link churn reported",
            ],
            params={
                "families": ["rule_induction", "compositional_transfer"],
                "explore_per_agent": 6,
                "rounds": 3,
                "pilot_tasks": 30,
                "min_n": 30,
                "max_n": 400,
                "treatment": "adaptive_graph",
                "baselines": ["fixed_graph"],
            },
        )
        t["budget"] = {
            "max_tokens": 0,
            "max_calls": 10_000_000,
            "max_wall_seconds": 3600,
            "max_message_bytes": 72,
        }
    if exp in ("E4", "E6"):
        common = {
            "pretrain_families": ["rule_induction", "compositional_transfer"],
            "support": 8,
            "n_pretrain": 80,
            "n_adapt": 60,
            "pretrain_steps": 300,
            "pretrain_lr": 0.1,
            "adapt_steps": 200,
            "adapt_lr": 0.2,
            "rank": 2,
            "pilot_tasks": 30,
            "min_n": 30,
            "max_n": 400,
            "tune_offset": 500_000,
        }
    if exp == "E4":
        t.update(
            hypothesis="One reversible adapter update (LoRA) trained on tune-split worlds improves "
            "held-out compositional transfer over the frozen base without regressing "
            "fit families beyond 2 points.",
            null_hypothesis="The LoRA candidate does not beat the frozen base by the threshold.",
            arms=["frozen_base", "lora_candidate", "memory_candidate"],
            primary_metric="compositional_transfer_success_diff_lora_minus_base",
            guardrails=["fit-family regression within floor", "exploration calls matched"],
            params={
                **common,
                "families": ["compositional_transfer"],
                "memory_alpha": 1.0,
                "fit_eval_n": 100,
                "fit_regression_floor": 0.02,
                "treatment": "lora_candidate",
                "baselines": ["frozen_base"],
            },
        )
    if exp == "E6":
        t.update(
            hypothesis="Original replay keeps recursive self-training from collapsing: at depth 3, "
            "training with replay beats training on synthetic data alone.",
            null_hypothesis="d3 with replay does not beat d3 without replay by the threshold.",
            arms=[
                "d1_replay",
                "d1_no_replay",
                "d2_replay",
                "d2_no_replay",
                "d3_replay",
                "d3_no_replay",
            ],
            primary_metric="success_diff_d3_replay_minus_d3_no_replay",
            guardrails=[
                "generated-task diversity above pilot floor at every depth",
                "d3_replay not below d1_replay beyond floor",
            ],
            params={
                **common,
                "families": ["rule_induction", "compositional_transfer"],
                "generations": 3,
                "diversity_floor_frac": 0.5,
                "non_collapse_floor": 0.02,
                "treatment": "d3_replay",
                "baselines": ["d3_no_replay"],
            },
        )
    if exp == "E4" and variant == "llm":
        p = dict(t["params"])  # type: ignore[call-overload]
        for k in ("pretrain_steps", "pretrain_lr", "n_pretrain", "memory_alpha"):
            p.pop(k, None)
        p.update(
            backbone="llm",
            n_adapt=60,
            n_replay=60,
            replay_frac=0.5,
            rank=8,
            lora_layers=8,
            lora_scale=20.0,
            adapt_steps=300,
            adapt_lr=1e-4,
            adapt_batch=8,
            pilot_tasks=20,
            min_n=30,
            max_n=200,
            fit_eval_n=60,
        )
        t.update(
            version=2,
            arms=["frozen_base", "lora_candidate"],
            hypothesis="A rank-8 LoRA adapter on a frozen Qwen2.5-1.5B-Instruct (4-bit), "
            "trained on "
            "tune-split worlds with 50% fit-world replay, improves held-out "
            "compositional transfer over the frozen model without regressing fit "
            "families beyond 2 points.",
            params=p,
        )
    if exp == "E6" and (variant == "llm" or variant in GEN_VARIANTS):
        p = dict(t["params"])  # type: ignore[call-overload]
        for k in ("pretrain_steps", "pretrain_lr", "n_pretrain"):
            p.pop(k, None)
        p.update(
            backbone="llm",
            n_adapt=50,
            n_replay=50,
            rank=8,
            lora_layers=8,
            lora_scale=20.0,
            adapt_steps=200,
            adapt_lr=1e-4,
            adapt_batch=8,
            gen_max_tokens=20,
            n_demos=2,
            pilot_tasks=15,
            min_n=30,
            max_n=100,
        )
        if variant in GEN_VARIANTS:
            p.update(eval_mode="generate", extra_arms=True, diversity_floor_frac=0.0)
        if variant in TERMINATED:  # v4+: stop marker (v3 never terminated)
            p.update(target_terminator="\n", adapt_steps=300)
        if variant in SEVEN_B:  # v5+: 7B generator, original data includes tune-type worlds
            p.update(model="qwen2.5-7b", n_replay_tune=50)
        if variant in GEN_VARIANTS:
            t["arms"] = [*t["arms"], "real_only", "frozen_base"]  # type: ignore[misc]
        t.update(
            version=VERSIONS[variant],
            hypothesis="With an unconstrained free-text generator (Qwen2.5-1.5B-Instruct), "
            "training "
            "each generation on the previous generation's own hypotheses collapses "
            "held-out success at depth 3 unless original labeled data is replayed: "
            "d3 with replay beats d3 without.",
            params=p,
        )
        if variant in GEN_VARIANTS:
            t["hypothesis"] = (
                "When held-out success is measured by the model's own free-text hypotheses (no "
                "consistency filter), recursive self-training degrades success by depth 3 unless "
                "original labeled data is replayed: d3 with replay beats d3 without."
            )
        if variant in GEN_VARIANTS - {
            "llm-gen",
            "llm-gen2",
            "llm-gen3",
        }:  # v6+: generator-level metric
            from hsi.paths import run_root

            src = "5ace07853340f87d"
            man = json.loads((run_root() / src / "learning" / "manifest.json").read_text())
            p.update(
                eval_mode="ground", reuse_from=src, min_n=100, max_n=400, pilot_tasks=30,
                diversity_floor_frac=0.5,
                pinned_adapters={a: c["digest"] for a, c in man["candidates"].items()},
            )  # fmt: skip
            t["primary_metric"] = "grounded_generation_rate_diff_d3_replay_minus_d3_no_replay"
            t["hypothesis"] = (
                "Measured directly on the generator (valid AND evidence-consistent hypothesis per "
                "held-out prompt, no planning), recursive self-training degrades generation "
                "quality by depth 3 unless original labeled data is replayed: d3 with replay "
                "beats d3 without."
            )
        if (
            variant == "llm-confirm"
        ):  # v7: confirmatory depth-2 contrast on unseen in-distribution worlds
            p.update(
                families=["rule_induction@tune", "compositional_transfer@tune"],
                depth=2, treatment="d2_replay", baselines=["d2_no_replay"],
            )  # fmt: skip
            t["primary_metric"] = "grounded_generation_rate_diff_d2_replay_minus_d2_no_replay"
            t["hypothesis"] = (
                "Confirmatory (pre-registered after the exploratory depth-2 signal in E6 v6): on "
                "unseen worlds of trained rule families, at recursion depth 2 the generator "
                "trained with original-data replay produces more valid, evidence-consistent "
                "hypotheses than the one trained on its own outputs alone: d2 replay beats d2 "
                "no-replay."
            )
            t["null_hypothesis"] = "d2 replay does not beat d2 no-replay by the threshold."
        if variant in (
            "llm-verified",
            "llm-torch",
        ):  # v8: does verifying the synthetic half rescue recursion?
            p.update(
                families=["rule_induction@tune", "compositional_transfer@tune"],
                depth=2, treatment="d2_verified", baselines=["d2_replay"],
                verified_chain=True, verify_pool_base=1000, verify_pool_worlds=600,
                guard_suffix="verified", avoid=[[1000, 1600]],
            )  # fmt: skip
            t["arms"] = [*t["arms"], "d1_verified", "d2_verified", "d3_verified"]  # type: ignore[misc]
            t["primary_metric"] = "grounded_generation_rate_diff_d2_verified_minus_d2_replay"
            t["hypothesis"] = (
                "Keeping only verified synthetic examples (a valid hypothesis whose re-execution "
                "reproduces the observed transitions), with the same original-data half and the "
                "same example count, yields more grounded generation at recursion depth 2 on "
                "unseen in-distribution worlds than keeping the unverified synthetic half: d2 "
                "verified beats d2 replay."
            )
            t["null_hypothesis"] = "d2 verified does not beat d2 replay by the threshold."
        if variant == "llm-torch":  # v11: v8 re-run on the PyTorch/CUDA runtime, fresh lock
            for k in ("reuse_from", "pinned_adapters"):
                p.pop(k, None)
            p["model"] = "qwen2.5-7b-hf"
            t["hypothesis"] = (
                "Replication on a different stack (PyTorch/CUDA, full-precision Qwen2.5-7B-"
                "Instruct, fresh held-out lock): " + str(t["hypothesis"])
            )
        if variant in ("llm-poolctl", "llm-ood"):  # v9/v10: follow-ups that reuse the v8 adapters
            from hsi.paths import run_root

            v8 = "f55ccd542e0e2ad4"
            m8 = json.loads((run_root() / v8 / "learning" / "manifest.json").read_text())
            p.update(
                reuse_from=v8, depth=2, treatment="d2_verified", verified_chain=True,
                verify_pool_base=1000, verify_pool_worlds=600, guard_suffix="verified",
                avoid=[[1000, 1600]],
                pinned_adapters={a: c["digest"] for a, c in m8["candidates"].items()},
            )  # fmt: skip
        if variant == "llm-poolctl":  # v9: pool-matched unverified control (in-distribution)
            p.update(
                families=["rule_induction@tune", "compositional_transfer@tune"],
                baselines=["d2_poolctl"], control_chain=True,
            )  # fmt: skip
            t["arms"] = [
                "d1_verified", "d2_verified", "d3_verified",
                "d1_poolctl", "d2_poolctl", "d3_poolctl",
                "d2_replay", "real_only", "frozen_base",
            ]  # fmt: skip
            t["primary_metric"] = "grounded_generation_rate_diff_d2_verified_minus_d2_poolctl"
            t["hypothesis"] = (
                "Pool-matched control for the verified-synthetic result: when the unverified "
                "synthetic examples are drawn from the same 600-world proposal pool and paired "
                "with the same original-data half and example count, keeping only verified "
                "examples still yields more grounded generation at depth 2 on unseen "
                "in-distribution worlds: d2 verified beats d2 pool-matched unverified."
            )
            t["null_hypothesis"] = (
                "d2 verified does not beat d2 pool-matched unverified by the threshold."
            )
        if variant == "llm-ood":  # v10: the verified chain on held-out rule TYPES
            p.update(families=["rule_induction", "compositional_transfer"], baselines=["d2_replay"])
            t["arms"] = [
                "d1_verified", "d2_verified", "d3_verified",
                "d1_replay", "d2_replay", "d3_replay",
                "d1_no_replay", "d2_no_replay", "d3_no_replay",
                "real_only", "frozen_base",
            ]  # fmt: skip
            t["primary_metric"] = (
                "grounded_generation_rate_diff_d2_verified_minus_d2_replay_heldout_types"
            )
            t["hypothesis"] = (
                "The advantage of verified over unverified synthetic examples at recursion depth 2 "
                "carries over to held-out rule types (mix, swap+copy) that never appeared in "
                "training: d2 verified beats d2 replay on held-out types."
            )
            t["null_hypothesis"] = (
                "d2 verified does not beat d2 replay on held-out types by the threshold."
            )
        if variant == "llm-noisyver":  # v12: ADR 0003, imperfect verifiers + a non-circular metric
            from hsi.paths import run_root

            v9 = "fda4e460ea88bf7b"  # holds every earlier adapter: replay, verified, poolctl
            m9 = json.loads((run_root() / v9 / "learning" / "manifest.json").read_text())
            p.update(
                eval_mode="probe", reuse_from=v9, depth=2, verifier_depth=2,
                families=["rule_induction@tune", "compositional_transfer@tune"],
                verified_chain=True, control_chain=True,
                verify_pool_base=1000, verify_pool_worlds=600,
                audit_base=1600, audit_worlds=300, avoid=[[1000, 1600], [1600, 1900]],
                min_n=100, max_n=400, pilot_tasks=30, diversity_floor_frac=0.5,
                guard_suffix="noisy10",
                pinned_adapters={a: c["digest"] for a, c in m9["candidates"].items()},
                verifier_chains=NOISYVER_CHAINS,
                treatment="d2_noisy10", baselines=["d2_poolctl"],
                secondary=NOISYVER_SECONDARY,
            )  # fmt: skip
            t["arms"] = NOISYVER_ARMS
            t["primary_metric"] = "probe_success_rate_diff_d2_noisy10_minus_d2_poolctl"
            t["hypothesis"] = (
                "A verifier that wrongly accepts 10% of the proposals that contradict the "
                "observed evidence still earns its keep: at recursion depth 2, on unseen "
                "in-distribution worlds, scored on fresh probe states that no verifier saw, a "
                "generator trained on the noisy-verified half beats one trained on the "
                "pool-matched unverified half (same original-data half, same example count)."
            )
            t["null_hypothesis"] = (
                "d2 with a 10%-false-accept verifier does not beat the pool-matched unverified "
                "control by the threshold."
            )
        if (
            variant == "llm-dose"
        ):  # v13: ADR 0008, fixed-size kept set, precision set by intervention
            from hsi.paths import run_root

            v9 = "fda4e460ea88bf7b"
            m9 = json.loads((run_root() / v9 / "learning" / "manifest.json").read_text())
            p.update(
                eval_mode="probe", reuse_from=v9, depth=2, verifier_depth=2,
                families=["rule_induction@tune", "compositional_transfer@tune"],
                verified_chain=True, control_chain=True,
                verify_pool_base=1000, verify_pool_worlds=600,
                audit_base=1600, audit_worlds=300, avoid=[[1000, 1600], [1600, 1900]],
                min_n=100, max_n=400, pilot_tasks=30, diversity_floor_frac=0.5,
                guard_suffix="dose100a",
                pinned_adapters={a: c["digest"] for a, c in m9["candidates"].items()},
                verifier_chains=DOSE_CHAINS,
                treatment="d2_dose100a", baselines=["d2_dose000a"],
                secondary=DOSE_SECONDARY,
            )  # fmt: skip
            t["arms"] = DOSE_ARMS
            t["threshold"] = 0.10
            t["primary_metric"] = "probe_success_rate_diff_d2_dose100a_minus_d2_dose000a"
            t["hypothesis"] = (
                "Verifier precision pays at constant kept-set size: with the number of synthetic "
                f"examples fixed at {DOSE_K} and the same original-data top-up, a generator "
                "trained at depth 2 on a kept set that is entirely probe-correct beats one "
                "trained on a kept "
                "set of equally consistent proposals that are all wrong, on fresh probe states in "
                "unseen in-distribution worlds."
            )
            t["null_hypothesis"] = (
                "d2 trained on an all-correct kept set does not beat d2 trained on an all-wrong "
                "(visible-consistent) kept set of the same size by the threshold."
            )
        if variant == "llm-share":  # v14: ADR 0009, synthetic share of a fixed total
            from hsi.paths import run_root

            v9 = "fda4e460ea88bf7b"
            m9 = json.loads((run_root() / v9 / "learning" / "manifest.json").read_text())
            p.update(
                eval_mode="probe", reuse_from=v9, depth=2, verifier_depth=2,
                families=["rule_induction@tune", "compositional_transfer@tune"],
                verified_chain=True, control_chain=True,
                verify_pool_base=1000, verify_pool_worlds=600,
                audit_base=1600, audit_worlds=300, avoid=[[1000, 1600], [1600, 1900]],
                min_n=300, max_n=600, pilot_tasks=30, diversity_floor_frac=0.5,
                guard_suffix="s050a",
                pinned_adapters={a: c["digest"] for a, c in m9["candidates"].items()},
                verifier_chains=SHARE_CHAINS,
                treatment="d2_s050a", baselines=["d2_s000a"],
                secondary=SHARE_SECONDARY,
            )  # fmt: skip
            t["arms"] = SHARE_ARMS
            t["threshold"] = 0.05
            t["primary_metric"] = "probe_success_rate_diff_d2_s050a_minus_d2_s000a"
            t["hypothesis"] = (
                "At a fixed total of 350 training examples, replacing half of the original "
                "labelled data with visible-verified synthetic examples improves depth-2 "
                "generation on fresh "
                "probe states in unseen in-distribution worlds: d2 at 50% synthetic beats d2 at 0% "
                "synthetic (a retrained all-real chain)."
            )
            t["null_hypothesis"] = (
                "d2 at 50% synthetic does not beat the all-real chain by the threshold."
            )
    if exp in ("E4", "E6"):
        t["budget"] = {
            "max_tokens": 0,
            "max_calls": 10_000_000,
            "max_wall_seconds": 3600,
            "max_message_bytes": 0,
        }
    return t


def freeze(src: Path) -> Path:
    packet = Packet.model_validate(load_json(src), strict=False)
    body = packet.model_dump(mode="json")
    frozen = {"packet": body, "packet_digest": canonical_digest(body)}
    dst = src.with_suffix(".frozen.json")
    if dst.exists():
        raise FileExistsError(str(dst))
    dst.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", "utf-8")
    dst.chmod(0o444)
    return dst


def check(path: Path) -> str:
    data = load_json(path)
    if set(data) != {"packet", "packet_digest"}:
        raise ValueError("frozen packet must have exactly packet and packet_digest")
    Packet.model_validate(data["packet"], strict=False)
    if canonical_digest(data["packet"]) != data["packet_digest"]:
        raise RuntimeError("digest mismatch")
    return str(data["packet_digest"])


def baselines(out: Path, n: int) -> Path:
    import random

    from hsi.accounting import Meter
    from hsi.agents import POLICIES, rollout
    from hsi.contracts.records import Split
    from hsi.environment import SEED_BASE, TASK_FAMILIES, RuleWorld
    from hsi.experience import Ledger
    from hsi.paths import refuse_in_repo

    out = refuse_in_repo(out)
    led, meter, table = Ledger(out / "baselines.ledger.jsonl"), Meter(), {}
    for name, pol in POLICIES.items():
        for fam in TASK_FAMILIES:
            for split in (Split.FIT, Split.TUNE):  # held-out is never touched before lock
                wins = 0
                for i in range(n):
                    w = RuleWorld(fam, split, SEED_BASE[split] + i)
                    ep = rollout(w, pol(w, random.Random(i)))
                    led.append(ep.model_copy(update={"episode_id": f"{name}-{ep.episode_id}"}))
                    meter.charge(calls=len(ep.steps))
                    wins += ep.outcome.success
                table[f"{name}/{fam}/{split.value}"] = wins / n
    dst = out / "baselines.json"
    dst.write_text(
        json.dumps(
            {"success": table, "resources": meter.snapshot(), "ledger_head": led.head},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "utf-8",
    )
    return dst


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hsi")
    sub = p.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("packet").add_subparsers(dest="action", required=True)
    n = pk.add_parser("new")
    n.add_argument("experiment", choices=sorted(EXPERIMENTS))
    n.add_argument("out", type=Path)
    n.add_argument(
        "--variant",
        default="",
        choices=["", "llm", "e8-rep", "e8-he", "e8-7b", *sorted(GEN_VARIANTS)],
    )
    pk.add_parser("freeze").add_argument("file", type=Path)
    pk.add_parser("check").add_argument("file", type=Path)
    b = sub.add_parser("baselines")
    b.add_argument("out", type=Path)
    b.add_argument("-n", type=int, default=50)
    for cmd in ("pilot", "lock", "run", "eval", "verify-receipt"):
        sub.add_parser(cmd).add_argument("path", type=Path)
    rp = sub.add_parser("reproduce")
    rp.add_argument("path", type=Path)
    rp.add_argument("--orig-run", type=Path, required=True)
    rp.add_argument("--orig-custody", type=Path, required=True)
    rp.add_argument("--out", type=Path)
    rp.add_argument("--at-commit", help="reproduce at this commit instead of source_commit")
    rp.add_argument(
        "--fetch",
        action="store_true",
        help="allow network for the hash-pinned dependency install only",
    )
    bd = sub.add_parser("bundle")
    bd.add_argument("path", type=Path)
    bd.add_argument("out", type=Path)
    wt = sub.add_parser("watch", help="live read-only terminal view of a run (or list runs)")
    wt.add_argument("path", type=Path, nargs="?")
    wt.add_argument("--once", action="store_true", help="print a text snapshot and exit")
    wt.add_argument(
        "--tab",
        default="all",
        choices=["all", "overview", "mechanism", "data", "receipt", "events"],
    )
    wt.add_argument("--width", type=int, default=100)
    wt.add_argument("--log", type=Path, help="also tail this log file in the events tab")
    asr = sub.add_parser("assurance")
    asr.add_argument("--write", type=Path)
    dc = sub.add_parser("decide")
    dc.add_argument("path", type=Path)
    dc.add_argument("outcome")
    a = p.parse_args(argv)
    try:
        if a.cmd == "baselines":
            print(baselines(a.out, a.n))
            return 0
        if a.cmd == "watch":
            from hsi.paths import run_root
            from hsi.watch import watch

            return watch(
                (a.path or run_root()).expanduser().resolve(), a.once, a.tab, a.width, a.log
            )
        if a.cmd == "reproduce":
            from hsi.assurance import reproduce

            rep = reproduce(a.path, a.orig_run, a.orig_custody, a.out, a.fetch, a.at_commit)
            print(
                f"{rep['experiment']}: matched {rep['aggregates_matched']}/"
                f"{rep['aggregates_compared']} aggregates; validator "
                f"{'ok' if rep['rerun_validator_ok'] else 'FAILED'}"
            )
            return 0 if rep["match"] and rep["rerun_validator_ok"] else 10
        if a.cmd == "bundle":
            from hsi.assurance import bundle

            print(bundle(a.path, a.out))
            return 0
        if a.cmd == "assurance":
            from hsi.assurance import assess, render

            text = render(assess())
            if a.write:
                a.write.write_text(text, "utf-8")
            print(text)
            return 0
        if a.cmd != "packet":
            fn = {
                "pilot": pipeline.pilot,
                "lock": pipeline.lock,
                "run": pipeline.run,
                "eval": pipeline.evaluate,
                "verify-receipt": pipeline.verify_receipt,
            }
            print(pipeline.decide(a.path, a.outcome) if a.cmd == "decide" else fn[a.cmd](a.path))
            return 0
        if a.action == "new":
            if a.out.exists():
                raise FileExistsError(str(a.out))
            a.out.write_text(
                json.dumps(template(a.experiment, a.variant), indent=2) + "\n", "utf-8"
            )
            print(a.out)
        elif a.action == "freeze":
            print(freeze(a.file))
        else:
            print(check(a.file))
        return 0
    except HsiError as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return e.code
    except FileExistsError as e:
        print(f"error: exists: {e}", file=sys.stderr)
        return EXIT_EXISTS
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_DIGEST
    except (ValueError, ValidationError) as e:
        print(f"error: invalid: {e}", file=sys.stderr)
        return EXIT_INVALID
