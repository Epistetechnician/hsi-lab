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
    "llm-poolctl", "llm-ood",
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
}


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
        if variant == "llm-verified":  # v8: does verifying the synthetic half rescue recursion?
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
        choices=["", "llm", *sorted(GEN_VARIANTS)],
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
