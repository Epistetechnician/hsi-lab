"""E6: recursive synthesis (depth 1-3) with and without original replay.

Generation g trains an adapter on pseudo-labels produced by the generation g-1 model (unverified
self-training on the tune-world pool). Replay arms swap half of the synthetic data for original
labeled data (same total example count). Treatment d3_replay vs baseline d3_no_replay; guardrails:
generated-task diversity above the pilot floor at every depth, and d3_replay not below d1_replay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hsi.accounting import BudgetExceeded
from hsi.contracts.packet import Packet
from hsi.contracts.records import Comparison, Split
from hsi.evaluation.base import TaskResult
from hsi.evaluation.learned import Stack, load_stack, prepare_e6, prior_for, run_prior_task


def sig_entropy(sigs: list[str]) -> float:
    """Normalized Shannon entropy of hypothesis signatures (21 classes: 20 shapes + invalid)."""
    import math
    from collections import Counter

    n = len(sigs)
    return (
        -sum(c / n * math.log(c / n) for c in Counter(sigs).values()) / math.log(21) if n else 0.0
    )


class E6Spec:
    stack: Stack | None = None

    manifest: dict[str, Any] | None = None
    run_dir: Path | None = None

    @staticmethod
    def _llm(packet: Packet) -> bool:
        return packet.params.get("backbone") == "llm"

    def prepare(self, packet: Packet, run_dir: Path) -> None:
        if self._llm(packet):
            from hsi.evaluation.llm_stack import prepare_e6_llm

            prepare_e6_llm(packet, run_dir)
        else:
            prepare_e6(packet, run_dir)

    def bind(self, packet: Packet, run_dir: Path) -> None:
        if self._llm(packet):
            from hsi.evaluation.llm_stack import bind_e6_llm

            self.manifest, self.run_dir = bind_e6_llm(packet, run_dir), run_dir
        else:
            self.stack = load_stack(packet, run_dir)
            self.manifest, self.run_dir = self.stack.manifest, None

    def model_digest(self) -> str | None:
        if self.run_dir is None:
            return None
        from hsi.evaluation.llm_stack import model_digest_e6

        return model_digest_e6(self.run_dir)

    def run_task(
        self, arm: str, family: str, split: Split, seed: int, packet: Packet
    ) -> TaskResult:
        if self._llm(packet):
            from hsi.evaluation.llm_stack import run_llm_e6_task

            return run_llm_e6_task(arm, family, split, seed, packet)
        assert self.stack is not None, "spec not bound"
        return run_prior_task(
            prior_for(self.stack, arm), arm, family, split, seed, int(packet.params["support"])
        )

    def check_budget(self, r: TaskResult, packet: Packet) -> None:
        if r.stats["explore_calls"] != packet.params["support"]:
            raise BudgetExceeded(f"{r.arm}: exploration not matched")

    def guardrails(
        self, results: list[TaskResult], comps: list[Comparison], packet: Packet
    ) -> tuple[dict[str, float], bool]:
        assert self.manifest is not None
        man = self.manifest
        cands = man["candidates"]
        assert isinstance(cands, dict)
        gens = int(packet.params.get("depth", packet.params["generations"]))
        suf = str(packet.params.get("guard_suffix", "replay"))  # arm family the guardrails track
        ground = packet.params.get("eval_mode") in ("ground", "probe")
        if ground:  # diversity of the model's own outputs on held-out prompts, per arm
            div = {
                a: sig_entropy([str(d["sig"]) for r in results if r.arm == a for d in r.decisions])
                for a in sorted({r.arm for r in results})
            }
            min_div = min(div[f"d{g}_{suf}"] for g in range(1, gens + 1))
            # reference level: the depth-1 no-replay arm when evaluated, else the guarded chain's
            # own depth-1 arm (packets such as the pool-matched control omit d1_no_replay)
            ref = "d1_no_replay" if "d1_no_replay" in div else f"d1_{suf}"
            floor = float(packet.params["diversity_floor_frac"]) * div[ref]
        else:  # diversity of the texts used to train each generation (from the manifest)
            min_div = min(float(cands[f"d{g}_{suf}"]["diversity"]) for g in range(1, gens + 1))
            floor = float(man["diversity_floor"])

        def rate(arm: str) -> float:
            rs = [r for r in results if r.arm == arm]
            return sum(r.success for r in rs) / len(rs)

        gv = {
            "min_replay_diversity": min_div,
            "diversity_floor": floor,
            f"rate_d1_{suf}": rate(f"d1_{suf}"),
            f"rate_d{gens}_{suf}": rate(f"d{gens}_{suf}"),
            "non_collapse_floor": float(packet.params["non_collapse_floor"]),
        }
        for arm in (f"d{gens}_{suf}", f"d{gens}_no_replay"):  # diagnostics (LLM generator only)
            for k in ("validity", "diversity"):
                if k in cands.get(arm, {}):
                    gv[f"{k}_{arm}"] = float(cands[arm][k])
        for arm in sorted({r.arm for r in results}):  # eval-time generation quality (gen mode)
            rs = [r for r in results if r.arm == arm and "gen_seen" in r.stats]
            if rs:
                tot = sum(r.stats["gen_seen"] for r in rs)
                gv[f"eval_valid_{arm}"] = sum(r.stats["gen_valid"] for r in rs) / tot
                gv[f"eval_consistent_{arm}"] = sum(r.stats["gen_consistent"] for r in rs) / tot
                if "gen_exact" in rs[0].stats:
                    gv[f"eval_exact_{arm}"] = sum(r.stats["gen_exact"] for r in rs) / tot
                if "gen_probe" in rs[0].stats:
                    gv[f"eval_probe_{arm}"] = sum(r.stats["gen_probe"] for r in rs) / tot
                if ground:
                    gv[f"eval_diversity_{arm}"] = div[arm]
        held = (
            min_div >= floor
            and gv[f"rate_d{gens}_{suf}"] - gv[f"rate_d1_{suf}"] >= -gv["non_collapse_floor"]
        )
        return gv, held
