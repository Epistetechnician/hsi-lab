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

    def run_task(self, arm: str, family: str, split: Split, seed: int,
                 packet: Packet) -> TaskResult:
        if self._llm(packet):
            from hsi.evaluation.llm_stack import run_llm_e6_task
            return run_llm_e6_task(arm, family, split, seed, packet)
        assert self.stack is not None, "spec not bound"
        return run_prior_task(prior_for(self.stack, arm), arm, family, split, seed,
                              int(packet.params["support"]))

    def check_budget(self, r: TaskResult, packet: Packet) -> None:
        if r.stats["explore_calls"] != packet.params["support"]:
            raise BudgetExceeded(f"{r.arm}: exploration not matched")

    def guardrails(self, results: list[TaskResult], comps: list[Comparison],
                   packet: Packet) -> tuple[dict[str, float], bool]:
        assert self.manifest is not None
        man = self.manifest
        cands = man["candidates"]
        assert isinstance(cands, dict)
        gens = int(packet.params["generations"])
        min_div = min(float(cands[f"d{g}_replay"]["diversity"]) for g in range(1, gens + 1))

        def rate(arm: str) -> float:
            rs = [r for r in results if r.arm == arm]
            return sum(r.success for r in rs) / len(rs)

        floor = float(man["diversity_floor"])
        gv = {"min_replay_diversity": min_div, "diversity_floor": floor,
              "rate_d1_replay": rate("d1_replay"), f"rate_d{gens}_replay": rate(f"d{gens}_replay"),
              "non_collapse_floor": float(packet.params["non_collapse_floor"])}
        for arm in (f"d{gens}_replay", f"d{gens}_no_replay"):  # diagnostics (LLM generator only)
            for k in ("validity", "diversity"):
                if k in cands[arm]:
                    gv[f"{k}_{arm}"] = float(cands[arm][k])
        held = min_div >= floor and gv[f"rate_d{gens}_replay"] - gv["rate_d1_replay"] \
            >= -gv["non_collapse_floor"]
        return gv, held
