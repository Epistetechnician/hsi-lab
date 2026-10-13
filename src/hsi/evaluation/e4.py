"""E4: one reversible adapter update (LoRA / external memory) on a frozen base.

Candidates are trained on tune-split worlds only; the primary metric is held-out compositional
transfer; the guardrail is regression on fresh fit-split worlds.
"""
from __future__ import annotations

from pathlib import Path

from hsi.accounting import BudgetExceeded
from hsi.contracts.packet import Packet
from hsi.contracts.records import Comparison, Split
from hsi.environment import SEED_BASE
from hsi.evaluation.base import TaskResult
from hsi.evaluation.learned import Stack, load_stack, prepare_e4, prior_for, run_prior_task


class E4Spec:
    stack: Stack | None = None

    def prepare(self, packet: Packet, run_dir: Path) -> None:
        prepare_e4(packet, run_dir)

    def bind(self, packet: Packet, run_dir: Path) -> None:
        self.stack = load_stack(packet, run_dir)

    def run_task(self, arm: str, family: str, split: Split, seed: int,
                 packet: Packet) -> TaskResult:
        assert self.stack is not None, "spec not bound"
        pr = prior_for(self.stack, arm, float(packet.params["memory_alpha"]))
        return run_prior_task(pr, arm, family, split, seed, int(packet.params["support"]))

    def check_budget(self, r: TaskResult, packet: Packet) -> None:
        if r.stats["explore_calls"] != packet.params["support"]:
            raise BudgetExceeded(f"{r.arm}: exploration not matched")

    def guardrails(self, results: list[TaskResult], comps: list[Comparison],
                   packet: Packet) -> tuple[dict[str, float], bool]:
        q = packet.params
        n = int(q["fit_eval_n"])
        fams = list(q["pretrain_families"])
        succ = {}
        for arm in ("frozen_base", "lora_candidate"):
            succ[arm] = sum(self.run_task(arm, f, Split.FIT, SEED_BASE[Split.FIT] + 500_000 + i,
                                          packet).success for f in fams for i in range(n))
        gv = {"fit_n": float(n * len(fams)), "fit_success_base": float(succ["frozen_base"]),
              "fit_success_candidate": float(succ["lora_candidate"]),
              "fit_regression_floor": float(q["fit_regression_floor"]),
              "compute_matched_fraction": sum(
                  r.stats["explore_calls"] == q["support"] for r in results) / len(results)}
        diff = (gv["fit_success_candidate"] - gv["fit_success_base"]) / gv["fit_n"]
        return gv, diff >= -gv["fit_regression_floor"] and gv["compute_matched_fraction"] == 1.0
