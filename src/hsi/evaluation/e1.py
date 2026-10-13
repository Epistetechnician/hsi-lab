"""E1 arm execution: codec-compressed teacher trace -> inductive learner -> plan -> rollout."""
from __future__ import annotations

import random

from hsi.accounting import BudgetExceeded
from hsi.agents.learner import Model
from hsi.agents.scripted import random_policy, rollout
from hsi.codec import CODECS, Trace, fit_to_budget
from hsi.contracts.packet import Packet
from hsi.contracts.records import Comparison, Split
from hsi.environment import RuleWorld
from hsi.evaluation.base import TaskResult


def teacher_trace(env: RuleWorld, length: int) -> Trace:
    """Random exploration of the same world, recorded as full-state transitions."""
    t = env.task
    rng = random.Random(f"teacher/{t.family}/{t.seed}")
    obs, _ = env.reset()
    init = tuple(obs)
    steps = []
    for _ in range(length):
        a = rng.randrange(4)
        obs, *_ = env.step(a)
        steps.append((a, tuple(obs)))
    return Trace(f"episode={t.family}-{t.split.value}-{t.seed} origin=observed", init, tuple(steps))


def run_task_e1(arm: str, family: str, split: Split, seed: int, byte_budget: int,
                trace_len: int) -> TaskResult:
    codec = CODECS[arm]
    teacher = teacher_trace(RuleWorld(family, split, seed), trace_len)
    payload = fit_to_budget(codec, teacher, byte_budget)
    decoded = codec.decode(payload) if payload else Trace(teacher.provenance, teacher.initial, ())
    exact = decoded.steps == teacher.steps[: len(decoded.steps)] and decoded.initial == teacher.initial
    env = RuleWorld(family, split, seed)
    obs, info = env.reset()
    goal = tuple(int(x) for x in info["goal"])  # type: ignore[attr-defined]
    plan = Model.from_trace(decoded).plan(tuple(obs), goal, env.task.horizon)
    rng = random.Random(f"act/{family}/{seed}")
    acts = [(0, a) for a in plan] if plan else random_policy(env, rng)
    ep = rollout(env, acts)
    ep = ep.model_copy(update={"episode_id": f"{arm}-{ep.episode_id}"})
    return TaskResult(arm, family, seed, ep.outcome.success, [ep],
                      {"payload_bytes": float(len(payload)), "recon_exact": float(exact),
                       "calls": float(len(ep.steps))})


class E1Spec:
    def run_task(self, arm: str, family: str, split: Split, seed: int,
                 packet: Packet) -> TaskResult:
        return run_task_e1(arm, family, split, seed, packet.budget.max_message_bytes,
                           int(packet.params["trace_len"]))

    def check_budget(self, r: TaskResult, packet: Packet) -> None:
        if r.stats["payload_bytes"] > packet.budget.max_message_bytes:
            raise BudgetExceeded(f"{r.arm} payload {r.stats['payload_bytes']:.0f} > "
                                 f"{packet.budget.max_message_bytes}")

    def guardrails(self, results: list[TaskResult], comps: list[Comparison],
                   packet: Packet) -> tuple[dict[str, float], bool]:
        gv = {"recon_exact_fraction": sum(r.stats["recon_exact"] for r in results) / len(results),
              "max_payload_bytes": max(r.stats["payload_bytes"] for r in results),
              "byte_budget": float(packet.budget.max_message_bytes)}
        return gv, gv["recon_exact_fraction"] == 1.0 and gv["max_payload_bytes"] <= gv["byte_budget"]
