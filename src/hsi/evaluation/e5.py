"""E5: adaptive vs fixed communication graph under a byte cap (four agents, one action each).

Agent k can only execute action k, so the planner (agent 0) needs the other agents' learned rules.
Both arms explore identically; only the graph protocol differs. Primary outcome: success within
the per-task byte cap (success per kilobyte is checked as a guardrail); link churn is reported.
"""
from __future__ import annotations

import random

from hsi.accounting import BudgetExceeded
from hsi.agents.learner import Model
from hsi.agents.scripted import random_policy, rollout
from hsi.collective import adaptive, fixed_gossip
from hsi.contracts.packet import Packet
from hsi.contracts.records import Comparison, Split
from hsi.environment import RuleWorld
from hsi.environment.world import N_ACTIONS
from hsi.evaluation.base import TaskResult

N_AGENTS = N_ACTIONS
PATH = [(0, 1), (1, 2), (2, 3)]


def _explore(family: str, split: Split, seed: int, agent: int, calls: int):  # type: ignore[no-untyped-def]
    env = RuleWorld(family, split, seed)
    obs, _ = env.reset()
    s, out = tuple(obs), []
    for _ in range(calls):
        o, *_ = env.step(agent)  # agent k owns action k
        out.append((s, agent, tuple(o)))
        s = tuple(o)
    return out


def run_task_e5(arm: str, family: str, split: Split, seed: int, packet: Packet) -> TaskResult:
    q = packet.params
    per, rounds, cap = int(q["explore_per_agent"]), int(q["rounds"]), packet.budget.max_message_bytes
    key = f"{arm}/{family}-{split.value}-{seed}"
    env = RuleWorld(family, split, seed)
    obs, info = env.reset()
    start, goal = tuple(obs), tuple(int(x) for x in info["goal"])  # type: ignore[attr-defined]
    know = []
    for k in range(N_AGENTS):
        m = Model(_explore(family, split, seed, k, per))
        know.append({a: h for a, h in m.hyp.items() if a == k})

    def believes_solvable(kn: dict) -> bool:  # type: ignore[type-arg]
        return Model.with_hyps(Model([]), kn).plan(start, goal, env.task.horizon) is not None

    if arm == "fixed_graph":
        net = fixed_gossip(know, PATH, rounds, cap, key)
    elif arm == "adaptive_graph":
        net = adaptive(know, 0, rounds, cap, key, believes_solvable)
    else:
        raise ValueError(f"unknown arm {arm}")
    model = Model.with_hyps(Model([]), know[0])
    plan = model.plan(start, goal, env.task.horizon)
    acts = [(a, a) for a in plan] if plan else random_policy(env, random.Random(f"act/{key}"))
    ep = rollout(env, acts)
    ep = ep.model_copy(update={"episode_id": key})
    stats = {"explore_calls": float(per * N_AGENTS), "message_bytes": float(net.used),
             "messages": float(len(net.messages)), "links_formed": float(net.formed),
             "links_dropped": float(net.dropped), "calls": float(len(ep.steps))}
    return TaskResult(arm, family, seed, ep.outcome.success, [ep], stats, messages=net.messages,
                      decisions=[{**e, "task": key} for e in net.events])


class E5Spec:
    def run_task(self, arm: str, family: str, split: Split, seed: int,
                 packet: Packet) -> TaskResult:
        return run_task_e5(arm, family, split, seed, packet)

    def check_budget(self, r: TaskResult, packet: Packet) -> None:
        if r.stats["explore_calls"] != packet.params["explore_per_agent"] * N_AGENTS:
            raise BudgetExceeded(f"{r.arm}: exploration not matched")
        if r.stats["message_bytes"] > packet.budget.max_message_bytes:
            raise BudgetExceeded(f"{r.arm}: message bytes {r.stats['message_bytes']:.0f} over cap")

    def guardrails(self, results: list[TaskResult], comps: list[Comparison],
                   packet: Packet) -> tuple[dict[str, float], bool]:
        gv: dict[str, float] = {
            "message_byte_cap": float(packet.budget.max_message_bytes),
            "max_message_bytes_used": max(r.stats["message_bytes"] for r in results),
            "compute_matched_fraction": sum(
                r.stats["explore_calls"] == packet.params["explore_per_agent"] * N_AGENTS
                for r in results) / len(results)}
        for arm in ("fixed_graph", "adaptive_graph"):
            rs = [r for r in results if r.arm == arm]
            gv[f"bytes_total_{arm}"] = sum(r.stats["message_bytes"] for r in rs)
            gv[f"link_churn_mean_{arm}"] = sum(
                r.stats["links_formed"] + r.stats["links_dropped"] for r in rs) / len(rs)

        def spk(arm: str) -> float:
            s = sum(r.success for r in results if r.arm == arm)
            return s / (gv[f"bytes_total_{arm}"] / 1024)

        gv["success_per_kb_fixed_graph"], gv["success_per_kb_adaptive_graph"] = (
            spk("fixed_graph"), spk("adaptive_graph"))
        held = (gv["success_per_kb_adaptive_graph"] >= gv["success_per_kb_fixed_graph"]
                and gv["max_message_bytes_used"] <= gv["message_byte_cap"]
                and gv["compute_matched_fraction"] == 1.0)
        return gv, held
