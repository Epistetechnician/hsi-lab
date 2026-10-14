"""E3: single vs independent vs centralized vs fixed-graph collective at matched exploration.

Agent k may only execute actions {2k, 2k+1}, so effects of the other half are discoverable only by
the other agent. Total exploration calls are equal across arms; message bytes are capped.
"""

from __future__ import annotations

import random

from hsi.accounting import BudgetExceeded
from hsi.agents.learner import Model, Transition, transitions_of
from hsi.agents.scripted import random_policy, rollout
from hsi.codec import CODECS
from hsi.collective import decode_hyps, encode_hyps, encode_log, message
from hsi.contracts.packet import Packet
from hsi.contracts.records import Comparison, Episode, Message, Split
from hsi.environment import RuleWorld
from hsi.evaluation.base import TaskResult


def _explore(
    family: str, split: Split, seed: int, agent: int, calls: int, key: str
) -> list[Transition]:
    env = RuleWorld(family, split, seed)
    allowed = sorted(env.allowed(agent))
    rng = random.Random(f"explore/{key}/{agent}")
    obs, _ = env.reset()
    s, out = tuple(obs), []
    for _ in range(calls):
        a = rng.choice(allowed)
        obs, *_ = env.step(a, agent)
        out.append((s, a, tuple(obs)))
        s = tuple(obs)
    return out


def _act(env: RuleWorld, model: Model, key: str, owner: dict[int, int] | None) -> Episode:
    obs, info = env.reset()
    goal = tuple(int(x) for x in info["goal"])  # type: ignore[attr-defined]
    plan = model.plan(tuple(obs), goal, env.task.horizon)
    if plan:
        acts = [((owner or {}).get(a, a // 2), a) for a in plan]
    else:
        acts = random_policy(env, random.Random(f"act/{key}"))
    return rollout(env, acts)


def run_task_e3(arm: str, family: str, split: Split, seed: int, packet: Packet) -> TaskResult:
    total, cap = int(packet.params["explore_total"]), packet.budget.max_message_bytes
    key = f"{family}-{split.value}-{seed}"
    env = RuleWorld(family, split, seed)
    msgs: list[Message] = []
    eps: list[Episode] = []
    explored = 0
    if arm == "single":
        t0 = _explore(family, split, seed, 0, total, key)
        explored = len(t0)
        eps.append(_act(env, Model(t0), key, None))
        ok = eps[0].outcome.success
    else:
        per = total // 2
        t = [_explore(family, split, seed, k, per, key) for k in (0, 1)]
        explored = sum(map(len, t))
        if arm == "independent":
            models = [Model(t[0]), Model(t[1])]
            for k in (0, 1):
                e = _act(env, models[k], f"{key}/{k}", dict.fromkeys(range(4), k))
                eps.append(e.model_copy(update={"episode_id": f"{e.episode_id}/agent{k}"}))
            ok = any(e.outcome.success for e in eps)
        else:
            refs = [f"explore/1/{i}" for i in range(per)]
            if arm == "centralized":
                payload = encode_log(t[1], cap)
                msgs.append(message(f"{arm}/{key}/0", 1, [0], "typed_event", payload, refs, total))
                got = transitions_of(CODECS["typed_event"].decode(payload)) if payload else []
                model = Model(t[0] + got)
            elif arm == "fixed_graph":
                payload = encode_hyps(Model(t[1]).hyp, cap)
                msgs.append(message(f"{arm}/{key}/0", 1, [0], "hyp_summary", payload, refs, total))
                model = Model.with_hyps(Model(t[0]), decode_hyps(payload))
            else:
                raise ValueError(f"unknown arm {arm}")
            eps.append(_act(env, model, key, None))
            ok = eps[0].outcome.success
    eps = [e.model_copy(update={"episode_id": f"{arm}/{e.episode_id}"}) for e in eps]
    stats = {
        "explore_calls": float(explored),
        "message_bytes": float(sum(m.bytes for m in msgs)),
        "messages": float(len(msgs)),
        "calls": float(sum(len(e.steps) for e in eps)),
    }
    return TaskResult(arm, family, seed, ok, eps, stats, messages=msgs)


class E3Spec:
    def run_task(
        self, arm: str, family: str, split: Split, seed: int, packet: Packet
    ) -> TaskResult:
        return run_task_e3(arm, family, split, seed, packet)

    def check_budget(self, r: TaskResult, packet: Packet) -> None:
        if r.stats["explore_calls"] != packet.params["explore_total"]:
            raise BudgetExceeded(f"{r.arm}: exploration {r.stats['explore_calls']:.0f} not matched")
        if r.stats["message_bytes"] > packet.budget.max_message_bytes:
            raise BudgetExceeded(f"{r.arm}: message bytes {r.stats['message_bytes']:.0f} over cap")

    def guardrails(
        self, results: list[TaskResult], comps: list[Comparison], packet: Packet
    ) -> tuple[dict[str, float], bool]:
        gv = {
            "max_message_bytes_used": max(r.stats["message_bytes"] for r in results),
            "message_byte_cap": float(packet.budget.max_message_bytes),
            "compute_matched_fraction": sum(
                r.stats["explore_calls"] == packet.params["explore_total"] for r in results
            )
            / len(results),
        }
        return gv, gv["max_message_bytes_used"] <= gv["message_byte_cap"] and gv[
            "compute_matched_fraction"
        ] == 1.0
