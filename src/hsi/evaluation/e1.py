"""E1 arm execution: codec-compressed teacher trace -> inductive learner -> plan -> rollout."""
from __future__ import annotations

import random
from dataclasses import dataclass

from hsi.agents.learner import Model
from hsi.agents.scripted import random_policy, rollout
from hsi.codec import CODECS, Trace, fit_to_budget
from hsi.contracts.records import Episode, Split
from hsi.environment import RuleWorld


@dataclass(frozen=True)
class TaskResult:
    arm: str
    family: str
    seed: int
    success: bool
    payload_bytes: int
    recon_exact: bool
    episode: Episode


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


def run_task(arm: str, family: str, split: Split, seed: int, byte_budget: int,
             trace_len: int) -> TaskResult:
    codec = CODECS[arm]
    teacher = teacher_trace(RuleWorld(family, split, seed), trace_len)
    payload = fit_to_budget(codec, teacher, byte_budget)
    decoded = codec.decode(payload) if payload else Trace(teacher.provenance, teacher.initial, ())
    exact = decoded.steps == teacher.steps[: len(decoded.steps)] and decoded.initial == teacher.initial
    env = RuleWorld(family, split, seed)
    obs, info = env.reset()
    goal = tuple(int(x) for x in info["goal"])  # type: ignore[attr-defined]
    plan = Model(decoded).plan(tuple(obs), goal, env.task.horizon)
    rng = random.Random(f"act/{family}/{seed}")
    acts = [(0, a) for a in plan] if plan else random_policy(env, rng)
    ep = rollout(env, acts)
    ep = ep.model_copy(update={"episode_id": f"{arm}-{ep.episode_id}"})
    return TaskResult(arm, family, seed, ep.outcome.success, len(payload), exact, ep)
