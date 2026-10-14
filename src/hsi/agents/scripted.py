"""Model-free scripted baseline policies (used for E1-E3)."""

from __future__ import annotations

import random
from collections.abc import Callable

from hsi.contracts.records import Eligibility, Episode, EpisodeOutcome, Origin, Step
from hsi.environment import ENV_VERSION, RuleWorld

Policy = Callable[[RuleWorld, random.Random], list[tuple[int, int]]]  # -> [(agent, action)]


def random_policy(env: RuleWorld, rng: random.Random) -> list[tuple[int, int]]:
    return [(rng.randrange(env.task.n_agents), rng.randrange(4)) for _ in range(env.task.horizon)]


def oracle_policy(env: RuleWorld, rng: random.Random) -> list[tuple[int, int]]:
    """Upper bound: plans with full knowledge of hidden rules (agent routing respected)."""
    plan = env.oracle_plan() or []
    return [(next(a for a in range(env.task.n_agents) if x in env.allowed(a)), x) for x in plan]


def greedy_policy(env: RuleWorld, rng: random.Random) -> list[tuple[int, int]]:
    """Probe-free greedy: random actions, but never repeats the last action."""
    out, last = [], -1
    for _ in range(env.task.horizon):
        a = rng.choice([x for x in range(4) if x != last])
        out.append((rng.randrange(env.task.n_agents), a))
        last = a
    return out


POLICIES: dict[str, Policy] = {
    "random": random_policy,
    "greedy": greedy_policy,
    "oracle": oracle_policy,
}


def rollout(
    env: RuleWorld, plan: list[tuple[int, int]], origin: Origin = Origin.OBSERVED
) -> Episode:
    obs, _ = env.reset()
    start = list(obs)
    steps: list[Step] = []
    ok = False
    for agent, a in plan:
        obs, r, done, trunc, _ = env.step(a, agent)
        steps.append(Step(action=a, agent=agent, observation=obs, reward=r))
        if done:
            ok = True
        if done or trunc:
            break
    t = env.task
    return Episode(
        episode_id=f"{t.family}-{t.split.value}-{t.seed}",
        env_version=ENV_VERSION,
        family=t.family,
        rule_family=t.rule_family,
        split=t.split,
        seed=t.seed,
        start=start,
        steps=steps,
        outcome=EpisodeOutcome(success=ok, steps=len(steps), oracle_steps=env.oracle_steps()),
        origin=origin,
        parent_ids=[],
        generation_depth=0,
        governance="synthetic-local",
        eligibility=Eligibility.ELIGIBLE,
        prev_digest="",
        digest="",
    )
