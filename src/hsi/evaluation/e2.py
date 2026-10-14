"""E2: real-only vs unverified synthetic vs verified synthetic, at equal example count."""

from __future__ import annotations

import random

from hsi.accounting import BudgetExceeded
from hsi.agents.learner import Model, Transition
from hsi.agents.scripted import random_policy, rollout
from hsi.contracts.packet import Packet
from hsi.contracts.records import (
    Comparison,
    Eligibility,
    Episode,
    EpisodeOutcome,
    Origin,
    Split,
    Step,
)
from hsi.environment import RuleWorld
from hsi.evaluation.base import TaskResult, arms_of
from hsi.evaluation.e1 import teacher_trace
from hsi.synthesis import Proposer, Quarantine
from hsi.verification import Verifier


def _real_example(
    arm: str,
    key: str,
    i: int,
    s: tuple[int, ...],
    a: int,
    s2: tuple[int, ...],
    family: str,
    split: Split,
    seed: int,
    rule_family: str,
) -> Episode:
    return Episode(
        episode_id=f"{arm}/real/{key}/{i}",
        env_version="rulegrid-v1",
        family=family,
        rule_family=rule_family,
        split=split,
        seed=seed,
        start=list(s),
        steps=[Step(action=a, agent=0, observation=list(s2), reward=0.0)],
        outcome=EpisodeOutcome(success=False, steps=1, oracle_steps=0),
        origin=Origin.OBSERVED,
        parent_ids=[],
        generation_depth=0,
        governance="synthetic-local",
        eligibility=Eligibility.ELIGIBLE,
        prev_digest="",
        digest="",
    )


def _transition(ep: Episode) -> Transition:
    st = ep.steps[0]
    return tuple(ep.start), st.action, tuple(st.observation)


def run_task_e2(arm: str, family: str, split: Split, seed: int, packet: Packet) -> TaskResult:
    q = packet.params
    k_total, k_real, max_prop = int(q["examples"]), int(q["real_seed"]), int(q["max_proposals"])
    env = RuleWorld(family, split, seed)
    key = f"{family}-{split.value}-{seed}"
    tr = teacher_trace(RuleWorld(family, split, seed), k_total)
    prev, real = tr.initial, []
    for i, (a, s) in enumerate(tr.steps):
        real.append(
            _real_example(arm, key, i, prev, a, s, family, split, seed, env.task.rule_family)
        )
        prev = s
    quarantine, decisions, proposed, rejected = Quarantine(), [], 0, 0
    topped = 0
    if arm == "real_only":
        examples = real[:k_total]
    else:
        seeds = real[:k_real]
        want = k_total - k_real
        gen = Proposer([_transition(e) for e in seeds], [e.episode_id for e in seeds], key)
        syn: list[Episode] = []
        if arm == "unverified_synthetic":
            for j in range(want):
                ep = gen.propose(f"{arm}/syn/{key}/{j}", family, split, seed)
                quarantine.submit(ep)
                syn.append(ep)  # trusted as-is: still eligibility=quarantined
            proposed = want
        else:
            ver = Verifier(env)
            while len(syn) < want and proposed < max_prop:
                ep = gen.propose(f"{arm}/syn/{key}/{proposed}", family, split, seed)
                proposed += 1
                quarantine.submit(ep)
                d = ver.verify(ep)
                decisions.append(d.record())
                if d.episode.eligibility is Eligibility.ELIGIBLE:
                    syn.append(d.episode)
                else:
                    rejected += 1
            topped = want - len(syn)
        extra = real[k_real : k_real + topped]  # top up with real data to keep counts equal
        examples = seeds + syn + extra
    model = Model([_transition(e) for e in examples])
    obs, info = env.reset()
    goal = tuple(int(x) for x in info["goal"])  # type: ignore[attr-defined]
    plan = model.plan(tuple(obs), goal, env.task.horizon)
    acts = [(0, a) for a in plan] if plan else random_policy(env, random.Random(f"act/{key}"))
    ep = rollout(env, acts)
    ep = ep.model_copy(update={"episode_id": f"{arm}/rollout/{key}"})
    ids = {e.episode_id for e in real}
    elig = {d["episode_id"] for d in decisions if d["eligibility"] == "eligible"}
    lineage = all(
        e.origin is Origin.OBSERVED
        or (
            bool(e.parent_ids)
            and set(e.parent_ids) <= ids
            and (arm != "verified_synthetic" or e.episode_id in elig)
        )
        for e in examples
    )
    stats = {
        "n_examples": float(len(examples)),
        "lineage_ok": float(lineage),
        "proposals": float(proposed),
        "failed_proposals": float(rejected),
        "topped_up": float(topped),
        "calls": float(len(ep.steps)),
    }
    return TaskResult(
        arm, family, seed, ep.outcome.success, [ep], stats, examples, quarantine.items, decisions
    )


class E2Spec:
    def run_task(
        self, arm: str, family: str, split: Split, seed: int, packet: Packet
    ) -> TaskResult:
        return run_task_e2(arm, family, split, seed, packet)

    def check_budget(self, r: TaskResult, packet: Packet) -> None:
        if r.stats["n_examples"] != packet.params["examples"]:
            raise BudgetExceeded(
                f"{r.arm}: {r.stats['n_examples']:.0f} examples != {packet.params['examples']}"
            )

    def guardrails(
        self, results: list[TaskResult], comps: list[Comparison], packet: Packet
    ) -> tuple[dict[str, float], bool]:
        from hsi.evaluation.stats import family_diff

        treat, _ = arms_of(packet)
        c = next(c for c in comps if c.baseline_arm == "real_only" and c.treatment_arm == treat)
        floor = float(packet.params["family_regression_floor"])
        gv = {
            "min_family_diff_vs_real_only": min(family_diff(p)[0] for p in c.pairs),
            "family_regression_floor": floor,
            "example_count_equal_fraction": sum(
                r.stats["n_examples"] == packet.params["examples"] for r in results
            )
            / len(results),
            "lineage_fraction": sum(r.stats["lineage_ok"] for r in results) / len(results),
        }
        held = (
            gv["min_family_diff_vs_real_only"] >= -floor
            and gv["example_count_equal_fraction"] == 1.0
            and gv["lineage_fraction"] == 1.0
        )
        return gv, held
