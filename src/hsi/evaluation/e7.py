"""E7: relationship dynamics on return and counterevidence (streams of context-tagged worlds).

relational_return: contexts A,B,C,D,A  (A returns after interludes; capacity 2 forces eviction)
counterevidence:   contexts A,B,A*     (cue A returns but its hidden rules have changed)
Only the last phase is scored; it grants every arm the same few fresh exploration transitions.
"""

from __future__ import annotations

import copy
import random

from hsi.accounting import BudgetExceeded
from hsi.agents.learner import Model, Transition, transitions_of
from hsi.agents.scripted import random_policy, rollout
from hsi.contracts.packet import Packet
from hsi.contracts.records import Comparison, Split
from hsi.environment import SEED_BASE, RuleWorld
from hsi.evaluation.base import TaskResult
from hsi.evaluation.e1 import teacher_trace
from hsi.relations import Policy, RelationStore

POLICY = {
    "no_relations": Policy.NONE,
    "static_relations": Policy.STATIC,
    "adaptive_relations": Policy.ADAPTIVE,
}
# (context label, world index, variant): world index identifies the hidden rules
STREAMS = {
    "relational_return": [("A", 0, 0), ("B", 1, 0), ("C", 2, 0), ("D", 3, 0), ("A", 0, 1)],
    "counterevidence": [("A", 0, 0), ("B", 1, 0), ("A", 4, 0)],
}


def _phase_seed(split: Split, seed: int, world: int) -> int:
    base = SEED_BASE[split]
    return base + ((seed - base) * 31 + world * 104_729) % 1_000_000


def _probe(
    store: RelationStore,
    ctx: str,
    model: Model,
    new: list[Transition],
    seed: int,
    n_probe: int,
    echo: bool,
) -> bool:
    """Verify the stored prediction on the first fresh transitions; True if trusted."""
    trusted = True
    for k, (s, a, s2) in enumerate(new[:n_probe]):
        pred = model.predict(s, a)
        if pred is None:
            continue  # not testable
        supports = pred == s2
        trusted &= supports
        store.observe(ctx, f"probe/{seed}/{k}", supports)
        if echo:  # second agent echoing the same source: must be counted once
            store.observe(ctx, f"probe/{seed}/{k}", supports)
    return trusted


def run_task_e7(arm: str, family: str, split: Split, seed: int, packet: Packet) -> TaskResult:
    q = packet.params
    full, small, cap, n_probe = (
        int(q[k]) for k in ("explore_full", "explore_final", "relation_cap", "probes")
    )
    store = RelationStore(POLICY[arm], cap)
    stream = STREAMS[family]
    explored = 0
    for i, (ctx, widx, var) in enumerate(stream[:-1]):
        w = RuleWorld("rule_induction", split, _phase_seed(split, seed, widx), var)
        tr = teacher_trace(w, full, tag=f"/p{i}")
        explored += full
        store.learn(ctx, transitions_of(tr), i)
    ctx, widx, var = stream[-1]
    last = len(stream) - 1
    fin = RuleWorld("rule_induction", split, _phase_seed(split, seed, widx), var)
    new = transitions_of(teacher_trace(fin, small, tag="/final"))
    explored += small
    rel = store.retrieve(ctx, last)
    dup_inflation = 0.0
    if rel is None:
        model = Model(new)
    else:
        stored = store.payload[rel.rel_id]
        if arm == "static_relations":
            model = Model(stored + new)
        else:
            ref = copy.deepcopy(store)
            trusted = _probe(store, ctx, Model(stored), new, seed, n_probe, echo=True)
            _probe(ref, ctx, Model(stored), new, seed, n_probe, echo=False)
            dup_inflation = float(
                store.rels[rel.rel_id].strength != ref.rels[rel.rel_id].strength
                or store.rels[rel.rel_id].state != ref.rels[rel.rel_id].state
            )
            model = Model(stored + new) if trusted else Model(new)
    obs, info = fin.reset()
    goal = tuple(int(x) for x in info["goal"])  # type: ignore[attr-defined]
    plan = model.plan(tuple(obs), goal, fin.task.horizon)
    acts = (
        [(0, a) for a in plan] if plan else random_policy(fin, random.Random(f"act/{arm}/{seed}"))
    )
    ep = rollout(fin, acts)
    ep = ep.model_copy(update={"episode_id": f"{arm}/{family}/{split.value}-{seed}"})
    stats = {
        "explore_calls": float(explored),
        "calls": float(len(ep.steps)),
        "relation_cost": store.cost,
        "relation_events": float(len(store.events)),
        "dup_ignored": float(store.dup_ignored),
        "dup_inflation": dup_inflation,
    }
    return TaskResult(
        arm, family, seed, ep.outcome.success, [ep], stats, relation_events=list(store.events)
    )


class E7Spec:
    def run_task(
        self, arm: str, family: str, split: Split, seed: int, packet: Packet
    ) -> TaskResult:
        return run_task_e7(arm, family, split, seed, packet)

    def check_budget(self, r: TaskResult, packet: Packet) -> None:
        q = packet.params
        want = (len(STREAMS[r.family]) - 1) * int(q["explore_full"]) + int(q["explore_final"])
        if r.stats["explore_calls"] != want:
            raise BudgetExceeded(f"{r.arm}: exploration {r.stats['explore_calls']:.0f} != {want}")

    def guardrails(
        self, results: list[TaskResult], comps: list[Comparison], packet: Packet
    ) -> tuple[dict[str, float], bool]:
        gv = {
            "duplicate_strength_updates": sum(r.stats["dup_inflation"] for r in results),
            "compute_matched_fraction": sum(
                r.stats["explore_calls"]
                == (len(STREAMS[r.family]) - 1) * int(packet.params["explore_full"])
                + int(packet.params["explore_final"])
                for r in results
            )
            / len(results),
        }
        return gv, gv["duplicate_strength_updates"] == 0 and gv["compute_matched_fraction"] == 1.0
