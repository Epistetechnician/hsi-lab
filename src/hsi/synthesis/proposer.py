"""Blind proposal generator (F8). Sees only real example transitions and an opaque stream key.

It must not import the verifier, the environment world, the evaluator, or any path/custody code
(enforced by tests/property/test_isolation.py). Writes only to the quarantine queue.
"""
from __future__ import annotations

import random

from hsi.agents.learner import Model, Transition
from hsi.contracts.records import Eligibility, Episode, EpisodeOutcome, Origin, Split, Step

N_CELLS, MOD, N_ACTIONS = 5, 7, 4  # public format constants, not task knowledge
SYN_ENV = "synthetic-proposal-v1"


class Proposer:
    def __init__(self, seed_examples: list[Transition], parent_ids: list[str], key: str) -> None:
        self._model = Model(seed_examples)
        self._parents = parent_ids
        self._rng = random.Random(f"gen/{key}")
        self._k = 0

    def propose(self, episode_id: str, family: str, split: Split, seed: int) -> Episode:
        """One-step trajectory from a fresh start state; claimed outcome from the generator's
        own induced model (which may be wrong). Actions are cycled for coverage."""
        start = tuple(self._rng.randrange(MOD) for _ in range(N_CELLS))
        a = self._k % N_ACTIONS
        self._k += 1
        claimed = self._model.next(start, a)
        return Episode(
            episode_id=episode_id, env_version=SYN_ENV, family=family, rule_family="unknown",
            split=split, seed=seed, start=list(start),
            steps=[Step(action=a, agent=0, observation=list(claimed), reward=0.0)],
            outcome=EpisodeOutcome(success=False, steps=1, oracle_steps=0), origin=Origin.GENERATED,
            parent_ids=list(self._parents), generation_depth=1, governance="synthetic-local",
            eligibility=Eligibility.QUARANTINED, prev_digest="", digest="")
