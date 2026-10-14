"""Executable-oracle verifier (F9). Moves quarantined proposals to eligible or rejected.

Never imports synthesis: it sees proposals as plain episodes, never generator state.
"""

from __future__ import annotations

from dataclasses import dataclass

from hsi.contracts.packet import canonical_digest
from hsi.contracts.records import Eligibility, Episode
from hsi.environment import RuleWorld

N_ACTIONS = 4


@dataclass(frozen=True)
class Decision:
    episode: Episode
    reason: str  # "ok" or a rejection reason

    def record(self) -> dict[str, object]:
        ep = self.episode
        return {
            "episode_id": ep.episode_id,
            "eligibility": ep.eligibility.value,
            "reason": self.reason,
            "parent_ids": ep.parent_ids,
            "content_digest": canonical_digest([ep.start, [s.model_dump() for s in ep.steps]]),
        }


class Verifier:
    def __init__(self, env: RuleWorld) -> None:
        self._env = env
        self._seen: set[tuple[tuple[int, ...], int]] = set()

    def verify(self, ep: Episode) -> Decision:
        def rej(reason: str) -> Decision:
            return Decision(ep.model_copy(update={"eligibility": Eligibility.REJECTED}), reason)

        if not ep.parent_ids:
            return rej("provenance_missing")
        if (
            len(ep.steps) != 1
            or not 0 <= ep.steps[0].action < N_ACTIONS
            or len(ep.start) != len(ep.steps[0].observation)
        ):
            return rej("invalid")
        start, a = tuple(ep.start), ep.steps[0].action
        if (start, a) in self._seen:
            return rej("ambiguous_duplicate")
        truth = self._env.transition(start, a)
        if truth != tuple(ep.steps[0].observation):
            return rej("outcome_mismatch")
        self._seen.add((start, a))
        return Decision(ep.model_copy(update={"eligibility": Eligibility.ELIGIBLE}), "ok")
