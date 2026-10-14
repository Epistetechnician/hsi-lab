"""Quarantine queue: the only place synthesis may write."""

from __future__ import annotations

from hsi.contracts.records import Eligibility, Episode, Origin


class Quarantine:
    def __init__(self) -> None:
        self.items: list[Episode] = []

    def submit(self, ep: Episode) -> None:
        if ep.origin is not Origin.GENERATED or ep.eligibility is not Eligibility.QUARANTINED:
            raise ValueError("only quarantined, generated proposals may be submitted")
        if ep.generation_depth < 1:
            raise ValueError("generated proposals need generation_depth >= 1")
        self.items.append(ep)
