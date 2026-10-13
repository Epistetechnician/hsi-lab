"""Codec interface (F6). Byte counts include schema header and provenance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

State = tuple[int, ...]


@dataclass(frozen=True)
class Trace:
    provenance: str
    initial: State
    steps: tuple[tuple[int, State], ...]  # (action, resulting state)

    def prefix(self, k: int) -> Trace:
        return Trace(self.provenance, self.initial, self.steps[:k])


class Codec(Protocol):
    name: str

    def encode(self, trace: Trace) -> bytes: ...
    def decode(self, data: bytes) -> Trace: ...


def fit_to_budget(codec: Codec, trace: Trace, budget: int) -> bytes:
    """Longest whole-step prefix whose encoding fits `budget` bytes; b'' if even the header doesn't."""
    best = b""
    for k in range(len(trace.steps) + 1):
        data = codec.encode(trace.prefix(k))
        if len(data) > budget:
            break
        best = data
    return best
