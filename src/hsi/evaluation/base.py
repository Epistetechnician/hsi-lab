"""Shared experiment plumbing: task results and the per-experiment spec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hsi.contracts.packet import Packet
from hsi.contracts.records import Comparison, Episode, Message, RelationEvent, Split


@dataclass(frozen=True)
class TaskResult:
    arm: str
    family: str
    seed: int
    success: bool
    episodes: list[Episode]  # rollouts
    stats: dict[str, float]
    examples: list[Episode] = field(default_factory=list)  # experience the agent used
    quarantine: list[Episode] = field(default_factory=list)  # synthetic proposals
    decisions: list[dict[str, Any]] = field(default_factory=list)  # eligibility decisions
    messages: list[Message] = field(default_factory=list)
    relation_events: list[RelationEvent] = field(default_factory=list)


class Spec(Protocol):
    """Optional hooks (not in the Protocol): prepare(packet, run_dir) trains candidates during
    `pilot`; bind(packet, run_dir) loads and digest-verifies them before run/eval."""

    def run_task(
        self, arm: str, family: str, split: Split, seed: int, packet: Packet
    ) -> TaskResult: ...

    def check_budget(self, r: TaskResult, packet: Packet) -> None: ...

    def guardrails(
        self, results: list[TaskResult], comps: list[Comparison], packet: Packet
    ) -> tuple[dict[str, float], bool]: ...


def arms_of(packet: Packet) -> tuple[str, list[str]]:
    """(treatment, baselines)."""
    t = str(packet.params.get("treatment", packet.arms[1]))
    b = list(packet.params.get("baselines", [packet.arms[0]]))
    return t, b
