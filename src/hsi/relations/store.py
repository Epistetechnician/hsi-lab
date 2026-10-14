"""Typed relationship layer (F12-F13): context -> remembered rule evidence.

Policies: NONE (no memory), STATIC (FIFO capacity, hard eviction, never updated), ADAPTIVE
(strengthen on confirming evidence, decouple/retire only on counterevidence, dormant over delete,
reactivate on return). Shared-source evidence is counted once. Every state change is logged as a
RelationEvent with a justification and source event ids.
"""

from __future__ import annotations

from enum import StrEnum

from hsi.agents.learner import Transition
from hsi.contracts.records import Relation, RelationEvent, RelOp, RelState

STRENGTH0, STEP_UP, STEP_DOWN, RETIRE_BELOW = 0.6, 0.2, 0.5, 0.25


class Policy(StrEnum):
    NONE = "none"
    STATIC = "static"
    ADAPTIVE = "adaptive"


class RelationStore:
    def __init__(self, policy: Policy, cap: int) -> None:
        self.policy, self.cap = policy, cap
        self.rels: dict[str, Relation] = {}
        self.payload: dict[str, list[Transition]] = {}
        self.events: list[RelationEvent] = []
        self._seen: set[str] = set()
        self._used: dict[str, int] = {}
        self.dup_ignored = 0

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def rid(context: str) -> str:
        return f"rel/{context}"

    def _emit(self, rid: str, op: RelOp, why: str, sources: list[str]) -> None:
        self.events.append(
            RelationEvent(
                event_id=f"ev/{len(self.events)}",
                rel_id=rid,
                op=op,
                justification=why,
                source_event_ids=sources,
            )
        )

    def _set(self, rid: str, **kw: object) -> None:
        self.rels[rid] = self.rels[rid].model_copy(update=kw)

    def _active(self) -> list[Relation]:
        return [r for r in self.rels.values() if r.state is RelState.ACTIVE]

    def _make_room(self, phase: int, src: str) -> None:
        act = self._active()
        if len(act) < self.cap:
            return
        if self.policy is Policy.ADAPTIVE:
            v = min(act, key=lambda r: (r.strength, self._used.get(r.rel_id, 0)))
            self._set(v.rel_id, state=RelState.DORMANT)
            self._emit(v.rel_id, RelOp.DORMANT, "capacity pressure: weakest/least-recent", [src])
        else:
            v = min(act, key=lambda r: r.valid_from)
            del self.rels[v.rel_id], self.payload[v.rel_id]
            self._emit(v.rel_id, RelOp.PRUNE, "static FIFO capacity eviction", [src])

    # -- operations ---------------------------------------------------------
    def learn(self, context: str, transitions: list[Transition], phase: int) -> None:
        if self.policy is Policy.NONE:
            return
        rid, src = self.rid(context), f"learn/{context}/{phase}"
        if rid in self.rels:
            if self.policy is Policy.ADAPTIVE and self.rels[rid].state is not RelState.RETIRED:
                self.payload[rid] = list(transitions)
                self._used[rid] = phase
            return
        self._make_room(phase, src)
        self.rels[rid] = Relation(
            rel_id=rid,
            type="context_rule",
            endpoints=[context, "rule-model"],
            strength=STRENGTH0,
            context=context,
            evidence_refs=[src],
            uncertainty=0.4,
            valid_from=phase,
            valid_to=None,
            state=RelState.ACTIVE,
            cost=float(len(transitions)),
        )
        self.payload[rid] = list(transitions)
        self._used[rid] = phase
        self._emit(rid, RelOp.CREATE, "first encounter of context", [src])

    def retrieve(self, context: str, phase: int) -> Relation | None:
        rid = self.rid(context)
        r = self.rels.get(rid)
        if r is None or self.policy is Policy.NONE or r.state is RelState.RETIRED:
            return None
        if r.state is RelState.DORMANT:
            self._make_room(phase, f"cue/{context}/{phase}")
            self._set(rid, state=RelState.ACTIVE)
            self._emit(rid, RelOp.REACTIVATE, "context cue returned", [f"cue/{context}/{phase}"])
        self._used[rid] = phase
        return self.rels[rid]

    def observe(self, context: str, evidence_id: str, supports: bool) -> None:
        """Adaptive only. Evidence from an already-seen source is ignored (counted once)."""
        if self.policy is not Policy.ADAPTIVE:
            return
        if evidence_id in self._seen:
            self.dup_ignored += 1
            return
        self._seen.add(evidence_id)
        rid = self.rid(context)
        r = self.rels[rid]
        if supports:
            self._set(
                rid,
                strength=min(1.0, r.strength + STEP_UP),
                evidence_refs=[*r.evidence_refs, evidence_id],
            )
            self._emit(rid, RelOp.STRENGTHEN, "probe confirmed stored prediction", [evidence_id])
            return
        s = max(0.0, r.strength - STEP_DOWN)
        self._set(rid, strength=s)
        self._emit(rid, RelOp.DECOUPLE, "probe contradicted stored prediction", [evidence_id])
        if s < RETIRE_BELOW:
            self._set(rid, state=RelState.RETIRED, valid_to=self._used.get(rid, 0))
            self._emit(
                rid, RelOp.RETIRE, "counterevidence drove strength below floor", [evidence_id]
            )

    @property
    def cost(self) -> float:
        return float(sum(r.cost for r in self.rels.values()))
