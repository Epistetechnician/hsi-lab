"""Scripted inductive learner: infer per-action primitives from decoded transitions, then plan."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from itertools import product

from hsi.codec.base import State, Trace
from hsi.environment.world import MOD, N_ACTIONS, N_CELLS, Prim, apply_prim

_R = range(N_CELLS)
_PRIMS: list[Prim] = (
    [("add", j, 0, c) for j in _R for c in range(1, MOD)]
    + [("swap", j, k, 0) for j in _R for k in _R if j < k]
    + [("copy", j, k, 0) for j in _R for k in _R if j != k]
    + [("mix", j, k, 0) for j in _R for k in _R if j != k]
)
_SINGLES: list[tuple[Prim, ...]] = [(p,) for p in _PRIMS]
HYPOTHESES: list[tuple[Prim, ...]] = _SINGLES + [a + b for a, b in product(_SINGLES, _SINGLES)]


def _run(h: tuple[Prim, ...], s: State) -> State:
    t = list(s)
    for p in h:
        apply_prim(p, t)
    return tuple(t)


Transition = tuple[State, int, State]
Chooser = Callable[[list[int]], int]  # picks one hypothesis index among the consistent ones


def consistent(pairs: list[tuple[State, State]], first_only: bool = False) -> list[int]:
    out = []
    for i, h in enumerate(HYPOTHESES):
        if all(_run(h, s) == s2 for s, s2 in pairs):
            out.append(i)
            if first_only:
                break
    return out


class Model:
    def __init__(
        self,
        transitions: list[Transition],
        chooser: Chooser | None = None,
        proposer: Callable[[list[tuple[State, State]]], tuple[Prim, ...] | None] | None = None,
    ) -> None:
        obs: dict[int, list[tuple[State, State]]] = {a: [] for a in range(N_ACTIONS)}
        for s, a, s2 in transitions:
            obs[a].append((s, s2))
        self.hyp: dict[int, tuple[Prim, ...]] = {}
        self.table: dict[tuple[State, int], State] = {}
        for a, pairs in obs.items():
            for s, s2 in pairs:
                self.table[(s, a)] = s2
            if pairs and proposer is not None:
                # free generation: whatever valid hypothesis is proposed is used, even if the
                # evidence contradicts it; an invalid proposal leaves the action unknown
                h = proposer(pairs)
                if h is not None:
                    self.hyp[a] = h
            elif pairs:
                c = consistent(pairs, first_only=chooser is None)
                if c:
                    ctx = getattr(chooser, "with_context", None)  # LLM choosers see the evidence
                    pick = (
                        c[0]
                        if chooser is None
                        else (ctx(c, pairs) if ctx is not None else chooser(c))
                    )
                    self.hyp[a] = HYPOTHESES[pick]

    @classmethod
    def from_trace(cls, trace: Trace) -> Model:
        out, prev = [], trace.initial
        for a, s in trace.steps:
            out.append((prev, a, s))
            prev = s
        return cls(out)

    @classmethod
    def with_hyps(cls, base: Model, hyps: dict[int, tuple[Prim, ...]]) -> Model:
        m = cls([])
        m.hyp = {**hyps, **base.hyp}
        m.table = dict(base.table)
        return m

    def predict(self, s: State, a: int) -> State | None:
        """Prediction if the model knows this action (hypothesis or exact table hit), else None."""
        if a in self.hyp:
            return _run(self.hyp[a], s)
        return self.table.get((s, a))

    def next(self, s: State, a: int) -> State:
        if a in self.hyp:
            return _run(self.hyp[a], s)
        return self.table.get((s, a), s)

    def plan(
        self, start: State, goal: State, limit: int, max_nodes: int = 50_000
    ) -> list[int] | None:
        prev: dict[State, tuple[State, int] | None] = {start: None}
        depth = {start: 0}
        q = deque([start])
        while q and len(prev) < max_nodes:
            c = q.popleft()
            if c == goal:
                out: list[int] = []
                while prev[c] is not None:
                    c, a = prev[c]  # type: ignore[misc]
                    out.append(a)
                return out[::-1]
            if depth[c] >= limit:
                continue
            for a in range(N_ACTIONS):
                n = self.next(c, a)
                if n not in prev:
                    prev[n], depth[n] = (c, a), depth[c] + 1
                    q.append(n)
        return None


def transitions_of(trace: Trace) -> list[Transition]:
    out, prev = [], trace.initial
    for a, s in trace.steps:
        out.append((prev, a, s))
        prev = s
    return out
