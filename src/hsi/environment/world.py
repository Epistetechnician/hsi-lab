"""Seeded procedural rule worlds with hidden rule combinations (F1-F3).

State: N_CELLS ints mod MOD. Each of N_ACTIONS actions applies 1-2 hidden primitives drawn
from the rule family. Gymnasium-style reset/step; exact transitions; BFS solvability oracle.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from hsi.contracts.records import Split

ENV_VERSION = "rulegrid-v1"
N_CELLS, MOD, N_ACTIONS = 5, 7, 4
HIDDEN = -1

# rule family -> split (split by rule family; seed ranges are partitioned per split too)
RULE_FAMILIES: dict[str, Split] = {
    "add": Split.FIT, "swap": Split.FIT, "add+swap": Split.FIT,
    "copy": Split.TUNE, "add+copy": Split.TUNE,
    "mix": Split.HELDOUT, "swap+copy": Split.HELDOUT,
}
SEED_BASE = {Split.FIT: 0, Split.TUNE: 1_000_000, Split.HELDOUT: 2_000_000}
TASK_FAMILIES = ("rule_induction", "compositional_transfer", "counterevidence",
                 "partial_observation", "collective_discovery", "relational_return")
# which rule families each task family draws from, per split
_SINGLE = {Split.FIT: ["add", "swap"], Split.TUNE: ["copy"], Split.HELDOUT: ["mix"]}
_COMPOSITE = {Split.FIT: ["add+swap"], Split.TUNE: ["add+copy"], Split.HELDOUT: ["swap+copy"]}

State = tuple[int, ...]
Prim = tuple[str, int, int, int]  # kind, j, k, c


def seed_split(seed: int) -> Split:
    return Split.HELDOUT if seed >= 2_000_000 else Split.TUNE if seed >= 1_000_000 else Split.FIT


def _apply(p: Prim, s: list[int]) -> None:
    kind, j, k, c = p
    if kind == "add":
        s[j] = (s[j] + c) % MOD
    elif kind == "swap":
        s[j], s[k] = s[k], s[j]
    elif kind == "copy":
        s[k] = s[j]
    elif kind == "mix":
        s[k] = (s[k] + s[j]) % MOD


def _prim(kind: str, rng: random.Random) -> Prim:
    j, k = rng.sample(range(N_CELLS), 2)
    return (kind, j, k, rng.randrange(1, MOD))


@dataclass(frozen=True)
class Task:
    family: str
    rule_family: str
    split: Split
    seed: int
    actions: tuple[tuple[Prim, ...], ...]
    guard: tuple[int, int] | None  # (action, cell): action is a no-op when s[cell]==0
    visible: tuple[int, ...]
    start: State
    goal: State
    horizon: int
    n_agents: int


class RuleWorld:
    """Gymnasium-style environment. Deterministic given (task_family, split, seed)."""

    def __init__(self, family: str, split: Split, seed: int) -> None:
        if family not in TASK_FAMILIES:
            raise ValueError(f"unknown family {family}")
        if seed_split(seed) != split:
            raise ValueError(f"seed {seed} not in {split.value} seed family")
        self.task = self._build(family, split, seed)
        self._s: list[int] = list(self.task.start)
        self._t = 0

    # -- construction -------------------------------------------------------
    @staticmethod
    def _build(family: str, split: Split, seed: int) -> Task:
        for attempt in range(100):  # resample degenerate worlds (no reachable unmet goal)
            task = RuleWorld._try_build(family, split, seed, attempt)
            if task is not None:
                return task
        raise RuntimeError("no non-degenerate world found")

    @staticmethod
    def _try_build(family: str, split: Split, seed: int, attempt: int) -> Task | None:
        rng = random.Random(f"{ENV_VERSION}/{family}/{seed}/{attempt}")
        pool = _COMPOSITE[split] if family == "compositional_transfer" else _SINGLE[split]
        rf = rng.choice(pool)
        kinds = rf.split("+")
        actions = tuple(
            tuple(_prim(kinds[i % len(kinds)] if len(kinds) > 1 else kinds[0], rng)
                  for i in range(len(kinds)))
            for _ in range(N_ACTIONS)
        )
        guard = (rng.randrange(N_ACTIONS), rng.randrange(N_CELLS)) if family == "counterevidence" else None
        visible = (0, 1, 2) if family == "partial_observation" else tuple(range(N_CELLS))
        n_agents = 2 if family == "collective_discovery" else 1
        start = tuple(rng.randrange(MOD) for _ in range(N_CELLS))
        probe = RuleWorld.__new__(RuleWorld)
        probe.task = Task(family, rf, split, seed, actions, guard, visible, start, start, 0, n_agents)
        # pick a goal reachable in 3..5 steps (guarantees solvability)
        dist = probe._bfs_all(start)
        unmet = [s for s in dist if any(s[i] != start[i] for i in visible)]
        cands = sorted(s for s in unmet if 3 <= dist[s] <= 5)
        if not cands:
            cands = sorted(unmet)
        if not cands:
            return None
        goal = rng.choice(cands)
        h = min((d for t, d in dist.items() if all(t[i] == goal[i] for i in visible)), default=0)
        task = Task(family, rf, split, seed, actions, guard, visible, start, goal, 3 * h + 4, n_agents)
        return task

    # -- dynamics -----------------------------------------------------------
    def _next(self, s: State, a: int) -> State:
        t = self.task
        if t.guard and t.guard[0] == a and s[t.guard[1]] == 0:
            return s
        out = list(s)
        for p in t.actions[a]:
            _apply(p, out)
        return tuple(out)

    def transition(self, s: State, a: int) -> State:
        """Oracle: exact next state (used by the verifier)."""
        return self._next(s, a)

    def _bfs_all(self, start: State) -> dict[State, int]:
        dist = {start: 0}
        q = deque([start])
        while q:
            s = q.popleft()
            for a in range(N_ACTIONS):
                n = self._next(s, a)
                if n not in dist:
                    dist[n] = dist[s] + 1
                    q.append(n)
        return dist

    def _reached(self, s: State) -> bool:
        v = self.task.visible
        return all(s[i] == self.task.goal[i] for i in v)

    def oracle_steps(self) -> int:
        """Minimum steps from the start to a goal-satisfying state (solvability oracle)."""
        dist = self._bfs_all(self.task.start)
        best = [d for s, d in dist.items() if self._reached(s)]
        return min(best)

    def oracle_plan(self, s: State | None = None) -> list[int] | None:
        s = s or tuple(self._s)
        prev: dict[State, tuple[State, int] | None] = {s: None}
        q = deque([s])
        while q:
            c = q.popleft()
            if self._reached(c):
                plan: list[int] = []
                while prev[c] is not None:
                    c, a = prev[c]  # type: ignore[misc]
                    plan.append(a)
                return plan[::-1]
            for a in range(N_ACTIONS):
                n = self._next(c, a)
                if n not in prev:
                    prev[n] = (c, a)
                    q.append(n)
        return None

    # -- gym-style API ------------------------------------------------------
    def allowed(self, agent: int) -> set[int]:
        if self.task.n_agents == 1:
            return set(range(N_ACTIONS))
        return {2 * agent, 2 * agent + 1}

    def _obs(self) -> list[int]:
        return [self._s[i] if i in self.task.visible else HIDDEN for i in range(N_CELLS)]

    def goal_obs(self) -> list[int]:
        return [self.task.goal[i] if i in self.task.visible else HIDDEN for i in range(N_CELLS)]

    def reset(self) -> tuple[list[int], dict[str, object]]:
        self._s, self._t = list(self.task.start), 0
        return self._obs(), {"goal": self.goal_obs(), "horizon": self.task.horizon}

    def step(self, action: int, agent: int = 0) -> tuple[list[int], float, bool, bool, dict[str, object]]:
        if not 0 <= action < N_ACTIONS:
            raise ValueError("bad action")
        blocked = action not in self.allowed(agent)
        if not blocked:
            self._s = list(self._next(tuple(self._s), action))
        self._t += 1
        done = self._reached(tuple(self._s))
        return self._obs(), 1.0 if done else 0.0, done, (not done and self._t >= self.task.horizon), {
            "blocked": blocked}


apply_prim = _apply
