"""Pure-Python hypothesis prior with a frozen base and a reversible LoRA adapter (F15).

score(h) = v . tanh(W phi(h) + b). The base trains W; the adapter adds a low-rank update
W' = W + B A (B starts at zero) and never touches W. Dropping the adapter restores the base
exactly. This is the stand-in for an MLX LLM backbone: same freeze / adapter / digest contract.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from hsi.agents.learner import HYPOTHESES
from hsi.contracts.packet import canonical_digest
from hsi.environment.world import Prim

KINDS = ("add", "swap", "copy", "mix")
D, H, RUNTIME = 14, 8, "hsi-prior-v1/python-pure"
_PHI: dict[int, list[float]] = {}


def signature(h: tuple[Prim, ...]) -> str:
    return ">".join(p[0] for p in h)


def phi(idx: int) -> list[float]:
    f = _PHI.get(idx)
    if f is None:
        h = HYPOTHESES[idx]
        f = [0.0] * D
        f[len(h) - 1] = 1.0  # 0: single, 1: pair
        for slot, p in enumerate(h):
            f[2 + 4 * slot + KINDS.index(p[0])] = 1.0
            f[10 + KINDS.index(p[0])] += 1.0
        _PHI[idx] = f
    return f


def canonical(p: Prim) -> Prim:
    kind, j, k, c = p
    if kind == "add":
        return (kind, j, 0, c)
    if kind == "swap":
        return (kind, min(j, k), max(j, k), 0)
    return (kind, j, k, 0)


def true_index(prims: tuple[Prim, ...]) -> int | None:
    key = tuple(canonical(p) for p in prims)
    return _INDEX.get(key)


_INDEX = {h: i for i, h in enumerate(HYPOTHESES)}


@dataclass
class Base:
    W: list[list[float]]
    b: list[float]
    v: list[float]
    _z0: dict[int, list[float]] = field(default_factory=dict, repr=False)

    @classmethod
    def init(cls, seed: int = 0) -> Base:
        r = random.Random(f"prior-init/{seed}")
        return cls([[r.gauss(0, 0.3) for _ in range(D)] for _ in range(H)],
                   [0.0] * H, [r.choice((-1.0, 1.0)) * r.uniform(0.5, 1.0) for _ in range(H)])

    def z(self, idx: int) -> list[float]:
        z = self._z0.get(idx)
        if z is None:
            f = phi(idx)
            z = [sum(w * x for w, x in zip(row, f, strict=True)) + bi
                 for row, bi in zip(self.W, self.b, strict=True)]
            self._z0[idx] = z
        return z

    def state(self) -> dict[str, object]:
        return {"W": self.W, "b": self.b, "v": self.v}

    @property
    def digest(self) -> str:
        return canonical_digest(self.state())


@dataclass
class Adapter:
    A: list[list[float]]  # r x D
    B: list[list[float]]  # H x r

    @classmethod
    def init(cls, rank: int, seed: int) -> Adapter:
        r = random.Random(f"adapter-init/{seed}")
        return cls([[r.gauss(0, 0.2) for _ in range(D)] for _ in range(rank)],
                   [[0.0] * rank for _ in range(H)])

    def state(self) -> dict[str, object]:
        return {"A": self.A, "B": self.B}

    @property
    def digest(self) -> str:
        return canonical_digest(self.state())


class Prior:
    def __init__(self, base: Base, adapter: Adapter | None = None,
                 memory: dict[str, int] | None = None, alpha: float = 0.0) -> None:
        self.base, self.adapter, self.memory, self.alpha = base, adapter, memory, alpha

    def _u(self, idx: int) -> list[float]:
        f = phi(idx)
        assert self.adapter is not None
        return [sum(a * x for a, x in zip(row, f, strict=True)) for row in self.adapter.A]

    def hidden(self, idx: int) -> list[float]:
        z = list(self.base.z(idx))
        if self.adapter is not None:
            u = self._u(idx)
            for i, row in enumerate(self.adapter.B):
                z[i] += sum(bv * uv for bv, uv in zip(row, u, strict=True))
        return [math.tanh(x) for x in z]

    def score(self, idx: int) -> float:
        s = sum(v * h for v, h in zip(self.base.v, self.hidden(idx), strict=True))
        if self.memory is not None:
            s += self.alpha * math.log1p(self.memory.get(signature(HYPOTHESES[idx]), 0))
        return s

    def choose(self, idxs: list[int]) -> int:
        return max(idxs, key=lambda i: (self.score(i), -i))


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    t = sum(e)
    return [x / t for x in e]
