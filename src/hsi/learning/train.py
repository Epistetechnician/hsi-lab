"""Training for the prior: base pretraining and LoRA adapters. Reads only the sample sets it is
handed; the caller gates which splits those come from (never held-out)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from hsi.learning.prior import Adapter, Base, D, H, Prior, phi, signature, softmax


@dataclass(frozen=True)
class Sample:
    key: str
    cands: tuple[int, ...]  # consistent hypothesis indices
    label: int
    pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()  # (before, after) evidence


def _batch_grads(samples, rng, cap, fwd, accum, vs) -> float:  # type: ignore[no-untyped-def]
    loss = 0.0
    for smp in samples:
        c = list(smp.cands)
        if len(c) > cap:
            c = rng.sample([x for x in c if x != smp.label], cap - 1) + [smp.label]
        hs = fwd(c)
        scores = [sum(v * h for v, h in zip(vs, hi, strict=True)) for hi in hs]
        p = softmax(scores)
        loss -= math.log(max(p[c.index(smp.label)], 1e-12))
        accum(c, hs, [(pi - (1.0 if ci == smp.label else 0.0)) for pi, ci in zip(p, c, strict=True)])
    return loss / max(1, len(samples))


def pretrain(base: Base, samples: list[Sample], steps: int, lr: float, seed: int,
             batch: int = 16, cap: int = 48) -> float:
    """Full update of W (and b) on labeled samples. v stays fixed."""
    rng = random.Random(f"pretrain/{seed}")
    last = 0.0
    for _ in range(steps):
        mb = rng.sample(samples, min(batch, len(samples)))
        gW = [[0.0] * D for _ in range(H)]
        gb = [0.0] * H

        def fwd(c: list[int]) -> list[list[float]]:
            hs = []
            for i in c:
                f = phi(i)
                hs.append([math.tanh(sum(w * x for w, x in zip(row, f, strict=True)) + bi)
                           for row, bi in zip(base.W, base.b, strict=True)])
            return hs

        def accum(c: list[int], hs: list[list[float]], dp: list[float]) -> None:
            for ci, hi, d in zip(c, hs, dp, strict=True):
                f = phi(ci)
                for i in range(H):
                    dz = d * base.v[i] * (1 - hi[i] ** 2)
                    gb[i] += dz
                    for k in range(D):
                        gW[i][k] += dz * f[k]

        last = _batch_grads(mb, rng, cap, fwd, accum, base.v)
        n = len(mb)
        for i in range(H):
            base.b[i] -= lr * gb[i] / n
            for k in range(D):
                base.W[i][k] -= lr * gW[i][k] / n
    base._z0.clear()
    return last


def train_adapter(base: Base, adapter: Adapter, samples: list[Sample], steps: int, lr: float,
                  seed: int, batch: int = 16, cap: int = 48) -> float:
    """Update only A and B. The base object is never mutated."""
    rng = random.Random(f"adapt/{seed}")
    r = len(adapter.A)
    prior = Prior(base, adapter)
    last = 0.0
    for _ in range(steps):
        mb = rng.sample(samples, min(batch, len(samples)))
        gA = [[0.0] * D for _ in range(r)]
        gB = [[0.0] * r for _ in range(H)]

        def fwd(c: list[int]) -> list[list[float]]:
            return [prior.hidden(i) for i in c]

        def accum(c: list[int], hs: list[list[float]], dp: list[float]) -> None:
            for ci, hi, d in zip(c, hs, dp, strict=True):
                f, u = phi(ci), prior._u(ci)
                dz = [d * base.v[i] * (1 - hi[i] ** 2) for i in range(H)]
                for i in range(H):
                    for k in range(r):
                        gB[i][k] += dz[i] * u[k]
                for k in range(r):
                    du = sum(adapter.B[i][k] * dz[i] for i in range(H))
                    for dd in range(D):
                        gA[k][dd] += du * f[dd]

        last = _batch_grads(mb, rng, cap, fwd, accum, base.v)
        n = len(mb)
        for i in range(H):
            for k in range(r):
                adapter.B[i][k] -= lr * gB[i][k] / n
        for k in range(r):
            for dd in range(D):
                adapter.A[k][dd] -= lr * gA[k][dd] / n
    return last


def norm_entropy(labels: list[int]) -> float:
    """Normalized Shannon entropy of hypothesis signatures (diversity of generated tasks)."""
    from hsi.agents.learner import HYPOTHESES

    sig: dict[str, int] = {}
    for x in labels:
        s = signature(HYPOTHESES[x])
        sig[s] = sig.get(s, 0) + 1
    n = len(labels)
    return -sum(c / n * math.log(c / n) for c in sig.values()) / math.log(20) if n else 0.0
