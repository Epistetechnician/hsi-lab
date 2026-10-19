"""Paired family-level statistics. Duplicated independently in validator/recompute.py."""

from __future__ import annotations

import math

from hsi.contracts.packet import Outcome
from hsi.contracts.records import PairedFamily

Z95 = 1.96


def family_diff(p: PairedFamily) -> tuple[float, float]:
    d = (p.only_treatment - p.only_baseline) / p.n
    var = (
        p.only_treatment + p.only_baseline - (p.only_treatment - p.only_baseline) ** 2 / p.n
    ) / p.n**2
    return d, var


def overall(pairs: list[PairedFamily], z: float = Z95) -> tuple[float, float, float]:
    f = len(pairs)
    ds = [family_diff(p) for p in pairs]
    mean = sum(d for d, _ in ds) / f
    se = math.sqrt(sum(v for _, v in ds)) / f
    return mean, mean - z * se, mean + z * se


def decide(cis: list[tuple[float, float]], threshold: float, guardrails_held: bool) -> Outcome:
    if not guardrails_held or any(hi < threshold for _, hi in cis):
        return Outcome.NO_CANDIDATE
    return Outcome.PROMOTE if all(lo >= threshold for lo, _ in cis) else Outcome.INCONCLUSIVE


def bonferroni_z(m: int) -> float:
    """Two-sided multiplier for a family-wise 95% level over m contrasts (m=1 gives 1.96)."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(1 - 0.025 / m)
