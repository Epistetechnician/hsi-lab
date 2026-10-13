"""Paired family-level statistics. Duplicated independently in validator/recompute.py."""
from __future__ import annotations

import math

from hsi.contracts.packet import Outcome
from hsi.contracts.records import PairedFamily

Z95 = 1.96


def family_diff(p: PairedFamily) -> tuple[float, float]:
    d = (p.only_treatment - p.only_baseline) / p.n
    var = (p.only_treatment + p.only_baseline - (p.only_treatment - p.only_baseline) ** 2 / p.n) / p.n**2
    return d, var


def overall(pairs: list[PairedFamily]) -> tuple[float, float, float]:
    f = len(pairs)
    ds = [family_diff(p) for p in pairs]
    mean = sum(d for d, _ in ds) / f
    se = math.sqrt(sum(v for _, v in ds)) / f
    return mean, mean - Z95 * se, mean + Z95 * se


def decide(lo: float, hi: float, threshold: float, guardrails_held: bool) -> Outcome:
    if not guardrails_held or hi < threshold:
        return Outcome.NO_CANDIDATE
    return Outcome.PROMOTE if lo >= threshold else Outcome.INCONCLUSIVE
