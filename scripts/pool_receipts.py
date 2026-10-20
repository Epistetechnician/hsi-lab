"""Pooled paired analysis across two evaluations of the same arms (exploratory, not a receipt).

usage: pool_receipts.py <results.jsonl> <results.jsonl> ...   (locked-evaluator per-task caches)
Families are weighted as in the packet estimator: mean of per-family paired differences, with the same
variance formula, over all families of all inputs. Stdlib only.
"""

from __future__ import annotations

import json
import math
import sys

PAIRS = [("d2_plus", "d2_ex"), ("d2_plus", "d2_none"), ("d2_ex", "d2_none"),
         ("d2_plus", "frozen_base"), ("d1_plus", "d1_ex")]  # fmt: skip


def load(paths: list[str]) -> dict[tuple[str, str, str], dict[int, bool]]:
    out: dict[tuple[str, str, str], dict[int, bool]] = {}
    for k, p in enumerate(paths):
        for ln in open(p, encoding="utf-8").read().splitlines()[1:]:
            d = json.loads(ln)
            out.setdefault((str(k), d["f"], d["a"]), {})[d["s"]] = bool(d["ok"])
    return out


def pooled(res: dict[tuple[str, str, str], dict[int, bool]], t: str, b: str, z: float = 1.96):
    fams = sorted({(k, f) for k, f, _ in res})
    ds, vs = [], []
    for k, f in fams:
        x, y = res[(k, f, t)], res[(k, f, b)]
        seeds = sorted(set(x) & set(y))
        n = len(seeds)
        ot = sum(x[s] and not y[s] for s in seeds)
        ob = sum(y[s] and not x[s] for s in seeds)
        ds.append((ot - ob) / n)
        vs.append((ot + ob - (ot - ob) ** 2 / n) / n**2)
    m = sum(ds) / len(ds)
    se = math.sqrt(sum(vs)) / len(ds)
    return m, m - z * se, m + z * se, sum(len(res[(k, f, t)]) for k, f in fams)


if __name__ == "__main__":
    r = load(sys.argv[1:])
    for t, b in PAIRS:
        m, lo, hi, n = pooled(r, t, b)
        print(f"{t} - {b}: {m:+.4f}  [{lo:+.3f}, {hi:+.3f}]  tasks={n}")
