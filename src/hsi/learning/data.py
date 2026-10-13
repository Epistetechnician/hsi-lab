"""Training-sample construction. Refuses held-out data (learning never reads it)."""
from __future__ import annotations

import random

from hsi.agents.learner import consistent
from hsi.contracts.records import Split
from hsi.environment import SEED_BASE, RuleWorld
from hsi.learning.prior import true_index
from hsi.learning.train import Sample


def build_samples(families: list[str], split: Split, seeds: range, support: int,
                  labeled: bool) -> tuple[list[Sample], list[dict[str, object]]]:
    """One sample per (world, action) with observations. Labels come from the oracle world
    (labeled) or are left at -1 for the caller to pseudo-label."""
    if split is Split.HELDOUT:
        raise PermissionError("learning may not read held-out data")
    out, log = [], []
    for fam in families:
        for i in seeds:
            seed = SEED_BASE[split] + i
            env = RuleWorld(fam, split, seed)
            rng = random.Random(f"train-walk/{fam}/{seed}")
            obs, _ = env.reset()
            s = tuple(obs)
            per: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {a: [] for a in range(4)}
            for _ in range(support):
                a = rng.randrange(4)
                o, *_ = env.step(a)
                per[a].append((s, tuple(o)))
                s = tuple(o)
            for a, pairs in per.items():
                if not pairs:
                    continue
                cands = consistent(pairs)
                lab = true_index(env.task.actions[a]) if labeled else -1
                if not cands or (labeled and lab not in cands):
                    continue
                key = f"{fam}-{split.value}-{seed}/a{a}"
                out.append(Sample(key, tuple(cands), -1 if lab is None else lab))
                log.append({"sample_id": key, "split": split.value,
                            "origin": "observed" if labeled else "unlabeled-world",
                            "eligibility": "eligible", "label_source": "oracle" if labeled else "none"})
    return out, log
