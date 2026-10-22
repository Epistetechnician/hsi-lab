"""E9 supply audit (ADR 0008): how many probe-correct and consistent-but-wrong proposals does the
600-world proposal pool actually yield, at depth 1 (frozen base) and at depth 2 (proxy adapters)?

No held-out data is read. The pool is the training pool of the v8-v12 chains (worlds 1000-1600).

usage: python scripts/e9_supply_audit.py <out.json> [stage ...]
       stages: base d1_verified d1_poolctl (default: all). Reads adapters from the v9 run root.
       A stage may carry a pool prefix limit, e.g. d1_verified:1000 (the first 1000 pool samples;
       supply scales linearly, so the full-pool supply is the count times pool / limit).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hsi.cli import template
from hsi.contracts.packet import Packet
from hsi.contracts.records import Split
from hsi.evaluation.llm_stack import _demos, _ensure_adapter, verify_texts
from hsi.learning.data import build_samples
from hsi.learning.llm import few_shot_prompt, parse_hypothesis
from hsi.paths import run_root
from hsi.verification.verifiers import probe_correct, true_rule

V9 = "fda4e460ea88bf7b"


def main(out: str, stages: list[str]) -> None:
    packet = Packet.model_validate(template("E6", "llm-noisyver"), strict=False)
    q = packet.params
    llm = _ensure_adapter(packet)
    fams, sup = list(q["pretrain_families"]), int(q["support"])
    base = int(q["verify_pool_base"])
    pool, _ = build_samples(
        fams, Split.TUNE, range(base, base + int(q["verify_pool_worlds"])), sup, False
    )
    demos = _demos(packet)
    path = Path(out)
    res: dict = json.loads(path.read_text()) if path.exists() else {}
    for spec in stages:
        st, _, lim = spec.partition(":")
        if spec in res:
            continue
        sub = pool[: int(lim)] if lim else pool
        if st == "base":
            llm.set_adapter(False)
        else:
            llm.load_adapter(run_root() / V9 / "learning" / f"{st}.safetensors")
            llm.set_adapter(True)
        n_valid = n_cons = n_corr = 0
        fam_stats: dict[str, list[int]] = {}
        for i, s in enumerate(sub):
            t = llm.generate(few_shot_prompt(s.pairs, demos), int(q["gen_max_tokens"])) or "unknown"
            h = parse_hypothesis(t)
            ok = verify_texts([t], [s])[0]
            c = probe_correct(h, true_rule(s.key), s.key)
            n_valid += h is not None
            n_cons += ok
            n_corr += c
            f = fam_stats.setdefault(s.key.rsplit("-", 2)[0], [0, 0, 0])
            f[0] += 1
            f[1] += ok
            f[2] += c
            if (i + 1) % 250 == 0:
                print(f"{spec} {i + 1}/{len(sub)} cons={n_cons} correct={n_corr}", flush=True)
        res[spec] = {
            "pool": len(sub), "full_pool": len(pool), "valid": n_valid,
            "consistent": n_cons, "correct": n_corr, "consistent_wrong": n_cons - n_corr,
            "by_family": {
                k: dict(zip(("n", "consistent", "correct"), v, strict=True))
                for k, v in fam_stats.items()
            },
        }  # fmt: skip
        path.write_text(json.dumps(res, indent=1, sort_keys=True))
        print(
            spec, json.dumps({k: v for k, v in res[spec].items() if k != "by_family"}), flush=True
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["base", "d1_verified", "d1_poolctl"])
