"""Verifier audit for ADR 0003 (no held-out data is read).

  kept       how correct were the examples the existing chains trained on? (offline, from logs)
  proposals  how good are candidate verifiers on fresh proposals from the frozen model? (MLX)

usage: python scripts/verifier_audit.py kept   <run_dir> <log> <out.json>
       python scripts/verifier_audit.py proposals <frozen_packet> <out.json> [n_proposals]
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from hsi.learning.llm import parse_hypothesis
from hsi.learning.prior import true_index
from hsi.verification.verifiers import probe_correct, true_rule


def _pc(text: str, key: str) -> tuple[bool, bool]:
    h = parse_hypothesis(text)
    return h is not None, probe_correct(h, true_rule(key), key)


def kept(run_dir: str, log: str, out: str) -> None:
    """Precision of the kept (generated) sets of an earlier run, per arm, against the truth."""
    from hsi.contracts.records import Split
    from hsi.learning.data import build_samples
    from hsi.verification.verifiers import world_of

    rows = [json.loads(x) for x in (Path(run_dir) / "learning" / log).read_text().splitlines()]
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("origin") == "generated":
            by[r["arm"]].append(r)
    res = {}
    for arm, rs in sorted(by.items()):
        n = len(rs)
        valid = sum(parse_hypothesis(r["text"]) is not None for r in rs)
        correct = sum(_pc(r["text"], r["sample_id"])[1] for r in rs)
        exact = 0
        cons = 0
        fam_seed: dict[tuple, list] = {}
        for r in rs:
            fam, split, seed, a = world_of(r["sample_id"])
            h = parse_hypothesis(r["text"])
            idx = true_index(h) if h else None
            smp = fam_seed.get((fam, seed))
            if smp is None:
                s, _ = build_samples(
                    [fam], Split(split), range(seed - 1_000_000, seed - 999_999), 8, False
                )
                smp = fam_seed[(fam, seed)] = {x.key: x for x in s}
            sm = smp.get(r["sample_id"])
            cons += bool(sm and idx is not None and idx in sm.cands)
            exact += idx is not None and idx == true_index(true_rule(r["sample_id"]))
        res[arm] = {
            "n": n, "valid": valid / n, "consistent": cons / n,
            "probe_correct": correct / n, "exact": exact / n,
        }  # fmt: skip
        print(arm, json.dumps(res[arm]))
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True))


def _auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def proposals(packet_path: str, out: str, n: int) -> None:
    """Generate from the frozen model on audit worlds and score every candidate verifier."""
    from hsi.cli import template
    from hsi.contracts.packet import Packet
    from hsi.contracts.records import Split
    from hsi.evaluation.llm_stack import _demos, get_llm, verify_texts
    from hsi.learning.data import build_samples
    from hsi.learning.llm import few_shot_prompt

    t = template("E6", packet_path)  # packet_path is a template variant, e.g. llm-poolctl
    t["params"] = {**t["params"], "audit_base": 1600, "audit_worlds": 300}
    packet = Packet.model_validate(t, strict=False)
    q = packet.params
    llm = get_llm(packet)
    llm.set_adapter(False) if llm._lora else None
    fams, sup = list(q["pretrain_families"]), int(q["support"])
    base = int(q["audit_base"])
    pool, _ = build_samples(
        fams, Split.TUNE, range(base, base + int(q["audit_worlds"])), sup, False
    )
    random.Random("audit/proposals").shuffle(pool)
    pool = pool[:n]
    demos = _demos(packet)
    rows = []
    for i, s in enumerate(pool):
        prompt = few_shot_prompt(s.pairs, demos)
        t = llm.generate(prompt, int(q["gen_max_tokens"])) or "unknown"
        h = parse_hypothesis(t)
        ok = verify_texts([t], [s])[0]
        lp = llm.mean_logprob(prompt, t) if h is not None else None
        rows.append(
            {
                "key": s.key, "text": t, "valid": h is not None, "consistent": ok,
                "correct": probe_correct(h, true_rule(s.key), s.key),
                "mean_logprob": lp,
            }
        )  # fmt: skip
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(pool)}", flush=True)
    valid = [r for r in rows if r["valid"]]
    rep: dict = {
        "n": len(rows),
        "valid": len(valid) / len(rows),
        "consistent": sum(r["consistent"] for r in rows) / len(rows),
        "correct": sum(r["correct"] for r in rows) / len(rows),
        "correct_given_consistent": (
            sum(r["correct"] for r in rows if r["consistent"])
            / max(1, sum(r["consistent"] for r in rows))
        ),
        "auc_logprob_vs_consistent": _auc(
            [r["mean_logprob"] for r in valid if r["consistent"]],
            [r["mean_logprob"] for r in valid if not r["consistent"]],
        ),
        "auc_logprob_vs_correct": _auc(
            [r["mean_logprob"] for r in valid if r["correct"]],
            [r["mean_logprob"] for r in valid if not r["correct"]],
        ),
    }
    lps = sorted(r["mean_logprob"] for r in valid)
    rep["confidence"] = {}
    for pct in (50, 70, 80, 90):
        tau = lps[min(len(lps) - 1, int(len(lps) * pct / 100))]
        keep = [r for r in valid if r["mean_logprob"] >= tau]
        rep["confidence"][str(pct)] = {
            "tau": tau, "kept": len(keep),
            "precision_consistent": sum(r["consistent"] for r in keep) / max(1, len(keep)),
            "precision_correct": sum(r["correct"] for r in keep) / max(1, len(keep)),
        }  # fmt: skip
    a = rep["consistent"]
    rep["noisy_contamination"] = {
        str(p): (p * (1 - a)) / (a + p * (1 - a)) for p in (0.03, 0.1, 0.3, 1.0)
    }
    Path(out).write_text(json.dumps({"summary": rep, "rows": rows}, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "kept":
        kept(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        proposals(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 600)
