"""E8 pre-run audit: how often do weak test tiers wrongly pass base-model code?

usage: code_audit.py <out.json> [samples_per_task=8] [temp=0.8]
Greedy on every non-broken task, plus K sampled proposals per tune task. Reports pass rates per tier
and the false-positive share of each tier against the strongest (plus). Rows carry the text.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hsi.learning.llm import LLM
from hsi.verification.codetests import TIERS, extract_code, passes, split_ids, tasks, user_prompt


def main(out: str, k: int, temp: float) -> None:
    llm = LLM(name="qwen2.5-1.5b")
    T = tasks()
    tune, held = split_ids()
    jobs = [(i, "greedy", 0, 0.0) for i in tune + held]
    jobs += [(i, "sample", j, temp) for i in tune for j in range(k)]
    texts = []
    for n, (i, _kind, j, tp) in enumerate(jobs):
        texts.append(llm.complete(llm.chat(user_prompt(T[i])), 320, tp, seed=1000 * i + j))
        if n % 50 == 0:
            print(f"{n}/{len(jobs)}", flush=True)

    def score(a: tuple[tuple[int, str, int, float], str]) -> dict[str, object]:
        (i, kind, j, _), t = a
        code = extract_code(t)
        return {"task": i, "kind": kind, "j": j, "text": t,
                **{tier: passes(code, T[i], tier) for tier in TIERS}}  # fmt: skip

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(score, zip(jobs, texts, strict=True)))
    rep: dict[str, object] = {}
    for kind in ("greedy", "sample"):
        for pool, ids in (("tune", tune), ("held", held)):
            rs = [r for r in rows if r["kind"] == kind and r["task"] in set(ids)]
            if not rs:
                continue
            rep[f"{kind}/{pool}"] = {
                "n": len(rs), **{t: sum(bool(r[t]) for r in rs) / len(rs) for t in TIERS},
                **{f"fp_{t}": 1 - sum(bool(r["plus"]) for r in rs if r[t])
                   / max(1, sum(bool(r[t]) for r in rs)) for t in TIERS[:-1]},
            }  # fmt: skip
    Path(out).write_text(json.dumps({"summary": rep, "rows": rows}))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8,
         float(sys.argv[3]) if len(sys.argv) > 3 else 0.8)  # fmt: skip
