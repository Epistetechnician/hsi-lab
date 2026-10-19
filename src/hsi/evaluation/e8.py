"""E8: does test strength matter when a code model trains on its own verified solutions?

Chains of LoRA adapters (depth G) on MBPP+ tune tasks. Generation g samples K solutions per tune
task with the previous model, keeps those that pass the chain's TEST TIER (none/ex/base/plus), fills
a fixed quota, and trains a fresh adapter on the kept solutions. Evaluation: greedy pass@1 on
held-out tasks under the strongest tier (plus). Held-out tasks are never read by any chain
(asserted); the lock only fixes the order. Treatment d2_plus vs baseline d2_ex (pre-registered).
"""

from __future__ import annotations

import hashlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from hsi.contracts.io import load_json
from hsi.contracts.packet import Packet, canonical_digest
from hsi.contracts.records import Comparison, Split
from hsi.environment import SEED_BASE
from hsi.evaluation.base import TaskResult
from hsi.evaluation.learned import _save
from hsi.progress import Progress
from hsi.verification import DecisionLog
from hsi.verification.codetests import (
    TIERS,
    Task,
    extract_code,
    passes,
    split_ids,
    tasks,
    user_prompt,
)

_STATE: dict[str, Any] = {}


def buckets(packet: Packet, pool: str) -> dict[str, list[int]]:
    """family -> task ids for the tune or held-out pool: equal-sized, hash-assigned."""
    tune, held = split_ids(float(packet.params["tune_frac"]))
    ids = tune if pool == "tune" else held
    ordered = sorted(ids, key=lambda i: hashlib.sha256(f"e8-bucket/{i}".encode()).hexdigest())
    fams = list(packet.params["families"])
    size = len(ordered) // len(fams)
    out = {f: sorted(ordered[k :: len(fams)][:size]) for k, f in enumerate(fams)}
    return out


def task_for(packet: Packet, family: str, split: Split, seed: int) -> Task:
    pool = "held" if split is Split.HELDOUT else "tune"
    ids = buckets(packet, pool)[family]
    mod = int(packet.params["holdout_mod"])
    if pool == "held" and len(ids) != mod:
        raise ValueError("holdout_mod must equal the held-out bucket size")
    return tasks()[ids[(seed - SEED_BASE[split]) % len(ids)]]


def _llm(packet: Packet) -> Any:
    from hsi.evaluation.llm_stack import get_llm

    return get_llm(packet)


def _ensure(packet: Packet) -> Any:
    from hsi.evaluation.llm_stack import _ensure_adapter

    return _ensure_adapter(packet)


def _sig(code: str) -> str:
    return hashlib.sha256("".join(code.split()).encode()).hexdigest()[:16]


def _chain(
    packet: Packet, run_dir: Path, llm: Any, log: DecisionLog, cands: dict[str, Any],
    spec: dict[str, Any], prog: Progress, persist: Any,
) -> None:  # fmt: skip
    q = packet.params
    ld = run_dir / "learning"
    T = tasks()
    tune_ids = sorted(i for b in buckets(packet, "tune").values() for i in b)
    held_ids = {i for b in buckets(packet, "held").values() for i in b}
    assert not held_ids & set(tune_ids), "tune pool overlaps held-out pool"
    name, tier = str(spec["name"]), str(spec["tier"])
    seed = int(packet.seeds[0]) + int(spec["seed_offset"])
    quota, k, per_task = int(q["quota"]), int(q["k"]), int(q["per_task"])
    llm.set_adapter(False)  # generation 1 is produced by the frozen base
    for g in range(1, int(q["depth"]) + 1):
        arm = f"d{g}_{name}"
        if arm in cands:
            llm.load_adapter(ld / f"{arm}.safetensors")  # resume: this generation already exists
            llm.set_adapter(True)
            continue
        jobs = [(i, j) for i in tune_ids for j in range(k)]
        cache = ld / "proposals.d1.json"
        if g == 1 and cache.exists():  # generation 1 is chain-independent: shared proposals
            texts = load_json(cache)["texts"]  # type: ignore[assignment,index]
            assert len(texts) == len(jobs)
        else:
            base_seed = int(packet.seeds[0]) if g == 1 else seed
            texts = []
            for n, (i, j) in enumerate(jobs):
                texts.append(
                    llm.complete(
                        llm.chat(user_prompt(T[i])),
                        int(q["max_tokens"]),
                        float(q["temp"]),
                        seed=base_seed * 100003 + g * 1009 + i * 31 + j,
                    )
                )
                if n % 20 == 0:
                    prog.update(
                        len(cands), int(q["n_adapters"]), f"{arm}: proposing {n}/{len(jobs)}"
                    )
            if g == 1:
                _save(cache, {"texts": texts})
        codes = [extract_code(t) for t in texts]
        with ThreadPoolExecutor(int(q["workers"])) as ex:
            ok = list(
                ex.map(lambda a: passes(a[1], T[a[0][0]], tier), zip(jobs, codes, strict=True))
            )
            truth = list(  # audit only: never feeds selection
                ex.map(
                    lambda a: bool(a[2]) and passes(a[1], T[a[0][0]], "plus"),
                    [(jb, c, o) for jb, c, o in zip(jobs, codes, ok, strict=True)],
                )
            )
        rng = random.Random(f"e8/{seed}/{g}")
        by_task: dict[int, list[int]] = {}
        for n, ((i, _), o) in enumerate(zip(jobs, ok, strict=True)):
            if o:
                by_task.setdefault(i, []).append(n)
        kept: list[int] = []
        for i, idx in sorted(by_task.items()):
            seen: set[str] = set()
            rng.shuffle(idx)
            for n in idx:
                s = _sig(str(codes[n]))
                if s not in seen and len([x for x in kept if jobs[x][0] == i]) < per_task:
                    seen.add(s)
                    kept.append(n)
        if len(kept) > quota:
            kept = sorted(rng.sample(kept, quota))
        data = [
            (llm.chat(user_prompt(T[jobs[n][0]])), f"```python\n{codes[n]}\n```<|im_end|>")
            for n in kept
        ]
        for n, ((i, j), t, o) in enumerate(zip(jobs, texts, ok, strict=True)):
            log.append(
                {"sample_id": f"{arm}/{i}/{j}", "arm": arm, "gen": g, "origin": "generated",
                 "eligibility": "eligible" if n in set(kept) else "rejected",
                 "label_source": f"tests:{tier}", "sig": _sig(t), "passed_tier": o}
            )  # fmt: skip
        llm.reset_lora(seed + g)
        steps = int(q["steps"])
        loss = llm.train_lora(
            data, steps, float(q["lr"]), seed + g, int(q["batch"]),
            on_step=lambda i, ls, a=arm, st=steps: prog.update(
                len(cands), int(q["n_adapters"]), f"{a}: step {i}/{st} loss {ls:.2f}"
            ),
            lead="",
        )  # fmt: skip
        digest = llm.save_adapter(ld / f"{arm}.safetensors")
        valid = sum(c is not None for c in codes) / len(codes)
        cands[arm] = {
            "generation": g, "loss": loss, "digest": digest, "chain": name, "tier": tier,
            "n_proposals": len(jobs), "validity": valid,
            "accept_rate": sum(ok) / len(ok),
            "pool_precision": sum(truth) / max(1, sum(ok)),  # P(plus | passed tier), audit
            "n_synthetic": len(data), "n_real": 0, "shortfall": max(0, quota - len(data)),
            "n_correct": sum(bool(truth[n]) for n in kept),
            "kept_precision": sum(bool(truth[n]) for n in kept) / max(1, len(kept)),
            "tasks_covered": len({jobs[n][0] for n in kept}),
            "diversity": len({_sig(str(codes[n])) for n in kept}) / max(1, len(kept)),
        }  # fmt: skip
        persist()
        # live weights now equal adapter g: the next generation proposes with it


class E8Spec:
    manifest: dict[str, Any] | None = None
    run_dir: Path | None = None

    def prepare(self, packet: Packet, run_dir: Path) -> None:
        from hsi.learning.llm import MODELS, weights_digest

        q = packet.params
        llm = _ensure(packet)
        ld = run_dir / "learning"
        ld.mkdir(parents=True, exist_ok=True)
        log = DecisionLog(ld / "training.jsonl")
        prog = Progress(run_dir, "pilot:prepare")
        pm = ld / "manifest.partial.json"
        cands: dict[str, Any] = load_json(pm)["candidates"] if pm.exists() else {}  # type: ignore[assignment]
        q = {**q, "n_adapters": len(q["chains"]) * int(q["depth"])}
        packet2 = packet.model_copy(update={"params": q})

        def persist() -> None:
            _save(pm, {"candidates": cands})

        for spec in q["chains"]:
            _chain(packet2, run_dir, llm, log, cands, spec, prog, persist)
        prog.finish()
        man: dict[str, object] = {
            "packet_digest": canonical_digest(packet.model_dump(mode="json")),
            "runtime": llm.runtime, "python": sys.version.split()[0], "seeds": packet.seeds,
            "backbone": "llm", "model_repo": MODELS[llm.name][0],
            "model_revision": MODELS[llm.name][1], "model_license": MODELS[llm.name][2],
            "model_name": llm.name, "weights_digest": weights_digest(llm.path),
            "candidates": cands, "training_log_head": log.head,
        }  # fmt: skip
        man["manifest_digest"] = canonical_digest(man)
        _save(ld / "manifest.json", man)

    def bind(self, packet: Packet, run_dir: Path) -> None:
        from hsi.evaluation.llm_stack import _STATE as S
        from hsi.evaluation.llm_stack import bind_e6_llm

        self.manifest, self.run_dir = bind_e6_llm(packet, run_dir), run_dir
        S["run_dir"] = run_dir
        _STATE["arm"] = None

    def model_digest(self) -> str | None:
        if self.run_dir is None:
            return None
        from hsi.evaluation.llm_stack import model_digest_e6

        return model_digest_e6(self.run_dir)

    def run_task(
        self, arm: str, family: str, split: Split, seed: int, packet: Packet
    ) -> TaskResult:
        from hsi.evaluation.llm_stack import _STATE as S

        llm = _llm(packet)
        if arm == "frozen_base":
            llm.set_adapter(False)
            _STATE["arm"] = None
        elif _STATE.get("arm") != arm:
            llm.load_adapter(S["run_dir"] / "learning" / f"{arm}.safetensors")
            llm.set_adapter(True)
            _STATE["arm"] = arm
        t = task_for(packet, family, split, seed)
        text = llm.complete(llm.chat(user_prompt(t)), int(packet.params["max_tokens"]), 0.0, 0)
        code = extract_code(text)
        res = {tier: passes(code, t, tier) for tier in TIERS[1:]}
        return TaskResult(
            arm, family, seed, res["plus"], [],
            {"calls": 1.0, "valid": float(code is not None), **{f"pass_{k}": float(v) for k, v in res.items()}},  # noqa: E501
            decisions=[{"task": t.task_id, "sig": _sig(code or ""), "valid": code is not None}],
        )  # fmt: skip

    def check_budget(self, r: TaskResult, packet: Packet) -> None:
        return None

    def guardrails(
        self, results: list[TaskResult], comps: list[Comparison], packet: Packet
    ) -> tuple[dict[str, float], bool]:
        assert self.manifest is not None
        q = packet.params
        gens, suf = int(q["depth"]), str(q["guard_suffix"])

        def rate(arm: str) -> float:
            rs = [r for r in results if r.arm == arm]
            return sum(r.success for r in rs) / len(rs)

        gv = {
            f"rate_d1_{suf}": rate(f"d1_{suf}"),
            f"rate_d{gens}_{suf}": rate(f"d{gens}_{suf}"),
            "non_collapse_floor": float(q["non_collapse_floor"]),
        }
        for arm in sorted({r.arm for r in results}):
            rs = [r for r in results if r.arm == arm]
            for k in ("valid", "pass_ex", "pass_base"):
                gv[f"eval_{k}_{arm}"] = sum(r.stats[k] for r in rs) / len(rs)
        held = gv[f"rate_d{gens}_{suf}"] - gv[f"rate_d1_{suf}"] >= -gv["non_collapse_floor"]
        return gv, held
