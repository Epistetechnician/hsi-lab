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
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from hsi.contracts.io import load_json
from hsi.contracts.packet import Packet, canonical_digest
from hsi.contracts.records import Comparison, Split
from hsi.environment import SEED_BASE
from hsi.errors import PreconditionError
from hsi.evaluation.base import TaskResult
from hsi.evaluation.learned import _save
from hsi.progress import Progress
from hsi.verification import DecisionLog
from hsi.verification.codetests import (
    TIERS,
    extract_code,
    passes,
    split_ids,
    tasks,
    user_prompt,
)

_STATE: dict[str, Any] = {}


def _equal_buckets(ids: list[Any], fams: list[str], key: str) -> dict[str, list[Any]]:
    ordered = sorted(ids, key=lambda i: hashlib.sha256(f"{key}/{i}".encode()).hexdigest())
    size = len(ordered) // len(fams)
    return {f: sorted(ordered[k :: len(fams)][:size]) for k, f in enumerate(fams)}


def buckets(packet: Packet, pool: str) -> dict[str, list[Any]]:
    """family -> task ids for the tune or held-out pool: equal-sized, hash-assigned.

    v1-v2: families split the MBPP+ held-out pool. v3: eval_pool="humaneval" (HumanEval+ only).
    v4: `eval_pools` maps pool name ("mbpp"/"he") to its families; every family holds exactly
    `holdout_mod` tasks (the largest equal share that fits). `tune_families` (possibly repeated, one
    per family) names the MBPP+ tune buckets that the pilot/run stages score."""
    q = packet.params
    tune, held = split_ids(float(q["tune_frac"]))
    if pool == "tune":
        fams = list(dict.fromkeys(q.get("tune_families", q["families"])))
        return _equal_buckets(tune, fams, "e8-bucket")
    if "eval_pools" in q:
        from hsi.verification.humaneval import pool as he_pool

        out: dict[str, list[Any]] = {}
        src: dict[str, tuple[list[Any], str]] = {
            "mbpp": (list(held), "e8-bucket"),
            "he": (list(he_pool()), "e8-he-bucket"),
        }
        for name, fams in q["eval_pools"].items():
            ids, key = src[name]
            b = _equal_buckets(ids, list(fams), key)
            out |= {f: v[: int(q["holdout_mod"])] for f, v in b.items()}
        return out
    if q.get("eval_pool") == "humaneval":
        from hsi.verification.humaneval import pool as he_pool

        return _equal_buckets(he_pool(), list(q["families"]), "e8-he-bucket")
    return _equal_buckets(held, list(q["families"]), "e8-bucket")


def is_he(packet: Packet, family: str, split: Split) -> bool:
    q = packet.params
    if split is not Split.HELDOUT:
        return False
    if "eval_pools" in q:
        return family in q["eval_pools"].get("he", [])
    return q.get("eval_pool") == "humaneval"


def task_for(packet: Packet, family: str, split: Split, seed: int) -> Any:
    """The Task (MBPP+) or HETask (HumanEval+) for a (family, split, seed)."""
    held = split is Split.HELDOUT
    fams = list(packet.params["families"])
    if not held and "tune_families" in packet.params:  # pilot/run stages score MBPP+ tune tasks
        family = packet.params["tune_families"][fams.index(family)]
    ids = buckets(packet, "held" if held else "tune")[family]
    mod = int(packet.params["holdout_mod"])
    if held and len(ids) != mod:
        raise ValueError("holdout_mod must equal the held-out bucket size")
    i = ids[(seed - SEED_BASE[split]) % len(ids)]
    if is_he(packet, family, split):
        from hsi.verification.humaneval import tasks as he_tasks

        return he_tasks()[i]
    return tasks()[i]


def expand(chains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One spec per replicate chain: name_r{i} when a chain has replicates > 1."""
    out = []
    for c in chains:
        reps = int(c.get("replicates", 1))
        for r in range(reps):
            out.append(
                {
                    "name": f"{c['name']}_r{r}" if reps > 1 else c["name"],
                    "tier": c["tier"],
                    "seed_offset": int(c["seed_offset"]) + 7 * r,
                }
            )
    return out


def adapter_for(packet: Packet, arm: str, task_id: int | str) -> str:
    """Adapter file stem for an arm. A replicated chain is a mixture: each task is answered by one
    replicate, chosen by a hash of (arm, task), so chain variation averages across tasks."""
    m = re.match(r"d(\d+)_(.+)$", arm)
    if m:
        reps = next(
            (int(c.get("replicates", 1)) for c in packet.params["chains"] if c["name"] == m[2]), 1
        )
        if reps > 1:
            h = int(hashlib.sha256(f"{arm}/{task_id}".encode()).hexdigest()[:8], 16)
            return f"{arm}_r{h % reps}"
    return arm


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
            texts: list[str] = list(load_json(cache)["texts"])  # type: ignore[call-overload]
            assert len(texts) == len(jobs)
        else:
            base_seed = int(packet.seeds[0]) if g == 1 else seed
            texts = []
            bs = int(q.get("gen_batch", 1))
            for n0 in range(0, len(jobs), bs):
                part = jobs[n0 : n0 + bs]
                seeds = [base_seed * 100003 + g * 1009 + i * 31 + j for i, j in part]
                texts += llm.complete_many(
                    [llm.chat(user_prompt(T[i])) for i, _ in part],
                    int(q["max_tokens"]), float(q["temp"]), seeds,
                )  # fmt: skip
                prog.update(
                    len(cands), int(q["n_adapters"]), f"{arm}: proposing {len(texts)}/{len(jobs)}"
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
        from hsi.evaluation.llm_stack import try_reuse_adapters

        if try_reuse_adapters(packet, run_dir):
            return
        if q.get("reuse_from"):  # never silently retrain what the packet pins
            raise PreconditionError(f"cannot reuse adapters from run {q['reuse_from']}")
        llm = _ensure(packet)
        ld = run_dir / "learning"
        ld.mkdir(parents=True, exist_ok=True)
        log = DecisionLog(ld / "training.jsonl")
        prog = Progress(run_dir, "pilot:prepare")
        pm = ld / "manifest.partial.json"
        cands: dict[str, Any] = load_json(pm)["candidates"] if pm.exists() else {}  # type: ignore[assignment]
        q = {**q, "n_adapters": len(expand(q["chains"])) * int(q["depth"])}
        packet2 = packet.model_copy(update={"params": q})

        def persist() -> None:
            _save(pm, {"candidates": cands})

        for spec in expand(q["chains"]):
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

    def _generate_one(self, llm: Any, packet: Packet, file: str, prompt: str, run_dir: Path) -> str:
        if file == "frozen_base":
            llm.set_adapter(False)
            _STATE["arm"] = None
        elif _STATE.get("arm") != file:
            llm.load_adapter(run_dir / "learning" / f"{file}.safetensors")
            llm.set_adapter(True)
            _STATE["arm"] = file
        return str(llm.complete(llm.chat(prompt), int(packet.params["max_tokens"]), 0.0, 0))

    def _prefetch(self, llm: Any, packet: Packet, arm: str, family: str, run_dir: Path) -> None:
        """Held-out evaluation visits every task of a family bucket once per arm: generate all of
        its greedy replies in batches, grouped by adapter file, then serve run_task
        from the cache."""
        key = (arm, family)
        if key in _STATE.setdefault("gen", {}):
            return
        he = is_he(packet, family, Split.HELDOUT)
        by_file: dict[str, list[Any]] = {}
        for tid in buckets(packet, "held")[family]:
            t = (
                __import__("hsi.verification.humaneval", fromlist=["tasks"]).tasks()[tid]
                if he
                else tasks()[tid]
            )
            f = "frozen_base" if arm == "frozen_base" else adapter_for(packet, arm, t.task_id)
            by_file.setdefault(f, []).append(t)
        out: dict[Any, str] = {}
        bs = int(packet.params.get("gen_batch", 1))
        for f, ts in by_file.items():
            if f == "frozen_base":
                llm.set_adapter(False)
            else:
                llm.load_adapter(run_dir / "learning" / f"{f}.safetensors")
                llm.set_adapter(True)
            _STATE["arm"] = None if f == "frozen_base" else f
            if he:
                from hsi.verification.humaneval import user_prompt as hp

                ps = [llm.chat(hp(t)) for t in ts]
            else:
                ps = [llm.chat(user_prompt(t)) for t in ts]
            for n0 in range(0, len(ps), bs):
                texts = llm.complete_many(
                    ps[n0 : n0 + bs],
                    int(packet.params["max_tokens"]),
                    0.0,
                    [0] * len(ps[n0 : n0 + bs]),
                )
                for t, x in zip(ts[n0 : n0 + bs], texts, strict=True):
                    out[t.task_id] = x
        _STATE["gen"][key] = out

    def run_task(
        self, arm: str, family: str, split: Split, seed: int, packet: Packet
    ) -> TaskResult:
        from hsi.evaluation.llm_stack import _STATE as S

        llm = _llm(packet)
        t = task_for(packet, family, split, seed)
        file = "frozen_base" if arm == "frozen_base" else adapter_for(packet, arm, t.task_id)
        he = is_he(packet, family, split)
        if split is Split.HELDOUT and int(packet.params.get("gen_batch", 1)) > 1:
            self._prefetch(llm, packet, arm, family, S["run_dir"])
            text = _STATE["gen"][(arm, family)][t.task_id]
        else:
            if he:
                from hsi.verification.humaneval import user_prompt as hp

                prompt = hp(t)
            else:
                prompt = user_prompt(t)
            text = self._generate_one(llm, packet, file, prompt, S["run_dir"])
        code = extract_code(text)
        if he:
            from hsi.verification.humaneval import passes as he_passes

            res = {"plus": he_passes(code, t)}
        else:
            res = {tier: passes(code, t, tier) for tier in TIERS[1:]}
        stats = {"calls": 1.0, "valid": float(code is not None)}
        stats |= {f"pass_{k}": float(v) for k, v in res.items()}
        return TaskResult(
            arm, family, seed, res["plus"], [], stats,
            decisions=[
                {
                    "task": t.task_id,
                    "sig": _sig(code or ""),
                    "valid": code is not None,
                    "adapter": file,
                }
            ],
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
                have = [r for r in rs if k in r.stats]
                if have:
                    gv[f"eval_{k}_{arm}"] = sum(r.stats[k] for r in have) / len(have)
        held = gv[f"rate_d{gens}_{suf}"] - gv[f"rate_d1_{suf}"] >= -gv["non_collapse_floor"]
        return gv, held
