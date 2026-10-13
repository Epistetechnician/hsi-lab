"""Candidate training/loading for E4 and E6 (producer side; isolated from held-out data)."""
from __future__ import annotations

import json
import random
import stat
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from hsi.agents.learner import HYPOTHESES, Model, transitions_of
from hsi.agents.scripted import random_policy, rollout
from hsi.contracts.io import load_json, write_json
from hsi.contracts.packet import Packet, canonical_digest
from hsi.contracts.records import Split
from hsi.environment import RuleWorld
from hsi.errors import DigestError
from hsi.evaluation.base import TaskResult
from hsi.evaluation.e1 import teacher_trace
from hsi.learning import (
    Adapter,
    Base,
    Prior,
    Sample,
    norm_entropy,
    pretrain,
    signature,
    train_adapter,
)
from hsi.learning.data import build_samples
from hsi.learning.prior import RUNTIME
from hsi.verification import DecisionLog


class Stack:
    """Loaded, digest-verified base + candidates."""

    def __init__(self, base: Base, adapters: dict[str, Adapter], memory: dict[str, int],
                 manifest: dict[str, object]) -> None:
        self.base, self.adapters, self.memory, self.manifest = base, adapters, memory, manifest


def _save(path: Path, obj: object, readonly: bool = False) -> str:
    write_json(path, obj)
    if readonly:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return canonical_digest(obj)


def _fit_samples(q: dict[str, object]) -> list[Sample]:
    s, _ = build_samples(list(q["pretrain_families"]), Split.FIT,  # type: ignore[call-overload]
                         range(int(q["n_pretrain"])), int(q["support"]), True)  # type: ignore[call-overload]
    return s


def prepare_common(packet: Packet, run_dir: Path) -> tuple[Base, list[Sample], DecisionLog, dict[str, object]]:
    q = packet.params
    ld = run_dir / "learning"
    ld.mkdir(parents=True, exist_ok=True)
    log = DecisionLog(ld / "training.jsonl")
    pre, plog = build_samples(list(q["pretrain_families"]), Split.FIT, range(int(q["n_pretrain"])),
                              int(q["support"]), True)
    for r in plog:
        log.append({**r, "used_by": "base"})
    base = Base.init(int(packet.seeds[0]))
    loss = pretrain(base, pre, int(q["pretrain_steps"]), float(q["pretrain_lr"]), int(packet.seeds[0]))
    man: dict[str, object] = {
        "packet_digest": canonical_digest(packet.model_dump(mode="json")), "runtime": RUNTIME,
        "python": sys.version.split()[0], "seeds": packet.seeds, "base_digest": base.digest,
        "base_loss": loss, "candidates": {}}
    _save(ld / "base.json", base.state(), readonly=True)
    return base, pre, log, man


def finish(ld: Path, log: DecisionLog, man: dict[str, object]) -> None:
    man["training_log_head"] = log.head
    man["manifest_digest"] = canonical_digest({k: v for k, v in man.items() if k != "manifest_digest"})
    _save(ld / "manifest.json", man)


def prepare_e4(packet: Packet, run_dir: Path) -> None:
    q = packet.params
    base, _pre, log, man = prepare_common(packet, run_dir)
    ld = run_dir / "learning"
    tune, tlog = build_samples(list(q["pretrain_families"]), Split.TUNE, range(int(q["n_adapt"])),
                               int(q["support"]), True)
    for r in tlog:
        log.append({**r, "used_by": "lora_candidate,memory_candidate"})
    ad = Adapter.init(int(q["rank"]), int(packet.seeds[0]))
    loss = train_adapter(base, ad, tune, int(q["adapt_steps"]), float(q["adapt_lr"]), int(packet.seeds[0]))
    mem: dict[str, int] = {}
    for smp in tune:
        sg = signature(HYPOTHESES[smp.label])
        mem[sg] = mem.get(sg, 0) + 1
    cands = man["candidates"]
    assert isinstance(cands, dict)
    cands["lora_candidate"] = {"kind": "lora", "rank": int(q["rank"]), "loss": loss,
                               "digest": _save(ld / "lora_candidate.json", ad.state())}
    cands["memory_candidate"] = {"kind": "memory", "alpha": float(q["memory_alpha"]),
                                 "digest": _save(ld / "memory_candidate.json", mem)}
    finish(ld, log, man)


def prepare_e6(packet: Packet, run_dir: Path) -> None:
    q = packet.params
    base, pre, log, man = prepare_common(packet, run_dir)
    ld = run_dir / "learning"
    syn, slog = build_samples(list(q["pretrain_families"]), Split.TUNE, range(int(q["n_adapt"])),
                              int(q["support"]), False)
    for r in slog:
        log.append({**r, "eligibility": "unverified", "used_by": "synthesis-pool"})
    n = len(syn)
    rng = random.Random(f"e6/{packet.seeds[0]}")
    cands = man["candidates"]
    assert isinstance(cands, dict)
    div: dict[str, float] = {}
    for mode in ("replay", "no_replay"):
        prior = Prior(base)
        for g in range(1, int(q["generations"]) + 1):
            take = syn if mode == "no_replay" else syn[: n // 2]
            labeled = [replace(s, label=prior.choose(list(s.cands))) for s in take]
            real = [] if mode == "no_replay" else rng.sample(pre, min(n - len(take), len(pre)))
            ad = Adapter.init(int(q["rank"]), int(packet.seeds[0]) + g)
            loss = train_adapter(base, ad, labeled + real, int(q["adapt_steps"]),
                                 float(q["adapt_lr"]), int(packet.seeds[0]) + g)
            arm = f"d{g}_{mode}"
            div[arm] = norm_entropy([s.label for s in labeled])
            for s in labeled + real:
                log.append({"sample_id": s.key, "arm": arm, "gen": g,
                            "origin": "generated" if s in labeled else "observed",
                            "eligibility": "unverified" if s in labeled else "eligible",
                            "label_source": "prior-pseudo-label" if s in labeled else "oracle"})
            cands[arm] = {"kind": "lora", "generation": g, "loss": loss,
                          "diversity": div[arm], "digest": _save(ld / f"{arm}.json", ad.state())}
            prior = Prior(base, ad)
    man["diversity_floor"] = float(q["diversity_floor_frac"]) * div["d1_no_replay"]
    finish(ld, log, man)


def load_stack(packet: Packet, run_dir: Path) -> Stack:
    ld = run_dir / "learning"
    if not (ld / "manifest.json").exists():
        raise DigestError("no learning manifest; run `hsi pilot` first")
    man = load_json(ld / "manifest.json")
    body = {k: v for k, v in man.items() if k != "manifest_digest"}
    if canonical_digest(body) != man["manifest_digest"]:
        raise DigestError("learning manifest digest mismatch")
    if man["packet_digest"] != canonical_digest(packet.model_dump(mode="json")):
        raise DigestError("candidates not bound to this packet")
    if man["runtime"] != RUNTIME:
        raise DigestError("runtime mismatch")
    bs = load_json(ld / "base.json")
    base = Base(bs["W"], bs["b"], bs["v"])  # type: ignore[arg-type]
    if base.digest != man["base_digest"]:
        raise DigestError("base digest mismatch")
    ads: dict[str, Adapter] = {}
    mem: dict[str, int] = {}
    cands = man["candidates"]
    assert isinstance(cands, dict)
    for arm, info in cands.items():
        obj = json.loads((ld / f"{arm}.json").read_text("utf-8"))
        if canonical_digest(obj) != info["digest"]:
            raise DigestError(f"candidate {arm} digest mismatch")
        if info["kind"] == "lora":
            ads[arm] = Adapter(obj["A"], obj["B"])
        else:
            mem = obj
    return Stack(base, ads, mem, man)


def prior_for(stack: Stack, arm: str, alpha: float = 1.0) -> Prior:
    if arm == "frozen_base":
        return Prior(stack.base)
    if arm in stack.adapters:
        return Prior(stack.base, stack.adapters[arm])
    if arm == "memory_candidate":
        return Prior(stack.base, memory=stack.memory, alpha=alpha)
    raise ValueError(f"unknown arm {arm}")


def run_prior_task(prior: Any, arm: str, family: str, split: Split, seed: int,
                   support: int) -> TaskResult:
    """`prior` is any chooser: a Prior (pure-Python) or an LLMPrior (exposes with_context)."""
    env = RuleWorld(family, split, seed)
    tr = teacher_trace(RuleWorld(family, split, seed), support, tag="/support")
    model = Model(transitions_of(tr), chooser=prior if hasattr(prior, "with_context")
                  else prior.choose)
    obs, info = env.reset()
    goal = tuple(int(x) for x in info["goal"])  # type: ignore[attr-defined]
    plan = model.plan(tuple(obs), goal, env.task.horizon)
    acts = [(0, a) for a in plan] if plan else random_policy(env, random.Random(f"act/{arm}/{seed}"))
    ep = rollout(env, acts)
    ep = ep.model_copy(update={"episode_id": f"{arm}/{family}/{split.value}-{seed}"})
    return TaskResult(arm, family, seed, ep.outcome.success, [ep],
                      {"explore_calls": float(support), "calls": float(len(ep.steps))})
