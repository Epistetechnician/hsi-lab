"""E4 with the MLX backbone: prepare (train the LoRA candidate), bind (verify + load), run."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

from hsi.contracts.io import load_json
from hsi.contracts.packet import Packet, canonical_digest
from hsi.contracts.records import Split
from hsi.environment import SEED_BASE
from hsi.errors import DigestError
from hsi.evaluation.base import TaskResult
from hsi.evaluation.learned import _save, run_prior_task
from hsi.learning.data import build_samples
from hsi.progress import Progress
from hsi.verification import DecisionLog

_STATE: dict[str, Any] = {}


def get_llm(packet: Packet | None = None) -> Any:
    from hsi.learning.llm import DEFAULT_MODEL, LLM

    name = (
        str(packet.params.get("model", DEFAULT_MODEL))
        if packet
        else _STATE.get("name", DEFAULT_MODEL)
    )
    if _STATE.get("name") != name or "llm" not in _STATE:
        _STATE.pop("llm", None)
        _STATE["llm"], _STATE["name"], _STATE["arm"] = LLM(name=name), name, None
    return _STATE["llm"]


def _lora_cfg(packet: Packet) -> tuple[int, int, float]:
    q = packet.params
    return int(q["rank"]), int(q["lora_layers"]), float(q["lora_scale"])


def _ensure_adapter(packet: Packet) -> Any:
    llm = get_llm(packet)
    if not llm._lora:
        r, layers, scale = _lora_cfg(packet)
        llm.attach_lora(r, layers, int(packet.seeds[0]), scale)
    return llm


def prepare_e4_llm(packet: Packet, run_dir: Path) -> None:
    from hsi.agents.learner import HYPOTHESES
    from hsi.learning.llm import (
        MODELS,
        RUNTIME,
        describe,
        render_prompt,
        weights_digest,
    )

    q = packet.params
    llm = _ensure_adapter(packet)
    llm.reset_lora(int(packet.seeds[0]))
    ld = run_dir / "learning"
    ld.mkdir(parents=True, exist_ok=True)
    log = DecisionLog(ld / "training.jsonl")
    fams = list(q["pretrain_families"])
    tune, tlog = build_samples(fams, Split.TUNE, range(int(q["n_adapt"])), int(q["support"]), True)
    for r in tlog:
        log.append({**r, "used_by": "lora_candidate"})
    fit, flog = build_samples(fams, Split.FIT, range(int(q["n_replay"])), int(q["support"]), True)
    n_replay = round(len(tune) * float(q["replay_frac"]) / (1 - float(q["replay_frac"])))
    rng = random.Random(f"e4llm-replay/{packet.seeds[0]}")
    fit = rng.sample(fit, min(n_replay, len(fit)))
    keep = {s.key for s in fit}
    for r in flog:
        if r["sample_id"] in keep:
            log.append({**r, "used_by": "lora_candidate", "role": "fit-replay"})
    data = [(render_prompt(s.pairs), describe(HYPOTHESES[s.label])) for s in [*tune, *fit]]
    loss = llm.train_lora(
        data,
        int(q["adapt_steps"]),
        float(q["adapt_lr"]),
        int(packet.seeds[0]),
        int(q["adapt_batch"]),
    )
    adapter_digest = llm.save_adapter(ld / "lora_candidate.safetensors")
    man: dict[str, object] = {
        "packet_digest": canonical_digest(packet.model_dump(mode="json")),
        "runtime": RUNTIME,
        "python": sys.version.split()[0],
        "seeds": packet.seeds,
        "backbone": "llm",
        "model_repo": MODELS[llm.name][0],
        "model_revision": MODELS[llm.name][1],
        "model_license": MODELS[llm.name][2],
        "model_name": llm.name,
        "weights_digest": weights_digest(llm.path),
        "adapter_digest": adapter_digest,
        "adapter": {
            "rank": int(q["rank"]),
            "layers": int(q["lora_layers"]),
            "scale": float(q["lora_scale"]),
        },
        "train": {
            "samples_tune": len(tune),
            "samples_fit_replay": len(fit),
            "final_loss": loss,
            "steps": int(q["adapt_steps"]),
            "lr": float(q["adapt_lr"]),
        },
        "training_log_head": log.head,
    }
    man["manifest_digest"] = canonical_digest(man)
    _save(ld / "manifest.json", man)


def bind_e4_llm(packet: Packet, run_dir: Path) -> dict[str, Any]:
    from hsi.learning.llm import RUNTIME, weights_digest

    ld = run_dir / "learning"
    if not (ld / "manifest.json").exists():
        raise DigestError("no learning manifest; run `hsi pilot` first")
    man = load_json(ld / "manifest.json")
    if (
        canonical_digest({k: v for k, v in man.items() if k != "manifest_digest"})
        != man["manifest_digest"]
    ):
        raise DigestError("learning manifest digest mismatch")
    if man["packet_digest"] != canonical_digest(packet.model_dump(mode="json")):
        raise DigestError("candidate not bound to this packet")
    llm = _ensure_adapter(packet)
    if man["runtime"] != RUNTIME or man["weights_digest"] != weights_digest(llm.path):
        raise DigestError("runtime or checkpoint mismatch")
    llm.load_adapter(ld / "lora_candidate.safetensors")
    if llm.adapter_digest() != man["adapter_digest"]:
        raise DigestError("adapter digest mismatch")
    return man


def model_digest(run_dir: Path) -> str:
    man = load_json(run_dir / "learning" / "manifest.json")
    return canonical_digest(
        {
            "weights": man["weights_digest"],
            "adapter": man["adapter_digest"],
            "runtime": man["runtime"],
        }
    )


def run_llm_task(arm: str, family: str, split: Split, seed: int, packet: Packet) -> TaskResult:
    from hsi.learning.llm import LLMPrior

    prior = LLMPrior(get_llm(packet), adapter=(arm == "lora_candidate"))
    return run_prior_task(prior, arm, family, split, seed, int(packet.params["support"]))


# ----------------------------------------------------------------------------- E6 (LLM)
def _text_entropy(texts: list[str]) -> float:
    import math

    from hsi.learning.llm import text_signature

    cnt: dict[str, int] = {}
    for t in texts:
        k = text_signature(t)
        cnt[k] = cnt.get(k, 0) + 1
    n = len(texts)
    return -sum(c / n * math.log(c / n) for c in cnt.values()) / math.log(21) if n else 0.0


def _demos(packet: Packet) -> list[tuple[Any, str]]:
    """Two fixed solved examples (fit-type worlds, seeds disjoint from everything else)."""
    from hsi.agents.learner import HYPOTHESES
    from hsi.learning.llm import describe

    s, _ = build_samples(
        ["rule_induction"], Split.FIT, range(700_000, 700_040), int(packet.params["support"]), True
    )
    picked = [x for x in s if len(x.pairs) >= 2][: int(packet.params.get("n_demos", 2))]
    return [(x.pairs, describe(HYPOTHESES[x.label])) for x in picked]


TRAIN_KEYS = (
    "model", "n_adapt", "n_replay", "n_replay_tune", "support", "rank", "lora_layers",
    "lora_scale", "adapt_steps", "adapt_lr", "adapt_batch", "gen_max_tokens", "n_demos",
    "target_terminator", "generations", "extra_arms", "pretrain_families",
)  # fmt: skip


def _check_pinned(cands: dict[str, Any], pinned: dict[str, str] | None) -> None:
    """Fail closed unless every adapter digest equals the packet's pinned digest."""
    if not pinned:
        return
    got = {a: i["digest"] for a, i in cands.items()}
    if got != pinned:
        bad = sorted(a for a in set(got) | set(pinned) if got.get(a) != pinned.get(a))
        raise DigestError(f"adapters differ from the packet's pinned digests: {bad}")


def try_reuse_adapters(packet: Packet, run_dir: Path) -> bool:
    """Reuse previously trained adapters (packet params `reuse_from`) when the training recipe is
    identical and their digests equal `pinned_adapters`. Returns False to fall back to training;
    raises if the source adapters exist but do not match the pinned digests."""
    import shutil

    from hsi.paths import run_root

    q = packet.params
    src_id, pinned = q.get("reuse_from"), q.get("pinned_adapters")
    if not src_id or not pinned:
        return False
    src = run_root() / str(src_id)
    sm, sp = (
        read_manifest(src / "learning" / "manifest.json"),
        load_json_safe(src / "packet.frozen.json"),
    )
    if sm is None or sp is None:
        return False
    old = sp["packet"]
    if old["seeds"] != packet.seeds or any(old["params"].get(k) != q.get(k) for k in TRAIN_KEYS):
        return False
    _check_pinned(sm["candidates"], pinned)
    for a in sm["candidates"]:
        if not (src / "learning" / f"{a}.safetensors").exists():
            return False
    ld = run_dir / "learning"
    ld.mkdir(parents=True, exist_ok=True)
    for a in sm["candidates"]:
        shutil.copyfile(src / "learning" / f"{a}.safetensors", ld / f"{a}.safetensors")
    shutil.copyfile(src / "learning" / "training.jsonl", ld / "training.jsonl")
    man = {k: v for k, v in sm.items() if k != "manifest_digest"}
    man["packet_digest"] = canonical_digest(packet.model_dump(mode="json"))
    man["reused_from"] = str(src_id)
    man["manifest_digest"] = canonical_digest(man)
    _save(ld / "manifest.json", man)
    prog = Progress(run_dir, "pilot:prepare")
    n = len(sm["candidates"])
    prog.update(n, n, f"reused {n} pinned adapters from run {src_id}")
    prog.finish()
    return True


def read_manifest(path: Path) -> dict[str, Any] | None:
    d = load_json_safe(path)
    if d is None:
        return None
    body = {k: v for k, v in d.items() if k != "manifest_digest"}
    return d if canonical_digest(body) == d.get("manifest_digest") else None


def load_json_safe(path: Path) -> dict[str, Any] | None:
    try:
        return dict(load_json(path))
    except (OSError, ValueError):
        return None


def verify_texts(texts: list[str], samples: list[Any]) -> list[bool]:
    """Verifier for generated hypotheses: valid AND consistent with the observed transitions
    (equivalently: re-executing the hypothesis on the evidence reproduces every observation)."""
    from hsi.learning.llm import parse_hypothesis
    from hsi.learning.prior import true_index

    out = []
    for t, smp in zip(texts, samples, strict=True):
        h = parse_hypothesis(t)
        idx = true_index(h) if h is not None else None
        out.append(idx is not None and idx in smp.cands)
    return out


def _real_pool(packet: Packet) -> list[Any]:
    q = packet.params
    fams, sup = list(q["pretrain_families"]), int(q["support"])
    pool, _ = build_samples(fams, Split.FIT, range(int(q["n_replay"])), sup, True)
    n_rt = int(q.get("n_replay_tune", 0))
    if n_rt:
        base = int(q["n_adapt"]) + 50
        extra, _ = build_samples(fams, Split.TUNE, range(base, base + n_rt), sup, True)
        pool = pool + extra
    return pool


def train_verified_chain(
    packet: Packet, run_dir: Path, llm: Any, log: DecisionLog, cands: dict[str, Any]
) -> None:
    """Chain d1..dG_verified: generation g keeps only VERIFIED self-generated examples (drawn from
    a larger proposal pool until the quota is met), plus original labelled data, so each arm has
    the same example count as the replay arms. Writes adapters; fills `cands`."""
    from hsi.agents.learner import HYPOTHESES
    from hsi.learning.llm import describe, few_shot_prompt, render_prompt

    q = packet.params
    fams, sup, seed = list(q["pretrain_families"]), int(q["support"]), int(packet.seeds[0])
    ld = run_dir / "learning"
    base = int(q["verify_pool_base"])
    pool, _ = build_samples(
        fams, Split.TUNE, range(base, base + int(q["verify_pool_worlds"])), sup, False
    )
    real_pool = _real_pool(packet)
    n_total = len(build_samples(fams, Split.TUNE, range(int(q["n_adapt"])), sup, False)[0])
    quota = n_total // 2  # same synthetic quota as the replay arms
    rng = random.Random(f"e6llm-verified/{seed}")
    demos, term = _demos(packet), str(q.get("target_terminator", ""))
    gens = int(q["generations"])
    prog = Progress(run_dir, "pilot:verified")
    prog.update(0, gens, "starting")
    llm.set_adapter(False)  # generation 1 is produced by the frozen base
    for g in range(1, gens + 1):
        order = list(pool)
        rng.shuffle(order)
        kept: list[tuple[Any, str]] = []
        proposed = 0
        for i in range(0, len(order), 8):  # proposals in small blocks; stop at the quota
            block = order[i : i + 8]
            texts = [
                llm.generate(few_shot_prompt(s.pairs, demos), int(q["gen_max_tokens"])) or "unknown"
                for s in block
            ]
            proposed += len(block)
            for ok, s, t in zip(verify_texts(texts, block), block, texts, strict=True):
                if ok and len(kept) < quota:
                    kept.append((s, t))
            prog.update(g - 1, gens, f"d{g}_verified: {len(kept)}/{quota} verified of {proposed}")
            if len(kept) >= quota:
                break
        short = quota - len(kept)
        real = rng.sample(real_pool, n_total - len(kept))  # top up with original data if short
        data = [(render_prompt(s.pairs), t + term) for s, t in kept]
        data += [(render_prompt(s.pairs), describe(HYPOTHESES[s.label]) + term) for s in real]
        arm = f"d{g}_verified"
        for s, t in kept:
            log.append(
                {
                    "sample_id": s.key,
                    "arm": arm,
                    "gen": g,
                    "origin": "generated",
                    "eligibility": "eligible",
                    "label_source": "self-verified",
                    "text": t,
                }
            )
        for s in real:
            log.append(
                {
                    "sample_id": s.key,
                    "arm": arm,
                    "gen": g,
                    "origin": "observed",
                    "eligibility": "eligible",
                    "label_source": "oracle",
                }
            )
        llm.reset_lora(seed + 100 + g)
        steps = int(q["adapt_steps"])
        loss = llm.train_lora(
            data, steps, float(q["adapt_lr"]), seed + 100 + g, int(q["adapt_batch"]),
            on_step=lambda i, ls, a=arm, st=steps, gg=g: prog.update(
                gg - 1, gens, f"{a}: step {i}/{st} loss {ls:.2f}"),
        )  # fmt: skip
        cands[arm] = {
            "generation": g, "loss": loss, "digest": llm.save_adapter(ld / f"{arm}.safetensors"),
            "diversity": _text_entropy([t for _, t in kept]), "validity": 1.0,
            "n_synthetic": len(kept), "n_real": len(real), "proposed": proposed,
            "yield": len(kept) / max(1, proposed), "shortfall": short,
        }  # fmt: skip
        # live weights now equal adapter g: the next generation is produced by it
    prog.finish()


def _extend_with_verified(packet: Packet, run_dir: Path) -> None:
    """After reusing the pinned adapters, train the verified chain and add it to the manifest."""
    llm = _ensure_adapter(packet)
    ld = run_dir / "learning"
    man: dict[str, Any] = dict(load_json(ld / "manifest.json"))
    log = DecisionLog(ld / "training.verified.jsonl")
    cands: dict[str, Any] = dict(man["candidates"])
    train_verified_chain(packet, run_dir, llm, log, cands)
    man.pop("manifest_digest", None)
    man["candidates"] = cands
    man["verified_log_head"] = log.head
    man["manifest_digest"] = canonical_digest(man)
    _save(ld / "manifest.json", man)


def prepare_e6_llm(packet: Packet, run_dir: Path) -> None:
    from hsi.agents.learner import HYPOTHESES
    from hsi.learning.llm import (
        MODELS,
        RUNTIME,
        describe,
        few_shot_prompt,
        parse_hypothesis,
        render_prompt,
        weights_digest,
    )
    from hsi.learning.prior import true_index

    q = packet.params
    if try_reuse_adapters(packet, run_dir):
        if q.get("verified_chain"):
            _extend_with_verified(packet, run_dir)
        return
    llm = _ensure_adapter(packet)
    ld = run_dir / "learning"
    ld.mkdir(parents=True, exist_ok=True)
    log = DecisionLog(ld / "training.jsonl")
    fams = list(q["pretrain_families"])
    syn, slog = build_samples(fams, Split.TUNE, range(int(q["n_adapt"])), int(q["support"]), False)
    for r in slog:
        log.append({**r, "eligibility": "unverified", "used_by": "synthesis-pool"})
    real_pool, _ = build_samples(
        fams, Split.FIT, range(int(q["n_replay"])), int(q["support"]), True
    )
    n_rt = int(q.get("n_replay_tune", 0))
    if (
        n_rt
    ):  # original labelled data also covers tune-type worlds (disjoint from the synthetic pool)
        base = int(q["n_adapt"]) + 50
        extra, _ = build_samples(
            fams, Split.TUNE, range(base, base + n_rt), int(q["support"]), True
        )
        real_pool = real_pool + extra
    n, seed = len(syn), int(packet.seeds[0])
    rng = random.Random(f"e6llm/{seed}")
    cands: dict[str, Any] = {}
    n_adapters = 2 * int(q["generations"]) + (1 if q.get("extra_arms") else 0)
    prog = Progress(run_dir, "pilot:prepare")
    prog.update(0, n_adapters, "starting")
    demos = _demos(packet)
    term = str(q.get("target_terminator", ""))  # end-of-hypothesis marker so generation stops
    for mode in ("no_replay", "replay"):
        llm.set_adapter(False)  # generation 1 is produced by the frozen base
        for g in range(1, int(q["generations"]) + 1):
            take = syn if mode == "no_replay" else syn[: n // 2]
            prog.update(len(cands), n_adapters, f"d{g}_{mode}: generating {len(take)} texts")
            texts = [
                llm.generate(few_shot_prompt(s.pairs, demos), int(q["gen_max_tokens"])) or "unknown"
                for s in take
            ]
            valid = [
                (h := parse_hypothesis(t)) is not None
                and (i := true_index(h)) is not None
                and i in s.cands
                for t, s in zip(texts, take, strict=True)
            ]
            data = [(render_prompt(s.pairs), t + term) for s, t in zip(take, texts, strict=True)]
            real = [] if mode == "no_replay" else rng.sample(real_pool, n - len(take))
            data += [(render_prompt(s.pairs), describe(HYPOTHESES[s.label]) + term) for s in real]
            arm = f"d{g}_{mode}"
            for s, t in zip(take, texts, strict=True):
                log.append(
                    {
                        "sample_id": s.key,
                        "arm": arm,
                        "gen": g,
                        "origin": "generated",
                        "eligibility": "unverified",
                        "label_source": "llm-free-text",
                        "text": t,
                    }
                )
            for s in real:
                log.append(
                    {
                        "sample_id": s.key,
                        "arm": arm,
                        "gen": g,
                        "origin": "observed",
                        "eligibility": "eligible",
                        "label_source": "oracle",
                    }
                )
            llm.reset_lora(seed + g)
            steps = int(q["adapt_steps"])
            loss = llm.train_lora(
                data,
                steps,
                float(q["adapt_lr"]),
                seed + g,
                int(q["adapt_batch"]),
                on_step=lambda i, ls, a=arm, st=steps: prog.update(
                    len(cands), n_adapters, f"{a}: step {i}/{st} loss {ls:.2f}"
                ),
            )
            digest = llm.save_adapter(ld / f"{arm}.safetensors")
            cands[arm] = {
                "generation": g,
                "loss": loss,
                "digest": digest,
                "diversity": _text_entropy(texts),
                "validity": sum(valid) / len(valid),
                "n_synthetic": len(take),
                "n_real": len(real),
            }
            # live weights now equal adapter g: the next loop iteration generates with it
    if q.get("extra_arms"):  # reference: SFT on original labelled data only (no synthetic)
        real_only = [
            (render_prompt(s.pairs), describe(HYPOTHESES[s.label]) + term)
            for s in rng.sample(real_pool, min(n, len(real_pool)))
        ]
        llm.reset_lora(seed)
        steps = int(q["adapt_steps"])
        loss = llm.train_lora(
            real_only,
            steps,
            float(q["adapt_lr"]),
            seed,
            int(q["adapt_batch"]),
            on_step=lambda i, ls: prog.update(
                len(cands), n_adapters, f"real_only: step {i}/{steps} loss {ls:.2f}"
            ),
        )
        cands["real_only"] = {
            "generation": 0,
            "loss": loss,
            "n_synthetic": 0,
            "n_real": len(real_only),
            "digest": llm.save_adapter(ld / "real_only.safetensors"),
        }
    prog.finish()
    _check_pinned(cands, q.get("pinned_adapters"))
    if q.get("verified_chain"):
        train_verified_chain(packet, run_dir, llm, log, cands)
    man: dict[str, object] = {
        "packet_digest": canonical_digest(packet.model_dump(mode="json")),
        "runtime": RUNTIME,
        "python": sys.version.split()[0],
        "seeds": packet.seeds,
        "backbone": "llm",
        "model_repo": MODELS[llm.name][0],
        "model_revision": MODELS[llm.name][1],
        "model_license": MODELS[llm.name][2],
        "model_name": llm.name,
        "weights_digest": weights_digest(llm.path),
        "candidates": cands,
        "diversity_floor": float(q["diversity_floor_frac"]) * cands["d1_no_replay"]["diversity"],
        "training_log_head": log.head,
    }
    man["manifest_digest"] = canonical_digest(man)
    _save(ld / "manifest.json", man)


def bind_e6_llm(packet: Packet, run_dir: Path) -> dict[str, Any]:
    from hsi.learning.llm import RUNTIME, weights_digest

    ld = run_dir / "learning"
    if not (ld / "manifest.json").exists():
        raise DigestError("no learning manifest; run `hsi pilot` first")
    man = load_json(ld / "manifest.json")
    if (
        canonical_digest({k: v for k, v in man.items() if k != "manifest_digest"})
        != man["manifest_digest"]
    ):
        raise DigestError("learning manifest digest mismatch")
    if man["packet_digest"] != canonical_digest(packet.model_dump(mode="json")):
        raise DigestError("candidates not bound to this packet")
    llm = _ensure_adapter(packet)
    if man["runtime"] != RUNTIME or man["weights_digest"] != weights_digest(llm.path):
        raise DigestError("runtime or checkpoint mismatch")
    for arm, info in man["candidates"].items():  # type: ignore[attr-defined]
        llm.load_adapter(ld / f"{arm}.safetensors")
        if llm.adapter_digest() != info["digest"]:
            raise DigestError(f"adapter {arm} digest mismatch")
    _STATE["run_dir"], _STATE["arm"] = run_dir, None
    return man


def model_digest_e6(run_dir: Path) -> str:
    man = load_json(run_dir / "learning" / "manifest.json")
    cands = dict(man["candidates"])  # type: ignore[call-overload]
    return canonical_digest(
        {
            "weights": man["weights_digest"],
            "runtime": man["runtime"],
            "adapters": {a: i["digest"] for a, i in cands.items()},
        }
    )


def run_llm_e6_task(arm: str, family: str, split: Split, seed: int, packet: Packet) -> TaskResult:
    from hsi.learning.llm import LLMPrior

    llm = get_llm(packet)
    if arm == "frozen_base":
        llm.set_adapter(False)
        _STATE["arm"] = None
    elif _STATE.get("arm") != arm:
        llm.load_adapter(_STATE["run_dir"] / "learning" / f"{arm}.safetensors")
        llm.set_adapter(True)
        _STATE["arm"] = arm
    if packet.params.get("eval_mode") == "generate":
        return run_generation_task(llm, arm, family, split, seed, packet)
    if packet.params.get("eval_mode") == "ground":
        return run_grounding_task(llm, arm, family, split, seed, packet)
    return run_prior_task(
        LLMPrior(llm, adapter=True), arm, family, split, seed, int(packet.params["support"])
    )


def run_generation_task(
    llm: Any, arm: str, family: str, split: Split, seed: int, packet: Packet
) -> TaskResult:
    """Free-generation evaluation: the model writes a hypothesis per action; valid ones are used
    as-is (no consistency filter). Records how often the model's text was valid / consistent."""
    from hsi.agents.learner import Model, consistent, transitions_of
    from hsi.agents.scripted import random_policy, rollout
    from hsi.environment import RuleWorld
    from hsi.evaluation.e1 import teacher_trace
    from hsi.learning.llm import parse_hypothesis, render_prompt
    from hsi.learning.prior import true_index

    support, max_tok = int(packet.params["support"]), int(packet.params["gen_max_tokens"])
    seen = valid = consist = 0

    def propose(pairs: list[Any]) -> Any:
        nonlocal seen, valid, consist
        seen += 1
        h = parse_hypothesis(llm.generate(render_prompt(pairs), max_tok))
        if h is not None:
            valid += 1
            idx = true_index(h)
            consist += idx is not None and idx in consistent(pairs)
        return h

    env = RuleWorld(family, split, seed)
    tr = teacher_trace(RuleWorld(family, split, seed), support, tag="/support")
    model = Model(transitions_of(tr), proposer=propose)
    obs, info = env.reset()
    goal = tuple(int(x) for x in info["goal"])  # type: ignore[attr-defined]
    plan = model.plan(tuple(obs), goal, env.task.horizon)
    acts = (
        [(0, a) for a in plan] if plan else random_policy(env, random.Random(f"act/{arm}/{seed}"))
    )
    ep = rollout(env, acts)
    ep = ep.model_copy(update={"episode_id": f"{arm}/{family}/{split.value}-{seed}"})
    return TaskResult(
        arm,
        family,
        seed,
        ep.outcome.success,
        [ep],
        {
            "explore_calls": float(support),
            "calls": float(len(ep.steps)),
            "gen_seen": float(seen),
            "gen_valid": float(valid),
            "gen_consistent": float(consist),
        },
    )


IN_DIST = "@tune"  # family suffix: unseen worlds from a TRAINED rule family, drawn from the lock


def _collides(o: int, packet: Packet) -> bool:
    """True if offset o falls in a seed range used for training, replay, demos or pilot."""
    q = packet.params
    n_adapt, n_rt = int(q["n_adapt"]), int(q.get("n_replay_tune", 0))
    off = int(q.get("tune_offset", 0))
    extra = any(lo <= o < hi for lo, hi in q.get("avoid", []))  # e.g. the verified proposal pool
    return o < n_adapt + 50 + n_rt or off <= o < off + 200 or 700_000 <= o < 700_100 or extra


def resolve_world(family: str, split: Split, seed: int, packet: Packet) -> tuple[str, Split, int]:
    """Map a packet family label to (rule-world family, split, seed).

    "<fam>@tune" under the locked (held-out) cohort keeps the lock's random offset but builds the
    world in the TUNE seed range, i.e. an unseen world of a rule family the model was trained on.
    Offsets that collide with training/pilot seeds are shifted deterministically (identical
    for every arm, so pairing is preserved).
    """
    if not family.endswith(IN_DIST):
        return family, split, seed
    base = family[: -len(IN_DIST)]
    if split is not Split.HELDOUT:
        return base, split, seed
    o = seed - SEED_BASE[Split.HELDOUT]
    while _collides(o, packet):
        o = (o + 7_919) % 1_000_000
    return base, Split.TUNE, SEED_BASE[Split.TUNE] + o


def run_grounding_task(
    llm: Any, arm: str, family: str, split: Split, seed: int, packet: Packet
) -> TaskResult:
    """Generator-level probe: one prompt per world. The model writes a hypothesis for one observed
    action; success = it is valid AND consistent with the observed evidence ("grounded"). No
    planning, so the metric has headroom wherever the generator is partly competent."""
    from hsi.agents.learner import consistent, transitions_of
    from hsi.environment import RuleWorld
    from hsi.evaluation.e1 import teacher_trace
    from hsi.learning.llm import parse_hypothesis, render_prompt, text_signature
    from hsi.learning.prior import true_index

    support = int(packet.params["support"])
    fam0, split0, seed0 = resolve_world(family, split, seed, packet)
    env = RuleWorld(fam0, split0, seed0)
    trans = transitions_of(teacher_trace(RuleWorld(fam0, split0, seed0), support, tag="/support"))
    per: dict[int, list[tuple[Any, Any]]] = {}
    for st, a, st2 in trans:
        per.setdefault(a, []).append((st, st2))
    a = random.Random(f"ground/{seed0}").choice(sorted(per))
    pairs = per[a]
    text = llm.generate(render_prompt(pairs), int(packet.params["gen_max_tokens"]))
    h = parse_hypothesis(text)
    idx = true_index(h) if h is not None else None
    grounded = idx is not None and idx in consistent(pairs)
    exact = idx is not None and idx == true_index(env.task.actions[a])
    stats = {
        "explore_calls": float(support), "calls": 0.0, "gen_seen": 1.0,
        "gen_valid": float(h is not None), "gen_consistent": float(grounded),
        "gen_exact": float(exact),
    }  # fmt: skip
    return TaskResult(
        arm,
        family,
        seed,
        bool(grounded),
        [],
        stats,
        decisions=[{"arm": arm, "sig": text_signature(text), "text": text[:60]}],
    )
