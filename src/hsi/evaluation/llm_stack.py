"""E4 with the MLX backbone: prepare (train the LoRA candidate), bind (verify + load), run."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

from hsi.contracts.io import load_json
from hsi.contracts.packet import Packet, canonical_digest
from hsi.contracts.records import Split
from hsi.errors import DigestError
from hsi.evaluation.base import TaskResult
from hsi.evaluation.learned import _save, run_prior_task
from hsi.learning.data import build_samples
from hsi.verification import DecisionLog

_STATE: dict[str, Any] = {}


def get_llm() -> Any:
    from hsi.learning.llm import LLM

    if "llm" not in _STATE:
        _STATE["llm"] = LLM()
    return _STATE["llm"]


def _lora_cfg(packet: Packet) -> tuple[int, int, float]:
    q = packet.params
    return int(q["rank"]), int(q["lora_layers"]), float(q["lora_scale"])


def _ensure_adapter(packet: Packet) -> Any:
    llm = get_llm()
    if not llm._lora:
        r, layers, scale = _lora_cfg(packet)
        llm.attach_lora(r, layers, int(packet.seeds[0]), scale)
    return llm


def prepare_e4_llm(packet: Packet, run_dir: Path) -> None:
    from hsi.agents.learner import HYPOTHESES
    from hsi.learning.llm import (
        MODEL_LICENSE,
        MODEL_REPO,
        MODEL_REVISION,
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
    loss = llm.train_lora(data, int(q["adapt_steps"]), float(q["adapt_lr"]), int(packet.seeds[0]),
                          int(q["adapt_batch"]))
    adapter_digest = llm.save_adapter(ld / "lora_candidate.safetensors")
    man: dict[str, object] = {
        "packet_digest": canonical_digest(packet.model_dump(mode="json")), "runtime": RUNTIME,
        "python": sys.version.split()[0], "seeds": packet.seeds, "backbone": "llm",
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION, "model_license": MODEL_LICENSE,
        "weights_digest": weights_digest(llm.path), "adapter_digest": adapter_digest,
        "adapter": {"rank": int(q["rank"]), "layers": int(q["lora_layers"]),
                    "scale": float(q["lora_scale"])},
        "train": {"samples_tune": len(tune), "samples_fit_replay": len(fit), "final_loss": loss,
                  "steps": int(q["adapt_steps"]), "lr": float(q["adapt_lr"])},
        "training_log_head": log.head}
    man["manifest_digest"] = canonical_digest(man)
    _save(ld / "manifest.json", man)


def bind_e4_llm(packet: Packet, run_dir: Path) -> dict[str, Any]:
    from hsi.learning.llm import RUNTIME, weights_digest

    ld = run_dir / "learning"
    if not (ld / "manifest.json").exists():
        raise DigestError("no learning manifest; run `hsi pilot` first")
    man = load_json(ld / "manifest.json")
    if canonical_digest({k: v for k, v in man.items() if k != "manifest_digest"}) \
            != man["manifest_digest"]:
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
    return canonical_digest({"weights": man["weights_digest"], "adapter": man["adapter_digest"],
                             "runtime": man["runtime"]})


def run_llm_task(arm: str, family: str, split: Split, seed: int, packet: Packet) -> TaskResult:
    from hsi.learning.llm import LLMPrior

    prior = LLMPrior(get_llm(), adapter=(arm == "lora_candidate"))
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

    s, _ = build_samples(["rule_induction"], Split.FIT, range(700_000, 700_040),
                         int(packet.params["support"]), True)
    picked = [x for x in s if len(x.pairs) >= 2][: int(packet.params.get("n_demos", 2))]
    return [(x.pairs, describe(HYPOTHESES[x.label])) for x in picked]


def prepare_e6_llm(packet: Packet, run_dir: Path) -> None:
    from hsi.agents.learner import HYPOTHESES
    from hsi.learning.llm import (
        MODEL_LICENSE,
        MODEL_REPO,
        MODEL_REVISION,
        RUNTIME,
        describe,
        parse_hypothesis,
        render_prompt,
        weights_digest,
    )
    from hsi.learning.prior import true_index

    q = packet.params
    llm = _ensure_adapter(packet)
    ld = run_dir / "learning"
    ld.mkdir(parents=True, exist_ok=True)
    log = DecisionLog(ld / "training.jsonl")
    fams = list(q["pretrain_families"])
    syn, slog = build_samples(fams, Split.TUNE, range(int(q["n_adapt"])), int(q["support"]), False)
    for r in slog:
        log.append({**r, "eligibility": "unverified", "used_by": "synthesis-pool"})
    real_pool, _ = build_samples(fams, Split.FIT, range(int(q["n_replay"])), int(q["support"]), True)
    n, seed = len(syn), int(packet.seeds[0])
    rng = random.Random(f"e6llm/{seed}")
    cands: dict[str, Any] = {}
    demos = _demos(packet)
    for mode in ("no_replay", "replay"):
        llm.set_adapter(False)  # generation 1 is produced by the frozen base
        for g in range(1, int(q["generations"]) + 1):
            take = syn if mode == "no_replay" else syn[: n // 2]
            texts = [llm.generate(few_shot_prompt(s.pairs, demos), int(q["gen_max_tokens"]))
                     or "unknown"
                     for s in take]
            valid = [
                (h := parse_hypothesis(t)) is not None and (i := true_index(h)) is not None
                and i in s.cands for t, s in zip(texts, take, strict=True)]
            data = [(render_prompt(s.pairs), t) for s, t in zip(take, texts, strict=True)]
            real = [] if mode == "no_replay" else rng.sample(real_pool, n - len(take))
            data += [(render_prompt(s.pairs), describe(HYPOTHESES[s.label])) for s in real]
            arm = f"d{g}_{mode}"
            for s, t in zip(take, texts, strict=True):
                log.append({"sample_id": s.key, "arm": arm, "gen": g, "origin": "generated",
                            "eligibility": "unverified", "label_source": "llm-free-text",
                            "text": t})
            for s in real:
                log.append({"sample_id": s.key, "arm": arm, "gen": g, "origin": "observed",
                            "eligibility": "eligible", "label_source": "oracle"})
            llm.reset_lora(seed + g)
            loss = llm.train_lora(data, int(q["adapt_steps"]), float(q["adapt_lr"]), seed + g,
                                  int(q["adapt_batch"]))
            digest = llm.save_adapter(ld / f"{arm}.safetensors")
            cands[arm] = {"generation": g, "loss": loss, "digest": digest,
                          "diversity": _text_entropy(texts), "validity": sum(valid) / len(valid),
                          "n_synthetic": len(take), "n_real": len(real)}
            # live weights now equal adapter g: the next loop iteration generates with it
    man: dict[str, object] = {
        "packet_digest": canonical_digest(packet.model_dump(mode="json")), "runtime": RUNTIME,
        "python": sys.version.split()[0], "seeds": packet.seeds, "backbone": "llm",
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION, "model_license": MODEL_LICENSE,
        "weights_digest": weights_digest(llm.path), "candidates": cands,
        "diversity_floor": float(q["diversity_floor_frac"]) * cands["d1_no_replay"]["diversity"],
        "training_log_head": log.head}
    man["manifest_digest"] = canonical_digest(man)
    _save(ld / "manifest.json", man)


def bind_e6_llm(packet: Packet, run_dir: Path) -> dict[str, Any]:
    from hsi.learning.llm import RUNTIME, weights_digest

    ld = run_dir / "learning"
    if not (ld / "manifest.json").exists():
        raise DigestError("no learning manifest; run `hsi pilot` first")
    man = load_json(ld / "manifest.json")
    if canonical_digest({k: v for k, v in man.items() if k != "manifest_digest"}) \
            != man["manifest_digest"]:
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
    return canonical_digest({"weights": man["weights_digest"], "runtime": man["runtime"],
                             "adapters": {a: i["digest"] for a, i in cands.items()}})


def run_llm_e6_task(arm: str, family: str, split: Split, seed: int, packet: Packet) -> TaskResult:
    from hsi.learning.llm import LLMPrior

    llm = get_llm()
    if _STATE.get("arm") != arm:
        llm.load_adapter(_STATE["run_dir"] / "learning" / f"{arm}.safetensors")
        _STATE["arm"] = arm
    return run_prior_task(LLMPrior(llm, adapter=True), arm, family, split, seed,
                          int(packet.params["support"]))
