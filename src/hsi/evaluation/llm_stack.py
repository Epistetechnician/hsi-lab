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
