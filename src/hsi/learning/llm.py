"""MLX language-model backbone for the hypothesis prior (F15), behind the same chooser contract
as the pure-Python prior.

The frozen base is a pinned open-weight checkpoint (Qwen2.5-1.5B-Instruct 4-bit, Apache-2.0).
A hypothesis is scored as log P(description | evidence prompt). A LoRA adapter (mlx-lm LoRALinear)
is trained by supervised fine-tuning on oracle-labelled non-held-out samples; base weights are
frozen and never written; the adapter can be switched off exactly (scale -> 0) or detached.

mlx is imported lazily so the model-free suite and the core install stay MLX-free.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from hsi.agents.learner import HYPOTHESES
from hsi.environment.world import Prim

MODELS: dict[str, tuple[str, str, str]] = {  # name -> (HF repo, pinned revision, license)
    "qwen2.5-1.5b": ("mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                     "8b403126fc14f14cfc99bb4cfa72ecbc129ea677", "apache-2.0"),
    "qwen2.5-7b": ("mlx-community/Qwen2.5-7B-Instruct-4bit",
                   "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed", "apache-2.0"),
}
DEFAULT_MODEL = "qwen2.5-1.5b"
MODEL_REPO, MODEL_REVISION, MODEL_LICENSE = MODELS[DEFAULT_MODEL]
RUNTIME = "hsi-llm-v1/mlx"
MODEL_HOME = Path("~/hsi-models").expanduser()

PREAMBLE = (
    "A machine has 5 cells holding digits 0-6. A button applies a hidden rule built from "
    "operations: 'add j c' (cell j += c mod 7), 'swap j k', 'copy j k' (cell k = cell j), "
    "'mix j k' (cell k += cell j mod 7). Cells are numbered 0-4. Observed presses "
    "(before -> after):\n"
)


def describe(h: tuple[Prim, ...]) -> str:
    parts = []
    for kind, j, k, c in h:
        parts.append(f"add {j} {c}" if kind == "add" else f"{kind} {j} {k}")
    return " ; ".join(parts)


def render_prompt(pairs: Sequence[tuple[Sequence[int], Sequence[int]]]) -> str:
    lines = [f"{' '.join(map(str, s))} -> {' '.join(map(str, s2))}" for s, s2 in pairs]
    return PREAMBLE + "\n".join(lines) + "\nThe rule is:"


def few_shot_prompt(pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
                    demos: Sequence[tuple[Sequence[tuple[Sequence[int], Sequence[int]]], str]]) -> str:
    """Generation prompt: solved demonstrations first, then the query (same format)."""
    shots = [f"{render_prompt(dp)} {txt}\n\n" for dp, txt in demos]
    return "".join(shots) + render_prompt(pairs)


def model_path(name: str = DEFAULT_MODEL, home: Path = MODEL_HOME) -> Path:
    repo, revision, _ = MODELS[name]
    p = home / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if not list(p.glob("*.safetensors")):
        raise FileNotFoundError(f"checkpoint not found at {p}")
    return p


def weights_digest(path: Path) -> str:
    """sha256 over the checkpoint's safetensors files in name order (cached beside the model)."""
    cache = path.parent.parent.parent / f"{path.name}.sha256.json"
    if cache.exists():
        return str(json.loads(cache.read_text())["sha256"])
    h = hashlib.sha256()
    for f in sorted(path.glob("*.safetensors")):
        with f.open("rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
    cache.write_text(json.dumps({"sha256": h.hexdigest(), "revision": path.name}))
    return h.hexdigest()


_OP = re.compile(r"\b(add|swap|copy|mix)\s+(\d)\s+(\d)\b")


def parse_hypothesis(text: str) -> tuple[Prim, ...] | None:
    """Canonical hypothesis for a generated description, or None if it is not a valid one."""
    parts = [x.strip() for x in text.split(";")]
    if not 1 <= len(parts) <= 2:
        return None
    out: list[Prim] = []
    for part in parts:
        m = _OP.fullmatch(part)
        if not m:
            return None
        kind, a, b = m[1], int(m[2]), int(m[3])
        if kind == "add":
            if a > 4 or not 1 <= b <= 6:
                return None
            out.append(("add", a, 0, b))
            continue
        if a > 4 or b > 4 or a == b:
            return None
        out.append(("swap", min(a, b), max(a, b), 0) if kind == "swap" else (kind, a, b, 0))
    return tuple(out)


def text_signature(text: str) -> str:
    h = parse_hypothesis(text)
    return "invalid" if h is None else ">".join(p[0] for p in h)


class LLM:
    """Loaded base model + tokenizer with scoring, LoRA attach/train/save/load, and an exact
    adapter off-switch."""

    def __init__(self, path: Path | None = None, name: str = DEFAULT_MODEL) -> None:
        import mlx.core as mx  # noqa: F401
        from mlx_lm.utils import load

        self.name = name
        self.path = path or model_path(name)
        res = load(str(self.path))
        self.model, self.tok = res[0], res[1]
        self.model.freeze()
        self._lora: list[Any] = []
        self.enabled = True

    # -- scoring --------------------------------------------------------------
    def _seq_logprobs(self, prompts: list[list[int]], cands: list[list[int]]) -> Any:
        """Per-row sum of log P(cand tokens | prompt). Logits only at candidate positions."""
        import mlx.core as mx

        n = len(prompts)
        total = [len(p) + len(c) for p, c in zip(prompts, cands, strict=True)]
        t_max, l_max = max(total), max(len(c) for c in cands)
        rows, gather, mask, targets = [], [], [], []
        for p, c in zip(prompts, cands, strict=True):
            seq = p + c
            rows.append(seq + [0] * (t_max - len(seq)))
            pos = [len(p) - 1 + j for j in range(l_max)]
            gather.append([min(x, t_max - 1) for x in pos])
            mask.append([1.0] * len(c) + [0.0] * (l_max - len(c)))
            targets.append(c + [0] * (l_max - len(c)))
        x = mx.array(rows)
        h = self.model.model(x)  # [n, t_max, d]
        g = mx.array(gather)
        hs = mx.take_along_axis(h, g[:, :, None], axis=1)  # [n, l_max, d]
        if getattr(self.model.args, "tie_word_embeddings", False):
            logits = self.model.model.embed_tokens.as_linear(hs)
        else:
            logits = self.model.lm_head(hs)
        lp = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1,
                                                       keepdims=True)
        tok_lp = mx.take_along_axis(lp, mx.array(targets)[:, :, None], axis=-1)[:, :, 0]
        return (tok_lp * mx.array(mask)).sum(axis=1).reshape(n)

    def score(self, prompt: str, texts: list[str], batch: int = 128) -> list[float]:
        """log P(text | prompt) for each text. The prompt is run once; its KV cache is broadcast
        across the candidate batch, so cost is ~ (prompt + n * len(text)) tokens."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        p = self.tok.encode(prompt)
        pc = make_prompt_cache(self.model)
        h0 = self.model(mx.array([p]), cache=pc)
        lp0 = h0[0, -1].astype(mx.float32)
        lp0 = lp0 - mx.logsumexp(lp0)
        mx.eval(lp0, [c.state for c in pc])
        out: list[float] = []
        for i in range(0, len(texts), batch):
            chunk = [self.tok.encode(" " + t) for t in texts[i:i + batch]]
            n, l_max = len(chunk), max(len(c) for c in chunk)
            cache = make_prompt_cache(self.model)
            for dst, src in zip(cache, pc, strict=True):
                k, v = src.keys[..., :src.offset, :], src.values[..., :src.offset, :]
                dst.keys = mx.broadcast_to(k, (n, *k.shape[1:]))
                dst.values = mx.broadcast_to(v, (n, *v.shape[1:]))
                dst.offset = src.offset
            toks = mx.array([c + [0] * (l_max - len(c)) for c in chunk])
            logits = self.model(toks, cache=cache).astype(mx.float32)  # [n, l_max, V]
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            first = lp0[mx.array([c[0] for c in chunk])]
            if l_max > 1:
                nxt = mx.take_along_axis(lp[:, :-1, :], toks[:, 1:, None], axis=-1)[:, :, 0]
                mask = mx.array([[1.0] * (len(c) - 1) + [0.0] * (l_max - len(c)) for c in chunk])
                total = first + (nxt * mask).sum(axis=1)
            else:
                total = first
            mx.eval(total)
            out += [float(v) for v in total.tolist()]
        return out

    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        """Greedy free-text continuation (first line only). Unconstrained: no validity filter."""
        from mlx_lm import generate

        out = generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens)
        return str(out).strip().split("\n")[0].strip()

    # -- LoRA -----------------------------------------------------------------
    def attach_lora(self, rank: int, layers: int, seed: int, scale: float = 20.0) -> None:
        import mlx.core as mx
        from mlx_lm.tuner.lora import LoRALinear
        from mlx_lm.tuner.utils import linear_to_lora_layers

        mx.random.seed(seed)
        self.model.freeze()
        linear_to_lora_layers(self.model, layers, {"rank": rank, "scale": scale, "dropout": 0.0})
        self._lora = [m for _, m in self.model.named_modules() if isinstance(m, LoRALinear)]
        self._scale = scale
        self.enabled = True

    def reset_lora(self, seed: int) -> None:
        """Re-initialise the adapter (A random, B zero) so training starts from identity."""
        import math

        import mlx.core as mx

        mx.random.seed(seed)
        for m in self._lora:
            d_in, r = m.lora_a.shape
            s = 1 / math.sqrt(d_in)
            m.lora_a = mx.random.uniform(low=-s, high=s, shape=(d_in, r))
            m.lora_b = mx.zeros(m.lora_b.shape)
        self.set_adapter(True)

    def set_adapter(self, on: bool) -> None:
        """Exact off-switch: with scale 0 the LoRA branch adds exactly zero."""
        for m in self._lora:
            m.scale = self._scale if on else 0.0
        self.enabled = on

    def adapter_params(self) -> dict[str, Any]:
        from mlx.utils import tree_flatten

        return dict(tree_flatten(self.model.trainable_parameters()))

    def adapter_digest(self) -> str:
        import mlx.core as mx
        import numpy as np

        h = hashlib.sha256()
        for k, v in sorted(self.adapter_params().items()):
            h.update(k.encode())
            h.update(np.asarray(v.astype(mx.float32)).tobytes())
        return h.hexdigest()

    def save_adapter(self, path: Path) -> str:
        import mlx.core as mx

        path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path), self.adapter_params())
        return self.adapter_digest()

    def load_adapter(self, path: Path) -> None:
        self.model.load_weights(str(path), strict=False)

    def train_lora(self, samples: list[tuple[str, str]], steps: int, lr: float, seed: int,
                   batch: int = 8, on_step: Callable[[int, float], None] | None = None) -> float:
        """Supervised fine-tuning of the adapter only: maximise log P(target | prompt)."""
        import random

        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx import nn

        rng = random.Random(f"llm-lora/{seed}")
        opt = optim.Adam(learning_rate=lr)

        def loss_fn(model: Any, prompts: list[list[int]], cands: list[list[int]]) -> Any:
            return -self._seq_logprobs(prompts, cands).mean()

        vg = nn.value_and_grad(self.model, loss_fn)  # type: ignore[attr-defined]
        last = 0.0
        for step in range(steps):
            mb = rng.sample(samples, min(batch, len(samples)))
            ps = [self.tok.encode(p) for p, _ in mb]
            cs = [self.tok.encode(" " + t) for _, t in mb]
            loss, grads = vg(self.model, ps, cs)
            opt.update(self.model, grads)
            mx.eval(self.model.parameters(), opt.state, loss)
            last = float(loss.item())
            if on_step is not None:
                on_step(step + 1, last)
        return last


class LLMPrior:
    """Chooser contract (see agents.learner.Model): picks the best-scoring consistent hypothesis."""

    def __init__(self, llm: LLM, adapter: bool = False) -> None:
        self.llm, self.adapter = llm, adapter

    def with_context(self, idxs: list[int], pairs: list[tuple[Any, Any]]) -> int:
        prompt = render_prompt(pairs)
        if self.llm._lora:
            self.llm.set_adapter(self.adapter)
        sc = self.llm.score(prompt, [describe(HYPOTHESES[i]) for i in idxs])
        best = max(range(len(idxs)), key=lambda j: (sc[j], -idxs[j]))
        return idxs[best]

    def __call__(self, idxs: list[int]) -> int:  # context-free fallback
        return idxs[0]
