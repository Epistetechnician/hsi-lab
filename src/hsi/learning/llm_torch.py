"""PyTorch/CUDA backend with the same contract as `hsi.learning.llm.LLM` (ADR 0002).

The base is the full-precision Qwen2.5-Instruct checkpoint (Apache-2.0) at a pinned revision. The
LoRA adapter is implemented here, not via PEFT, so its parameterisation is identical to the MLX
one: y = W x + scale * (x A) B on the q/v projections of the last N layers, A ~ U(-1/sqrt(d_in),
1/sqrt(d_in)), B = 0, scale -> 0 as an exact off-switch. Numerics differ from MLX/Metal, so receipts
carry the runtime tag and cross-backend agreement is a replication of conclusions, never A3.

torch and transformers are imported lazily so the core install stays free of them.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

RUNTIME = "hsi-llm-v1/torch"
LORA_KEYS = ("self_attn.q_proj", "self_attn.v_proj")


def _lora_cls() -> Any:
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Module, rank: int, scale: float) -> None:
            super().__init__()
            self.base = base
            d_in = base.in_features
            d_out = base.out_features
            self.lora_a = nn.Parameter(torch.zeros(d_in, rank, dtype=torch.float32))
            self.lora_b = nn.Parameter(torch.zeros(rank, d_out, dtype=torch.float32))
            self.base_scale = scale
            self.scale = scale

        def forward(self, x: Any) -> Any:
            y = self.base(x)
            if self.scale == 0.0:
                return y
            z = (x.to(torch.float32) @ self.lora_a) @ self.lora_b
            return y + (self.scale * z).to(y.dtype)

    return LoRALinear


class TorchLLM:
    def __init__(self, path: Path | None = None, name: str = "qwen2.5-1.5b-hf") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from hsi.learning.llm import model_path

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.name, self.runtime = name, RUNTIME
        self.path = path or model_path(name)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.dev == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(str(self.path))
        self.model = AutoModelForCausalLM.from_pretrained(str(self.path), dtype=dtype)
        self.model.to(self.dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._lora: list[Any] = []
        self._scale = 20.0
        self.enabled = True

    def encode(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False).input_ids)

    @staticmethod
    def peak_gb() -> float:
        import torch

        return float(torch.cuda.max_memory_allocated()) / 1e9 if torch.cuda.is_available() else 0.0

    # -- scoring --------------------------------------------------------------
    def _seq_logprobs(self, prompts: list[list[int]], cands: list[list[int]]) -> Any:
        import torch

        n = len(prompts)
        t_max = max(len(p) + len(c) for p, c in zip(prompts, cands, strict=True))
        l_max = max(len(c) for c in cands)
        rows, gather, mask, targets = [], [], [], []
        for p, c in zip(prompts, cands, strict=True):
            seq = p + c
            rows.append(seq + [0] * (t_max - len(seq)))
            gather.append([min(len(p) - 1 + j, t_max - 1) for j in range(l_max)])
            mask.append([1.0] * len(c) + [0.0] * (l_max - len(c)))
            targets.append(c + [0] * (l_max - len(c)))
        x = torch.tensor(rows, device=self.dev)
        h = self.model.model(input_ids=x).last_hidden_state  # [n, t, d]
        g = torch.tensor(gather, device=self.dev)
        hs = torch.gather(h, 1, g[:, :, None].expand(-1, -1, h.shape[-1]))
        logits = self.model.lm_head(hs).float()
        lp = torch.log_softmax(logits, dim=-1)
        tok_lp = torch.gather(lp, 2, torch.tensor(targets, device=self.dev)[:, :, None])[:, :, 0]
        m = torch.tensor(mask, device=self.dev)
        return (tok_lp * m).sum(dim=1).reshape(n)

    def score(self, prompt: str, texts: list[str], batch: int = 128) -> list[float]:
        """log P(text given prompt); the prompt runs once, its KV cache is shared over the batch."""
        import torch

        p = self.encode(prompt)
        out: list[float] = []
        with torch.no_grad():
            o0 = self.model(input_ids=torch.tensor([p], device=self.dev), use_cache=True)
            lp0 = torch.log_softmax(o0.logits[0, -1].float(), dim=-1)
            for i in range(0, len(texts), batch):
                chunk = [self.encode(" " + t) for t in texts[i : i + batch]]
                n, l_max = len(chunk), max(len(c) for c in chunk)
                cache = copy.deepcopy(o0.past_key_values)
                cache.batch_repeat_interleave(n)
                toks = torch.tensor([c + [0] * (l_max - len(c)) for c in chunk], device=self.dev)
                att = torch.ones(n, len(p) + l_max, dtype=torch.long, device=self.dev)
                logits = self.model(
                    input_ids=toks, past_key_values=cache, attention_mask=att, use_cache=True
                ).logits.float()
                lp = torch.log_softmax(logits, dim=-1)
                first = lp0[torch.tensor([c[0] for c in chunk], device=self.dev)]
                if l_max > 1:
                    nxt = torch.gather(lp[:, :-1, :], 2, toks[:, 1:, None])[:, :, 0]
                    mask = torch.tensor(
                        [[1.0] * (len(c) - 1) + [0.0] * (l_max - len(c)) for c in chunk],
                        device=self.dev,
                    )
                    total = first + (nxt * mask).sum(dim=1)
                else:
                    total = first
                out += [float(v) for v in total.tolist()]
        return out

    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        import torch

        ids = torch.tensor([self.encode(prompt)], device=self.dev)
        with torch.no_grad():
            o = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tok.eos_token_id,
            )
        text = self.tok.decode(o[0, ids.shape[1] :], skip_special_tokens=True)
        return str(text).strip().split("\n")[0].strip()

    # -- LoRA -----------------------------------------------------------------
    def attach_lora(self, rank: int, layers: int, seed: int, scale: float = 20.0) -> None:
        import torch

        cls = _lora_cls()
        torch.manual_seed(seed)
        blocks = self.model.model.layers[-layers:] if layers > 0 else []
        self._lora, self._names = [], []
        base = len(self.model.model.layers) - len(blocks)
        for off, blk in enumerate(blocks):
            for key in LORA_KEYS:
                parent_name, leaf = key.split(".")
                parent = getattr(blk, parent_name)
                m = cls(getattr(parent, leaf), rank, scale).to(self.dev)
                setattr(parent, leaf, m)
                self._lora.append(m)
                self._names.append(f"model.layers.{base + off}.{key}")
        self._scale = scale
        self.reset_lora(seed)

    def reset_lora(self, seed: int) -> None:
        import torch

        g = torch.Generator(device="cpu").manual_seed(seed)
        for m in self._lora:
            d_in = m.lora_a.shape[0]
            s = 1 / math.sqrt(d_in)
            a = (torch.rand(m.lora_a.shape, generator=g) * 2 - 1) * s
            m.lora_a.data.copy_(a.to(self.dev))
            m.lora_b.data.zero_()
            m.lora_a.requires_grad_(True)
            m.lora_b.requires_grad_(True)
        self.set_adapter(True)

    def set_adapter(self, on: bool) -> None:
        for m in self._lora:
            m.scale = self._scale if on else 0.0
        self.enabled = on

    def adapter_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, m in zip(self._names, self._lora, strict=True):
            out[f"{name}.lora_a"] = m.lora_a.detach().cpu()
            out[f"{name}.lora_b"] = m.lora_b.detach().cpu()
        return out

    def adapter_digest(self) -> str:
        h = hashlib.sha256()
        for k, v in sorted(self.adapter_params().items()):
            h.update(k.encode())
            h.update(v.float().numpy().tobytes())
        return h.hexdigest()

    def save_adapter(self, path: Path) -> str:
        from safetensors.torch import save_file

        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in self.adapter_params().items()}, str(path))
        return self.adapter_digest()

    def load_adapter(self, path: Path) -> None:
        from safetensors.torch import load_file

        w = load_file(str(path))
        for name, m in zip(self._names, self._lora, strict=True):
            m.lora_a.data.copy_(w[f"{name}.lora_a"].to(self.dev))
            m.lora_b.data.copy_(w[f"{name}.lora_b"].to(self.dev))

    def train_lora(
        self,
        samples: list[tuple[str, str]],
        steps: int,
        lr: float,
        seed: int,
        batch: int = 8,
        on_step: Callable[[int, float], None] | None = None,
    ) -> float:
        import torch

        rng = random.Random(f"llm-lora/{seed}")
        params = [p for m in self._lora for p in (m.lora_a, m.lora_b)]
        opt = torch.optim.Adam(params, lr=lr)
        last = 0.0
        for step in range(steps):
            mb = rng.sample(samples, min(batch, len(samples)))
            ps = [self.encode(p) for p, _ in mb]
            cs = [self.encode(" " + t) for _, t in mb]
            opt.zero_grad()
            loss = -self._seq_logprobs(ps, cs).mean()
            loss.backward()
            opt.step()
            last = float(loss.item())
            if on_step is not None:
                on_step(step + 1, last)
        return last
