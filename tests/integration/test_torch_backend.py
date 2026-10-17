"""PyTorch backbone checks (ADR 0002 gate B1). Gated: need `--extra torch`, the pinned
full-precision checkpoint under ~/hsi-models, and HSI_TORCH_TESTS=1."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("HSI_TORCH_TESTS") != "1",
    reason="set HSI_TORCH_TESTS=1 (needs torch + checkpoint)",
)

PAIRS = [((1, 2, 3, 4, 5), (1, 2, 3, 4, 5)), ((0, 1, 2, 3, 4), (0, 1, 2, 3, 0))]


@pytest.fixture(scope="module")
def llm():  # type: ignore[no-untyped-def]
    from hsi.learning.llm_torch import TorchLLM

    return TorchLLM()


def _scores(llm):  # type: ignore[no-untyped-def]
    from hsi.agents.learner import HYPOTHESES
    from hsi.learning.llm import describe, render_prompt

    return llm.score(render_prompt(PAIRS), [describe(HYPOTHESES[i]) for i in range(30)])


def test_cached_scoring_matches_full_forward(llm) -> None:  # type: ignore[no-untyped-def]
    import torch

    from hsi.agents.learner import HYPOTHESES
    from hsi.learning.llm import describe, render_prompt

    prompt = render_prompt(PAIRS)
    texts = [describe(HYPOTHESES[i]) for i in range(30)]
    fast = llm.score(prompt, texts)
    p, c = llm.encode(prompt), [llm.encode(" " + t) for t in texts]
    with torch.no_grad():
        full = llm._seq_logprobs([p] * len(c), c).tolist()
    assert max(abs(a - b) for a, b in zip(fast, full, strict=True)) < (
        0.25 if llm.dev == "cuda" else 1e-3
    )  # bf16 on GPU, fp32 on CPU


def test_adapter_identity_at_init_and_exact_off_switch(llm) -> None:  # type: ignore[no-untyped-def]
    import torch

    base = _scores(llm)
    llm.attach_lora(rank=4, layers=2, seed=0, scale=20.0)
    assert _scores(llm) == base  # B = 0: bit-exact identity
    for m in llm._lora:  # make the adapter non-trivial
        m.lora_b.data.normal_(0, 0.5)
    assert _scores(llm) != base
    llm.set_adapter(False)
    assert _scores(llm) == base  # scale 0: bit-exact off-switch
    llm.set_adapter(True)
    llm.reset_lora(0)
    assert _scores(llm) == base
    assert torch.isfinite(torch.tensor(base)).all()


def test_train_save_load_roundtrip(llm, tmp_path) -> None:  # type: ignore[no-untyped-def]
    llm.attach_lora(rank=4, layers=2, seed=1, scale=20.0) if not llm._lora else llm.reset_lora(1)
    samples = [
        ("A machine. 1 2 3 -> ", "add 0 1"),
        ("A machine. 4 5 6 -> ", "swap 0 1"),
    ]
    first = llm.train_lora(samples, steps=1, lr=1e-3, seed=3, batch=2)
    last = llm.train_lora(samples, steps=15, lr=1e-3, seed=3, batch=2)
    assert last < first
    d = llm.save_adapter(tmp_path / "a.safetensors")
    llm.reset_lora(1)
    assert llm.adapter_digest() != d
    llm.load_adapter(tmp_path / "a.safetensors")
    assert llm.adapter_digest() == d
    assert isinstance(llm.generate("add 0 1 or", max_tokens=5), str)
