"""MLX backbone checks. Gated: need `uv sync --extra mlx`, the pinned checkpoint under
~/hsi-models, and HSI_LLM_TESTS=1 (keeps the default model-free suite fast)."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("HSI_LLM_TESTS") != "1", reason="set HSI_LLM_TESTS=1 (needs mlx + checkpoint)"
)


@pytest.fixture(scope="module")
def llm():  # type: ignore[no-untyped-def]
    from hsi.learning.llm import LLM

    return LLM()


def test_checkpoint_is_pinned_and_apache(llm) -> None:  # type: ignore[no-untyped-def]
    from hsi.learning.llm import MODEL_LICENSE, MODEL_REVISION, weights_digest

    assert llm.path.name == MODEL_REVISION and MODEL_LICENSE == "apache-2.0"
    assert len(weights_digest(llm.path)) == 64


def test_cached_scoring_matches_full_forward(llm) -> None:  # type: ignore[no-untyped-def]
    import mlx.core as mx

    from hsi.agents.learner import HYPOTHESES
    from hsi.learning.llm import describe, render_prompt

    prompt = render_prompt([((1, 2, 3, 4, 5), (1, 2, 3, 4, 5)), ((0, 1, 2, 3, 4), (0, 1, 2, 3, 0))])
    texts = [describe(HYPOTHESES[i]) for i in range(30)]
    fast = llm.score(prompt, texts)
    p, c = llm.tok.encode(prompt), [llm.tok.encode(" " + t) for t in texts]
    full = llm._seq_logprobs([p] * len(c), c)
    mx.eval(full)
    assert max(abs(a - b) for a, b in zip(fast, full.tolist(), strict=True)) < 0.1
    assert llm.score(prompt, texts) == fast  # deterministic


def test_lora_identity_reversible_and_base_frozen(llm, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from hsi.agents.learner import HYPOTHESES
    from hsi.learning.llm import describe, render_prompt, weights_digest

    pairs = [((0, 1, 2, 3, 4), (0, 1, 2, 3, 0))]
    prompt = render_prompt(pairs)
    texts = [describe(HYPOTHESES[i]) for i in range(12)]
    base = llm.score(prompt, texts)
    before = weights_digest(llm.path)
    llm.attach_lora(rank=4, layers=2, seed=0)
    assert llm.score(prompt, texts) == base  # B = 0: exact identity at init
    loss = llm.train_lora(
        [(prompt, texts[3]), (prompt, texts[4])], steps=3, lr=1e-3, seed=0, batch=2
    )
    assert 0 <= loss < 1e6  # finite
    assert llm.score(prompt, texts) != base  # adapter acts
    llm.set_adapter(False)
    assert llm.score(prompt, texts) == base  # exact off-switch
    llm.set_adapter(True)
    d = llm.save_adapter(tmp_path / "a.safetensors")
    assert d == llm.adapter_digest() and weights_digest(llm.path) == before
    assert (tmp_path / "a.safetensors").exists()
