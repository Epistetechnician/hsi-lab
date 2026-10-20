"""Torch backend code-generation contract (CPU, 1.5B-hf); skipped without torch or the model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def llm():  # type: ignore[no-untyped-def]
    from hsi.learning.llm_torch import TorchLLM

    try:
        return TorchLLM(name="qwen2.5-1.5b-hf")
    except FileNotFoundError:
        pytest.skip("qwen2.5-1.5b-hf checkpoint not present")


def test_chat_template_matches_the_instruct_format(llm) -> None:  # type: ignore[no-untyped-def]
    c = llm.chat("hello")
    assert c.startswith("<|im_start|>system") and c.endswith("<|im_start|>assistant\n")


def test_batched_greedy_equals_unbatched(llm) -> None:  # type: ignore[no-untyped-def]
    ps = [
        llm.chat("Say hi in one word."),
        llm.chat("List the first three primes, then stop. Be brief."),
    ]
    solo = [llm.complete(p, 12, 0.0, 0) for p in ps]
    both = llm.complete_many(ps, 12, 0.0, [0, 0])
    assert both == solo and all(isinstance(x, str) and x for x in both)


def test_sampling_is_reproducible_and_differs_from_greedy_seeds(llm) -> None:  # type: ignore[no-untyped-def]
    p = [llm.chat("Write one sentence about the sea.")]
    a = llm.complete_many(p, 20, 0.9, [1])
    b = llm.complete_many(p, 20, 0.9, [1])
    c = llm.complete_many(p, 20, 0.9, [2])
    assert a == b and a != c


def test_adapter_off_switch_and_lead_param(llm) -> None:  # type: ignore[no-untyped-def]
    llm.attach_lora(4, 2, 0)
    p = llm.chat("Say hi in one word.")
    base_on = llm.complete(p, 8, 0.0, 0)
    llm.set_adapter(False)
    assert llm.complete(p, 8, 0.0, 0) == base_on  # B starts at zero: adapter is identity
    llm.set_adapter(True)
    loss = llm.train_lora([(p, "```python\nx\n```<|im_end|>")], 2, 1e-3, 0, 1, lead="")
    assert loss > 0
