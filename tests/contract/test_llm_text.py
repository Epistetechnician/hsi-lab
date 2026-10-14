import pytest

from hsi.agents.learner import HYPOTHESES
from hsi.learning.llm import (
    describe,
    few_shot_prompt,
    parse_hypothesis,
    render_prompt,
    text_signature,
)
from hsi.learning.prior import true_index


@pytest.mark.parametrize("i", range(0, 6480, 97))
def test_describe_parse_roundtrip(i: int) -> None:
    assert true_index(parse_hypothesis(describe(HYPOTHESES[i])) or ()) == i


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "swap 1",
        "swap 2 2",
        "add 1 0",
        "add 9 1",
        "copy 1 7",
        "swap 1 2 ; add 1 1 ; mix 0 1",
        "'add 0 4' -> 'swap 0 4' -> 'copy 0",
        "hello",
    ],
)
def test_invalid_texts_rejected(bad: str) -> None:
    assert parse_hypothesis(bad) is None and text_signature(bad) == "invalid"


def test_swap_is_canonical_and_signature() -> None:
    assert parse_hypothesis("swap 3 1") == (("swap", 1, 3, 0),)
    assert text_signature("swap 1 3 ; copy 0 2") == "swap>copy"


def test_few_shot_prompt_ends_with_query() -> None:
    q = [((1, 2, 3, 4, 5), (1, 2, 3, 4, 0))]
    d = [([((0, 0, 0, 0, 0), (0, 0, 0, 0, 0))], "swap 1 2")]
    p = few_shot_prompt(q, d)
    assert p.startswith(render_prompt(d[0][0])) and p.endswith(render_prompt(q))
    assert "The rule is: swap 1 2\n\n" in p
