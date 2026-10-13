from hypothesis import given
from hypothesis import strategies as st

from hsi.codec import CODECS, Trace, fit_to_budget

state = st.tuples(*[st.integers(0, 6)] * 5)
trace = st.builds(Trace, st.just("episode=x origin=observed"), state,
                  st.lists(st.tuples(st.integers(0, 3), state), max_size=30).map(tuple))


@given(trace, st.sampled_from(sorted(CODECS)))
def test_roundtrip_lossless(t: Trace, name: str) -> None:
    d = CODECS[name].decode(CODECS[name].encode(t))
    assert d.initial == t.initial and d.steps == t.steps


@given(trace, st.integers(0, 300), st.sampled_from(sorted(CODECS)))
def test_budget_respected(t: Trace, b: int, name: str) -> None:
    assert len(fit_to_budget(CODECS[name], t, b)) <= b


def test_typed_denser_than_text() -> None:
    t = Trace("episode=x origin=observed", (1, 2, 3, 4, 5), tuple((i % 4, (i % 7,) * 5) for i in range(20)))
    assert len(CODECS["typed_event"].encode(t)) < len(CODECS["text"].encode(t)) / 3
