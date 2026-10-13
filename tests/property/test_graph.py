from hypothesis import given
from hypothesis import strategies as st

from hsi.collective import adaptive, fixed_gossip

PATH = [(0, 1), (1, 2), (2, 3)]


def know(owned: set[int]) -> list[dict]:  # type: ignore[type-arg]
    return [{k: (("add", 0, 0, k + 1),)} if k in owned else {} for k in range(4)]


@given(st.sets(st.integers(0, 3)), st.integers(0, 150))
def test_cap_never_exceeded_and_bytes_match_messages(owned: set[int], cap: int) -> None:
    for res in (fixed_gossip(know(owned), PATH, 3, cap, "k"),
                adaptive(know(owned), 0, 3, cap, "k", lambda kn: False)):
        assert res.used <= cap and res.used == sum(m.bytes for m in res.messages)


def test_gossip_delivers_everything_given_budget() -> None:
    k = know({0, 1, 2, 3})
    fixed_gossip(k, PATH, 3, 10_000, "k")
    assert all(set(x) == {0, 1, 2, 3} for x in k)


def test_gossip_is_delta_sync() -> None:
    # 4 one-chunk (6 B) summaries each reach 3 other agents exactly once: 12 chunk sends.
    res = fixed_gossip(know({0, 1, 2, 3}), PATH, 3, 10_000, "k")
    assert res.used == 12 * 6 + 2 * len(res.messages)


def test_adaptive_forms_and_drops_links_on_demand() -> None:
    k = know({0, 1, 2, 3})
    res = adaptive(k, 0, 3, 1000, "k", lambda kn: len(kn) == 4)
    assert set(k[0]) == {0, 1, 2, 3} and res.formed == res.dropped == 3
    assert [e["event"] for e in res.events] == ["link_open", "link_drop"] * 3


def test_adaptive_stops_early_and_skips_useless_links() -> None:
    k = know({0, 1, 2, 3})
    assert adaptive(k, 0, 3, 1000, "k", lambda kn: True).used == 0
    k2 = know({0, 3})  # agents 1 and 2 know nothing: links opened, dropped without delivery
    res = adaptive(k2, 0, 3, 1000, "k", lambda kn: False)
    drops = [e for e in res.events if e["event"] == "link_drop"]
    assert [d["delivered"] for d in drops] == [False, False, True]
