import pytest
from hypothesis import given
from hypothesis import strategies as st

from hsi.collective import decode_hyps, encode_hyps
from hsi.contracts.records import RelOp, RelState, Split
from hsi.environment import RuleWorld
from hsi.relations import Policy, RelationStore

T = [((0, 0, 0, 0, 0), 0, (1, 0, 0, 0, 0))]


def test_dormant_over_delete_and_reactivate() -> None:
    s = RelationStore(Policy.ADAPTIVE, cap=2)
    for i, c in enumerate("ABC"):
        s.learn(c, T, i)
    assert s.rels["rel/A"].state is RelState.DORMANT  # weakest/least-recent, not deleted
    assert s.retrieve("A", 3) is not None and s.rels["rel/A"].state is RelState.ACTIVE
    ops = [e.op for e in s.events]
    assert RelOp.DORMANT in ops and RelOp.REACTIVATE in ops and RelOp.PRUNE not in ops


def test_static_evicts_and_forgets() -> None:
    s = RelationStore(Policy.STATIC, cap=2)
    for i, c in enumerate("ABC"):
        s.learn(c, T, i)
    assert s.retrieve("A", 3) is None
    assert [e.op for e in s.events].count(RelOp.PRUNE) == 1


def test_retire_only_on_counterevidence() -> None:
    s = RelationStore(Policy.ADAPTIVE, cap=2)
    s.learn("A", T, 0)
    s.observe("A", "e1", True)
    assert s.rels["rel/A"].state is RelState.ACTIVE
    s.observe("A", "e2", False)  # strength 0.8 -> 0.3: decoupled, not retired
    assert s.rels["rel/A"].state is RelState.ACTIVE
    s.observe("A", "e3", False)
    assert s.rels["rel/A"].state is RelState.RETIRED
    assert all(e.source_event_ids for e in s.events)


@given(st.integers(1, 5))
def test_shared_source_counted_once(k: int) -> None:
    a, b = RelationStore(Policy.ADAPTIVE, 2), RelationStore(Policy.ADAPTIVE, 2)
    a.learn("A", T, 0)
    b.learn("A", T, 0)
    a.observe("A", "src", True)
    for _ in range(k):
        b.observe("A", "src", True)
    assert a.rels["rel/A"].strength == pytest.approx(b.rels["rel/A"].strength)
    assert b.dup_ignored == k - 1


def test_none_policy_stores_nothing() -> None:
    s = RelationStore(Policy.NONE, 2)
    s.learn("A", T, 0)
    assert s.retrieve("A", 1) is None and not s.events


def test_variant_keeps_rules_changes_task() -> None:
    a, b = RuleWorld("rule_induction", Split.FIT, 5), RuleWorld("rule_induction", Split.FIT, 5, 1)
    assert a.task.actions == b.task.actions and (a.task.start, a.task.goal) != (
        b.task.start,
        b.task.goal,
    )


@given(
    st.dictionaries(
        st.integers(0, 3),
        st.lists(
            st.tuples(
                st.sampled_from(["add", "swap", "copy", "mix"]),
                st.integers(0, 4),
                st.integers(0, 4),
                st.integers(0, 6),
            ),
            min_size=1,
            max_size=2,
        ).map(tuple),
        max_size=4,
    ),
    st.integers(2, 40),
)
def test_hyp_message_roundtrip_and_cap(h, cap) -> None:  # type: ignore[no-untyped-def]
    data = encode_hyps(h, cap)
    assert len(data) <= cap
    got = decode_hyps(data) if data else {}
    assert all(h[a] == v for a, v in got.items())


def test_proposer_hook_uses_valid_proposals_even_if_wrong_and_skips_invalid() -> None:
    from hsi.agents.learner import Model

    t = [((0, 0, 0, 0, 0), 0, (1, 0, 0, 0, 0)), ((0, 0, 0, 0, 0), 1, (0, 1, 0, 0, 0))]
    wrong = (("swap", 0, 1, 0),)  # contradicts the evidence for action 0
    m = Model(t, proposer=lambda pairs: wrong if pairs[0][1][0] == 1 else None)
    assert m.hyp == {0: wrong}  # used as-is; the invalid proposal for action 1 leaves it unknown
