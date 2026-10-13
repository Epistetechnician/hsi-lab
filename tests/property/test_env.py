from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hsi.agents import POLICIES, rollout
from hsi.contracts.records import Split
from hsi.environment import RULE_FAMILIES, SEED_BASE, TASK_FAMILIES, RuleWorld
from hsi.experience import Ledger, LedgerError

fam = st.sampled_from(TASK_FAMILIES)
spl = st.sampled_from(list(Split))
off = st.integers(0, 999_999)


@settings(max_examples=60, deadline=None)
@given(fam, spl, off)
def test_determinism(f: str, s: Split, o: int) -> None:
    a, b = RuleWorld(f, s, SEED_BASE[s] + o), RuleWorld(f, s, SEED_BASE[s] + o)
    assert a.task == b.task
    e1 = rollout(a, POLICIES["oracle"](a, None))  # type: ignore[arg-type]
    e2 = rollout(b, POLICIES["oracle"](b, None))  # type: ignore[arg-type]
    assert e1 == e2


@settings(max_examples=60, deadline=None)
@given(fam, spl, off)
def test_oracle_solves(f: str, s: Split, o: int) -> None:
    w = RuleWorld(f, s, SEED_BASE[s] + o)
    ep = rollout(w, POLICIES["oracle"](w, None))  # type: ignore[arg-type]
    assert ep.outcome.success and ep.outcome.steps == ep.outcome.oracle_steps


@settings(max_examples=40, deadline=None)
@given(fam, spl, off)
def test_split_rule_families_disjoint(f: str, s: Split, o: int) -> None:
    w = RuleWorld(f, s, SEED_BASE[s] + o)
    assert RULE_FAMILIES[w.task.rule_family] == s


def test_wrong_seed_family_rejected() -> None:
    with pytest.raises(ValueError):
        RuleWorld("rule_induction", Split.FIT, SEED_BASE[Split.HELDOUT])


def test_split_sets_disjoint() -> None:
    by = {s: {k for k, v in RULE_FAMILIES.items() if v == s} for s in Split}
    assert not (by[Split.FIT] & by[Split.HELDOUT]) and not (by[Split.TUNE] & by[Split.HELDOUT])
    assert not (by[Split.FIT] & by[Split.TUNE])


def test_ledger_chain_and_tamper(tmp_path: Path) -> None:
    led = Ledger(tmp_path / "l.jsonl")
    for i in range(5):
        w = RuleWorld("rule_induction", Split.FIT, i)
        led.append(rollout(w, POLICIES["random"](w, __import__("random").Random(i))))
    assert Ledger(tmp_path / "l.jsonl").head == led.head
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    lines[2] = lines[2].replace('"success": false', '"success": true').replace(
        '"success":false', '"success":true')
    if lines[2] == (tmp_path / "l.jsonl").read_text().splitlines()[2]:
        lines[2] = lines[2].replace('"seed": 2', '"seed": 9')
    (tmp_path / "l.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError):
        Ledger(tmp_path / "l.jsonl")
