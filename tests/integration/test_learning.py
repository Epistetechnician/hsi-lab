import copy
import json
from pathlib import Path

import pytest

from hsi.agents.learner import consistent
from hsi.cli import main, template
from hsi.contracts.records import Split
from hsi.learning import Adapter, Base, Prior, pretrain, train_adapter
from hsi.learning.data import build_samples

FAM = ["rule_induction", "compositional_transfer"]


def test_adapter_is_reversible_and_base_frozen() -> None:
    base = Base.init(0)
    pre, _ = build_samples(FAM, Split.FIT, range(6), 8, True)
    pretrain(base, pre, 10, 0.1, 0)
    before, d0 = copy.deepcopy(base.state()), base.digest
    tune, _ = build_samples(FAM, Split.TUNE, range(6), 8, True)
    ad = Adapter.init(2, 0)
    train_adapter(base, ad, tune, 10, 0.2, 0)
    assert base.state() == before and base.digest == d0  # base untouched
    cands = list(tune[0].cands)
    assert any(Prior(base, ad).score(i) != Prior(base).score(i) for i in cands)  # adapter acts
    assert all(
        Prior(base, None).score(i) == Prior(Base(**dict(before.items()))).score(i) for i in cands
    )  # dropping the adapter restores the base exactly


def test_lora_zero_init_is_identity() -> None:
    base = Base.init(1)
    ad = Adapter.init(2, 1)
    for i in (0, 100, 5000):
        assert Prior(base, ad).score(i) == Prior(base).score(i)


def test_learning_refuses_heldout() -> None:
    with pytest.raises(PermissionError):
        build_samples(FAM, Split.HELDOUT, range(2), 8, True)


def test_samples_labels_are_consistent() -> None:
    s, _ = build_samples(FAM, Split.FIT, range(4), 8, True)
    assert s and all(x.label in x.cands for x in s)
    assert consistent([((0, 0, 0, 0, 0), (0, 0, 0, 0, 0))])  # identity explained by something


def tiny(exp: str) -> dict[str, object]:
    t = template(exp)
    p = dict(t["params"])  # type: ignore[call-overload]
    p.update(
        n_pretrain=5,
        n_adapt=5,
        pretrain_steps=8,
        adapt_steps=8,
        pilot_tasks=3,
        min_n=6,
        max_n=6,
        fit_eval_n=3,
    )
    t["params"] = p
    return t


@pytest.mark.parametrize("exp", ["E4", "E6"])
def test_pipeline_and_tamper(exp: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HSI_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HSI_CUSTODY_ROOT", str(tmp_path / "custody"))
    (tmp_path / "p.json").write_text(json.dumps(tiny(exp)))
    assert main(["packet", "freeze", str(tmp_path / "p.json")]) == 0
    f = str(tmp_path / "p.frozen.json")
    for cmd in ("pilot", "lock", "run"):
        assert main([cmd, f]) == 0
    rd = next((tmp_path / "runs").iterdir())
    assert (rd / "learning" / "base.json").stat().st_mode & 0o222 == 0  # base is read-only
    lines = (rd / "learning" / "training.jsonl").read_text().splitlines()
    assert not any(json.loads(x).get("split") == "heldout" for x in lines)
    assert main(["eval", str(rd)]) == 0
    rc = rd / "eval" / "receipt.json"
    assert main(["verify-receipt", str(rc)]) == 0
    r = json.loads(rc.read_text())
    r["guardrails_held"] = not r["guardrails_held"]
    rc.write_text(json.dumps(r))
    assert main(["verify-receipt", str(rc)]) == 10
    # tampered candidate snapshot is refused (fail closed)
    cand = rd / "learning" / ("lora_candidate.json" if exp == "E4" else "d1_replay.json")
    d = json.loads(cand.read_text())
    d["A"][0][0] += 1.0
    cand.write_text(json.dumps(d))
    (rd / "eval" / "receipt.json").unlink()
    assert main(["eval", str(rd)]) == 3
