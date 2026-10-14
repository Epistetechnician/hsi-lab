import pytest
from pydantic import ValidationError

from hsi.accounting import BudgetExceeded, Meter
from hsi.contracts.records import Message
from hsi.paths import REPO, refuse_in_repo


def test_message_rejects_unknown_and_missing() -> None:
    good = {
        "msg_id": "m",
        "sender": 0,
        "receivers": [1],
        "payload_codec": "text",
        "bytes": 3,
        "evidence_refs": [],
        "confidence": 0.5,
        "expires_at_step": 4,
    }
    Message.model_validate(good)
    with pytest.raises(ValidationError):
        Message.model_validate({**good, "x": 1})
    good.pop("bytes")
    with pytest.raises(ValidationError):
        Message.model_validate(good)


def test_budget_enforced() -> None:
    m = Meter()
    with pytest.raises(BudgetExceeded):
        m.charge(bytes=11, limits={"bytes": 10})


def test_repo_paths_refused() -> None:
    with pytest.raises(ValueError):
        refuse_in_repo(REPO / "runs")
