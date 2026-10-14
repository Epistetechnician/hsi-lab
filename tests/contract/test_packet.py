import json
from pathlib import Path

import pytest

from hsi.cli import main, template


def write(tmp: Path, obj: object) -> Path:
    p = tmp / "p.json"
    p.write_text(json.dumps(obj))
    return p


def test_freeze_and_check_roundtrip(tmp_path: Path) -> None:
    p = write(tmp_path, template("E1"))
    assert main(["packet", "freeze", str(p)]) == 0
    assert main(["packet", "check", str(tmp_path / "p.frozen.json")]) == 0


def test_refreeze_refused(tmp_path: Path) -> None:
    p = write(tmp_path, template("E1"))
    main(["packet", "freeze", str(p)])
    assert main(["packet", "freeze", str(p)]) == 4


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(extra=1),
        lambda d: d.pop("threshold"),
        lambda d: d.update(experiment="E9"),
        lambda d: d.update(arms=["a", "a"]),
    ],
)
def test_invalid_rejected(tmp_path: Path, mutate) -> None:  # type: ignore[no-untyped-def]
    d = template("E2")
    mutate(d)
    assert main(["packet", "freeze", str(write(tmp_path, d))]) == 2


def test_duplicate_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "p.json"
    p.write_text('{"version": 1, "version": 2}')
    assert main(["packet", "freeze", str(p)]) == 2


def test_tampered_digest(tmp_path: Path) -> None:
    main(["packet", "freeze", str(write(tmp_path, template("E3")))])
    f = tmp_path / "p.frozen.json"
    d = json.loads(f.read_text())
    d["packet"]["threshold"] = 0.5
    f.chmod(0o644)
    f.write_text(json.dumps(d))
    assert main(["packet", "check", str(f)]) == 3


def test_validator_has_no_hsi_imports() -> None:
    root = Path(__file__).resolve().parents[2] / "validator"
    for py in root.rglob("*.py"):
        src = py.read_text()
        assert "import hsi" not in src and "from hsi" not in src
