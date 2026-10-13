from __future__ import annotations

import json
from pathlib import Path

from hsi.contracts.packet import Packet, canonical_digest
from hsi.errors import DigestError, InvalidError


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k}")
        out[k] = v
    return out


def load_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text("utf-8"), object_pairs_hook=_reject_duplicates)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON must be an object")  # noqa: TRY004
    return data


def load_frozen(path: Path) -> tuple[Packet, str]:
    try:
        data = load_json(path)
    except ValueError as e:
        raise InvalidError(str(e)) from e
    if set(data) != {"packet", "packet_digest"}:
        raise InvalidError("not a frozen packet")
    packet = Packet.model_validate(data["packet"], strict=False)
    if canonical_digest(data["packet"]) != data["packet_digest"]:
        raise DigestError("packet digest mismatch")
    return packet, str(data["packet_digest"])


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", "utf-8")
