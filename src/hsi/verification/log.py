"""Digest-chained decision log."""
from __future__ import annotations

import json
from pathlib import Path

from hsi.contracts.packet import canonical_digest


class DecisionLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        self.head = "0" * 64

    def append(self, rec: dict[str, object]) -> None:
        body = {**rec, "prev": self.head}
        body["digest"] = canonical_digest(body)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(body, sort_keys=True) + "\n")
        self.head = str(body["digest"])
