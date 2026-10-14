"""Append-only, digest-chained episode ledger (F5)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from hsi.contracts.packet import canonical_digest
from hsi.contracts.records import Episode

GENESIS = "0" * 64


class LedgerError(RuntimeError):
    pass


def seal(ep: Episode, prev: str) -> Episode:
    body = ep.model_copy(update={"prev_digest": prev, "digest": ""}).model_dump(mode="json")
    body.pop("digest")
    return ep.model_copy(update={"prev_digest": prev, "digest": canonical_digest(body)})


def _check(ep: Episode, prev: str) -> None:
    body = ep.model_dump(mode="json")
    d = body.pop("digest")
    if ep.prev_digest != prev:
        raise LedgerError(f"broken chain at {ep.episode_id}")
    if canonical_digest(body) != d:
        raise LedgerError(f"digest mismatch at {ep.episode_id}")


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.head = self.verify()

    def read(self) -> list[Episode]:
        out = []
        for line in self.path.read_text("utf-8").splitlines():
            out.append(Episode.model_validate_json(line, strict=False))
        return out

    def verify(self) -> str:
        prev, ids = GENESIS, set()
        for ep in self.read():
            _check(ep, prev)
            if ep.episode_id in ids:
                raise LedgerError(f"duplicate id {ep.episode_id}")
            ids.add(ep.episode_id)
            prev = ep.digest
        return prev

    def append(self, ep: Episode) -> Episode:
        sealed = seal(ep, self.head)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sealed.model_dump(mode="json"), sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.head = sealed.digest
        return sealed
