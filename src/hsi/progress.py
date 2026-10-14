"""Stage progress for long runs. Producers write; `hsi watch` reads. Counts only (never partial
scores), so watching a run cannot leak held-out outcomes.

One small JSON file per stage under <run_dir>/progress/, replaced atomically on each update.
Writes are best-effort: progress must never be able to fail a run.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class Progress:
    def __init__(self, run_dir: Path, stage: str) -> None:
        self.path = run_dir / "progress" / f"{stage.replace(':', '-')}.json"
        self.stage = stage
        self.started = time.time()
        self.done = 0
        self.total = 0

    def update(self, done: int, total: int, note: str = "") -> None:
        self.done, self.total = done, total
        self._write({"stage": self.stage, "done": done, "total": total, "note": note,
                     "started": self.started, "updated": time.time(), "finished": None})

    def finish(self, note: str = "") -> None:
        self._write({"stage": self.stage, "done": self.total, "total": self.total, "note": note,
                     "started": self.started, "updated": time.time(), "finished": time.time()})

    def _write(self, obj: dict[str, object]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(obj), "utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass
