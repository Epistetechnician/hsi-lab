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
        self._last_t, self._last_done, self.rate = self.started, 0, 0.0  # rate: EMA of s/unit

    def update(self, done: int, total: int, note: str = "") -> None:
        now = time.time()
        if done > self._last_done and now > self._last_t:
            inst = (now - self._last_t) / (done - self._last_done)
            if self.rate:
                inst = min(inst, 3 * self.rate)  # a stall must not blow up the estimate
            self.rate = inst if self.rate == 0 else 0.8 * self.rate + 0.2 * inst
            self._last_t, self._last_done = now, done
        self.done, self.total = done, total
        self._write(
            {
                "stage": self.stage,
                "done": done,
                "total": total,
                "note": note,
                "started": self.started,
                "updated": now,
                "finished": None,
                "rate": self.rate,
            }
        )

    def finish(self, note: str = "") -> None:
        self._write(
            {
                "stage": self.stage,
                "done": self.total,
                "total": self.total,
                "note": note,
                "started": self.started,
                "updated": time.time(),
                "finished": time.time(),
            }
        )

    def _write(self, obj: dict[str, object]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(obj), "utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass
