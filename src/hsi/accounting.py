"""Resource accounting (F19) and budget conservation."""

from __future__ import annotations

import resource
import sys
import time
from dataclasses import asdict, dataclass, field


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Meter:
    tokens: int = 0
    calls: int = 0
    bytes: int = 0
    retries: int = 0
    failed_proposals: int = 0
    _t0: float = field(default_factory=time.monotonic, repr=False)

    def snapshot(self) -> dict[str, float]:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
        d = {k: float(v) for k, v in asdict(self).items() if not k.startswith("_")}
        d["wall_seconds"] = time.monotonic() - self._t0
        d["peak_mem_mb"] = peak_mb
        return d

    def charge(
        self,
        *,
        tokens: int = 0,
        calls: int = 0,
        bytes: int = 0,
        limits: dict[str, int] | None = None,
    ) -> None:
        self.tokens += tokens
        self.calls += calls
        self.bytes += bytes
        for k, lim in (limits or {}).items():
            if getattr(self, k) > lim:
                raise BudgetExceeded(f"{k} {getattr(self, k)} > {lim}")
