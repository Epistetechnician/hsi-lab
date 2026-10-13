"""Run/custody roots live outside the repo; the CLI refuses repo-internal paths."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def refuse_in_repo(p: Path) -> Path:
    rp = p.expanduser().resolve()
    if rp == REPO or REPO in rp.parents:
        raise ValueError(f"path inside repo not allowed: {rp}")
    return rp


def run_root() -> Path:
    return refuse_in_repo(Path(os.environ.get("HSI_RUN_ROOT", "~/hsi-runs")))


def custody_root() -> Path:
    return refuse_in_repo(Path(os.environ.get("HSI_CUSTODY_ROOT", "~/hsi-custody")))
