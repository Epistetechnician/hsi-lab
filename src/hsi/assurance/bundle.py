"""Hand-off bundle for the independent validator: validator/ + receipt + frozen packet only."""
from __future__ import annotations

import shutil
from pathlib import Path

from hsi.errors import PreconditionError
from hsi.paths import REPO, refuse_in_repo

README = """Independent validation bundle (no producer source).
Run from this directory:
    python3 -I recompute.py receipt.json packet.frozen.json
Exit 0 and a line starting with OK means every aggregate, interval, guardrail and decision in the
receipt was recomputed from its own counts and matches. Standard library only.
"""


def bundle(receipt: Path, out: Path) -> Path:
    out = refuse_in_repo(out)
    pf = receipt.parent.parent / "packet.frozen.json"
    if not pf.exists():
        raise PreconditionError(f"frozen packet not found next to run: {pf}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / "validator" / "recompute.py", out / "recompute.py")
    shutil.copyfile(receipt, out / "receipt.json")
    shutil.copyfile(pf, out / "packet.frozen.json")
    (out / "README.txt").write_text(README, "utf-8")
    return out
