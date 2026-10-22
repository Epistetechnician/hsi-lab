"""Recompute every committed receipt with the stdlib-only validator. No install needed beyond Python 3.

usage: python3 scripts/verify_all.py        (from a fresh clone; exit code 1 if any receipt fails)
Each receipt is matched to its frozen packet by digest and checked in a fresh `python -I` process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    frozen = {}
    for p in sorted((REPO / "packets").glob("*.frozen.json")):
        frozen[json.loads(p.read_text("utf-8"))["packet_digest"]] = p
    bad = 0
    for rp in sorted((REPO / "docs" / "experiments").glob("*-receipt.json")):
        r = json.loads(rp.read_text("utf-8"))
        pf = frozen.get(r["packet_digest"])
        if pf is None:
            print(f"NO PACKET  {rp.name}")
            bad += 1
            continue
        v = subprocess.run(
            [sys.executable, "-I", str(REPO / "validator" / "recompute.py"), str(rp), str(pf)],
            capture_output=True, text=True, check=False, cwd="/",
        )  # fmt: skip
        ok = v.returncode == 0
        bad += not ok
        print(f"{'OK        ' if ok else 'FAIL      '}{rp.name}  {pf.name}")
        if not ok:
            print("   ", (v.stdout + v.stderr).strip().splitlines()[-1][:160])
    print(f"\n{'all receipts recompute' if not bad else str(bad) + ' failed'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
