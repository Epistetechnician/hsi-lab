"""Locked evaluator: its own process; the only reader of the custody root."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from hsi.accounting import Meter
from hsi.contracts.io import load_frozen, load_json, write_json
from hsi.contracts.packet import canonical_digest
from hsi.contracts.records import Comparison, FamilyResult, PairedFamily, Receipt, Split
from hsi.environment import SEED_BASE
from hsi.errors import DigestError, HsiError, LockError, PreconditionError
from hsi.evaluation.base import TaskResult, arms_of
from hsi.evaluation.registry import SPECS
from hsi.evaluation.stats import decide, overall
from hsi.paths import REPO, custody_root


def _commit() -> str:
    def g(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True,
                              check=False).stdout.strip()

    return g("rev-parse", "HEAD") + ("+dirty" if g("status", "--porcelain") else "")


def main(run_dir: Path) -> Path:
    from hsi.pipeline import _commitment, _lock_digest, holdout_offsets

    man = load_json(run_dir / "run" / "manifest.json")
    packet, digest = load_frozen(run_dir / "packet.frozen.json")
    if man["packet_digest"] != digest:
        raise DigestError("manifest/packet mismatch")
    sec_path = custody_root() / f"{digest[:16]}.lock.json"
    if not sec_path.exists():
        raise PreconditionError("no custody lock")
    sec = load_json(sec_path)
    n = int(sec["n_per_family"])  # type: ignore[call-overload]
    offs = holdout_offsets(str(sec["salt"]), n)
    ld = _lock_digest(digest, n, _commitment(str(sec["salt"]), offs))
    if ld != man["lock_digest"] or n != man["n_per_family"]:
        raise LockError("lock digest mismatch")
    q = packet.params
    fams = list(q["families"])
    spec = SPECS[packet.experiment]
    if hasattr(spec, "bind"):
        spec.bind(packet, run_dir)
    treat, bases = arms_of(packet)
    meter, rows = Meter(), []
    res: dict[tuple[str, str], list[TaskResult]] = {}
    for fam in fams:
        for arm in packet.arms:
            rs = [spec.run_task(arm, fam, Split.HELDOUT, SEED_BASE[Split.HELDOUT] + o, packet)
                  for o in offs]
            res[(fam, arm)] = rs
            for r in rs:
                meter.charge(calls=int(r.stats.get("calls", 0)),
                             bytes=int(r.stats.get("payload_bytes", 0)) + int(r.stats.get("message_bytes", 0)))
                meter.failed_proposals += int(r.stats.get("failed_proposals", 0))
                rows.append([fam, r.seed, arm, r.success])
    fr = [FamilyResult(family=fam, arm=arm, n=n, successes=sum(r.success for r in res[(fam, arm)]))
          for fam in fams for arm in packet.arms]
    comps = []
    for base in bases:
        pairs = []
        for fam in fams:
            b, t = res[(fam, base)], res[(fam, treat)]
            both = sum(x.success and y.success for x, y in zip(b, t, strict=True))
            ot = sum(y.success and not x.success for x, y in zip(b, t, strict=True))
            ob = sum(x.success and not y.success for x, y in zip(b, t, strict=True))
            pairs.append(PairedFamily(family=fam, n=n, both=both, only_treatment=ot,
                                      only_baseline=ob, neither=n - both - ot - ob))
        mean, lo, hi = overall(pairs)
        comps.append(Comparison(baseline_arm=base, treatment_arm=treat, pairs=pairs,
                                paired_diff=mean, ci_low=lo, ci_high=hi))
    allr = [r for rs in res.values() for r in rs]
    gv, held = spec.guardrails(allr, comps, packet)
    body = Receipt(
        packet_digest=digest, source_commit=_commit(), lock_digest=ld,
        model_digest=hashlib.sha256(b"scripted:learner-v1:codecs-v1").hexdigest(),
        data_digest=canonical_digest(rows), seeds=packet.seeds, budget_used=meter.snapshot(),
        family_results=fr, comparisons=comps, guardrail_values=gv, threshold=packet.threshold,
        guardrails_held=held,
        decision=decide([(c.ci_low, c.ci_high) for c in comps], packet.threshold, held),
        nonclaims=packet.nonclaims, receipt_digest="")
    d = body.model_dump(mode="json")
    d.pop("receipt_digest")
    d["receipt_digest"] = canonical_digest(d)
    out = run_dir / "eval" / "receipt.json"
    write_json(out, d)
    return out


if __name__ == "__main__":
    try:
        print(main(Path(sys.argv[1])))
    except HsiError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(e.code)
