"""Locked evaluator: its own process; the only reader of the custody root."""

from __future__ import annotations

import hashlib
import json
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
from hsi.evaluation.stats import bonferroni_z, decide, overall
from hsi.paths import REPO, custody_root
from hsi.progress import Progress


def _commit() -> str:
    def g(*a: str) -> str:
        return subprocess.run(
            ["git", *a], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.strip()

    return g("rev-parse", "HEAD") + ("+dirty" if g("status", "--porcelain") else "")


def _load_cache(path: Path, digest: str, lock: str) -> dict[tuple[str, str, int], TaskResult]:
    """Per-task results saved by an earlier attempt of the SAME packet and lock, if any."""
    out: dict[tuple[str, str, int], TaskResult] = {}
    try:
        lines = path.read_text("utf-8").splitlines()
    except OSError:
        return out
    try:
        head = json.loads(lines[0])
    except (IndexError, ValueError):
        return {}
    if head.get("packet_digest") != digest or head.get("lock_digest") != lock:
        path.rename(path.with_suffix(".stale"))  # never mix results across locks
        return {}
    for ln in lines[1:]:
        try:
            d = json.loads(ln)
        except ValueError:
            continue  # a half-written last line
        if "f" not in d:
            continue
        out[(d["f"], d["a"], d["s"])] = TaskResult(
            d["a"], d["f"], d["s"], d["ok"], [], d["stats"], decisions=d["dec"]
        )
    return out


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
    offs = holdout_offsets(str(sec["salt"]), n, int(packet.params.get("holdout_mod", 0)))
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
    prog = Progress(run_dir, "eval")
    total, done = len(fams) * len(packet.arms) * len(offs), 0
    cache_path = run_dir / "eval" / "results.jsonl"
    cache = _load_cache(cache_path, digest, ld)  # resume after a crash: finished tasks are kept
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not cache_path.exists() or cache_path.stat().st_size == 0
    with cache_path.open("a", encoding="utf-8") as cf:
        if fresh:
            cf.write(json.dumps({"packet_digest": digest, "lock_digest": ld}) + "\n")
        for fam in fams:
            for arm in packet.arms:
                rs = []
                for o in offs:
                    seed = SEED_BASE[Split.HELDOUT] + o
                    hit = cache.get((fam, arm, seed))
                    if hit is not None:
                        rs.append(hit)
                    else:
                        r = spec.run_task(arm, fam, Split.HELDOUT, seed, packet)
                        cf.write(
                            json.dumps(
                                {
                                    "f": fam,
                                    "a": arm,
                                    "s": seed,
                                    "ok": r.success,
                                    "stats": r.stats,
                                    "dec": r.decisions,
                                }
                            )
                            + "\n"
                        )
                        cf.flush()
                        rs.append(r)
                    done += 1
                    if done % 5 == 0 or done == total:
                        prog.update(done, total, f"{arm} · {fam}")  # counts only, no scores
                res[(fam, arm)] = rs
                for r in rs:
                    meter.charge(
                        calls=int(r.stats.get("calls", 0)),
                        bytes=int(r.stats.get("payload_bytes", 0))
                        + int(r.stats.get("message_bytes", 0)),
                    )
                    meter.failed_proposals += int(r.stats.get("failed_proposals", 0))
                    rows.append([fam, r.seed, arm, r.success])
    fr = [
        FamilyResult(family=fam, arm=arm, n=n, successes=sum(r.success for r in res[(fam, arm)]))
        for fam in fams
        for arm in packet.arms
    ]
    comps = []
    for base in bases:
        pairs = []
        for fam in fams:
            b, t = res[(fam, base)], res[(fam, treat)]
            both = sum(x.success and y.success for x, y in zip(b, t, strict=True))
            ot = sum(y.success and not x.success for x, y in zip(b, t, strict=True))
            ob = sum(x.success and not y.success for x, y in zip(b, t, strict=True))
            pairs.append(
                PairedFamily(
                    family=fam,
                    n=n,
                    both=both,
                    only_treatment=ot,
                    only_baseline=ob,
                    neither=n - both - ot - ob,
                )
            )
        mean, lo, hi = overall(pairs)
        comps.append(
            Comparison(
                baseline_arm=base,
                treatment_arm=treat,
                pairs=pairs,
                paired_diff=mean,
                ci_low=lo,
                ci_high=hi,
            )
        )
    sec_comps, sec_z = [], None
    sec_spec = list(q.get("secondary", []))  # exploratory contrasts: reported, never decide
    if sec_spec:
        sec_z = bonferroni_z(len(sec_spec))
        for c in sec_spec:
            pairs = []
            for fam in fams:
                b, t = res[(fam, c["baseline"])], res[(fam, c["treatment"])]
                ot = sum(y.success and not x.success for x, y in zip(b, t, strict=True))
                ob = sum(x.success and not y.success for x, y in zip(b, t, strict=True))
                both = sum(x.success and y.success for x, y in zip(b, t, strict=True))
                pairs.append(
                    PairedFamily(
                        family=fam, n=n, both=both, only_treatment=ot, only_baseline=ob,
                        neither=n - both - ot - ob,
                    )
                )  # fmt: skip
            mean, lo, hi = overall(pairs, sec_z)
            sec_comps.append(
                Comparison(
                    baseline_arm=c["baseline"],
                    treatment_arm=c["treatment"],
                    pairs=pairs,
                    paired_diff=mean,
                    ci_low=lo,
                    ci_high=hi,
                )  # fmt: skip
            )
    allr = [r for rs in res.values() for r in rs]
    gv, held = spec.guardrails(allr, comps, packet)
    body = Receipt(
        packet_digest=digest,
        source_commit=_commit(),
        lock_digest=ld,
        model_digest=(
            getattr(spec, "model_digest", lambda: None)()
            or hashlib.sha256(b"scripted:learner-v1:codecs-v1").hexdigest()
        ),
        data_digest=canonical_digest(rows),
        seeds=packet.seeds,
        budget_used=meter.snapshot(),
        family_results=fr,
        comparisons=comps,
        secondary_comparisons=sec_comps,
        secondary_z=sec_z,
        guardrail_values=gv,
        threshold=packet.threshold,
        guardrails_held=held,
        decision=decide([(c.ci_low, c.ci_high) for c in comps], packet.threshold, held),
        nonclaims=packet.nonclaims,
        receipt_digest="",
    )
    d = body.model_dump(mode="json")
    if not sec_comps:  # receipts of packets without secondary contrasts keep their old shape
        d.pop("secondary_comparisons")
        d.pop("secondary_z")
    d.pop("receipt_digest")
    d["receipt_digest"] = canonical_digest(d)
    out = run_dir / "eval" / "receipt.json"
    write_json(out, d)
    prog.finish()
    return out


if __name__ == "__main__":
    try:
        print(main(Path(sys.argv[1])))
    except HsiError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(e.code)
