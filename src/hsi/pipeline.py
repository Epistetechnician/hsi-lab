"""packet -> pilot -> lock -> run -> eval -> verify-receipt -> decide (E1)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

from hsi.accounting import BudgetExceeded, Meter
from hsi.contracts.io import load_frozen, load_json, write_json
from hsi.contracts.packet import Packet, canonical_digest
from hsi.contracts.records import Split
from hsi.environment import SEED_BASE
from hsi.errors import (
    BudgetError,
    DigestError,
    ExistsError,
    InvalidError,
    LockError,
    NotVerifiedError,
    PreconditionError,
    RecomputeError,
)
from hsi.evaluation.base import arms_of
from hsi.evaluation.registry import SPECS
from hsi.experience import Ledger
from hsi.paths import REPO, custody_root, run_root
from hsi.verification import DecisionLog

OUTCOMES = {"PROMOTE", "NO_CANDIDATE", "INCONCLUSIVE"}


def rundir(digest: str) -> Path:
    return run_root() / digest[:16]


def _setup(packet_file: Path) -> tuple[Packet, str, Path]:
    packet, digest = load_frozen(packet_file)
    if packet.experiment not in SPECS:
        raise PreconditionError(f"{packet.experiment} not implemented yet ({sorted(SPECS)})")
    d = rundir(digest)
    d.mkdir(parents=True, exist_ok=True)
    pf = d / "packet.frozen.json"
    if not pf.exists():
        pf.write_bytes(packet_file.read_bytes())
    return packet, digest, d


def _p(packet: Packet) -> tuple[list[str], int, int, int]:
    q = packet.params
    return list(q["families"]), int(q["pilot_tasks"]), int(q["min_n"]), int(q["max_n"])


def pilot(packet_file: Path) -> Path:
    packet, digest, d = _setup(packet_file)
    fams, npilot, lo, hi = _p(packet)
    spec = SPECS[packet.experiment]
    if hasattr(spec, "prepare"):
        spec.prepare(packet, d)
    if hasattr(spec, "bind"):
        spec.bind(packet, d)
    off = int(packet.params.get("tune_offset", 0))
    treat, bases = arms_of(packet)
    per_family: dict[str, float] = {}
    sigma = 0.0
    for base in bases:
        sq: list[float] = []
        for fam in fams:
            fd = []
            for split in (Split.FIT, Split.TUNE):
                for i in range(npilot):
                    seed = SEED_BASE[split] + off + i
                    b = spec.run_task(base, fam, split, seed, packet)
                    t = spec.run_task(treat, fam, split, seed, packet)
                    fd.append(float(t.success) - float(b.success))
            mean = sum(fd) / len(fd)
            per_family[f"{base}/{fam}"] = mean
            sq += [(x - mean) ** 2 for x in fd]
        sigma = max(sigma, math.sqrt(sum(sq) / max(1, len(sq) - len(fams))))
    need = math.ceil((1.96 * sigma / (packet.threshold / 2)) ** 2 / len(fams))
    n = min(hi, max(lo, need))
    out = {"packet_digest": digest, "sigma": sigma, "per_family_mean_diff": per_family,
           "required_n": need, "n_per_family": n, "underpowered": need > hi,
           "pilot_tasks_per_family_per_split": npilot}
    write_json(d / "pilot.json", out)
    return d / "pilot.json"


def _commitment(salt: str, offsets: list[int]) -> str:
    return hashlib.sha256((salt + ":" + ",".join(map(str, offsets))).encode()).hexdigest()


def _lock_digest(digest: str, n: int, commitment: str) -> str:
    return canonical_digest({"packet_digest": digest, "n_per_family": n, "commitment": commitment})


def holdout_offsets(salt: str, n: int) -> list[int]:
    return random.Random(salt).sample(range(1_000_000), n)


def lock(packet_file: Path) -> Path:
    _, digest, d = _setup(packet_file)
    if not (d / "pilot.json").exists():
        raise PreconditionError("pilot missing; run `hsi pilot` first")
    if (d / "lock.public.json").exists():
        raise ExistsError("already locked")
    croot = custody_root()
    croot.mkdir(parents=True, exist_ok=True)
    croot.chmod(0o700)
    if croot.stat().st_mode & 0o077:
        raise LockError("custody root is group/world accessible")
    n = int(load_json(d / "pilot.json")["n_per_family"])  # type: ignore[call-overload]
    salt = os.urandom(16).hex()
    offs = holdout_offsets(salt, n)
    com = _commitment(salt, offs)
    secret = croot / f"{digest[:16]}.lock.json"
    secret.write_text(json.dumps({"packet_digest": digest, "salt": salt, "n_per_family": n}))
    secret.chmod(0o600)
    ld = _lock_digest(digest, n, com)
    write_json(d / "lock.public.json", {"packet_digest": digest, "n_per_family": n,
                                         "commitment": com, "lock_digest": ld})
    return d / "lock.public.json"


def run(packet_file: Path) -> Path:
    """Producer side. Never reads the custody root; exercises all arms on the tune split."""
    packet, digest, d = _setup(packet_file)
    if not (d / "lock.public.json").exists():
        raise PreconditionError("not locked; run `hsi lock` first")
    lk = load_json(d / "lock.public.json")
    if lk["packet_digest"] != digest:
        raise DigestError("lock/packet mismatch")
    fams, _, _, _ = _p(packet)
    spec = SPECS[packet.experiment]
    n = min(int(lk["n_per_family"]), 20)  # type: ignore[call-overload]
    off = int(packet.params.get("tune_offset", 0))
    if hasattr(spec, "bind"):
        spec.bind(packet, d)
    meter = Meter()
    rd = d / "run"
    if (rd / "episodes.jsonl").exists():
        raise ExistsError("run already exists")
    led, exl, qled = (Ledger(rd / f"{x}.jsonl") for x in ("episodes", "examples", "quarantine"))
    dlog = DecisionLog(rd / "decisions.jsonl")
    mlog, rlog = DecisionLog(rd / "messages.jsonl"), DecisionLog(rd / "relation_events.jsonl")
    try:
        for arm in packet.arms:
            for fam in fams:
                for i in range(n):
                    r = spec.run_task(arm, fam, Split.TUNE, SEED_BASE[Split.TUNE] + off + i, packet)
                    meter.charge(calls=int(r.stats.get("calls", 0)),
                                 bytes=int(r.stats.get("payload_bytes", 0)),
                                 limits={"calls": packet.budget.max_calls})
                    meter.failed_proposals += int(r.stats.get("failed_proposals", 0))
                    spec.check_budget(r, packet)
                    for ep in r.episodes:
                        led.append(ep)
                    for ep in r.examples:
                        exl.append(ep)
                    for ep in r.quarantine:
                        qled.append(ep)
                    for rec in r.decisions:
                        dlog.append(rec)
                    for m in r.messages:
                        mlog.append(m.model_dump(mode="json"))
                    for ev in r.relation_events:
                        rlog.append({**ev.model_dump(mode="json"), "task": r.family + str(r.seed),
                                     "arm": r.arm})
                    meter.bytes += int(r.stats.get("message_bytes", 0))
    except BudgetExceeded as e:
        raise BudgetError(str(e)) from e
    manifest = {"packet_digest": digest, "lock_digest": lk["lock_digest"],
                "n_per_family": lk["n_per_family"], "arms": packet.arms,
                "ledger_head": led.head, "decision_log_head": dlog.head, "resources": meter.snapshot()}
    manifest["run_digest"] = canonical_digest({k: v for k, v in manifest.items() if k != "resources"})
    write_json(d / "run" / "manifest.json", manifest)
    return d


def evaluate(run_dir: Path) -> Path:
    """Spawn the locked evaluator in a separate process."""
    if not (run_dir / "run" / "manifest.json").exists():
        raise PreconditionError("no run manifest")
    r = subprocess.run([sys.executable, "-m", "hsi.evaluation.locked", str(run_dir)],
                       capture_output=True, text=True, cwd=REPO, check=False)
    if r.returncode != 0:
        msg = (r.stderr.strip().splitlines() or ["evaluator failed"])[-1]
        cls = {3: DigestError, 7: LockError, 9: PreconditionError}.get(r.returncode, InvalidError)
        raise cls(msg)
    return run_dir / "eval" / "receipt.json"


def verify_receipt(receipt: Path) -> str:
    rd = receipt.parent.parent
    pf = rd / "packet.frozen.json"
    r = subprocess.run([sys.executable, "-I", str(REPO / "validator" / "recompute.py"),
                        str(receipt), str(pf)], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RecomputeError((r.stdout + r.stderr).strip())
    digest = str(load_json(receipt)["receipt_digest"])
    (rd / "receipt.verified").write_text(digest)
    return digest


def decide(receipt: Path, outcome: str) -> Path:
    if outcome not in OUTCOMES:
        raise InvalidError(f"outcome must be one of {sorted(OUTCOMES)}")
    rd = receipt.parent.parent
    digest = str(load_json(receipt)["receipt_digest"])
    marker = rd / "receipt.verified"
    if not marker.exists() or marker.read_text() != digest:
        raise NotVerifiedError("receipt not verified")
    with (rd / "decisions.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"receipt_digest": digest, "owner_decision": outcome}) + "\n")
    return rd / "decisions.jsonl"
