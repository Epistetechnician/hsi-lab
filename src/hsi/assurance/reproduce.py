"""Clean-environment reproduction (M6): rerun a receipt's packet from a fresh clone at the
receipt's source commit, with a fresh run root, a fresh venv from uv.lock (offline), and the
ORIGINAL held-out lock, then compare every aggregate."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from hsi.contracts.io import load_json, write_json
from hsi.errors import PreconditionError, RecomputeError
from hsi.paths import REPO

CORE = ["schema_version", "packet_digest", "source_commit", "lock_digest", "model_digest",
        "data_digest", "seeds", "family_results", "comparisons", "guardrail_values", "threshold",
        "guardrails_held", "decision", "nonclaims"]
BUDGET = ["calls", "bytes", "tokens", "failed_proposals", "retries"]  # wall time / memory excluded


def _flat(prefix: str, v: Any, out: dict[str, Any]) -> None:
    if isinstance(v, dict):
        for k, x in v.items():
            _flat(f"{prefix}.{k}", x, out)
    elif isinstance(v, list):
        for i, x in enumerate(v):
            _flat(f"{prefix}[{i}]", x, out)
    else:
        out[prefix] = v


def leaves(receipt: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in CORE:
        _flat(k, receipt[k], out)
    for k in BUDGET:
        _flat(f"budget_used.{k}", receipt["budget_used"][k], out)
    return out


def compare(orig: dict[str, Any], rerun: dict[str, Any]) -> tuple[int, int, list[str]]:
    a, b = leaves(orig), leaves(rerun)
    bad = sorted(k for k in a.keys() | b.keys() if a.get(k, "<missing>") != b.get(k, "<missing>"))
    return len(a.keys() | b.keys()) - len(bad), len(a.keys() | b.keys()), bad


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> str:
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RecomputeError(f"{' '.join(cmd)} failed ({r.returncode}): "
                             f"{(r.stderr or r.stdout).strip()[-400:]}")
    return r.stdout.strip()


def reproduce(receipt_path: Path, orig_run: Path, orig_custody: Path, out: Path | None,
              fetch: bool = False) -> dict[str, Any]:
    orig = load_json(receipt_path)
    commit = str(orig["source_commit"])
    if commit.endswith("+dirty"):
        raise PreconditionError("receipt was produced from a dirty tree; cannot reproduce")
    pd = str(orig["packet_digest"])
    pub, secret = orig_run / "lock.public.json", orig_custody / f"{pd[:16]}.lock.json"
    if not pub.exists() or not secret.exists():
        raise PreconditionError("original lock (run lock.public.json + custody secret) not found")
    if load_json(pub)["lock_digest"] != orig["lock_digest"]:
        raise PreconditionError("original lock does not match the receipt's lock_digest")
    tmp = Path(tempfile.mkdtemp(prefix="hsi-repro-"))
    try:
        src = tmp / "src"
        _run(["git", "clone", "--quiet", "--no-local", str(REPO), str(src)], tmp, dict(os.environ))
        _run(["git", "checkout", "--quiet", commit], src, dict(os.environ))
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        env.update(HSI_RUN_ROOT=str(tmp / "runs"), HSI_CUSTODY_ROOT=str(tmp / "custody"),
                   PYTHONHASHSEED="random")
        (tmp / "custody").mkdir(mode=0o700)
        # Dependency install is the only step that may use the network (hash-pinned by uv.lock,
        # only with --fetch); every hsi command below runs with --offline.
        extras = ["--all-extras"] if "[project.optional-dependencies]" in (
            src / "pyproject.toml").read_text() else []
        _run(["uv", "sync", "--frozen", "--quiet", *extras, *([] if fetch else ["--offline"])],
             src, env)
        frozen = next((p for p in sorted((src / "packets").glob("*.frozen.json"))
                       if json.loads(p.read_text())["packet_digest"] == pd), None)
        if frozen is None:
            raise PreconditionError("no frozen packet with the receipt's digest at that commit")
        hsi = ["uv", "run", "--frozen", "--offline", "--quiet", *extras, "hsi"]
        _run([*hsi, "pilot", str(frozen)], src, env)
        _run([*hsi, "lock", str(frozen)], src, env)
        rd = tmp / "runs" / pd[:16]
        # import the ORIGINAL lock so the held-out cohort is identical to the receipt's
        shutil.copyfile(pub, rd / "lock.public.json")
        sec = tmp / "custody" / secret.name
        shutil.copyfile(secret, sec)
        sec.chmod(0o600)
        _run([*hsi, "run", str(frozen)], src, env)
        _run([*hsi, "eval", str(rd)], src, env)
        rerun_path = rd / "eval" / "receipt.json"
        rerun = load_json(rerun_path)
        vr = subprocess.run([sys.executable, "-I", str(REPO / "validator" / "recompute.py"),
                             str(rerun_path), str(rd / "packet.frozen.json")],
                            capture_output=True, text=True, check=False)
        ok, total, bad = compare(orig, rerun)
        report: dict[str, Any] = {
            "experiment": json.loads(frozen.read_text())["packet"]["experiment"],
            "packet_digest": pd, "source_commit": commit,
            "original_receipt_digest": orig["receipt_digest"],
            "rerun_receipt_digest": rerun["receipt_digest"],
            "aggregates_compared": total, "aggregates_matched": ok, "mismatches": bad,
            "match_fraction": ok / total, "match": not bad,
            "rerun_validator_ok": vr.returncode == 0,
            "uv_lock_sha256": hashlib.sha256((src / "uv.lock").read_bytes()).hexdigest(),
            "python": platform.python_version(), "platform": platform.platform(),
            "excluded": ["budget_used.wall_seconds", "budget_used.peak_mem_mb", "receipt_digest"],
            "method": "fresh clone at source_commit, fresh uv venv from uv.lock, fresh run root, "
                      "original held-out lock imported",
        }
        if out:
            write_json(out, report)
        return report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
