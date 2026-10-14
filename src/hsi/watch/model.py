"""Read-only loader for a run directory. Tolerant of missing, partial, or mid-write files.

Only reads the run root (never the custody root). Heavy files are cached by (mtime, size).
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hsi.paths import REPO

STAGES = ["packet", "pilot", "lock", "run", "eval", "verify", "decide"]
_CACHE: dict[tuple[str, int, int], Any] = {}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """All parseable lines (a half-written final line is skipped). Cached per (mtime, size)."""
    try:
        st = path.stat()
    except OSError:
        return []
    key = (str(path), int(st.st_mtime_ns), st.st_size)
    if key in _CACHE:
        return _CACHE[key]  # type: ignore[no-any-return]
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text("utf-8", errors="replace").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                out.append(d)
    except OSError:
        return []
    for k in [k for k in _CACHE if k[0] == key[0]]:
        del _CACHE[k]
    _CACHE[key] = out
    return out


def count_lines(path: Path) -> int:
    try:
        return path.read_bytes().count(b"\n")
    except OSError:
        return 0


@dataclass
class Stage:
    name: str
    state: str  # done | running | stalled | pending
    detail: str = ""


@dataclass
class RunView:
    path: Path
    packet: dict[str, Any] | None = None
    packet_digest: str = ""
    pilot: dict[str, Any] | None = None
    lock: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    learning: dict[str, Any] | None = None
    verified: bool = False
    owner_decisions: list[dict[str, Any]] = field(default_factory=list)
    progress: dict[str, dict[str, Any]] = field(default_factory=dict)
    ledgers: dict[str, int] = field(default_factory=dict)
    procs: list[str] = field(default_factory=list)
    repro: dict[str, Any] | None = None
    stages: list[Stage] = field(default_factory=list)
    updated: float = 0.0

    @property
    def digest16(self) -> str:
        return self.path.name

    @property
    def experiment(self) -> str:
        return str(self.packet["experiment"]) if self.packet else "?"

    @property
    def version(self) -> int:
        return int(self.packet["version"]) if self.packet else 0

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.packet["params"]) if self.packet else {}

    @property
    def arms(self) -> list[str]:
        return list(self.packet["arms"]) if self.packet else []

    def rows(self, name: str) -> list[dict[str, Any]]:
        return read_jsonl(self.path / name)

    @property
    def title(self) -> str:
        bb = self.params.get("backbone")
        return f"{self.experiment} v{self.version}" + (
            f" ({self.params.get('model', 'llm')})" if bb == "llm" else ""
        )


def _procs(v: RunView) -> list[str]:
    names = {
        p.name
        for p in (REPO / "packets").glob("*.frozen.json")
        if (read_json(p) or {}).get("packet_digest", "").startswith(v.digest16)
    }
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,etime=,command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for ln in out.splitlines():
        if "hsi" not in ln or "ps -axo" in ln or " watch" in ln:
            continue
        if v.digest16 in ln or any(n in ln for n in names):
            m = re.search(r"hsi (pilot|lock|run|eval|verify-receipt|reproduce|decide)\b", ln)
            if m and "python" in ln:
                pid, et = ln.split()[:2]
                rows.append(f"{m[1]} (pid {pid}, up {et})")
    return rows


def _repro_for(digest: str) -> dict[str, Any] | None:
    for p in sorted((REPO / "docs" / "experiments").glob("*-repro.json")):
        d = read_json(p)
        if d and d.get("packet_digest") == digest:
            return d
    return None


def _stages(v: RunView, now: float) -> list[Stage]:
    def prog(prefix: str) -> list[dict[str, Any]]:
        return [p for k, p in v.progress.items() if k.startswith(prefix)]

    def live(ps: list[dict[str, Any]]) -> tuple[str, str]:
        act = [p for p in ps if p.get("finished") is None]
        if not act:
            return "pending", ""
        p = max(act, key=lambda x: x.get("updated", 0))
        fresh = now - float(p.get("updated", 0)) < 180 or bool(v.procs)
        return ("running" if fresh else "stalled"), f"{p.get('done', 0)}/{p.get('total', 0)}"

    st: list[Stage] = [Stage("packet", "done", f"{v.title}")]
    pilot_ps = prog("pilot")
    if v.pilot:
        st.append(
            Stage(
                "pilot",
                "done",
                f"n={v.pilot.get('n_per_family')}"
                + (" underpowered" if v.pilot.get("underpowered") else ""),
            )
        )
    else:
        s, d = live(pilot_ps)
        st.append(
            Stage(
                "pilot",
                s if pilot_ps else ("running" if any("pilot" in p for p in v.procs) else "pending"),
                d,
            )
        )
    st.append(
        Stage(
            "lock", "done" if v.lock else "pending", f"n={v.lock['n_per_family']}" if v.lock else ""
        )
    )
    if v.manifest:
        st.append(Stage("run", "done", ""))
    else:
        s, d = live(prog("run"))
        st.append(Stage("run", s, d))
    if v.receipt:
        st.append(Stage("eval", "done", str(v.receipt.get("decision", ""))))
    else:
        s, d = live(prog("eval"))
        st.append(Stage("eval", s, d))
    st.append(Stage("verify", "done" if v.verified else "pending", ""))
    st.append(
        Stage(
            "decide",
            "done" if v.owner_decisions else "pending",
            str(v.owner_decisions[-1].get("owner_decision", "")) if v.owner_decisions else "",
        )
    )
    # a stage is "running" only if it is the first unfinished one
    seen_open = False
    for stage in st:
        if stage.state != "done":
            if seen_open and stage.state == "running":
                stage.state = "pending"
            seen_open = True
    return st


def load(path: Path) -> RunView:
    v = RunView(path=path)
    pk = read_json(path / "packet.frozen.json")
    if pk:
        v.packet, v.packet_digest = pk.get("packet"), str(pk.get("packet_digest", ""))
    v.pilot = read_json(path / "pilot.json")
    v.lock = read_json(path / "lock.public.json")
    v.manifest = read_json(path / "run" / "manifest.json")
    v.receipt = read_json(path / "eval" / "receipt.json")
    v.learning = read_json(path / "learning" / "manifest.json")
    v.owner_decisions = read_jsonl(path / "decisions.jsonl")
    if v.receipt:
        try:
            v.verified = (path / "receipt.verified").read_text() == v.receipt.get("receipt_digest")
        except OSError:
            v.verified = False
    for p in sorted((path / "progress").glob("*.json")) if (path / "progress").exists() else []:
        d = read_json(p)
        if d:
            v.progress[p.stem] = d
    for f in sorted((path / "run").glob("*.jsonl")) if (path / "run").exists() else []:
        v.ledgers[f.stem] = count_lines(f)
    if (path / "learning" / "training.jsonl").exists():
        v.ledgers["training"] = count_lines(path / "learning" / "training.jsonl")
    v.procs = _procs(v)
    v.repro = _repro_for(v.packet_digest) if v.packet_digest else None
    now = time.time()
    v.stages = _stages(v, now)
    mt = [p.stat().st_mtime for p in path.rglob("*") if p.is_file()][:2000] or [now]
    v.updated = max(mt)
    return v


def discover(root: Path) -> list[Path]:
    runs = (
        [p for p in sorted(root.iterdir()) if (p / "packet.frozen.json").exists()]
        if root.exists()
        else []
    )
    return sorted(
        runs, key=lambda p: max((x.stat().st_mtime for x in p.iterdir()), default=0), reverse=True
    )
