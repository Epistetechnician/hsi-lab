"""HumanEval+ (EvalPlus, 164 tasks) as an extra held-out pool for the code-domain study.

Same sandboxed execution as MBPP+ (hsi.verification.codetests). The model sees the function stub and
replies with the full function; the program is  stub + reply + EvalPlus test script + check(entry).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from hsi.verification.codetests import _SANDBOX, TIMEOUT_S, _limits

DATA = Path(__file__).resolve().parents[3] / "data" / "humanevalplus.jsonl.gz"
BROKEN = frozenset({"HumanEval/32"})  # reference fails its own script (float root-finding)
DATA_SHA256 = "20c489316f5de0c76ffa016cd57964bbcd81836df66a41efff0376e19ab585af"


@dataclass(frozen=True)
class HETask:
    task_id: str
    prompt: str
    solution: str  # reference body (never shown to a model)
    entry_point: str
    test: str


@cache
def tasks() -> dict[str, HETask]:
    raw = DATA.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("data/humanevalplus.jsonl.gz does not match its pinned digest")
    out: dict[str, HETask] = {}
    for ln in gzip.decompress(raw).decode("utf-8").splitlines():
        d = json.loads(ln)
        out[d["task_id"]] = HETask(
            d["task_id"], d["prompt"], d["canonical_solution"], d["entry_point"], d["test"]
        )
    return out


def pool() -> list[str]:
    return sorted(set(tasks()) - BROKEN)


def user_prompt(t: HETask) -> str:
    return (
        "Complete this Python function. Reply with the full function (including its signature) "
        f"in a ```python code block.\n\n```python\n{t.prompt}```"
    )


def passes(code: str | None, t: HETask) -> bool:
    """Run the reply against the EvalPlus tests in a scratch dir with limits. Never raises."""
    if code is None:
        return False
    prog = f"{t.prompt}\n\n{code}\n\n{t.test}\n\ncheck({t.entry_point})\n"
    with tempfile.TemporaryDirectory(prefix="hsi-he-") as tmp:
        tmp = os.path.realpath(tmp)
        f = Path(tmp) / "main.py"
        f.write_text(prog, encoding="utf-8")
        cmd = [sys.executable, "-I", str(f)]
        if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists():
            cmd = ["/usr/bin/sandbox-exec", "-p", _SANDBOX.format(tmp=tmp), *cmd]
        try:
            r = subprocess.run(
                cmd, cwd=tmp, capture_output=True, timeout=TIMEOUT_S * 3, check=False,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"}, preexec_fn=_limits,
            )  # fmt: skip
        except (subprocess.TimeoutExpired, OSError):
            return False
        return r.returncode == 0
