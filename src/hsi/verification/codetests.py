"""Code-domain verifiers (E8): MBPP+ tasks, extraction, sandboxed test execution, test tiers.

Tiers, weakest to strongest (all run the same candidate against a different test set):
  none   no execution: any extractable code passes
  ex     the single example assertion shown in the prompt
  base   the original MBPP assertions (usually 3)
  plus   the EvalPlus test script (hundreds of inputs; also covers base)
Hidden evaluation always uses `plus` on tasks no chain ever trained on.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data" / "mbppplus.jsonl"
DATA_SHA256 = "d1fc99cd220d2b3960aa756e99c2b06ebebbc8b85b0589f916cd54c83e01e72e"
TIERS = ("none", "ex", "base", "plus")
TIMEOUT_S = 10
BROKEN = frozenset({590, 599})  # reference solution fails its own EvalPlus script (flaky/env)

_SANDBOX = (
    "(version 1)(allow default)(deny network*)"
    '(deny file-write* (require-not (subpath "{tmp}")))'
)  # macOS: no network, writes only in the scratch dir


@dataclass(frozen=True)
class Task:
    task_id: int
    prompt: str
    code: str  # reference solution (never shown to a model)
    test_list: tuple[str, ...]
    test: str  # EvalPlus script
    imports: tuple[str, ...]


@cache
def tasks() -> dict[int, Task]:
    raw = DATA.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("data/mbppplus.jsonl does not match its pinned digest")
    out: dict[int, Task] = {}
    for ln in raw.decode("utf-8").splitlines():
        d = json.loads(ln)
        out[int(d["task_id"])] = Task(
            int(d["task_id"]), d["prompt"], d["code"], tuple(d["test_list"]), d["test"],
            tuple(d["test_imports"]),
        )  # fmt: skip
    return out


def split_ids(tune_frac: float = 0.4) -> tuple[list[int], list[int]]:
    """Deterministic tune/held-out split by task-id hash: the tune pool feeds the chains, the
    held-out pool is only ever scored."""
    ids = sorted(set(tasks()) - BROKEN)
    key = lambda i: hashlib.sha256(f"e8-split/{i}".encode()).hexdigest()  # noqa: E731
    order = sorted(ids, key=key)
    k = round(len(ids) * tune_frac)
    return sorted(order[:k]), sorted(order[k:])


def user_prompt(t: Task) -> str:
    return (
        f"{t.prompt}\nYour code should pass this test:\n{t.test_list[0]}\n"
        "Reply with only the Python function in a ```python code block."
    )


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    m = _FENCE.findall(text)
    code = (m[0] if m else "").strip("\n")
    if not m and "def " in text:  # an unfenced function body
        code = text.split("<|im_end|>")[0].strip("\n")
    return code if "def " in code else None


def tests_for(t: Task, tier: str) -> str | None:
    if tier == "none":
        return None
    if tier == "ex":
        return t.test_list[0]
    if tier == "base":
        return "\n".join(t.test_list)
    if tier == "plus":
        return t.test
    raise ValueError(tier)


def passes(code: str | None, t: Task, tier: str) -> bool:
    """Run `code` against the tier's tests in a scratch dir with limits. Never raises."""
    if code is None:
        return False
    tests = tests_for(t, tier)
    if tests is None:
        return True
    prog = "\n".join([*t.imports, code, "", tests]) + "\n"
    with tempfile.TemporaryDirectory(prefix="hsi-code-") as tmp:
        tmp = os.path.realpath(tmp)
        f = Path(tmp) / "main.py"
        f.write_text(prog, encoding="utf-8")
        cmd = [sys.executable, "-I", str(f)]
        if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists():
            cmd = ["/usr/bin/sandbox-exec", "-p", _SANDBOX.format(tmp=tmp), *cmd]
        try:
            r = subprocess.run(
                cmd, cwd=tmp, capture_output=True, timeout=TIMEOUT_S, check=False,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                preexec_fn=_limits,
            )  # fmt: skip
        except (subprocess.TimeoutExpired, OSError):
            return False
        return r.returncode == 0


def _limits() -> None:  # pragma: no cover - runs in the child
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
