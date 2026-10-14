from __future__ import annotations

import re

from hsi.codec.base import State, Trace

_STEP = re.compile(r"step (\d+): action (\d+) -> cells \[([\d, ]+)\]")


def _cells(s: State) -> str:
    return "[" + ", ".join(map(str, s)) + "]"


class TextCodec:
    name = "text"

    def encode(self, trace: Trace) -> bytes:
        out = [
            f"hsi-text-v1 provenance={trace.provenance}\n",
            f"start cells {_cells(trace.initial)}\n",
        ]
        out += [
            f"step {i}: action {a} -> cells {_cells(s)}\n" for i, (a, s) in enumerate(trace.steps)
        ]
        return "".join(out).encode("utf-8")

    def decode(self, data: bytes) -> Trace:
        lines = data.decode("utf-8").splitlines()
        prov = lines[0].split("provenance=", 1)[1]
        init = tuple(int(x) for x in lines[1].split("[", 1)[1].rstrip("]").split(","))
        steps = []
        for ln in lines[2:]:
            m = _STEP.fullmatch(ln)
            if not m:
                raise ValueError(f"bad line {ln!r}")
            steps.append((int(m[2]), tuple(int(x) for x in m[3].split(","))))
        return Trace(prov, init, tuple(steps))
