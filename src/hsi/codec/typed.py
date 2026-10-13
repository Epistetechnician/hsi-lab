"""Typed-event codec: header + start state + per-step (action, changed-cell diffs)."""
from __future__ import annotations

import hashlib

from hsi.codec.base import State, Trace

MAGIC = b"T1"


class TypedEventCodec:
    name = "typed_event"

    def encode(self, trace: Trace) -> bytes:
        out = bytearray(MAGIC)
        out += hashlib.sha256(trace.provenance.encode()).digest()[:8]
        out += bytes(trace.initial)
        prev = trace.initial
        for a, s in trace.steps:
            diff = [(i, v) for i, (u, v) in enumerate(zip(prev, s, strict=True)) if u != v]
            out.append(a << 4 | len(diff))
            out += bytes(i << 4 | v for i, v in diff)
            prev = s
        return bytes(out)

    def decode(self, data: bytes) -> Trace:
        if data[:2] != MAGIC:
            raise ValueError("bad magic")
        prov = data[2:10].hex()
        n = 5
        cur: State = tuple(data[10:10 + n])
        init, pos, steps = cur, 10 + n, []
        while pos < len(data):
            a, k = data[pos] >> 4, data[pos] & 15
            lst = list(cur)
            for b in data[pos + 1:pos + 1 + k]:
                lst[b >> 4] = b & 15
            cur = tuple(lst)
            steps.append((a, cur))
            pos += 1 + k
        return Trace(prov, init, tuple(steps))
