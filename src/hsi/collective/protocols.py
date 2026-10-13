"""Message formats for collectives (F10). Every message is byte-counted."""
from __future__ import annotations

from hsi.agents.learner import Transition
from hsi.codec import CODECS, Trace, fit_to_budget
from hsi.contracts.records import Message
from hsi.environment.world import Prim

_KIND = {"add": 0, "swap": 1, "copy": 2, "mix": 3}
_NAME = {v: k for k, v in _KIND.items()}


def encode_hyps(hyps: dict[int, tuple[Prim, ...]], cap: int) -> bytes:
    """Learned-summary message: per action [a, n, (kind j k c)*n]; whole actions only."""
    out = b"H1"
    for a, h in sorted(hyps.items()):
        chunk = bytes([a, len(h)]) + b"".join(bytes([_KIND[k], j, kk, c]) for k, j, kk, c in h)
        if len(out) + len(chunk) > cap:
            break
        out += chunk
    return out if len(out) <= cap else b""


def decode_hyps(data: bytes) -> dict[int, tuple[Prim, ...]]:
    out: dict[int, tuple[Prim, ...]] = {}
    i = 2
    while data[:2] == b"H1" and i < len(data):
        a, n = data[i], data[i + 1]
        i += 2
        out[a] = tuple((_NAME[data[i + 4 * t]], data[i + 4 * t + 1], data[i + 4 * t + 2],
                        data[i + 4 * t + 3]) for t in range(n))
        i += 4 * n
    return out


def encode_log(trans: list[Transition], cap: int) -> bytes:
    """Raw experience log with the typed-event codec, truncated to the cap."""
    if not trans:
        return b""
    tr = Trace("agent-log", trans[0][0], tuple((a, s2) for _, a, s2 in trans))
    return fit_to_budget(CODECS["typed_event"], tr, cap)


def message(msg_id: str, sender: int, receivers: list[int], codec: str, payload: bytes,
            refs: list[str], expires: int) -> Message:
    return Message(msg_id=msg_id, sender=sender, receivers=receivers, payload_codec=codec,
                   bytes=len(payload), evidence_refs=refs, confidence=1.0, expires_at_step=expires)
