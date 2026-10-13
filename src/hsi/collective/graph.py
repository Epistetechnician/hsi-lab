"""Communication-graph protocols for N agents that each own one action (F10-F11).

fixed_gossip: static topology, delta-sync gossip (send only chunks the neighbour lacks).
adaptive:     demand-driven links: the planner opens a link to the owner of an action it lacks,
              drops it once it delivered (or proved useless), and stops when its model can plan.
Both share one byte cap per task; a message is truncated to whole chunks that fit.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hsi.collective.protocols import decode_hyps, encode_hyps
from hsi.contracts.records import Message
from hsi.environment.world import Prim

Know = dict[int, tuple[Prim, ...]]
QUERY = 81  # b"Q" + action id


@dataclass
class NetResult:
    messages: list[Message] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    used: int = 0
    formed: int = 0
    dropped: int = 0


def _msg(res: NetResult, key: str, s: int, d: int, codec: str, n: int, refs: list[str],
         rounds: int) -> None:
    res.messages.append(Message(msg_id=f"{key}/m{len(res.messages)}", sender=s, receivers=[d],
                                payload_codec=codec, bytes=n, evidence_refs=refs,
                                confidence=1.0, expires_at_step=rounds))
    res.used += n


def fixed_gossip(know: list[Know], edges: list[tuple[int, int]], rounds: int, cap: int,
                 key: str) -> NetResult:
    res = NetResult()
    sent: dict[tuple[int, int], set[int]] = {}
    for i, j in edges:
        sent[(i, j)], sent[(j, i)] = set(), set()
    for _ in range(rounds):
        for i, j in edges:
            for s, d in ((i, j), (j, i)):
                new = {a: h for a, h in know[s].items() if a not in sent[(s, d)]}
                if not new:
                    continue
                payload = encode_hyps(new, cap - res.used)
                if len(payload) <= 2:
                    continue
                got = decode_hyps(payload)
                know[d].update(got)
                sent[(s, d)] |= got.keys()
                sent[(d, s)] |= got.keys()
                _msg(res, key, s, d, "hyp_summary", len(payload), [f"explore/{a}" for a in got],
                     rounds)
    return res


def adaptive(know: list[Know], planner: int, rounds: int, cap: int, key: str,
             done: Callable[[Know], bool]) -> NetResult:
    res = NetResult()
    queried: set[int] = set()
    for r in range(rounds):
        if done(know[planner]):
            break
        targets = [j for j in range(len(know)) if j != planner and j not in know[planner]
                   and j not in queried]
        if not targets or res.used + 2 > cap:
            break
        j = min(targets)
        queried.add(j)
        res.formed += 1
        res.events.append({"event": "link_open", "round": r, "a": planner, "b": j})
        _msg(res, key, planner, j, "query", 2, [], rounds)
        reply = encode_hyps({j: know[j][j]}, cap - res.used) if j in know[j] else b""
        if len(reply) > 2:
            know[planner].update(decode_hyps(reply))
            _msg(res, key, j, planner, "hyp_summary", len(reply), [f"explore/{j}"], rounds)
        res.dropped += 1
        res.events.append({"event": "link_drop", "round": r, "a": planner, "b": j,
                           "delivered": len(reply) > 2})
    return res
