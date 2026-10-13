"""`hsi` command line. M0: `packet new|freeze|check`. M1: `baselines`. M2: `pilot|lock|run|eval|verify-receipt|decide` (E1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from hsi import pipeline
from hsi.contracts.io import load_json
from hsi.contracts.packet import EXPERIMENTS, SCHEMA_VERSION, Packet, canonical_digest
from hsi.errors import HsiError

EXIT_INVALID, EXIT_DIGEST, EXIT_EXISTS = 2, 3, 4


E1_PARAMS = {"families": ["rule_induction", "compositional_transfer", "counterevidence",
                          "relational_return"],
             "trace_len": 40, "pilot_tasks": 30, "min_n": 30, "max_n": 400}


def template(exp: str) -> dict[str, object]:
    t: dict[str, object] = {
        "schema_version": SCHEMA_VERSION, "experiment": exp, "version": 1,
        "hypothesis": "TODO", "null_hypothesis": "TODO", "arms": ["baseline", "treatment"],
        "primary_metric": "held_out_success_rate", "threshold": 0.05,
        "guardrails": ["TODO"],
        "budget": {"max_tokens": 0, "max_calls": 0, "max_wall_seconds": 3600,
                   "max_message_bytes": 0},
        "seeds": [0], "stop_rules": ["any digest, schema, split, or budget mismatch"],
        "claim_ceiling": "LocalSyntheticExperimentOnly",
        "nonclaims": ["no general intelligence claim", "not a proof"], "params": {},
    }
    if exp == "E1":
        t.update(
            hypothesis="The typed-event codec yields higher held-out success than the text codec "
                       "at an equal byte budget.",
            null_hypothesis="Held-out success difference (typed_event - text) is below the threshold.",
            arms=["text", "typed_event"], primary_metric="heldout_success_diff_typed_minus_text",
            guardrails=["lossless_roundtrip", "payload_within_byte_budget"],
            params=E1_PARAMS)
        t["budget"] = {"max_tokens": 0, "max_calls": 10_000_000, "max_wall_seconds": 3600,
                       "max_message_bytes": 200}
    return t


def freeze(src: Path) -> Path:
    packet = Packet.model_validate(load_json(src), strict=False)
    body = packet.model_dump(mode="json")
    frozen = {"packet": body, "packet_digest": canonical_digest(body)}
    dst = src.with_suffix(".frozen.json")
    if dst.exists():
        raise FileExistsError(str(dst))
    dst.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", "utf-8")
    dst.chmod(0o444)
    return dst


def check(path: Path) -> str:
    data = load_json(path)
    if set(data) != {"packet", "packet_digest"}:
        raise ValueError("frozen packet must have exactly packet and packet_digest")
    Packet.model_validate(data["packet"], strict=False)
    if canonical_digest(data["packet"]) != data["packet_digest"]:
        raise RuntimeError("digest mismatch")
    return str(data["packet_digest"])


def baselines(out: Path, n: int) -> Path:
    import random

    from hsi.accounting import Meter
    from hsi.agents import POLICIES, rollout
    from hsi.contracts.records import Split
    from hsi.environment import SEED_BASE, TASK_FAMILIES, RuleWorld
    from hsi.experience import Ledger
    from hsi.paths import refuse_in_repo

    out = refuse_in_repo(out)
    led, meter, table = Ledger(out / "baselines.ledger.jsonl"), Meter(), {}
    for name, pol in POLICIES.items():
        for fam in TASK_FAMILIES:
            for split in (Split.FIT, Split.TUNE):  # held-out is never touched before lock
                wins = 0
                for i in range(n):
                    w = RuleWorld(fam, split, SEED_BASE[split] + i)
                    ep = rollout(w, pol(w, random.Random(i)))
                    led.append(ep.model_copy(update={"episode_id": f"{name}-{ep.episode_id}"}))
                    meter.charge(calls=len(ep.steps))
                    wins += ep.outcome.success
                table[f"{name}/{fam}/{split.value}"] = wins / n
    dst = out / "baselines.json"
    dst.write_text(json.dumps({"success": table, "resources": meter.snapshot(),
                               "ledger_head": led.head}, indent=2, sort_keys=True) + "\n", "utf-8")
    return dst


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hsi")
    sub = p.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("packet").add_subparsers(dest="action", required=True)
    n = pk.add_parser("new")
    n.add_argument("experiment", choices=sorted(EXPERIMENTS))
    n.add_argument("out", type=Path)
    pk.add_parser("freeze").add_argument("file", type=Path)
    pk.add_parser("check").add_argument("file", type=Path)
    b = sub.add_parser("baselines")
    b.add_argument("out", type=Path)
    b.add_argument("-n", type=int, default=50)
    for cmd in ("pilot", "lock", "run", "eval", "verify-receipt"):
        sub.add_parser(cmd).add_argument("path", type=Path)
    dc = sub.add_parser("decide")
    dc.add_argument("path", type=Path)
    dc.add_argument("outcome")
    a = p.parse_args(argv)
    try:
        if a.cmd == "baselines":
            print(baselines(a.out, a.n))
            return 0
        if a.cmd != "packet":
            fn = {"pilot": pipeline.pilot, "lock": pipeline.lock, "run": pipeline.run,
                  "eval": pipeline.evaluate, "verify-receipt": pipeline.verify_receipt}
            print(pipeline.decide(a.path, a.outcome) if a.cmd == "decide" else fn[a.cmd](a.path))
            return 0
        if a.action == "new":
            if a.out.exists():
                raise FileExistsError(str(a.out))
            a.out.write_text(json.dumps(template(a.experiment), indent=2) + "\n", "utf-8")
            print(a.out)
        elif a.action == "freeze":
            print(freeze(a.file))
        else:
            print(check(a.file))
        return 0
    except HsiError as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return e.code
    except FileExistsError as e:
        print(f"error: exists: {e}", file=sys.stderr)
        return EXIT_EXISTS
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_DIGEST
    except (ValueError, ValidationError) as e:
        print(f"error: invalid: {e}", file=sys.stderr)
        return EXIT_INVALID
