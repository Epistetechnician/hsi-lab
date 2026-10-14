"""Pure renderers: RunView -> list of (text, style) lines. No curses here, so they are testable
and also drive `hsi watch --once`. Styles: '', ok, warn, bad, dim, hdr, acc."""

from __future__ import annotations

import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from hsi.paths import REPO
from hsi.watch.model import RunView

Line = tuple[str, str]
GLYPH = {"done": "✓", "running": "▶", "stalled": "!", "pending": "·"}
STYLE = {"done": "ok", "running": "acc", "stalled": "bad", "pending": "dim"}
TABS = ["overview", "mechanism", "data", "receipt", "events"]


def bar(frac: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    return "█" * full + "░" * (width - full)


def spark(vals: list[float], lo: float = 0.0, hi: float = 1.0) -> str:
    ticks = "▁▂▃▄▅▆▇█"
    return "".join(ticks[max(0, min(7, int((v - lo) / ((hi - lo) or 1) * 7.99)))] for v in vals)


def eta(
    done: float, total: float, started: float, now: float | None = None, rate: float = 0.0
) -> str:
    """ETA from the recent seconds-per-unit EMA if the producer reports it, else the average."""
    now = now or time.time()
    if done <= 0 or total <= 0 or done >= total:
        return ""
    s = (total - done) * (rate if rate > 0 else (now - started) / done)
    return (
        f"eta {int(s // 3600)}h{int(s % 3600 // 60):02d}m"
        if s >= 3600
        else f"eta {int(s // 60)}m{int(s % 60):02d}s"
    )


def ci_bar(
    diff: float, lo: float, hi: float, thr: float, width: int = 41, span: float = 0.3
) -> str:
    """Axis [-span, +span]: '|' zero, 'T' threshold, '[---o---]' the 95% interval."""

    def pos(x: float) -> int:
        return max(0, min(width - 1, round((x + span) / (2 * span) * (width - 1))))

    cells = [" "] * width
    cells[pos(0)] = "|"
    cells[pos(thr)] = "T"
    a, b = pos(lo), pos(hi)
    for i in range(a, b + 1):
        if cells[i] == " ":
            cells[i] = "─"
    cells[a], cells[b] = "[", "]"
    cells[pos(diff)] = "●"
    return "".join(cells)


def _arm_of(eid: str, arms: list[str]) -> str:
    best = ""
    for a in arms:
        if (eid.startswith((a + "/", a + "-"))) and len(a) > len(best):
            best = a
    return best or "?"


def _role(v: RunView, arm: str) -> str:
    p = v.params
    if arm == p.get("treatment", v.arms[1] if len(v.arms) > 1 else ""):
        return "treatment"
    if arm in p.get("baselines", v.arms[:1]):
        return "baseline"
    return "ref"


def _table(rows: list[list[str]], head: list[str]) -> list[Line]:
    cols = [head, *rows]
    w = [max(len(str(r[i])) for r in cols) for i in range(len(head))]
    out = [("  " + "  ".join(str(h).ljust(w[i]) for i, h in enumerate(head)), "dim")]
    out += [("  " + "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)), "") for r in rows]
    return out


# ---------------------------------------------------------------------------------- overview
def overview(v: RunView, width: int = 100) -> list[Line]:
    out: list[Line] = []
    pk = v.packet or {}
    out.append((f"{v.title}   run {v.digest16}   packet {v.packet_digest[:12]}", "hdr"))
    if pk:
        out.append((f"H: {pk.get('hypothesis', '')}", ""))
        out.append(
            (
                (
                    f"threshold +{pk.get('threshold')}   primary: {pk.get('primary_metric')}   "
                    f"claim ceiling: {pk.get('claim_ceiling')}"
                ),
                "dim",
            )
        )
    out.append(("", ""))
    out.append(("PIPELINE", "hdr"))
    line = ""
    for s in v.stages:
        line += f"[{GLYPH[s.state]} {s.name}] "
    out.append((line, "acc" if any(s.state == "running" for s in v.stages) else ""))
    for s in v.stages:
        if s.state in ("running", "stalled") or (s.detail and s.state == "done"):
            out.append((f"  {GLYPH[s.state]} {s.name:<7} {s.state:<8} {s.detail}", STYLE[s.state]))
    out.append(("", ""))
    live = [p for p in v.progress.values() if p.get("finished") is None]
    done = [p for p in v.progress.values() if p.get("finished") is not None]
    if live:
        out.append(("LIVE", "hdr"))
        for p in sorted(live, key=lambda x: x.get("stage", "")):
            frac = p["done"] / p["total"] if p.get("total") else 0.0
            age = int(time.time() - p.get("updated", 0))
            out.append(
                (
                    (
                        f"  {p['stage']:<15} {bar(frac, 24)} {p['done']}/{p['total']} "
                        f"{eta(p['done'], p['total'], p['started'], rate=p.get('rate', 0.0))}"
                    ),
                    "acc",
                )
            )
            if p.get("note"):
                out.append((f"    {p['note']}  (updated {age}s ago)", "dim"))
    elif not v.receipt and v.stages[1].state == "running" and expected_adapters(v):
        d, n, cur = adapter_status(v)
        out.append(("LIVE  (inferred from files; this run predates progress reporting)", "hdr"))
        out.append((f"  adapters        {bar(d / n, 24)} {d}/{n}   now: {cur or '—'}", "acc"))
    if v.procs:
        out.append(("PROCESSES  " + " · ".join(v.procs), "acc"))
    elif not v.receipt:
        out.append(("PROCESSES  none running for this packet", "dim"))
    for p in sorted(done, key=lambda x: x.get("started", 0)):
        dur = int(p["finished"] - p["started"])
        out.append(
            (f"  ✓ {p['stage']:<15} {p['total']} units in {dur // 60}m{dur % 60:02d}s", "dim")
        )
    if v.receipt:
        r = v.receipt
        out.append(("", ""))
        sty = {"PROMOTE": "ok", "NO_CANDIDATE": "bad", "INCONCLUSIVE": "warn"}.get(
            r["decision"], ""
        )
        out.append(
            (
                f"RESULT  {r['decision']}   "
                f"guardrails {'held' if r['guardrails_held'] else 'FAILED'}"
                f"   verified {'yes' if v.verified else 'no'}"
                + (
                    f"   reproduced {v.repro['aggregates_matched']}/"
                    f"{v.repro['aggregates_compared']}"
                    if v.repro
                    else ""
                ),
                sty,
            )
        )
    return out


# ---------------------------------------------------------------------------------- mechanism
def _dev_success(v: RunView) -> list[Line]:
    eps = v.rows("run/episodes.jsonl")
    if not eps:
        return [("  (no dev-split rollouts yet)", "dim")]
    n: Counter[tuple[str, str]] = Counter()
    ok: Counter[tuple[str, str]] = Counter()
    for e in eps:
        k = (_arm_of(str(e.get("episode_id", "")), v.arms), str(e.get("family", "")))
        n[k] += 1
        ok[k] += bool(e.get("outcome", {}).get("success"))
    fams = sorted({f for _, f in n})
    rows = []
    for a in v.arms:
        cells = [f"{ok[(a, f)] / n[(a, f)]:.2f}({n[(a, f)]})" if n[(a, f)] else "—" for f in fams]
        tot, hits = sum(n[(a, f)] for f in fams), sum(ok[(a, f)] for f in fams)
        rows.append([a, _role(v, a), *cells, f"{hits / tot:.2f}" if tot else "—"])
    return [
        ("DEV-SPLIT ROLLOUTS (tune seeds; held-out stays locked)", "hdr"),
        *_table(rows, ["arm", "role", *fams, "all"]),
    ]


def expected_adapters(v: RunView) -> list[str]:
    """Adapter files this packet will produce, in training order (LLM E4/E6 only)."""
    p = v.params
    if v.experiment == "E4" and p.get("backbone") == "llm":
        return ["lora_candidate"]
    if v.experiment == "E6" and p.get("backbone") == "llm":
        g = int(p.get("generations", 3))
        order = [f"d{i}_{m}" for m in ("no_replay", "replay") for i in range(1, g + 1)]
        return order + (["real_only"] if p.get("extra_arms") else [])
    return []


def adapter_status(v: RunView) -> tuple[int, int, str]:
    """(done, total, current) from files + training log, so runs without progress files work too."""
    exp = expected_adapters(v)
    done = [a for a in exp if (v.path / "learning" / f"{a}.safetensors").exists()]
    seen = [str(r.get("arm")) for r in v.rows("learning/training.jsonl") if r.get("arm")]
    cur = next((a for a in reversed(seen) if a not in done), "")
    return len(done), len(exp), cur


def _learning(v: RunView) -> list[Line]:
    out: list[Line] = []
    m = v.learning
    if not m:
        if not expected_adapters(v):
            return []
        out.append(("LEARNING  (training in progress)", "hdr"))
        _d, _n, cur = adapter_status(v)
        for a in expected_adapters(v):
            done = (v.path / "learning" / f"{a}.safetensors").exists()
            st = "✓ trained" if done else ("▶ training / generating" if a == cur else "· pending")
            out.append((f"  {a:<16} {st}", "ok" if done else "acc" if a == cur else "dim"))
        for p in v.progress.values():
            if p.get("stage", "").startswith("pilot:prepare") and p.get("note"):
                out.append((f"  now: {p['note']}", "acc"))
        return out
    out.append(("BACKBONE", "hdr"))
    if m.get("model_repo"):
        out.append(
            (
                (
                    f"  {m['model_repo']} @ {str(m['model_revision'])[:8]} "
                    f"({m.get('model_license')})"
                    f"  weights {str(m['weights_digest'])[:12]}"
                ),
                "",
            )
        )
    else:
        out.append((f"  runtime {m.get('runtime')}  base {str(m.get('base_digest', ''))[:12]}", ""))
    cands = m.get("candidates", {})
    if m.get("adapter_digest"):
        out.append(
            (
                (
                    f"  adapter {str(m['adapter_digest'])[:12]}  {m.get('adapter')}  "
                    f"train {m.get('train')}"
                ),
                "",
            )
        )
    if cands:
        out.append(("ADAPTERS / CANDIDATES", "hdr"))
        rows = []
        for a, i in cands.items():
            rows.append(
                [
                    a,
                    str(i.get("generation", "")),
                    f"{i.get('loss', 0):.2f}",
                    f"{i['diversity']:.2f}" if "diversity" in i else "—",
                    f"{i['validity']:.2f}" if "validity" in i else "—",
                    f"{i.get('n_synthetic', '')}/{i.get('n_real', '')}",
                    str(i.get("digest", ""))[:10],
                ]
            )
        out += _table(rows, ["arm", "gen", "loss", "diversity", "validity", "syn/real", "digest"])
        chains = defaultdict(list)
        for a, i in cands.items():
            if a.startswith("d") and "_" in a and "diversity" in i:
                chains[a.split("_", 1)[1]].append(
                    (i["generation"], i["diversity"], i.get("validity", 0.0))
                )
        if chains:
            out.append(("GENERATOR HEALTH BY DEPTH (diversity ▁→█ | validity ▁→█)", "hdr"))
            for mode, xs in sorted(chains.items()):
                xs.sort()
                out.append(
                    (
                        (
                            f"  {mode:<10} diversity {spark([x[1] for x in xs], 0, 0.5)}  "
                            f"validity {spark([x[2] for x in xs], 0, 0.3)}   "
                            f"({' → '.join(f'{x[1]:.2f}/{x[2]:.2f}' for x in xs)})"
                        ),
                        "",
                    )
                )
            if "diversity_floor" in m:
                out.append((f"  diversity floor {m['diversity_floor']:.3f}", "dim"))
    return out


def _messages(v: RunView) -> list[Line]:
    rows = v.rows("run/messages.jsonl")
    if not rows:
        return []
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in rows:
        by[_arm_of(str(m.get("msg_id", "")), v.arms)].append(m)
    cap = (v.packet or {}).get("budget", {}).get("max_message_bytes", 0)
    out: list[Line] = [(f"MESSAGES (cap {cap} B per task)", "hdr")]
    trs = []
    for a, ms in sorted(by.items()):
        b = sum(m["bytes"] for m in ms)
        trs.append(
            [
                a,
                str(len(ms)),
                str(b),
                f"{b / len(ms):.1f}",
                ",".join(sorted({m["payload_codec"] for m in ms})),
            ]
        )
    out += _table(trs, ["arm", "msgs", "bytes", "B/msg", "codecs"])
    return out


def _links(v: RunView) -> list[Line]:
    ev = [d for d in v.rows("run/decisions.jsonl") if d.get("event") in ("link_open", "link_drop")]
    if not ev:
        return []
    c = Counter(e["event"] for e in ev)
    deliv = sum(1 for e in ev if e["event"] == "link_drop" and e.get("delivered"))
    tasks = len({e.get("task") for e in ev})
    return [
        ("LINK CHURN (adaptive graph)", "hdr"),
        (
            (
                f"  opened {c['link_open']}  dropped {c['link_drop']}  "
                f"delivered {deliv}/{c['link_drop']}"
                f"  over {tasks} tasks  "
                f"({(c['link_open'] + c['link_drop']) / max(1, tasks):.1f} events/task)"
            ),
            "",
        ),
    ]


def _verification(v: RunView) -> list[Line]:
    d = [x for x in v.rows("run/decisions.jsonl") if "reason" in x]
    if not d:
        return []
    c = Counter(x["reason"] for x in d)
    tot = len(d)
    out: list[Line] = [("VERIFIER DECISIONS (quarantine → eligible / rejected)", "hdr")]
    for r, k in c.most_common():
        out.append(
            (
                f"  {r:<22} {bar(k / tot, 24)} {k:>5} {100 * k / tot:5.1f}%",
                "ok" if r == "ok" else "warn",
            )
        )
    out.append(
        (
            (
                f"  quarantine queue {v.ledgers.get('quarantine', 0)} · eligible examples "
                f"{v.ledgers.get('examples', 0)}"
            ),
            "dim",
        )
    )
    return out


def _relations(v: RunView) -> list[Line]:
    ev = v.rows("run/relation_events.jsonl")
    if not ev:
        return []
    by: dict[str, Counter[str]] = defaultdict(Counter)
    for e in ev:
        by[str(e.get("arm", "?"))][str(e.get("op"))] += 1
    ops = ["create", "strengthen", "decouple", "dormant", "reactivate", "retire", "prune"]
    return [
        ("RELATION EVENTS BY ARM", "hdr"),
        *_table([[a, *[str(c[o]) for o in ops]] for a, c in sorted(by.items())], ["arm", *ops]),
    ]


def mechanism(v: RunView, width: int = 100) -> list[Line]:
    out: list[Line] = [(f"MECHANISM · {v.title}", "hdr")]
    pk = v.packet or {}
    out.append(("arms: " + " · ".join(f"{a} [{_role(v, a)}]" for a in v.arms), ""))
    out.append((f"budget: {pk.get('budget')}", "dim"))
    out.append(("", ""))
    for fn in (_learning, _verification, _messages, _links, _relations, _dev_success):
        try:
            part = fn(v)
        except Exception as e:  # one odd artifact must not take down the view
            part = [
                (f"({fn.__name__.strip('_')} panel unavailable: {type(e).__name__}: {e})", "bad")
            ]
        if part:
            out += [*part, ("", "")]
    return out


# ---------------------------------------------------------------------------------- data
def data(v: RunView, width: int = 100) -> list[Line]:
    out: list[Line] = [("DATA & LINEAGE", "hdr")]
    if v.ledgers:
        out.append(("LEDGERS (digest-chained, append-only)", "hdr"))
        for name, n in sorted(v.ledgers.items()):
            out.append((f"  {name:<18} {n:>8} records", ""))
    tr = v.rows("learning/training.jsonl")
    if tr:
        out.append(("", ""))
        out.append(("TRAINING DATA COMPOSITION (origin / eligibility)", "hdr"))
        by: dict[str, Counter[str]] = defaultdict(Counter)
        for r in tr:
            by[str(r.get("arm") or r.get("used_by") or "?")][
                f"{r.get('origin')}/{r.get('eligibility')}"
            ] += 1
        for a, c in sorted(by.items()):
            out.append((f"  {a:<22} " + "  ".join(f"{k}:{n}" for k, n in c.most_common(4)), ""))
        splits = Counter(str(r.get("split")) for r in tr if r.get("split"))
        if splits:
            ok = "heldout" not in splits
            out.append(
                (
                    (
                        f"  splits touched: {dict(splits)}  "
                        f"held-out untouched: {'yes' if ok else 'NO'}"
                    ),
                    "ok" if ok else "bad",
                )
            )
    out.append(("", ""))
    out.append(("DIGESTS", "hdr"))
    items = [
        ("packet", v.packet_digest),
        ("lock", (v.lock or {}).get("lock_digest", "")),
        ("run", (v.manifest or {}).get("run_digest", "")),
        ("model", (v.receipt or {}).get("model_digest", "")),
        ("data", (v.receipt or {}).get("data_digest", "")),
        ("receipt", (v.receipt or {}).get("receipt_digest", "")),
    ]
    for k, d in items:
        out.append((f"  {k:<8} {d[:24] or '—'}", "" if d else "dim"))
    if v.receipt:
        out.append((f"  source   {v.receipt.get('source_commit')}", ""))
    if v.manifest:
        res = v.manifest.get("resources", {})
        out.append(("", ""))
        out.append(("RESOURCES (dev run)", "hdr"))
        out.append(("  " + "  ".join(f"{k}={v2:.0f}" for k, v2 in res.items() if v2), ""))
    return out


# ---------------------------------------------------------------------------------- receipt
_VALIDATOR: dict[str, str] = {}


def validator_status(v: RunView) -> str:
    if not v.receipt:
        return ""
    key = str(v.receipt.get("receipt_digest"))
    if key not in _VALIDATOR:
        pf = v.path / "packet.frozen.json"
        r = subprocess.run(
            [
                sys.executable,
                "-I",
                str(REPO / "validator" / "recompute.py"),
                str(v.path / "eval" / "receipt.json"),
                str(pf),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        _VALIDATOR[key] = "OK" if r.returncode == 0 else "MISMATCH: " + r.stdout.strip()[:80]
    return _VALIDATOR[key]


def receipt(v: RunView, width: int = 100) -> list[Line]:
    r = v.receipt
    if not r:
        return [
            ("RECEIPT", "hdr"),
            ("  not produced yet — the locked evaluator has not finished", "dim"),
        ]
    sty = {"PROMOTE": "ok", "NO_CANDIDATE": "bad", "INCONCLUSIVE": "warn"}.get(r["decision"], "")
    out: list[Line] = [
        (
            (
                f"RECEIPT · {r['decision']}   threshold +{r['threshold']}   "
                f"guardrails {'HELD' if r['guardrails_held'] else 'FAILED'}"
            ),
            sty,
        )
    ]
    vs = validator_status(v)
    out.append(
        (
            f"independent recompute: {vs}"
            + (
                f"   clean-env rerun: {v.repro['aggregates_matched']}/"
                f"{v.repro['aggregates_compared']} aggregates"
                if v.repro
                else "   clean-env rerun: not done"
            ),
            "ok" if vs == "OK" else "bad",
        )
    )
    out.append(("", ""))
    out.append(("COMPARISONS   axis -0.30 … +0.30   | zero   T threshold   [─●─] 95% CI", "hdr"))
    for c in r["comparisons"]:
        clear = c["ci_low"] >= r["threshold"]
        excl = c["ci_high"] < r["threshold"]
        out.append((f"  {c['baseline_arm']} → {c['treatment_arm']}", ""))
        out.append(
            (
                (
                    f"  {ci_bar(c['paired_diff'], c['ci_low'], c['ci_high'], r['threshold'])}  "
                    f"{c['paired_diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  "
                    f"{'clears' if clear else 'excludes' if excl else 'straddles'} threshold"
                ),
                "ok" if clear else "bad" if excl else "warn",
            )
        )
    out.append(("", ""))
    out.append(("GUARDRAILS", "hdr"))
    for k, val in r["guardrail_values"].items():
        if k.startswith(("eval_", "validity_", "diversity_d")):
            continue
        out.append((f"  {k:<34} {val:.4g}", ""))
    gen = {k: val for k, val in r["guardrail_values"].items() if k.startswith("eval_valid_")}
    if gen:
        out.append(
            ("EVAL-TIME GENERATION  valid | consistent with evidence | exact | diversity", "hdr")
        )
        gvals = r["guardrail_values"]
        for k, val in gen.items():
            a = k[len("eval_valid_") :]
            extra = ""
            if "eval_exact_" + a in gvals:
                extra += f" | {gvals['eval_exact_' + a]:.2f}"
            if "eval_diversity_" + a in gvals:
                extra += f" | {gvals['eval_diversity_' + a]:.2f}"
            out.append(
                (
                    f"  {a:<14} {bar(val, 16)} {val:.2f} | {gvals.get('eval_consistent_' + a, 0):.2f}"
                    + extra,
                    "",
                )
            )
    out.append(("", ""))
    out.append(("SUCCESS BY ARM (held-out)", "hdr"))
    n: Counter[str] = Counter()
    ok: Counter[str] = Counter()
    for f in r["family_results"]:
        n[f["arm"]] += f["n"]
        ok[f["arm"]] += f["successes"]
    for a in v.arms or list(n):
        if n[a]:
            out.append(
                (f"  {a:<22} {bar(ok[a] / n[a], 28)} {ok[a] / n[a]:.3f}  [{_role(v, a)}]", "")
            )
    bu = r.get("budget_used", {})
    out.append(("", ""))
    out.append(
        (
            (
                f"budget used: wall {bu.get('wall_seconds', 0):.0f}s · "
                f"calls {bu.get('calls', 0):.0f} · "
                f"bytes {bu.get('bytes', 0):.0f} · "
                f"failed proposals {bu.get('failed_proposals', 0):.0f} · "
                f"peak mem {bu.get('peak_mem_mb', 0):.0f} MB"
            ),
            "dim",
        )
    )
    out.append((f"nonclaims: {'; '.join(r.get('nonclaims', []))}", "dim"))
    return out


# ---------------------------------------------------------------------------------- events
def events(v: RunView, width: int = 100, limit: int = 40) -> list[Line]:
    """Newest records across this run's jsonl streams (by file mtime, then position)."""
    out: list[Line] = [("EVENT STREAM (newest last)", "hdr")]
    files = sorted(
        [*(v.path / "run").glob("*.jsonl"), *(v.path / "learning").glob("*.jsonl")],
        key=lambda p: p.stat().st_mtime,
    )
    recs: list[tuple[str, dict[str, Any]]] = []
    for f in files:
        for d in v.rows(str(f.relative_to(v.path)))[-limit:]:
            recs.append((f.stem, d))
    for name, d in recs[-limit:]:
        keys = (
            "episode_id",
            "sample_id",
            "msg_id",
            "event",
            "op",
            "reason",
            "eligibility",
            "arm",
            "bytes",
            "text",
        )
        body = " ".join(f"{k}={str(d[k])[:36]}" for k in keys if k in d)
        out.append((f"  {name[:14]:<14} {body}"[: width - 1], "dim"))
    if len(out) == 1:
        out.append(("  (no records yet)", "dim"))
    return out


def _safe(fn: Callable[[RunView, int], list[Line]]) -> Callable[[RunView, int], list[Line]]:
    def wrapped(v: RunView, width: int = 100) -> list[Line]:
        try:
            return fn(v, width)
        except Exception as e:
            return [(f"{fn.__name__}: unavailable ({type(e).__name__}: {e})", "bad")]

    wrapped.__name__ = fn.__name__
    return wrapped


PANELS = {
    k: _safe(f)
    for k, f in {
        "overview": overview,
        "mechanism": mechanism,
        "data": data,
        "receipt": receipt,
        "events": events,
    }.items()
}


def list_rows(views: list[RunView]) -> list[Line]:
    out: list[Line] = [
        (f"{'run':<17}{'experiment':<22}{'stage':<18}{'result':<14}{'updated'}", "hdr")
    ]
    for v in views:
        cur = next((s for s in v.stages if s.state in ("running", "stalled")), None)
        stage = (
            f"{GLYPH[cur.state]} {cur.name} {cur.detail}"
            if cur
            else "✓ done"
            if v.receipt
            else "· idle"
        )
        res = (v.receipt or {}).get("decision", "—")
        sty = {"PROMOTE": "ok", "NO_CANDIDATE": "bad", "INCONCLUSIVE": "warn"}.get(res, "dim")
        age = int(time.time() - v.updated)
        ago = (
            f"{age // 3600}h{age % 3600 // 60:02d}m ago"
            if age >= 3600
            else f"{age // 60}m ago"
            if age >= 60
            else f"{age}s ago"
        )
        out.append(
            (
                f"{v.digest16:<17}{v.title[:21]:<22}{stage[:17]:<18}{res:<14}{ago}",
                sty if not cur else "acc",
            )
        )
    return out
