"""Standalone receipt recomputation. Imports nothing from `hsi`.

usage: python -I validator/recompute.py <receipt.json> <packet.frozen.json>
"""
import hashlib
import json
import math
import sys


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def check(receipt, frozen):
    errs = []
    body = {k: v for k, v in receipt.items() if k != "receipt_digest"}
    if digest(body) != receipt["receipt_digest"]:
        errs.append("receipt_digest")
    if digest(frozen["packet"]) != frozen["packet_digest"]:
        errs.append("packet digest (frozen file)")
    if receipt["packet_digest"] != frozen["packet_digest"]:
        errs.append("packet_digest binding")
    pk = frozen["packet"]
    if receipt["threshold"] != pk["threshold"] or receipt["nonclaims"] != pk["nonclaims"]:
        errs.append("threshold/nonclaims differ from packet")
    if [receipt["baseline_arm"], receipt["treatment_arm"]] != pk["arms"][:2]:
        errs.append("arms differ from packet")
    succ = {(r["family"], r["arm"]): r["successes"] for r in receipt["family_results"]}
    ds, vs = [], []
    for p in receipt["pairs"]:
        n, ot, ob = p["n"], p["only_treatment"], p["only_baseline"]
        if p["both"] + ot + ob + p["neither"] != n:
            errs.append(f"pair table sums {p['family']}")
        f = p["family"]
        if succ[(f, receipt["treatment_arm"])] != p["both"] + ot:
            errs.append(f"treatment successes {f}")
        if succ[(f, receipt["baseline_arm"])] != p["both"] + ob:
            errs.append(f"baseline successes {f}")
        ds.append((ot - ob) / n)
        vs.append((ot + ob - (ot - ob) ** 2 / n) / n**2)
    k = len(ds)
    mean = sum(ds) / k
    se = math.sqrt(sum(vs)) / k
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    for name, got, want in (("paired_diff", receipt["paired_diff"], mean),
                            ("ci_low", receipt["ci_low"], lo), ("ci_high", receipt["ci_high"], hi)):
        if not close(got, want):
            errs.append(f"{name}: receipt {got} != recomputed {want}")
    g = receipt["guardrail_values"]
    held = g["recon_exact_fraction"] == 1.0 and g["max_payload_bytes"] <= g["byte_budget"]
    if held != receipt["guardrails_held"]:
        errs.append("guardrails_held")
    if g["byte_budget"] != pk["budget"]["max_message_bytes"]:
        errs.append("byte budget differs from packet")
    thr = pk["threshold"]
    if not held or hi < thr:
        dec = "NO_CANDIDATE"
    else:
        dec = "PROMOTE" if lo >= thr else "INCONCLUSIVE"
    if dec != receipt["decision"]:
        errs.append(f"decision: receipt {receipt['decision']} != recomputed {dec}")
    return errs


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        rec = json.load(f)
    with open(sys.argv[2]) as f:
        frz = json.load(f)
    errors = check(rec, frz)
    if errors:
        print("MISMATCH: " + "; ".join(errors))
        sys.exit(1)
    print("OK " + rec["receipt_digest"])
