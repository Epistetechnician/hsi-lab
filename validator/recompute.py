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


def stats(pairs, errs, succ, base, treat):
    ds, vs = [], []
    for p in pairs:
        n, ot, ob = p["n"], p["only_treatment"], p["only_baseline"]
        f = p["family"]
        if p["both"] + ot + ob + p["neither"] != n:
            errs.append(f"pair table sums {f}")
        if succ[(f, treat)] != p["both"] + ot:
            errs.append(f"treatment successes {f}")
        if succ[(f, base)] != p["both"] + ob:
            errs.append(f"baseline successes {f}")
        ds.append((ot - ob) / n)
        vs.append((ot + ob - (ot - ob) ** 2 / n) / n**2)
    k = len(ds)
    mean = sum(ds) / k
    se = math.sqrt(sum(vs)) / k
    return ds, mean, mean - 1.96 * se, mean + 1.96 * se


def guardrails(exp, pk, receipt, comps_ds, errs):
    g = receipt["guardrail_values"]
    if exp == "E1":
        if g["byte_budget"] != pk["budget"]["max_message_bytes"]:
            errs.append("byte budget differs from packet")
        return g["recon_exact_fraction"] == 1.0 and g["max_payload_bytes"] <= g["byte_budget"]
    if exp == "E2":
        treat = pk["params"].get("treatment", pk["arms"][1])
        ds = comps_ds[("real_only", treat)]
        floor = pk["params"]["family_regression_floor"]
        if not close(g["min_family_diff_vs_real_only"], min(ds)):
            errs.append("min_family_diff_vs_real_only")
        if g["family_regression_floor"] != floor:
            errs.append("floor differs from packet")
        return (min(ds) >= -floor and g["example_count_equal_fraction"] == 1.0
                and g["lineage_fraction"] == 1.0)
    if exp == "E3":
        if g["message_byte_cap"] != pk["budget"]["max_message_bytes"]:
            errs.append("message cap differs from packet")
        return g["max_message_bytes_used"] <= g["message_byte_cap"] \
            and g["compute_matched_fraction"] == 1.0
    if exp == "E4":
        q = pk["params"]
        if g["fit_regression_floor"] != q["fit_regression_floor"]:
            errs.append("fit floor differs from packet")
        diff = (g["fit_success_candidate"] - g["fit_success_base"]) / g["fit_n"]
        return diff >= -g["fit_regression_floor"] and g["compute_matched_fraction"] == 1.0
    if exp == "E6":
        q = pk["params"]
        gens = q["generations"]
        tot, ok = {}, {}
        for r in receipt["family_results"]:
            tot[r["arm"]] = tot.get(r["arm"], 0) + r["n"]
            ok[r["arm"]] = ok.get(r["arm"], 0) + r["successes"]
        r1, rg = ok["d1_replay"] / tot["d1_replay"], ok[f"d{gens}_replay"] / tot[f"d{gens}_replay"]
        if not (close(g["rate_d1_replay"], r1) and close(g[f"rate_d{gens}_replay"], rg)):
            errs.append("replay rates differ from family_results")
        if g["non_collapse_floor"] != q["non_collapse_floor"]:
            errs.append("non_collapse_floor differs from packet")
        return g["min_replay_diversity"] >= g["diversity_floor"] and rg - r1 >= -g["non_collapse_floor"]
    if exp == "E7":
        return g["duplicate_strength_updates"] == 0 and g["compute_matched_fraction"] == 1.0
    errs.append(f"no validator rule for {exp}")
    return False


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
    treat = pk["params"].get("treatment", pk["arms"][1])
    bases = pk["params"].get("baselines", [pk["arms"][0]])
    if [(c["baseline_arm"], c["treatment_arm"]) for c in receipt["comparisons"]] != [
            (b, treat) for b in bases]:
        errs.append("comparisons differ from packet")
    succ = {(r["family"], r["arm"]): r["successes"] for r in receipt["family_results"]}
    cis, comps_ds = [], {}
    for c in receipt["comparisons"]:
        ds, mean, lo, hi = stats(c["pairs"], errs, succ, c["baseline_arm"], c["treatment_arm"])
        comps_ds[(c["baseline_arm"], c["treatment_arm"])] = ds
        for name, got, want in (("paired_diff", c["paired_diff"], mean), ("ci_low", c["ci_low"], lo),
                                ("ci_high", c["ci_high"], hi)):
            if not close(got, want):
                errs.append(f"{c['baseline_arm']} {name}: receipt {got} != recomputed {want}")
        cis.append((lo, hi))
    held = guardrails(pk["experiment"], pk, receipt, comps_ds, errs)
    if held != receipt["guardrails_held"]:
        errs.append("guardrails_held")
    thr = pk["threshold"]
    if not held or any(hi < thr for _, hi in cis):
        dec = "NO_CANDIDATE"
    else:
        dec = "PROMOTE" if all(lo >= thr for lo, _ in cis) else "INCONCLUSIVE"
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
