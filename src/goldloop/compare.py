"""Compare two runs case by case, and decide whether a change may ship.

The mean can hold steady while one case breaks and another gets fixed. So the
unit of comparison is the case, not the average: which cases regressed, which
were fixed, which went flaky.

"Did it actually help?" is answered with an exact McNemar test on the cases
that changed. Going 41/42 -> 42/42 is one case moving; p = 1.0. That is a
real fix worth keeping, and it is not evidence the system got better overall.
"""
from __future__ import annotations

from math import comb


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar: b = pass->fail, c = fail->pass."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(base: dict, cand: dict) -> dict:
    bc, cc = base["cases"], cand["cases"]
    shared = sorted(set(bc) & set(cc))
    out = {"base": base["id"], "cand": cand["id"],
           "added": sorted(set(cc) - set(bc)), "removed": sorted(set(bc) - set(cc)),
           "changed_checks": [], "regressions": [], "fixes": [], "newly_flaky": [],
           "steadied": [], "degraded": [], "improved": [], "unscored": []}
    b = c = 0
    for cid in shared:
        sb, sc = bc[cid]["summary"], cc[cid]["summary"]
        if bc[cid].get("fp") and cc[cid].get("fp") and bc[cid]["fp"] != cc[cid]["fp"]:
            out["changed_checks"].append(cid)       # different exam: don't count it either way
            continue
        if sb["status"] == "unavailable" or sc["status"] == "unavailable":
            out["unscored"].append(cid)
            continue
        if sb["passed_all"] and not sc["passed_all"]:
            out["regressions"].append(cid)
            b += 1
        elif not sb["passed_all"] and sc["passed_all"]:
            out["fixes"].append(cid)
            c += 1
        if sc.get("flaky") and not sb.get("flaky"):
            out["newly_flaky"].append(cid)
        if sb.get("flaky") and not sc.get("flaky") and sc["passed_all"]:
            out["steadied"].append(cid)
        d = (sc["mean"] or 0) - (sb["mean"] or 0)
        if d < -1e-9 and sc["passed_all"] == sb["passed_all"]:
            out["degraded"].append(cid)
        if d > 1e-9 and sc["passed_all"] == sb["passed_all"]:
            out["improved"].append(cid)

    mb, mc = base["metrics"], cand["metrics"]

    def delta(key):
        x, y = mb.get(key), mc.get(key)
        return None if x is None or y is None else y - x

    out["delta"] = {"mean": delta("mean"), "pass_all": delta("pass_all"),
                    "cost_usd": delta("cost_usd"),
                    "latency_p95_ms": mc["latency_ms"]["p95"] - mb["latency_ms"]["p95"]}
    out["mcnemar"] = {"pass_to_fail": b, "fail_to_pass": c, "p": mcnemar_p(b, c)}
    out["verdict"] = _verdict(b, c, out)
    return out


def _verdict(b: int, c: int, cmp: dict) -> str:
    p = cmp["mcnemar"]["p"]
    if b == 0 and c == 0:
        return "no case changed outcome"
    if b and c:
        return f"mixed: {c} fixed, {b} regressed (p={p:.2f})"
    if p < 0.05:
        return f"{'better' if c else 'worse'}: {max(b, c)} cases moved one way (p={p:.3f})"
    return (f"{'fixed' if c else 'broke'} {max(b, c)} case(s), too few to call the system "
            f"{'better' if c else 'worse'} overall (p={p:.2f})")


DEFAULT_POLICY = {"min_mean": None, "min_pass_all": None, "max_flaky": None,
                  "max_regressions": 0, "require_complete": True}


def gate(run: dict, baseline: dict | None, policy: dict | None = None) -> tuple[bool, list[str], dict | None]:
    pol = {**DEFAULT_POLICY, **(policy or {})}
    m, reasons = run["metrics"], []
    if pol["require_complete"] and not m["complete"]:
        reasons.append(f"incomplete: {m['unavailable_runs']} run(s) unavailable. That is an infra "
                       "problem, not a score; fix access and rerun")
    if pol["min_mean"] is not None and (m["mean"] or 0) < pol["min_mean"]:
        reasons.append(f"mean {m['mean']:.1%} < {pol['min_mean']:.0%}")
    if pol["min_pass_all"] is not None and (m["pass_all"] or 0) < pol["min_pass_all"]:
        reasons.append(f"pass^k {m['pass_all']:.1%} < {pol['min_pass_all']:.0%}")
    if pol["max_flaky"] is not None and len(m["flaky"]) > pol["max_flaky"]:
        reasons.append(f"{len(m['flaky'])} flaky case(s) > {pol['max_flaky']}: {', '.join(m['flaky'])}")
    cmp = None
    if baseline is not None:
        cmp = compare(baseline, run)
        if len(cmp["regressions"]) > pol["max_regressions"]:
            reasons.append(f"{len(cmp['regressions'])} case(s) regressed vs baseline: "
                           + ", ".join(cmp["regressions"]))
    return (not reasons), reasons, cmp
