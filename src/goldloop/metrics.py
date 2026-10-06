"""Aggregate run results into the numbers worth quoting.

mean     average case score: partial credit for partial answers
pass^k   share of cases that passed EVERY repeat. The honest number: a case
         that passes 2 of 3 times fails for a real user one time in three.
flaky    cases that passed some repeats and failed others
CI       Wilson 95% interval on pass^k, so "97.6%" on 42 cases is read as
         what it is: 88-100%.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .util import percentile


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def case_summary(runs: list[dict]) -> dict:
    scored = [r for r in runs if r["status"] != "unavailable"]
    if not scored:
        return {"status": "unavailable", "mean": None, "passed_all": False, "flaky": False,
                "n_scored": 0, "n_unavailable": len(runs)}
    passes = [r["passed"] for r in scored]
    return {"status": "error" if all(r["status"] == "error" for r in scored) else "ok",
            "mean": sum(r["score"] for r in scored) / len(scored),
            "passed_all": all(passes), "passed_any": any(passes),
            "flaky": any(passes) and not all(passes),
            "n_scored": len(scored), "n_unavailable": len(runs) - len(scored)}


def cost_of(usage: dict, pricing: dict, fallback_model: str = "") -> float | None:
    """pricing: {model-substring: {input: $/Mtok, output: $/Mtok}}. Prices live in
    config, not in this library, because hard-coded prices go stale silently."""
    if not usage or not pricing:
        return None
    by_model = usage.get("by_model") or {fallback_model: usage}
    total, priced = 0.0, False
    for model, u in by_model.items():
        rate = next((r for key, r in pricing.items() if key in (model or "")), None)
        if rate:
            priced = True
            total += u.get("input_tokens", 0) / 1e6 * rate.get("input", 0)
            total += u.get("output_tokens", 0) / 1e6 * rate.get("output", 0)
    return total if priced else None


def aggregate(cases: dict, pricing: dict | None = None, model: str = "") -> dict:
    """cases: {case_id: {"tags": [...], "runs": [...], "summary": {...}}}"""
    sums = [c["summary"] for c in cases.values()]
    live = [s for s in sums if s["status"] != "unavailable"]
    all_runs = [r for c in cases.values() for r in c["runs"]]
    lat = [r["latency_ms"] for r in all_runs if r["status"] == "ok"]
    tok_in = sum(r.get("usage", {}).get("input_tokens", 0) for r in all_runs)
    tok_out = sum(r.get("usage", {}).get("output_tokens", 0) for r in all_runs)
    costs = [cost_of(r.get("usage", {}), pricing or {}, model) for r in all_runs]
    known = [c for c in costs if c is not None]
    k = sum(s["passed_all"] for s in live)
    n = len(live)

    by_tag: dict = defaultdict(list)
    for c in cases.values():
        if c["summary"]["status"] != "unavailable":
            for t in c.get("tags") or ["untagged"]:
                by_tag[t].append(c["summary"])

    return {
        "n_cases": len(sums), "n_scored": n,
        "complete": n == len(sums) and not any(s["n_unavailable"] for s in sums),
        "mean": sum(s["mean"] for s in live) / n if n else None,
        "pass_all": k / n if n else None, "pass_all_k": k,
        "pass_all_ci95": wilson(k, n),
        "flaky": sorted(cid for cid, c in cases.items() if c["summary"].get("flaky")),
        "errors": sorted(cid for cid, c in cases.items() if any(r["status"] == "error" for r in c["runs"])),
        "unavailable_runs": sum(s["n_unavailable"] for s in sums),
        "latency_ms": {"p50": percentile(lat, 50), "p95": percentile(lat, 95)},
        "tokens": {"input": tok_in, "output": tok_out},
        "cost_usd": sum(known) if known else None,
        "cost_per_case_usd": (sum(known) / len(known)) if known else None,
        "by_tag": {t: {"n": len(v), "mean": sum(s["mean"] for s in v) / len(v),
                       "pass_all": sum(s["passed_all"] for s in v) / len(v)}
                   for t, v in sorted(by_tag.items())},
    }
