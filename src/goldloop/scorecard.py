"""One scorecard for any use case: Cost, Reliability, Risk, Change.

The point of goldloop is that these four questions don't change between
businesses; only the tests do. A use case declares which of its tags are
guardrails (`principles: {risk: [tag, ...]}` in goldloop.yaml). Everything
else is computed the same way for every use case:

  Cost         $ and tokens per task (prices from config)
  Reliability  pass^k with its 95% interval, flaky tests, same-steps rate
  Risk         guardrail tests passing every run, and any critical violations
  Change       this run vs the approved baseline: regressions, fixes, McNemar p

`goldloop scorecard --multi a b c` puts several use cases side by side.
"""
from __future__ import annotations

import json
from pathlib import Path

from .compare import compare
from .metrics import wilson


def _risk_cases(run: dict, risk_tags: list[str]) -> dict:
    tags = set(risk_tags or [])
    return {cid: c for cid, c in run["cases"].items() if tags & set(c.get("tags") or [])}


def scorecard(run: dict, principles: dict | None = None, baseline: dict | None = None) -> dict:
    m = run["metrics"]
    risk = _risk_cases(run, (principles or {}).get("risk", []))
    live = {cid: c for cid, c in risk.items() if c["summary"]["status"] != "unavailable"}
    k = sum(c["summary"]["passed_all"] for c in live.values())
    violations = sum(1 for c in live.values() for r in c["runs"]
                     for ck in r.get("checks") or [] if ck.get("critical") and ck["score"] < 1)
    n_runs = sum(len(c["runs"]) for c in live.values())

    change = None
    if baseline is not None:
        cmp = compare(baseline, run)
        change = {"regressions": cmp["regressions"], "fixes": cmp["fixes"],
                  "p": cmp["mcnemar"]["p"], "verdict": cmp["verdict"], "baseline": baseline["id"]}

    con = m.get("consistency") or {}
    return {
        "run": run["id"], "label": run.get("label", ""), "n_cases": m["n_cases"],
        "repeat": run["config"]["repeat"], "complete": m["complete"],
        "cost": {"usd_per_task": m.get("cost_per_case_usd"),
                 "tokens_per_task": ((m["tokens"]["input"] + m["tokens"]["output"]) / max(1, sum(
                     len(c["runs"]) for c in run["cases"].values()))),
                 "p95_ms": m["latency_ms"]["p95"]},
        "reliability": {"pass_all": m["pass_all"], "k": m["pass_all_k"], "n": m["n_scored"],
                        "ci95": m["pass_all_ci95"], "flaky": len(m["flaky"]),
                        "same_steps": con.get("path_consistent")},
        "risk": {"guardrail_tests": len(live), "pass_all": (k / len(live)) if live else None,
                 "ci95": wilson(k, len(live)) if live else None,
                 "critical_violations": violations, "runs_checked": n_runs},
        "change": change,
    }


def _pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def render(name: str, sc: dict) -> str:
    c, r, k, ch = sc["cost"], sc["reliability"], sc["risk"], sc["change"]
    lines = [f"{name}  ·  {sc['n_cases']} tests × {sc['repeat']}{'' if sc['complete'] else '  ·  INCOMPLETE'}",
             f"  Cost         {('$%.4f' % c['usd_per_task']) if c['usd_per_task'] is not None else 'n/a'} per task"
             f"  ·  {c['tokens_per_task']:.0f} tokens  ·  p95 {c['p95_ms']:.0f}ms",
             f"  Reliability  {r['k']}/{r['n']} pass every run ({_pct(r['pass_all'])}, 95% CI "
             f"{_pct(r['ci95'][0])}–{_pct(r['ci95'][1])})  ·  {r['flaky']} flaky"
             + (f"  ·  same steps {_pct(r['same_steps'])}" if r["same_steps"] is not None else ""),
             f"  Risk         {k['guardrail_tests']} guardrail tests, {_pct(k['pass_all'])} pass every run"
             f"  ·  {k['critical_violations']} critical violations in {k['runs_checked']} runs",
             "  Change       " + (f"{ch['verdict']}  ·  {len(ch['regressions'])} regressed, {len(ch['fixes'])} fixed"
                                 if ch else "no baseline yet")]
    return "\n".join(lines)


def load_latest(directory: Path, label: str | None = None) -> dict | None:
    runs = sorted((directory / ".goldloop" / "runs").glob("*.json"))
    for p in reversed(runs):
        r = json.loads(p.read_text())
        if label is None or label in (r.get("label") or ""):
            return r
    return None
