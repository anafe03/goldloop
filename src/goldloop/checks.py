"""Checks: small scorers that each look at one thing in an observation.

Every check returns (score in [0, 1], detail). A case's score is the weighted
mean of its checks, except that a failed `critical: true` check zeroes the
whole case. That models the difference between "missed one field" and "routed
a cancellation as a normal reply": the second is not 80% correct.

Add your own with @register("name") or `type: custom, ref: module:fn`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .util import load_ref

_MISSING = object()


@dataclass
class Observation:
    """What the target produced on one run, plus what the harness saw around it."""
    output: Any
    trajectory: list[str] = field(default_factory=list)   # node names, in order (LangGraph)
    usage: dict = field(default_factory=dict)             # input_tokens, output_tokens, model
    latency_ms: float = 0.0
    events: list[dict] = field(default_factory=list)      # structured trace: {"name", "attrs", ...}


def resolve(obj: Any, path: str | None) -> Any:
    """Dotted path into dicts, lists and attributes. `updates.doors`, `messages.-1.content`."""
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if cur is None:
            return _MISSING
        if isinstance(cur, dict):
            if part not in cur:
                return _MISSING
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and re.fullmatch(r"-?\d+", part):
            i = int(part)
            if not -len(cur) <= i < len(cur):
                return _MISSING
            cur = cur[i]
        elif hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            return _MISSING
    return cur


def _text(v: Any) -> str:
    if v is _MISSING or v is None:
        return ""
    if hasattr(v, "content") and isinstance(getattr(v, "content"), str):
        return v.content          # a LangChain message: grade what it says
    return str(v)


def _empty(v: Any) -> bool:
    if v is _MISSING or v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict, tuple, set)):
        return len(v) == 0
    return False


def _norm(s: str, check: dict) -> str:
    return s if check.get("case_sensitive") else s.lower().strip()


REGISTRY: dict[str, Callable[[Observation, dict], tuple[float, str]]] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def _show(v: Any) -> str:
    if v is _MISSING:
        return "<missing>"
    s = _text(v)
    return repr(s if len(s) <= 80 else s[:77] + "...")


@register("contains")
def _contains(obs, c):
    v = resolve(obs.output, c.get("path"))
    hay = _norm(_text(v), c)
    needles = c["value"] if isinstance(c["value"], list) else [c["value"]]
    hits = [n for n in needles if _norm(str(n), c) in hay]
    return len(hits) / len(needles), f"{c.get('path') or 'output'}={_show(v)} needs {needles}"


@register("not_contains")
def _not_contains(obs, c):
    v = resolve(obs.output, c.get("path"))
    hay = _norm(_text(v), c)
    bad = c["value"] if isinstance(c["value"], list) else [c["value"]]
    leaked = [n for n in bad if _norm(str(n), c) in hay]
    return (0.0 if leaked else 1.0), (f"leaked {leaked}" if leaked else "clean")


@register("equals")
def _equals(obs, c):
    v = resolve(obs.output, c.get("path"))
    if v is _MISSING:
        return 0.0, f"{c.get('path')} missing"
    want = c["value"]
    if isinstance(want, str):
        ok = _norm(_text(v), c) == _norm(want, c)
    else:
        ok = v == want
    return float(ok), f"{c.get('path') or 'output'}={_show(v)} want {want!r}"


@register("one_of")
def _one_of(obs, c):
    """For genuinely ambiguous fields: any listed answer is right."""
    v = resolve(obs.output, c.get("path"))
    got = _norm(_text(v), c) if v is not _MISSING and v is not None else None
    allowed = [(_norm(str(a), c) if a is not None else None) for a in c["values"]]
    return float(got in allowed), f"{c.get('path')}={_show(v)} allowed {c['values']}"


@register("regex")
def _regex(obs, c):
    v = resolve(obs.output, c.get("path"))
    flags = 0 if c.get("case_sensitive") else re.IGNORECASE
    ok = re.search(c["pattern"], _text(v), flags) is not None
    return float(ok), f"{c.get('path') or 'output'}={_show(v)} ~ /{c['pattern']}/"


@register("absent")
def _absent(obs, c):
    """Negative control: the target must NOT answer this. An invented answer
    costs score; without these checks a model that fills every field wins."""
    v = resolve(obs.output, c.get("path"))
    return float(_empty(v)), f"{c.get('path')}={_show(v)} must be empty"


@register("present")
def _present(obs, c):
    v = resolve(obs.output, c.get("path"))
    return float(not _empty(v)), f"{c.get('path')}={_show(v)} must be set"


@register("number")
def _number(obs, c):
    v = resolve(obs.output, c.get("path"))
    try:
        x = float(re.sub(r"[^\d.\-]", "", _text(v)) if not isinstance(v, (int, float)) else v)
    except ValueError:
        return 0.0, f"{c.get('path')}={_show(v)} is not a number"
    ok = True
    if "value" in c:
        ok &= abs(x - float(c["value"])) <= float(c.get("tolerance", 0))
    if "min" in c:
        ok &= x >= float(c["min"])
    if "max" in c:
        ok &= x <= float(c["max"])
    return float(ok), f"{c.get('path')}={x}"


@register("length")
def _length(obs, c):
    v = resolve(obs.output, c.get("path"))
    n = 0 if _empty(v) else len(v if not hasattr(v, "content") else v.content)
    ok = c.get("min", 0) <= n <= c.get("max", float("inf"))
    return float(ok), f"len({c.get('path') or 'output'})={n}"


# ── Trajectory: what the agent did, not only what it said ───────────────────

@register("visited")
def _visited(obs, c):
    return float(c["node"] in obs.trajectory), f"path {obs.trajectory}"


@register("not_visited")
def _not_visited(obs, c):
    """'Never next to a payment tool' as a test, not a hope."""
    return float(c["node"] not in obs.trajectory), f"path {obs.trajectory}"


@register("sequence")
def _sequence(obs, c):
    """The nodes appear in this order (other nodes may sit between them)."""
    it = iter(obs.trajectory)
    ok = all(any(step == n for step in it) for n in c["nodes"])
    return float(ok), f"path {obs.trajectory} needs order {c['nodes']}"


# ── Events: grade the trace, not only the answer ────────────────────────────
# An event is {"name": str, "attrs": {...}}. `match` keys are matched against
# attrs (or "name"); string values match case-insensitively by equality.

def _match(ev: dict, name: str | None, match: dict | None) -> bool:
    if name and ev.get("name") != name:
        return False
    attrs = ev.get("attrs") or {}
    for k, want in (match or {}).items():
        got = attrs.get(k, _MISSING)
        if isinstance(want, str) and isinstance(got, str):
            if got.lower() != want.lower():
                return False
        elif got != want:
            return False
    return True


def _events(obs, c):
    return [e for e in obs.events if _match(e, c.get("name"), c.get("match"))]


@register("event_present")
def _event_present(obs, c):
    n = len(_events(obs, c))
    return float(n > 0), f"{n} event(s) {c.get('name')} {c.get('match') or ''}"


@register("event_absent")
def _event_absent(obs, c):
    """E.g. no side_effect.commit for issue_refund without an approval event."""
    hits = _events(obs, c)
    return float(not hits), f"{len(hits)} event(s) {c.get('name')} {c.get('match') or ''} (want 0)"


@register("event_count")
def _event_count(obs, c):
    n = len(_events(obs, c))
    ok = c.get("min", 0) <= n <= c.get("max", float("inf"))
    if "value" in c:
        ok = n == c["value"]
    return float(ok), f"{n} event(s) {c.get('name')} {c.get('match') or ''}"


@register("event_order")
def _event_order(obs, c):
    """Each step (a {name, match} dict) occurs after the previous one."""
    it = iter(obs.events)
    ok = all(any(_match(e, step.get("name"), step.get("match")) for e in it) for step in c["steps"])
    return float(ok), f"order {[s.get('name') for s in c['steps']]}"


# ── Budgets ─────────────────────────────────────────────────────────────────

@register("max_latency_ms")
def _max_latency(obs, c):
    return float(obs.latency_ms <= c["value"]), f"{obs.latency_ms:.0f}ms (max {c['value']})"


@register("max_tokens")
def _max_tokens(obs, c):
    t = obs.usage.get("input_tokens", 0) + obs.usage.get("output_tokens", 0)
    return float(t <= c["value"]), f"{t} tokens (max {c['value']})"


@register("custom")
def _custom(obs, c):
    fn = load_ref(c["ref"])
    args = {k: v for k, v in c.items() if k not in ("type", "ref", "weight", "critical", "name")}
    r = fn(obs, **args)
    if isinstance(r, tuple):
        return float(r[0]), str(r[1])
    return float(r), c["ref"]


@register("judge")
def _judge(obs, c):
    from .judge import judge_check
    return judge_check(obs, c)


def run_check(obs: Observation, check: dict) -> dict:
    fn = REGISTRY.get(check["type"])
    if fn is None:
        raise KeyError(f"unknown check type {check['type']!r} (known: {sorted(REGISTRY)})")
    score, detail = fn(obs, check)
    return {"type": check["type"], "name": check.get("name") or check.get("path") or check.get("node") or "",
            "score": max(0.0, min(1.0, float(score))), "detail": detail,
            "critical": bool(check.get("critical")), "weight": float(check.get("weight", 1.0))}


def score_case(results: list[dict]) -> float:
    if any(r["critical"] and r["score"] < 1.0 for r in results):
        return 0.0
    graded = [r for r in results if not r["critical"]] or results
    total = sum(r["weight"] for r in graded)
    return sum(r["score"] * r["weight"] for r in graded) / total if total else 1.0
