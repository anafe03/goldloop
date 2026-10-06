#!/usr/bin/env python3
"""Are the evals any good? Break the harness on purpose and see if they notice.

A suite that passes is only evidence if it would FAIL when the thing it
guards breaks. Each mutation below removes one guarantee from the harness;
the suite must turn red for every one of them, and the cases that catch it
should be the ones written for that guarantee. A mutation that survives
means a missing case, not a working harness.

    python mutation_check.py
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_target                                   # noqa: E402
from goldloop.cases import load_suite                # noqa: E402
from goldloop.runner import run_suite                # noqa: E402
from goldloop.targets import make_target             # noqa: E402
from harness import agent, inference, memory, policy  # noqa: E402

CASES = load_suite(HERE / "cases")


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def refunds_allowed():
    """Refunds no longer need a person."""
    orig = policy.Policy.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.tiers["issue_refund"] = policy.ALLOW
    return patched(policy.Policy, "__init__", init)


def no_ownership_check():
    orig = policy.decide
    return patched(agent, "decide", lambda *a, **k: orig(*a, **{**k, "owner_ok": True}))


def no_refund_limit():
    orig = policy.Policy.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.max_refund = 1e9
    return patched(policy.Policy, "__init__", init)


@contextlib.contextmanager
def approval_any_amount():
    """Approval keyed by order only: 'approve a refund on A1001' covers any amount."""
    orig_key, orig_run = policy.approval_key, eval_target.run

    def key(tool, args):
        k = orig_key(tool, args)
        return k.rsplit(":", 1)[0] if tool == "issue_refund" else k

    def run(inp, params):
        h = inp.get("human")
        if h:
            h = {**h, "approve": [k.rsplit(":", 1)[0] if k.startswith("issue_refund:") else k
                                  for k in h.get("approve", [])]}
        return orig_run({**inp, "human": h} if h else inp, params)
    # decide() looks approval_key up at call time, so patching the module is enough.
    with patched(policy, "approval_key", key), patched(eval_target, "run", run):
        yield


def latest_write_wins():
    """Memory without authority: the newest statement always wins (hearsay included)."""
    return patched(memory, "AUTHORITY", {k: 1 for k in memory.AUTHORITY})


def no_expiry():
    orig = memory.Memory.write

    def write(self, *a, **k):
        k["valid_until"] = None
        return orig(self, *a, **k)
    return patched(memory.Memory, "write", write)


def retract_is_noop():
    return patched(memory.Memory, "retract", lambda self, *a, **k: 0)


def no_staleness():
    return patched(memory, "SHELF_LIFE_DAYS", {})


def failover_on_everything():
    return patched(inference, "is_infra", lambda e: True)


def no_circuit_breaker():
    return patched(inference.Router, "__post_init__", lambda self: setattr(self, "cooldown_s", 0.0),
                   ) if hasattr(inference.Router, "__post_init__") else _cooldown_zero()


def _cooldown_zero():
    orig = eval_target._router

    def r(params, log):
        rt = orig(params, log)
        rt.cooldown_s = 0.0
        return rt
    return patched(eval_target, "_router", r)


def status_code_only_classifier():
    """The classifier before 2026-10-06: trusts the status code, ignores the message."""
    def is_infra(e):
        st = getattr(e, "status_code", None)
        return st in (408, 429) or (isinstance(st, int) and st >= 500) or \
            any(inference._INFRA.search(c.__name__) for c in type(e).__mro__)
    return patched(inference, "is_infra", is_infra)


def guess_instead_of_asking():
    """The agent never asks: a missing order or amount gets filled with a guess."""
    import json as _json
    from harness import rules
    from harness.ledger import SEED_ORDERS
    orig = rules.plan_factory

    def factory(gullible):
        plan = orig(gullible)

        def guessing(user):
            out = _json.loads(plan(user))
            if out.get("ask") and not out.get("actions"):
                cust = _json.loads(user)["customer"]
                oid, o = next((k, v) for k, v in SEED_ORDERS.items() if v["customer"] == cust)
                out = {"intent": "guess", "ask": "",
                       "actions": [{"tool": "issue_refund", "args": {"order_id": oid, "amount": o["total"]}}]}
            return _json.dumps(out)
        return guessing
    return patched(rules, "plan_factory", factory)


def history_without_dates():
    """The record keeps only current status: 'what did we believe on March 18'
    gets answered with today's facts."""
    def as_of(self, subject, attribute, at):
        live = [f for f in self.facts if f["subject"] == subject and f["attribute"] == attribute
                and f["status"] == "active"]
        return max(live, key=lambda f: f["observed_at"]) if live else None
    return patched(memory.Memory, "as_of", as_of)


def naive_ledger():
    return contextlib.nullcontext({"idempotent": False})


MUTATIONS = [
    ("refunds run without approval", refunds_allowed, "trust"),
    ("no ownership check", no_ownership_check, "trust"),
    ("no refund limit", no_refund_limit, "trust"),
    ("approval not scoped to amount", approval_any_amount, "trust"),
    ("side effects not idempotent", naive_ledger, "durability"),
    ("newest memory write always wins", latest_write_wins, "memory"),
    ("temporary facts never expire", no_expiry, "memory"),
    ("'forget that' is ignored", retract_is_noop, "memory"),
    ("no staleness check", no_staleness, "memory"),
    ("fail over on every error", failover_on_everything, "routing"),
    ("no circuit breaker", _cooldown_zero, "routing"),
    ("billing 400 read as a bad request", status_code_only_classifier, "routing"),
    ("guess instead of asking", guess_instead_of_asking, "restraint"),
    ("history forgets when facts changed", history_without_dates, "memory-over-time"),
]


def run(params=None):
    t = make_target({"kind": "python", "ref": "eval_target:run",
                     "params": {"model": "rules", "fallback": "rules", "idempotent": True, **(params or {})}})
    r = run_suite(CASES, t, threshold=1.0, concurrency=4)
    return sorted(cid for cid, c in r["cases"].items() if not c["summary"]["passed_all"]), r


def main():
    failing, _ = run()
    print(f"baseline: {len(CASES) - len(failing)}/{len(CASES)} pass")
    if failing:
        sys.exit(f"baseline must be green first: {failing}")
    survived = []
    for name, mut, area in MUTATIONS:
        with mut() as extra:
            failing, _ = run(extra if isinstance(extra, dict) else None)
        tags = {t for c in CASES if c.id in failing for t in c.tags}
        mark = "caught" if failing else "SURVIVED"
        print(f"  {mark:8} {name:34} by {len(failing)} case(s){'' if area in tags or not failing else '  (not by the ' + area + ' cases!)'}"
              + (f": {', '.join(failing[:3])}{'...' if len(failing) > 3 else ''}" if failing else ""))
        if not failing:
            survived.append(name)
    print(f"\n{len(MUTATIONS) - len(survived)}/{len(MUTATIONS)} mutations caught")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
