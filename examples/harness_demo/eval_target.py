"""goldloop target for the harness demo: one case = one customer conversation.

Case input:
    customer, now, message, history (earlier messages, for memory),
    human: {"approve": [approval keys], "by": "..."}   (answers an approval pause)

Params (the knobs a run varies):
    model         rules | gullible | <provider:model>  primary planner/extractor
    fallback      rules | <provider:model> | none      second provider
    fault         none | stall_primary | quota_primary | all_down | bad_request
    deadline_s    stall deadline per model call
    idempotent    true | false     (false = the naive ledger, to watch it double-refund)
    crash_after_effects  N         simulate process death after the Nth side effect

A crash is followed by a restart: new graph, new SQLite connection to the
same checkpoint file, resume from the last finished node. The event log
stands in for a durable event sink and survives the restart.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.checkpoint.sqlite import SqliteSaver   # noqa: E402
from langgraph.types import Command                    # noqa: E402

from goldloop import Result, Unavailable               # noqa: E402
from harness.agent import Crash, Deps, _say, build     # noqa: E402
from harness.events import EventLog                    # noqa: E402
from harness.inference import (AllProvidersDown, Faulty, LangChainProvider,  # noqa: E402
                               Router, RulesProvider)
from harness.ledger import Ledger                      # noqa: E402
from harness import rules                              # noqa: E402

PRICING = {"gpt-4o-mini": {"input": 0.15, "output": 0.60}, "gpt-4o": {"input": 2.50, "output": 10.00},
           "claude-haiku-4-5": {"input": 1.0, "output": 5.0}}


def _provider(spec: str):
    if spec in ("rules", "careful"):
        return RulesProvider(rules.handlers(False), name="rules")
    if spec == "gullible":
        return RulesProvider(rules.handlers(True), name="rules-gullible", model="rules-gullible-v1")
    return LangChainProvider(spec)


def _router(params: dict, log: EventLog) -> Router:
    primary = _provider(params.get("model", "rules"))
    fb = params.get("fallback", "rules")
    secondary = None if fb in (None, "none") else _provider(fb)
    if secondary is not None and secondary.name == primary.name:
        secondary.name += "-2"
    fault = params.get("fault", "none")
    if fault in ("stall_primary", "all_down"):
        d = float(params.get("deadline_s", 1.0))
        primary = Faulty(primary, "stall", stall_s=d * 3, deadline_s=d)
    elif fault == "quota_primary":
        primary = Faulty(primary, "quota")
    elif fault == "billing_400_primary":
        primary = Faulty(primary, "billing_400")
    elif fault == "bad_request":
        primary = Faulty(primary, "bad_request")
    if fault == "all_down" and secondary is not None:
        secondary = Faulty(secondary, "quota")
    providers = [primary] + ([secondary] if secondary else [])
    # deadline_s in a scenario is the STALLING provider's deadline; healthy
    # providers keep a realistic one.
    return Router(providers, log, deadline_s=20.0, pricing=PRICING)


def _drive(graph, payload, config, nodes: list):
    final, interrupted = None, False
    for mode, chunk in graph.stream(payload, config, stream_mode=["updates", "values"]):
        if mode == "updates":
            for n in chunk:
                if n == "__interrupt__":
                    interrupted = True
                elif not n.startswith("__"):
                    nodes.append(n)
        else:
            final = chunk
    return final, interrupted


def run(inp: dict, params: dict) -> Result:
    # A case can pin scenario knobs (a fault, a crash point, a compromised
    # model): those are part of the test, not something the run varies.
    params = {**params, **(inp.get("scenario") or {})}
    log = EventLog()
    work = Path(tempfile.mkdtemp(prefix="harness-"))
    ledger = Ledger(work / "ledger.db", idempotent=bool(params.get("idempotent", True)))
    router = _router(params, log)
    run_id = uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": run_id}}
    state = {"run_id": run_id, "customer": inp["customer"], "now": inp.get("now", "2026-10-06"),
             "history": inp.get("history") or [], "message": inp["message"], "facts": []}
    crash_n = params.get("crash_after_effects")
    nodes: list[str] = []

    def graph_for(crash):
        conn = sqlite3.connect(work / "checkpoints.db", check_same_thread=False)
        return build(Deps(router, ledger, log, crash_after_effects=crash,
                          plan_prompt=params.get("plan_prompt", ""),
                          spotlight=bool(params.get("spotlight", False))), SqliteSaver(conn))

    graph, payload = graph_for(crash_n), state
    human = inp.get("human")
    restarts = 0
    try:
        while True:
            try:
                final, interrupted = _drive(graph, payload, config, nodes)
            except Crash:
                restarts += 1
                if restarts > 3:
                    raise
                log.emit("process.restart", attempt=restarts)
                graph, payload = graph_for(None), None          # new process, same checkpoint file
                continue
            if interrupted and human is not None:
                payload, human = Command(resume=human), None     # the person answers
                continue
            break
    except AllProvidersDown as e:
        raise Unavailable(str(e)) from e
    except Exception as e:  # noqa: BLE001
        # A non-infra failure (bad request, a bug): surface it as the outcome,
        # with the trace, so a case can check it failed CLEANLY (no failover,
        # no side effect) instead of the harness swallowing it.
        log.emit("run.error", **{"error.type": type(e).__name__, "message": str(e)[:200]})
        return Result(output={"status": "error", "error": type(e).__name__, "reply": "",
                              "ledger": {"refund_count": ledger.count("issue_refund")}},
                      trajectory=nodes, events=[{"name": x["name"], "attrs": x["attrs"]} for x in log.events])

    final = final or {}
    snap = graph.get_state(config).values
    final = {**snap, **final}
    decisions = final.get("decisions") or []
    served = [e["attrs"].get("gen_ai.provider.name") for e in log.events
              if e["name"] == "gen_ai.client.operation" and e["attrs"].get("outcome") == "ok"]
    out = {
        "reply": final.get("reply") or _paused_reply(decisions),
        "status": "awaiting_approval" if interrupted else "done",
        "plan": [a.get("tool") for a in final.get("plan") or []],
        "decisions": {d["key"]: d["verdict"] for d in decisions},
        "verdicts": sorted({d["verdict"] for d in decisions}),
        "pending": [d["key"] for d in decisions if d["verdict"] == "needs_approval"],
        "ledger": {"refund_count": ledger.count("issue_refund"),
                   "refunded": {o: ledger.refunded_total(o) for o in ("A1001", "A1002", "A1003", "A1004", "A1005")
                                if ledger.refunded_total(o)}},
        "memory": {a: _fact_view(final, inp, a) for a in ("address", "phone")},
        "served_by": sorted(set(served)),
        "restarts": restarts,
    }
    usage = {"input_tokens": sum(e["attrs"].get("gen_ai.usage.input_tokens", 0) for e in log.events),
             "output_tokens": sum(e["attrs"].get("gen_ai.usage.output_tokens", 0) for e in log.events)}
    by_model: dict = {}
    for e in log.events:
        a = e["attrs"]
        if e["name"] == "gen_ai.client.operation" and a.get("outcome") == "ok":
            m = by_model.setdefault(a["gen_ai.request.model"], {"input_tokens": 0, "output_tokens": 0})
            m["input_tokens"] += a.get("gen_ai.usage.input_tokens", 0)
            m["output_tokens"] += a.get("gen_ai.usage.output_tokens", 0)
    usage["by_model"] = by_model
    # The path that matters for consistency is what the agent DID: each tool it
    # chose and what the policy decided, in order (graph nodes never vary).
    actions = [f"{e['attrs']['tool']}:{e['attrs']['decision']}" for e in log.events
               if e["name"] == "policy.decision" and e["attrs"].get("actor") == "agent"]
    actions += ["ask"] if any(e["name"] == "agent.ask" for e in log.events) else []
    out["nodes"] = nodes
    return Result(output=out, usage=usage, trajectory=actions,
                  events=[{"name": e["name"], "attrs": e["attrs"]} for e in log.events])


def _paused_reply(decisions: list[dict]) -> str:
    """Paused for a person: tell the customer what's waiting and what won't happen."""
    lines = [f"{_say(d)} needs a team member's approval. I've asked for it."
             for d in decisions if d["verdict"] == "needs_approval"]
    lines += [f"I can't do that here ({_say(d)}): {d['reason']}." for d in decisions if d["verdict"] == "blocked"]
    return " ".join(lines)


def _fact_view(final: dict, inp: dict, attr: str) -> dict:
    facts = [f for f in final.get("facts") or [] if f["attribute"] == attr]
    from harness.memory import Memory
    cur = Memory(facts, EventLog()).get(inp["customer"], attr, inp.get("now", "2026-10-06"))
    return {"current": cur["value"] if cur else None, "stale": bool(cur and cur["stale"]),
            "source_ref": cur["source_ref"] if cur else None,
            "superseded": [f["value"] for f in facts if f["status"] == "superseded"],
            "disputed": [f["value"] for f in facts if f["status"] == "disputed"],
            "retracted": [f["value"] for f in facts if f["status"] == "retracted"]}
