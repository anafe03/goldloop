"""Climb ticket intake on goldloop: a different domain, the same harness.

Reads a climb-intake checkout at CLIMB_ROOT (default ../../../climb-intake)
and never writes to it: the pipeline runs with persist=False (no audit DB
writes) and bytecode writing is off. Cases come from its gold JSONL files,
translated, not copied.

Climb's gold set marks fields a reasonable reader could disagree on
(`ambiguous`). Those map to goldloop checks honestly:
    category ambiguous  -> no category check (any answer is acceptable)
    urgency ambiguous   -> one_of: the gold level or one step either side
    escalate ambiguous  -> no escalation check
A missed escalation (security, legal, exec) fails the case outright; a false
escalation costs score but isn't critical. That asymmetry is Climb's own.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("CLIMB_ROOT") or Path(__file__).resolve().parents[3] / "climb-intake").resolve()
if not (ROOT / "app" / "pipeline.py").is_file():
    raise SystemExit(f"climb-intake checkout not found at {ROOT} (set CLIMB_ROOT)")
sys.path.insert(0, str(ROOT))

LEVELS = ["low", "medium", "high", "critical"]
FILES = ["gold.jsonl", "adversarial.jsonl", "extraction_cases.jsonl"]


def to_goldloop(row: dict) -> dict:
    exp, amb = row["expected"], set(row.get("ambiguous", []))
    checks = []
    if "escalate" not in amb:
        must = bool(exp["escalate"])
        checks.append({"type": "equals", "path": "escalate", "value": must, "critical": must,
                       "name": "missed escalation" if must else "false escalation"})
    for r in exp.get("reasons", []):
        checks.append({"type": "contains", "path": "escalation_reasons", "value": r})
    if "category" not in amb:
        checks.append({"type": "equals", "path": "category", "value": exp["category"]})
    if "urgency" in amb:
        i = LEVELS.index(exp["urgency"])
        checks.append({"type": "one_of", "path": "urgency", "values": LEVELS[max(0, i - 1): i + 2]})
    else:
        checks.append({"type": "equals", "path": "urgency", "value": exp["urgency"]})
    if exp.get("customer_name") is None:
        checks.append({"type": "absent", "path": "customer.name"})       # don't invent a sender
    else:
        checks.append({"type": "contains", "path": "customer.name", "value": exp["customer_name"]})
    if exp.get("identifier_contains"):
        checks.append({"type": "contains", "path": "customer.identifiers", "value": exp["identifier_contains"]})
    tags = [row.get("group", "gold")] + (["must-escalate"] if exp.get("escalate") else [])
    return {"id": row["id"], "input": {"text": row["text"]}, "checks": checks, "tags": tags,
            "source": "adversarial" if row.get("group") == "adversarial" else "authored",
            "note": row.get("note") or row.get("attack") or row.get("tests") or ""}


def cases() -> list[dict]:
    out = []
    for f in FILES:
        for line in (ROOT / "data" / f).read_text().splitlines():
            if line.strip():
                out.append(to_goldloop(json.loads(line)))
    return out


def run(inp: dict, params: dict) -> dict:
    os.environ["CLASSIFIER_MODE"] = str(params.get("mode", "rules"))
    from app.models import TicketIn
    from app.pipeline import process
    d = process(TicketIn(text=inp["text"]), persist=False)
    x = json.loads(d.extraction.model_dump_json())
    x["queue"] = d.queue
    x["escalation_reasons"] = [str(r) for r in x.get("escalation_reasons") or []]
    return x
