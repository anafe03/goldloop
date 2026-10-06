"""Close the loop: production feedback -> candidate cases -> reviewed suite.

Collection and shaping are automatic. Promotion is a reviewed commit: the gate
never ingests a label nobody looked at, because one wrong label teaches the
tuner the wrong lesson and then guards it forever.

Feedback record (JSONL, one per line), whatever your product stores:

    {"input": {...},                     # what the agent was given
     "output": {...},                    # what it produced
     "corrections": {"updates.doors": "6:30", "updates.parking": null},
     "verdict": "edited" | "rejected" | "wrong",
     "note": "venue corrected itself mid-thread", "at": "2026-08-03T..."}

A correction with a value becomes a `contains` check. A correction to null
("this should have been empty") becomes an `absent` check: a negative control
earned from a real hallucination.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml

from .cases import Case, _expand
from .util import load_ref, short_hash


def read_feedback(source: str) -> Iterable[dict]:
    """A .jsonl path, or 'module:fn' returning an iterable of records (e.g. a
    query against your observability store)."""
    if ":" in source and not Path(source).exists():
        yield from load_ref(source)()
        return
    for line in Path(source).read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def candidate_from(fb: dict) -> dict:
    checks, todo = [], []
    for path, value in (fb.get("corrections") or {}).items():
        if value is None or (isinstance(value, str) and not value.strip()):
            checks.append({"type": "absent", "path": path})
        else:
            checks.append({"type": "contains", "path": path, "value": value})
            if isinstance(value, str) and len(value) > 24:
                todo.append(f"shorten {path} to the part that must appear")
    if not checks:
        todo.append("no corrections recorded: write the checks by hand")
    was = json.dumps(fb.get("output"), default=str)
    return {
        "id": fb.get("id") or f"promoted-{short_hash(fb.get('input'))}",
        "source": "promoted",
        "review": "pending",
        "review_notes": todo,
        "note": "; ".join(x for x in [fb.get("note", ""), f"verdict={fb.get('verdict', '?')}",
                                       f"at={fb.get('at', '?')}"] if x),
        "was": was if len(was) <= 600 else was[:597] + "...",
        "input": fb.get("input"),
        "checks": checks,
    }


def promote(source: str, existing: list[Case], out: Path, limit: int | None = None) -> dict:
    seen = {short_hash(c.input) for c in existing}
    prior = yaml.safe_load(out.read_text()) if out.exists() else []
    prior = prior or []
    seen |= {short_hash(c.get("input")) for c in prior}
    new, dupes = [], 0
    for fb in read_feedback(source):
        if fb.get("input") is None:
            continue
        h = short_hash(fb["input"])
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        new.append(candidate_from(fb))
        if limit and len(new) >= limit:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(prior + new, sort_keys=False, allow_unicode=True, width=100))
    return {"new": len(new), "duplicates_skipped": dupes, "pending_total": len(prior) + len(new),
            "file": str(out)}


def accept(candidates: Path, suite_file: Path) -> dict:
    """Move candidates marked `review: ok` into the suite. Everything else stays."""
    items = yaml.safe_load(candidates.read_text()) or []
    keep, moved, rejected = [], [], 0
    for item in items:
        status = item.get("review")
        if status == "ok":
            clean = {k: v for k, v in item.items() if k not in ("review", "review_notes", "was")}
            _expand(clean, str(candidates))           # validate before it can gate anything
            moved.append(clean)
        elif status == "reject":
            rejected += 1
        else:
            keep.append(item)
    if moved:
        existing = yaml.safe_load(suite_file.read_text()) if suite_file.exists() else []
        suite_file.parent.mkdir(parents=True, exist_ok=True)
        suite_file.write_text(yaml.safe_dump((existing or []) + moved, sort_keys=False,
                                             allow_unicode=True, width=100))
    candidates.write_text(yaml.safe_dump(keep, sort_keys=False, allow_unicode=True, width=100))
    return {"accepted": len(moved), "rejected": rejected, "still_pending": len(keep),
            "suite_file": str(suite_file)}
