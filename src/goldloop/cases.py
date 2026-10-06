"""Cases: an input, the checks its output must pass, and where the case came from.

A suite is a directory (or file) of YAML / JSON / JSONL. Cases are plain data
on purpose: a reviewer reads a promoted case in a diff, not in a database.

Two shorthands, because most extraction cases are "these fields must say X,
these must stay empty":

    expect: {updates.doors: "8"}     -> contains checks
    forbid: [updates.catering]       -> absent checks (negative controls)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SOURCES = ("authored", "promoted", "regression", "adversarial")


@dataclass
class Case:
    id: str
    input: Any
    checks: list[dict]
    tags: list[str] = field(default_factory=list)
    source: str = "authored"
    note: str = ""
    file: str = ""

    def fingerprint(self) -> str:
        blob = json.dumps({"input": self.input, "checks": self.checks}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


class CaseError(ValueError):
    pass


def _expand(raw: dict, where: str) -> Case:
    if "id" not in raw:
        raise CaseError(f"{where}: case has no id")
    if "input" not in raw:
        raise CaseError(f"{where}: case {raw['id']!r} has no input")
    checks = [dict(c) for c in raw.get("checks") or []]
    for path, value in (raw.get("expect") or {}).items():
        checks.append({"type": "contains", "path": path, "value": value})
    for path in raw.get("forbid") or []:
        checks.append({"type": "absent", "path": path})
    if not checks:
        # A case with no checks always passes, which is worse than no case.
        raise CaseError(f"{where}: case {raw['id']!r} has no checks")
    for c in checks:
        if "type" not in c:
            raise CaseError(f"{where}: case {raw['id']!r} has a check with no type: {c}")
    source = raw.get("source", "authored")
    if source not in SOURCES:
        raise CaseError(f"{where}: case {raw['id']!r} has unknown source {source!r}")
    return Case(id=str(raw["id"]), input=raw["input"], checks=checks,
                tags=list(raw.get("tags") or []), source=source,
                note=str(raw.get("note", "")), file=where)


def _read(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if data is None:
        return []
    if isinstance(data, dict):
        data = data.get("cases", [data])
    return list(data)


def load_suite(paths: str | Path | list) -> list[Case]:
    """Paths (dirs or files), or 'module:fn' returning raw case dicts. The second
    form lets a repo whose golden set already lives in code adopt goldloop
    without exporting a copy that can drift."""
    if not isinstance(paths, list):
        paths = [paths]
    files: list[Path] = []
    raws: list[tuple[dict, str]] = []
    for item in paths:
        p = Path(item)
        if isinstance(item, str) and ":" in item and not p.exists():
            from .util import load_ref
            raws += [(r, item) for r in load_ref(item)()]
        elif p.is_dir():
            files += sorted(f for f in p.rglob("*")
                            if f.suffix in (".yaml", ".yml", ".json", ".jsonl"))
        elif p.exists():
            files.append(p)
        else:
            raise CaseError(f"suite path not found: {p}")
    raws += [(raw, str(f)) for f in files for raw in _read(f)]
    cases, seen = [], {}
    for raw, where in raws:
        c = _expand(raw, where)
        if c.id in seen:
            raise CaseError(f"duplicate case id {c.id!r} in {where} and {seen[c.id]}")
        seen[c.id] = where
        cases.append(c)
    if not cases:
        raise CaseError(f"no cases found in {paths}")
    return cases


def suite_fingerprint(cases: list[Case]) -> str:
    h = hashlib.sha256()
    for c in sorted(cases, key=lambda c: c.id):
        h.update(f"{c.id}:{c.fingerprint()}\n".encode())
    return h.hexdigest()[:12]


def split(cases: list[Case], holdout: float) -> tuple[list[Case], list[Case]]:
    """Deterministic train/holdout split by case id, so the same case lands on
    the same side on every machine and every run."""
    if holdout <= 0:
        return cases, []
    train, held = [], []
    for c in cases:
        bucket = int(hashlib.sha256(c.id.encode()).hexdigest(), 16) % 1000 / 1000
        (held if bucket < holdout else train).append(c)
    return train, held


NEGATIVE = {"absent", "not_contains", "event_absent", "not_visited"}


def lint(cases: list[Case]) -> list[str]:
    """Cases that can pass for the wrong reason.

    Only-negative checks ("no refund happened", "field is empty") pass when
    the system is broken and does nothing at all. Every such case needs one
    positive check that proves the run actually happened."""
    out = []
    for c in cases:
        if all(ch["type"] in NEGATIVE for ch in c.checks):
            out.append(f"{c.id}: only negative checks; passes vacuously if the target does nothing")
    seen: dict = {}
    for c in cases:
        k = json.dumps(c.input, sort_keys=True, default=str)
        if k in seen:
            out.append(f"{c.id}: same input as {seen[k]}")
        seen.setdefault(k, c.id)
    return out
