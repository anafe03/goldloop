"""Every agent action is a structured event you can query, export and audit.

Names and attribute keys follow the OpenTelemetry GenAI semantic conventions
where one exists (gen_ai.operation.name, gen_ai.provider.name,
gen_ai.request.model, gen_ai.usage.*, gen_ai.tool.name), so the same trace can
be shipped to any OTel backend later. Harness-specific events (policy
decisions, side effects, memory writes, failovers) use a small dotted
vocabulary of their own.

Events are append-only and carry a trace id (one per agent run) and a span id
(one per step), so "what did the agent do, and why" is a query, not a log grep.
"""
from __future__ import annotations

import contextlib
import json
import threading
import time
import uuid
from pathlib import Path


class EventLog:
    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.events: list[dict] = []
        self._lock = threading.Lock()
        self._stack: list[str] = []

    def emit(self, name: str, **attrs) -> dict:
        ev = {"ts": round(time.time(), 3), "trace_id": self.trace_id,
              "span_id": uuid.uuid4().hex[:8], "parent_id": self._stack[-1] if self._stack else None,
              "name": name, "attrs": attrs}
        with self._lock:
            self.events.append(ev)
        return ev

    @contextlib.contextmanager
    def span(self, name: str, **attrs):
        """A step with a duration. Children emitted inside it point at it."""
        ev = self.emit(name, **attrs)
        self._stack.append(ev["span_id"])
        t0 = time.perf_counter()
        try:
            yield ev
            ev["attrs"]["status"] = ev["attrs"].get("status", "ok")
        except BaseException as e:
            ev["attrs"]["status"] = "error"
            ev["attrs"]["error.type"] = type(e).__name__
            raise
        finally:
            ev["attrs"]["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self._stack.pop()

    def find(self, name: str | None = None, **match) -> list[dict]:
        return [e for e in self.events
                if (name is None or e["name"] == name)
                and all(e["attrs"].get(k) == v for k, v in match.items())]

    def export_jsonl(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            for e in self.events:
                f.write(json.dumps(e, default=str) + "\n")
        return p
