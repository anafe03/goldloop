"""Side effects that happen exactly once, even when the agent crashes and resumes.

The failure this prevents: a checkpointed agent saves state *between* steps.
If the process dies after a tool moved money but before the step's checkpoint
was written, resuming re-runs that step, and the refund happens twice. Retries
cause the same thing. The checkpoint can't fix it (it never learned the refund
happened); the side effect has to be idempotent.

So every side effect carries an idempotency key derived from WHAT it is (run,
step, tool, arguments), not when it ran. A replay with the same key returns
the original result instead of acting again. Same idea as the Stripe
Idempotency-Key header, and as Meridian's double-pay fix.

SQLite with a UNIQUE key: the database, not the Python process, is the source
of truth, so this holds across process restarts and concurrent workers.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from .events import EventLog

SEED_ORDERS = {
    "A1001": {"customer": "dana@example.com", "item": "Trail runner shoes", "total": 129.00, "status": "delivered"},
    "A1002": {"customer": "lee@example.com", "item": "Rain shell", "total": 89.50, "status": "in transit"},
    "A1003": {"customer": "sam@example.com", "item": "Tent, 2-person", "total": 349.00, "status": "delivered"},
    "A1004": {"customer": "dana@example.com", "item": "Headlamp", "total": 39.00, "status": "delivered"},
    "A1005": {"customer": "sam@example.com", "item": "Expedition pack", "total": 640.00, "status": "delivered"},
}


def idem_key(run_id: str, step: str, tool: str, args: dict) -> str:
    blob = json.dumps({"run": run_id, "step": step, "tool": tool, "args": args}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class Ledger:
    def __init__(self, path: str | Path = ":memory:", idempotent: bool = True):
        self.path, self.idempotent = str(path), idempotent
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS effects (
            id INTEGER PRIMARY KEY, idem TEXT, tool TEXT, args TEXT, result TEXT, at REAL)""")
        # The UNIQUE index is the whole trick. Without it (idempotent=False) the
        # table records every attempt, which is what a naive tool does.
        if idempotent:
            self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS effects_idem ON effects(idem)")

    def apply(self, tool: str, args: dict, key: str, log: EventLog) -> dict:
        with self._lock:
            if self.idempotent:
                row = self.db.execute("SELECT result FROM effects WHERE idem=?", (key,)).fetchone()
                if row:
                    log.emit("side_effect.replayed", tool=tool, idempotency_key=key)
                    return json.loads(row[0])
            result = self._perform(tool, args)
            self.db.execute("INSERT INTO effects (idem, tool, args, result, at) VALUES (?,?,?,?,?)",
                            (key, tool, json.dumps(args, sort_keys=True), json.dumps(result), time.time()))
        log.emit("side_effect.commit", tool=tool, idempotency_key=key, **{"gen_ai.tool.name": tool})
        return result

    def _perform(self, tool: str, args: dict) -> dict:
        if tool == "issue_refund":
            return {"refund_id": f"rf_{args['order_id']}_{int(time.time() * 1000) % 100000}",
                    "order_id": args["order_id"], "amount": float(args["amount"])}
        if tool == "send_email":
            return {"queued_to": args.get("to")}
        raise ValueError(f"no side effect for {tool}")

    def count(self, tool: str, **where) -> int:
        rows = self.db.execute("SELECT args FROM effects WHERE tool=?", (tool,)).fetchall()
        return sum(1 for (a,) in rows if all(json.loads(a).get(k) == v for k, v in where.items()))

    def refunded_total(self, order_id: str) -> float:
        rows = self.db.execute("SELECT args FROM effects WHERE tool='issue_refund'").fetchall()
        return sum(float(json.loads(a)["amount"]) for (a,) in rows if json.loads(a)["order_id"] == order_id)
