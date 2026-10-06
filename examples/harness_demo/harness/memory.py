"""Accountable memory: every fact knows where it came from, how sure we are, and whether it still holds.

The failure modes this is built against:
  stale facts        the customer moved; the agent keeps shipping to the old address
  confident fiction  something a third party said ("I think she moved to Denver")
                     gets stated back as fact
  zombie facts       the customer said "forget that" and it comes back next week
  temporary-as-permanent  "ship to my hotel until Friday" overwrites home forever

So a fact is not a string in a vector store. It is a record:
  value + provenance (source message id, quote, who said it, when)
        + authority (who is allowed to establish it) + confidence
        + lifecycle: active -> superseded | retracted | disputed, and an
          optional valid_until for things that are true only for a while.

Writes follow rules, and every write is an event:
  * same value again            -> reinforce (no duplicate row)
  * new value, authority >= old -> supersede the old one (kept, not deleted:
                                   "what did we believe on Sept 3?" stays answerable)
  * new value, authority <  old -> stored as DISPUTED, never served as truth
  * valid_until set             -> an overlay that expires; the permanent fact
                                   underneath is untouched and comes back after
  * retraction                  -> the active fact is retracted, and stays dead
Reads return the fact with its provenance and a `stale` flag when it is older
than the attribute's shelf life, so the agent can say "on file since March,
still right?" instead of asserting it.

Facts live in agent state (plain dicts), so they are checkpointed with it.
"""
from __future__ import annotations

import uuid
from datetime import date

from .events import EventLog

AUTHORITY = {"user": 3, "agent_confirmed": 3, "system": 2, "third_party": 1, "inferred": 0}
SHELF_LIFE_DAYS = {"address": 365, "phone": 730, "preference": 180}


def _d(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


class Memory:
    def __init__(self, facts: list[dict] | None, log: EventLog):
        self.facts = facts if facts is not None else []
        self.log = log

    def _active(self, subject: str, attribute: str, overlay: bool) -> list[dict]:
        return [f for f in self.facts if f["subject"] == subject and f["attribute"] == attribute
                and f["status"] == "active" and bool(f.get("valid_until")) == overlay]

    def write(self, subject: str, attribute: str, value: str, *, source: str, source_ref: str,
              quote: str, observed_at: str, confidence: float = 0.9,
              valid_until: str | None = None) -> dict:
        fact = {"id": uuid.uuid4().hex[:8], "subject": subject, "attribute": attribute,
                "value": value.strip(), "source": source, "source_ref": source_ref, "quote": quote,
                "authority": AUTHORITY.get(source, 0), "confidence": confidence,
                "observed_at": observed_at, "valid_until": valid_until, "status": "active",
                "superseded_by": None}
        olds = self._active(subject, attribute, overlay=bool(valid_until))
        same = next((o for o in olds if o["value"].lower() == fact["value"].lower()), None)
        if same:
            same["confidence"] = max(same["confidence"], confidence)
            same["last_confirmed"] = observed_at
            self.log.emit("memory.reinforce", attribute=attribute, fact_id=same["id"], source_ref=source_ref)
            return same
        stronger = [o for o in olds if o["authority"] > fact["authority"]]
        if stronger:
            fact["status"] = "disputed"
            self.facts.append(fact)
            self.log.emit("memory.conflict", attribute=attribute, kept=stronger[0]["id"], disputed=fact["id"],
                          reason=f"{source} can't override {stronger[0]['source']}", source_ref=source_ref)
            return fact
        for o in olds:
            o["status"], o["superseded_by"] = "superseded", fact["id"]
            o["superseded_at"] = observed_at
            self.log.emit("memory.supersede", attribute=attribute, old=o["id"], new=fact["id"], source_ref=source_ref)
        self.facts.append(fact)
        self.log.emit("memory.write", attribute=attribute, fact_id=fact["id"], source=source,
                      source_ref=source_ref, temporary=bool(valid_until))
        return fact

    def retract(self, subject: str, attribute: str, *, source: str, source_ref: str,
                at: str | None = None) -> int:
        n = 0
        for f in self.facts:
            if f["subject"] == subject and f["attribute"] == attribute and f["status"] == "active" \
                    and AUTHORITY.get(source, 0) >= f["authority"]:
                f["status"], f["retracted_by"] = "retracted", source_ref
                f["retracted_at"] = at
                n += 1
        self.log.emit("memory.retract", attribute=attribute, count=n, source_ref=source_ref)
        return n

    def as_of(self, subject: str, attribute: str, at: str) -> dict | None:
        """What we believed on date `at`, reconstructed from the record: facts
        observed by then, not yet superseded or retracted by then, and (for
        temporary ones) still in their window. This is the audit question,
        "what did the agent think on March 18, and why?", and it must agree with
        what the agent actually said on March 18."""
        day = _d(at)

        def held(f):
            if f["subject"] != subject or f["attribute"] != attribute or f["status"] == "disputed":
                return False
            if _d(f["observed_at"]) > day:
                return False
            if f.get("superseded_at") and _d(f["superseded_at"]) <= day:
                return False
            if f.get("retracted_at") and _d(f["retracted_at"]) <= day:
                return False
            return not f.get("valid_until") or _d(f["valid_until"]) >= day
        live = [f for f in self.facts if held(f)]
        if not live:
            return None
        return max(live, key=lambda f: (bool(f.get("valid_until")), f["authority"], f["observed_at"]))

    def get(self, subject: str, attribute: str, at: str) -> dict | None:
        """The fact to act on at date `at`: an unexpired overlay if any, else the
        permanent fact. With provenance and a staleness flag. None if nothing holds."""
        today = _d(at)
        live = [f for f in self.facts if f["subject"] == subject and f["attribute"] == attribute
                and f["status"] == "active" and _d(f["observed_at"]) <= today
                and (not f.get("valid_until") or _d(f["valid_until"]) >= today)]
        if not live:
            self.log.emit("memory.read", attribute=attribute, hit=False)
            return None
        best = max(live, key=lambda f: (bool(f.get("valid_until")), f["authority"], f["observed_at"]))
        seen = _d(best.get("last_confirmed") or best["observed_at"])
        stale = (today - seen).days > SHELF_LIFE_DAYS.get(attribute, 10_000)
        self.log.emit("memory.read", attribute=attribute, hit=True, fact_id=best["id"], stale=stale,
                      source_ref=best["source_ref"])
        return {**best, "stale": stale}
