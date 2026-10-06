"""An assistant that drafts replies for a small bakery and learns from the owner's corrections.

The business: Rosa runs a bakery. Customers message about pickups, wholesale,
dietary options, delivery and problems. The assistant drafts replies; Rosa
fixes the bad ones. Each fix can become something the assistant knows:

  memory   a fact       "Saturday pickups end at 2pm"
  skill    a procedure  "On a quality complaint: apologize, set aside a replacement"

Knowledge items carry where they came from (the correction), when, and what
triggers them, so a bad one can be traced and removed.

Two guardrails live in code, not in what was learned:
  * never share another customer's order
  * never offer more than 10% off; anything bigger is cut to 10% and logged
A learned item that keeps hitting a guardrail is harmful even when the reply
that goes out is safe, and the learning loop rejects it.

Two "models" sit behind the same interface: a deterministic stand-in (free,
offline, what CI runs) and any LangChain chat model (real runs).
"""
from __future__ import annotations

import json
import re

HOURS = "We're open Tuesday to Sunday, 7am to 3pm, and closed Mondays."
STOP = set("""about after again also anyway before being could from have here just know like
make more much need only other over really should some than thanks that their them then there
these they this those today tomorrow very want what when where which while will with would your
yours hello please order""".split())

OTHER_PERSON = re.compile(r"\b(neighbou?r|friend|someone else|sam|other customer)\b.*\b(order|ordered|buy|bought)\b"
                          r"|\bwhat did \w+ order\b", re.I)
DISCOUNT = re.compile(r"(\d{1,2})\s?% off", re.I)
MAX_DISCOUNT = 10


def guard(reply: str, log: list) -> str:
    """Code-level guardrail: cap discounts at 10%, whatever was learned."""
    def cap(m):
        pct = int(m.group(1))
        if pct > MAX_DISCOUNT:
            log.append({"name": "guardrail.blocked", "attrs": {"rule": "max_discount", "asked": pct}})
            return f"{MAX_DISCOUNT}% off"
        return m.group(0)
    return DISCOUNT.sub(cap, reply)


def fires(item: dict, message: str) -> bool:
    words = set(re.findall(r"[a-z]+", message.lower()))
    return bool(words & set(item["triggers"]))


# ── the stand-in model ──────────────────────────────────────────────────────

def rules_reply(message: str, knowledge: list[dict], log: list) -> dict:
    m = message.lower()
    if OTHER_PERSON.search(message):
        log.append({"name": "guardrail.blocked", "attrs": {"rule": "other_customer"}})
        return {"reply": "Sorry, I can't share other customers' orders.", "used": []}
    used = [k for k in knowledge if fires(k, message)]
    parts = [k["says"] for k in used]
    if "monday" in m and re.search(r"\b(order|pick|cake|bread)\b", m):
        parts.insert(0, "We're closed Mondays. Could Tuesday work instead?")
    if re.search(r"\bopen\b|\bhours\b", m):
        parts.insert(0, HOURS)
    if not parts:
        if re.search(r"\bthanks?\b|\bthank you\b", m):
            parts = ["You're welcome, see you soon!"]
        else:
            parts = ["Thanks for your message! Rosa will get back to you shortly."]
    return {"reply": " ".join(parts), "used": [k["id"] for k in used]}


def rules_learn(correction: dict) -> dict:
    """Turn one correction into a knowledge item, the naive way a learner
    generalizes: the fix becomes what to say, the customer's words become
    the trigger. That's exactly how one-off fixes turn into bad rules."""
    words = [w for w in re.findall(r"[a-z]+", correction["message"].lower()) if len(w) >= 5 and w not in STOP]
    kind = "skill" if re.search(r"\b(always|whenever|when|on a|if)\b", correction.get("owner_note", ""), re.I) else "memory"
    return {"id": f"k-{correction['id']}", "kind": kind, "says": correction["fixed"],
            "triggers": sorted(set(words)), "source": correction["id"], "learned_at": correction["at"],
            "owner_note": correction.get("owner_note", "")}


# ── a real model behind the same interface ──────────────────────────────────

REPLY_SYS = """You draft short, friendly replies for Rosa's Bakery. Bakery hours: Tuesday to Sunday,
7am to 3pm, closed Mondays. Use the KNOWLEDGE below when it applies; it was learned from Rosa's own
corrections. Never share another customer's order. Return ONLY JSON:
{"reply": "<the reply>", "used": ["<knowledge ids you relied on>"]}"""

LEARN_SYS = """Rosa corrected a reply her assistant drafted. Turn the correction into ONE reusable
knowledge item, as general as it is safe to be and no more. A one-off situation should stay narrow.
Return ONLY JSON: {"kind": "memory" | "skill", "says": "<what to say or do>", "triggers": ["<lowercase words that signal it applies>"]}"""


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}


def llm_reply(model, message: str, knowledge: list[dict], log: list) -> dict:
    if OTHER_PERSON.search(message):
        log.append({"name": "guardrail.blocked", "attrs": {"rule": "other_customer"}})
        return {"reply": "Sorry, I can't share other customers' orders.", "used": []}
    kn = "\n".join(f"[{k['id']}] ({k['kind']}) {k['says']}" for k in knowledge) or "(none yet)"
    out = model.invoke([("system", REPLY_SYS + "\n\nKNOWLEDGE:\n" + kn), ("human", message)])
    data = _json(out.content if isinstance(out.content, str) else str(out.content))
    return {"reply": str(data.get("reply", "")), "used": list(data.get("used") or [])}


def llm_learn(model, correction: dict) -> dict:
    out = model.invoke([("system", LEARN_SYS), ("human", json.dumps(
        {k: correction[k] for k in ("message", "draft", "fixed", "owner_note") if k in correction}))])
    d = _json(out.content if isinstance(out.content, str) else str(out.content))
    return {"id": f"k-{correction['id']}", "kind": d.get("kind", "memory"), "says": d.get("says", correction["fixed"]),
            "triggers": [t.lower() for t in d.get("triggers") or []], "source": correction["id"],
            "learned_at": correction["at"], "owner_note": correction.get("owner_note", "")}
