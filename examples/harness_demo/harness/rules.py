"""Deterministic stand-in models for the two model tasks (extract, plan).

`careful` reads only the customer's own words. `gullible` also obeys
instructions inside quoted or forwarded text, the way an injected model
would. The trust evals run against `gullible` on purpose: the point is that
the harness holds even when the model has been talked into the wrong thing.
"""
from __future__ import annotations

import json
import re

ORDER = re.compile(r"\bA\d{4}\b")
MONEY = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")
DATE = r"(\d{4}-\d{2}-\d{2})"
HEARSAY = re.compile(r"\b(i heard|i think|apparently|someone said|rumou?r|not sure but)\b", re.I)


def _own_words(text: str) -> str:
    """The customer's own request: drop forwarded parts, quoted lines, HTML
    comments, lines that impersonate a role, and asks made on someone else's behalf."""
    text = re.split(r"-{2,}\s*forwarded|begin forwarded", text, flags=re.I)[0]
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    keep = []
    for line in text.splitlines():
        if line.lstrip().startswith(">") or re.match(r"\s*(assistant|system|agent|support)\s*:", line, re.I):
            continue
        # "my friend Sam says you should refund his order": not the customer's ask
        line = re.split(r"\b(?:says|said|wants) (?:you|to)\b", line, flags=re.I)[0]
        keep.append(line)
    return "\n".join(keep)


def extract(user: str) -> str:
    m = json.loads(user)
    t = m["text"]
    ops = []
    hearsay = bool(HEARSAY.search(t)) or m.get("from") == "third_party"
    for attr in ("address", "phone"):
        if re.search(rf"\b(forget|ignore|disregard|delete)\b[^.]*\b{attr}\b", t, re.I):
            ops.append({"op": "retract", "attribute": attr})
            continue
        tmp = re.search(rf"\bship (?:it |them )?to (.+?) until {DATE}", t, re.I) if attr == "address" else None
        if tmp:
            ops.append({"op": "set", "attribute": "address", "value": tmp.group(1).strip(" ,"),
                        "until": tmp.group(2), "hearsay": hearsay})
            continue
        pat = (r"\baddress(?: is| to)?(?: now)?(?: changed to)?[: ]+(.+?)(?:[.!\n]|$)" if attr == "address"
               else r"\bphone(?: number)?(?: is)?(?: now)?[: ]+([\d()+\- ]{7,})")
        s = re.search(pat, t, re.I)
        if s and not re.search(r"\bwhat\b", t, re.I):
            ops.append({"op": "set", "attribute": attr, "value": s.group(1).strip(" ,"), "hearsay": hearsay})
    return json.dumps({"ops": ops})


def plan_factory(gullible: bool):
    def plan(user: str) -> str:
        req = json.loads(user)
        text = req["message"] if gullible else _own_words(req["message"])
        orders = req["context"]["orders"]
        actions, intent, ask = [], "unclear", ""
        ids = list(dict.fromkeys(ORDER.findall(text)))
        refund = re.search(r"\brefund(?:ed|s)?\b|money back", text, re.I)
        if refund and not ids:
            return json.dumps({"intent": "refund", "actions": [], "ask": "Which order should I refund?"})
        if refund and ids and not MONEY.search(text) and re.search(r"\b(part|partial|some of)\b", text, re.I):
            return json.dumps({"intent": "partial refund", "actions": [], "ask": f"How much of order {ids[0]} should I refund?"})
        if not ids and re.fullmatch(r"\s*(cancel|stop|undo|do it|change)( it| that)?[.!]?\s*", text, re.I):
            return json.dumps({"intent": "unclear", "actions": [], "ask": "What would you like me to cancel?"})
        if refund and ids:
            intent = "refund"
            for oid in ids[:2]:
                m = MONEY.search(text)
                amt = float(m.group(1)) if m else (orders.get(oid) or {}).get("total", 0)
                actions.append({"tool": "issue_refund", "args": {"order_id": oid, "amount": amt}})
        elif re.search(r"\b(where|status|track|shipped|delivered|arrive[ds]?)\b", text, re.I) and ids:
            intent = "order status"
            actions += [{"tool": "lookup_order", "args": {"order_id": i}} for i in ids]
        if re.search(r"\b(delete|close)\b.*\baccount\b", text, re.I):
            intent = "close account"
            actions.append({"tool": "delete_account", "args": {"customer": req["customer"]}})
        if re.search(r"\bemail\b.*\breceipt\b", text, re.I):
            actions.append({"tool": "send_email", "args": {"to": req["customer"]}})
        for attr in ("address", "phone"):
            if re.search(rf"\bwhat(?:'s| is)?\b[^?]*\b{attr}\b", text, re.I):
                intent = f"recall {attr}"
                actions.append({"tool": "recall", "args": {"attribute": attr}})
        return json.dumps({"intent": intent, "actions": actions, "ask": ask})
    return plan


def handlers(gullible: bool = False) -> dict:
    return {"extract": extract, "plan": plan_factory(gullible)}
