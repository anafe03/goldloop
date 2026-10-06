"""The trust boundary: what the agent may do alone, what needs a person, what never happens.

The rule this module exists to enforce: **permission is decided in code, at
the tool boundary, never by the model.** A prompt can be injected; a model can
be talked into calling `issue_refund`. It doesn't matter, because the call
still has to pass `decide()`, and `decide()` doesn't read the conversation.

Three tiers, safe by default:
    allow   runs on its own (read-only lookups)
    ask     runs only with a matching human approval (money, outbound messages)
    forbid  never runs, whatever anyone says (account deletion, from this agent)
A tool not in the policy is `forbid`. New capability starts closed.

Approvals are scoped: "issue_refund:A1001:25.00" approves that refund on that
order for that amount, not "refunds in general". Over-broad approval is how a
human-in-the-loop turns into a rubber stamp. Every decision is an event, so a
company can prove afterwards what was allowed, by whom, and why.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .events import EventLog

ALLOW, ASK, FORBID = "allow", "ask", "forbid"


@dataclass
class Policy:
    tiers: dict[str, str] = field(default_factory=lambda: {
        "lookup_order": ALLOW,
        "recall": ALLOW,
        "issue_refund": ASK,
        "send_email": ASK,
        "delete_account": FORBID,
    })
    # Limits that hold even WITH approval: approval is a person's judgement,
    # limits are the company's.
    max_refund: float = 500.0

    def tier(self, tool: str) -> str:
        return self.tiers.get(tool, FORBID)


def approval_key(tool: str, args: dict) -> str:
    """The exact thing a human approves. Order + amount for refunds."""
    if tool == "issue_refund":
        return f"issue_refund:{args.get('order_id')}:{float(args.get('amount', 0)):.2f}"
    if tool == "send_email":
        return f"send_email:{args.get('to')}"
    return f"{tool}:" + ":".join(f"{k}={args[k]}" for k in sorted(args))


@dataclass
class Decision:
    tool: str
    args: dict
    verdict: str            # "run" | "needs_approval" | "blocked"
    reason: str
    key: str


def decide(policy: Policy, tool: str, args: dict, approvals: set[str], log: EventLog,
           *, actor: str = "agent", owner_ok: bool = True) -> Decision:
    """owner_ok=False: the action targets something the requester doesn't own
    (a refund on another customer's order). Blocked even with approval: an
    approver sees one ticket; the harness sees whose order it is."""
    tier = policy.tier(tool)
    key = approval_key(tool, args)
    if tier != FORBID and not owner_ok:
        d = Decision(tool, args, "blocked", "not this customer's order", key)
    elif tier == FORBID:
        d = Decision(tool, args, "blocked",
                     "not in policy" if tool not in policy.tiers else "forbidden for this agent", key)
    elif tool == "issue_refund" and float(args.get("amount", 0)) > policy.max_refund:
        d = Decision(tool, args, "blocked", f"over the {policy.max_refund:.0f} limit, even with approval", key)
    elif tier == ASK and key not in approvals:
        d = Decision(tool, args, "needs_approval", "needs a person to say yes", key)
    else:
        d = Decision(tool, args, "run", "allowed" if tier == ALLOW else "approved", key)
    log.emit("policy.decision", **{"gen_ai.tool.name": tool, "tool": tool, "tier": tier,
                                   "decision": d.verdict, "reason": d.reason,
                                   "approval_key": key, "actor": actor})
    if d.verdict == "run" and tier == ASK:
        log.emit("approval.used", tool=tool, approval_key=key)
    return d
