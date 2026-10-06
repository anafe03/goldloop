"""Many providers behind one interface, with failover that doesn't lie about cost or continuity.

What a router has to get right:
  * A STALL is a failure. A provider that accepts the request and then sits
    there is worse than one that errors, because naive code waits forever.
    Every call gets a deadline; missing it fails over.
  * Only INFRA failures fail over (timeouts, 429/quota, 5xx, connection). A
    malformed request fails everywhere; retrying it on provider #2 just
    doubles the bill and hides the bug.
  * A CIRCUIT BREAKER keeps a failed provider out of rotation for a cooldown,
    so a long session doesn't pay the stall timeout on every single call.
    That's continuity: after the first failover the session stays on the
    working provider instead of flapping.
  * Every attempt is an event with provider, model, outcome, tokens and cost,
    so "which model answered this, and what did it cost" has an answer.
  * When every provider is down, say so (AllProvidersDown). Returning an
    empty answer would be scored as a quality failure, and it isn't one.
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .events import EventLog

_POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="provider")


class AllProvidersDown(Exception):
    """Infra, not quality. goldloop records this as UNAVAILABLE, not 0%."""


class Stall(Exception):
    pass


class QuotaExceeded(Exception):
    status_code = 429


class ServerError(Exception):
    status_code = 500


class BillingAs400(Exception):
    """What an empty account actually looks like on some providers: a 400."""
    status_code = 400


_INFRA = re.compile(r"RateLimit|Quota|Timeout|Stall|APIConnection|InternalServer|Overloaded|ServiceUnavailable")
# Account-level failures that some providers send as a plain 400. Found the
# hard way (2026-10-06): an Anthropic account with no credit answers
# "400 invalid_request_error: Your credit balance is too low", which a
# status-code-only router reads as "bad request, don't fail over".
_BILLING = re.compile(r"credit balance is too low|insufficient_quota|billing|exceeded your current quota", re.I)


def is_infra(e: BaseException) -> bool:
    status = getattr(e, "status_code", None)
    if status in (402, 408, 429) or (isinstance(status, int) and status >= 500):
        return True
    if _BILLING.search(str(e)):
        return True
    return any(_INFRA.search(c.__name__) for c in type(e).__mro__)


@dataclass
class Reply:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class Provider(Protocol):
    name: str
    model: str

    def complete(self, task: str, system: str, user: str) -> Reply: ...


class RulesProvider:
    """A deterministic stand-in model: each task is a plain Python function.
    Lets the whole harness (and its evals) run offline, free, in CI."""

    def __init__(self, handlers: dict[str, Callable[[str], str]], name: str = "rules", model: str = "rules-v1"):
        self.handlers, self.name, self.model = handlers, name, model

    def complete(self, task, system, user):
        return Reply(self.handlers[task](user), self.name, self.model, len(user) // 4, 20)


class LangChainProvider:
    """Any model init_chat_model knows: 'openai:gpt-4o-mini', 'anthropic:claude-haiku-4-5'."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.name, _, self.model = model_id.partition(":")
        self._llm = None

    def complete(self, task, system, user):
        if self._llm is None:
            from langchain.chat_models import init_chat_model
            # No sampling params: current Claude models reject temperature.
            self._llm = init_chat_model(self.model_id, max_tokens=1500)
        msg = self._llm.invoke([("system", system), ("human", user)])
        text = msg.content if isinstance(msg.content, str) else "".join(
            b.get("text", "") for b in msg.content if isinstance(b, dict))
        u = getattr(msg, "usage_metadata", None) or {}
        return Reply(text, self.name, self.model, u.get("input_tokens", 0), u.get("output_tokens", 0))


class Faulty:
    """Fault injection for evals: make a provider stall, run out of quota, or break."""

    def __init__(self, inner: Provider, mode: str, stall_s: float = 5.0, deadline_s: float | None = None):
        self.inner, self.mode, self.stall_s, self.deadline_s = inner, mode, stall_s, deadline_s
        self.name, self.model = inner.name, inner.model
        self.calls = 0

    def complete(self, task, system, user):
        self.calls += 1
        if self.mode == "stall":
            time.sleep(self.stall_s)
            raise Stall("no first token")
        if self.mode == "quota":
            raise QuotaExceeded(f"{self.name}: insufficient_quota")
        if self.mode == "server_error":
            raise ServerError(f"{self.name}: 500 internal error")
        if self.mode == "billing_400":
            raise BillingAs400(f"Error code: 400 - {self.name}: Your credit balance is too low to access the API.")
        if self.mode == "bad_request":
            raise ValueError("400: malformed request")    # not infra: must NOT fail over
        return self.inner.complete(task, system, user)


@dataclass
class Router:
    providers: list[Provider]
    log: EventLog
    deadline_s: float = 20.0
    cooldown_s: float = 60.0
    pricing: dict = field(default_factory=dict)          # model-substring -> {input, output} $/1M
    _open_until: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _cost(self, r: Reply) -> float | None:
        rate = next((v for k, v in self.pricing.items() if k in r.model), None)
        if not rate:
            return None
        return r.input_tokens / 1e6 * rate["input"] + r.output_tokens / 1e6 * rate["output"]

    def complete(self, task: str, system: str, user: str) -> Reply:
        tried = []
        for i, p in enumerate(self.providers):
            route = f"{p.name}:{p.model}"          # breaker per route, not per vendor
            with self._lock:
                if self._open_until.get(route, 0) > time.monotonic():
                    self.log.emit("route.skip", **{"gen_ai.provider.name": p.name, "reason": "circuit open"})
                    continue
            t0 = time.perf_counter()
            try:
                # Per-provider deadline: set it from that provider's normal p99,
                # not one number for everyone.
                deadline = getattr(p, "deadline_s", None) or self.deadline_s
                r = _POOL.submit(p.complete, task, system, user).result(timeout=deadline)
            except FutureTimeout:
                err: BaseException = Stall(f"{p.name}: no reply within {deadline}s")
            except Exception as e:  # noqa: BLE001
                err = e
            else:
                cost = self._cost(r)
                self.log.emit("gen_ai.client.operation", **{
                    "gen_ai.operation.name": "chat", "gen_ai.provider.name": r.provider,
                    "gen_ai.request.model": r.model, "gen_ai.usage.input_tokens": r.input_tokens,
                    "gen_ai.usage.output_tokens": r.output_tokens, "task": task, "outcome": "ok",
                    "attempt": len(tried) + 1, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    **({"cost_usd": cost} if cost is not None else {})})
                return r
            ms = round((time.perf_counter() - t0) * 1000, 1)
            self.log.emit("gen_ai.client.operation", **{
                "gen_ai.operation.name": "chat", "gen_ai.provider.name": p.name,
                "gen_ai.request.model": p.model, "task": task, "outcome": "error",
                "error.type": type(err).__name__, "attempt": len(tried) + 1, "latency_ms": ms})
            if not is_infra(err):
                raise err                                 # same request would fail anywhere
            with self._lock:
                self._open_until[route] = time.monotonic() + self.cooldown_s
            tried.append(p.name)
            nxt = next((q.name for q in self.providers[i + 1:]), None)
            if nxt:
                self.log.emit("route.failover", **{"from": p.name, "to": nxt, "reason": type(err).__name__})
        raise AllProvidersDown(f"no provider answered (tried {tried or 'none: all circuits open'})")
