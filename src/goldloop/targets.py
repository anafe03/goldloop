"""Targets: the system under test, behind one interface.

    python     any function: fn(input) or fn(input, params)
    langgraph  a compiled graph, or a builder fn(params) -> graph. Records the
               node path so checks can grade what the agent DID.
    http       a deployed endpoint. Cannot drift from production, because it is
               production.

`params` is the knob set a run varies: model, prompt text, temperature. The
tuner and the model matrix work by changing params, never by editing code.

Infra failures raise Unavailable. A rate limit or an empty billing account is
not a quality result, and scoring it as 0% is how a good model gets "proven" bad.
"""
from __future__ import annotations

import contextlib
import inspect
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .checks import Observation
from .util import load_ref


class Unavailable(Exception):
    """The target could not be reached or refused for infra reasons. Not a score."""


@dataclass
class Result:
    """Optional richer return value for python targets."""
    output: Any
    usage: dict = field(default_factory=dict)
    trajectory: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


_INFRA_NAMES = re.compile(
    r"RateLimit|Authentication|PermissionDenied|APIConnection|APITimeout|"
    r"InternalServer|ServiceUnavailable|Overloaded|InsufficientQuota|ConnectionError|Timeout")


def is_infra_error(e: BaseException) -> bool:
    if isinstance(e, Unavailable):
        return True
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status in (401, 402, 403, 408, 429) or (isinstance(status, int) and status >= 500):
        return True
    if any(_INFRA_NAMES.search(c.__name__) for c in type(e).__mro__):
        return True
    # Some providers report an empty account as a plain 400 ("credit balance
    # is too low"). That's an outage of your account, not a quality result.
    return bool(re.search(r"insufficient_quota|credit balance is too low|exceeded your current quota", str(e), re.I))


@contextlib.contextmanager
def _capture_usage():
    """Collect token usage from every LangChain model call made inside the block,
    without the target having to cooperate."""
    box: dict = {}
    try:
        from langchain_core.callbacks import get_usage_metadata_callback
    except ImportError:
        yield box
        return
    with get_usage_metadata_callback() as cb:
        yield box
        box.update(_flatten_usage(cb.usage_metadata))


def _flatten_usage(by_model: dict) -> dict:
    if not by_model:
        return {}
    return {"input_tokens": sum(u.get("input_tokens", 0) for u in by_model.values()),
            "output_tokens": sum(u.get("output_tokens", 0) for u in by_model.values()),
            "by_model": {m: {"input_tokens": u.get("input_tokens", 0),
                             "output_tokens": u.get("output_tokens", 0)} for m, u in by_model.items()}}


class Target:
    name: str = "target"

    def call(self, case_input: Any) -> Observation:
        raise NotImplementedError

    def describe(self) -> dict:
        return {}


class PythonTarget(Target):
    def __init__(self, fn: Callable, params: dict, ref: str = ""):
        self.fn, self.params, self.ref = fn, params, ref
        try:
            n = len([p for p in inspect.signature(fn).parameters.values()
                     if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        except (TypeError, ValueError):
            n = 1
        self._takes_params = n >= 2

    def call(self, case_input):
        t0 = time.perf_counter()
        try:
            with _capture_usage() as usage:
                r = self.fn(case_input, self.params) if self._takes_params else self.fn(case_input)
        except Exception as e:
            if is_infra_error(e):
                raise Unavailable(f"{type(e).__name__}: {e}") from e
            raise
        ms = (time.perf_counter() - t0) * 1000
        if isinstance(r, Result):
            return Observation(r.output, list(r.trajectory), r.usage or usage, ms, list(r.events))
        return Observation(r, [], usage, ms)

    def describe(self):
        return {"kind": "python", "ref": self.ref}


class LangGraphTarget(Target):
    def __init__(self, graph_or_builder: Any, params: dict, ref: str = "",
                 input_key: str | None = None, output_path: str | None = None):
        self.params, self.ref, self.input_key, self.output_path = params, ref, input_key, output_path
        if hasattr(graph_or_builder, "stream") and hasattr(graph_or_builder, "invoke"):
            self.graph = graph_or_builder
        else:
            self.graph = graph_or_builder(params)

    def call(self, case_input):
        from .checks import resolve
        state_in = {self.input_key: case_input} if self.input_key else case_input
        config = {"configurable": dict(self.params), "recursion_limit": int(self.params.get("recursion_limit", 25))}
        trajectory, final = [], None
        t0 = time.perf_counter()
        try:
            with _capture_usage() as usage:
                for mode, chunk in self.graph.stream(state_in, config, stream_mode=["updates", "values"]):
                    if mode == "updates" and isinstance(chunk, dict):
                        trajectory += [n for n in chunk if not n.startswith("__")]
                    elif mode == "values":
                        final = chunk
        except Exception as e:
            if is_infra_error(e):
                raise Unavailable(f"{type(e).__name__}: {e}") from e
            raise
        ms = (time.perf_counter() - t0) * 1000
        out = resolve(final, self.output_path) if self.output_path else final
        return Observation(out, trajectory, usage, ms)

    def describe(self):
        return {"kind": "langgraph", "ref": self.ref, "input_key": self.input_key}


def _expand_env(s: str) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), s)


class HttpTarget(Target):
    def __init__(self, url: str, params: dict, method: str = "POST", headers: dict | None = None,
                 timeout: float = 90, body_template: dict | None = None, send_params: bool = False):
        self.url, self.params, self.method = _expand_env(url), params, method
        self.send_params = send_params
        self.headers = {k: _expand_env(str(v)) for k, v in (headers or {}).items()}
        self.timeout, self.body_template = timeout, body_template

    def call(self, case_input):
        body = dict(self.body_template or {}, **case_input) if isinstance(case_input, dict) else case_input
        if self.send_params and isinstance(body, dict):
            # Only for endpoints built to accept overrides (e.g. an eval-only
            # route that takes a candidate prompt). Off by default.
            body = {**body, "params": self.params}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method=self.method,
                                     headers={"Content-Type": "application/json", **self.headers})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 403, 408, 429) or e.code >= 500:
                raise Unavailable(f"HTTP {e.code}") from e
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]!r}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise Unavailable(f"{type(e).__name__}: {e}") from e
        ms = (time.perf_counter() - t0) * 1000
        try:
            out = json.loads(raw)
        except ValueError:
            out = raw.decode(errors="replace")
        usage = out.pop("_usage", {}) if isinstance(out, dict) else {}
        return Observation(out, [], usage, ms)

    def describe(self):
        return {"kind": "http", "url": re.sub(r"//[^/@]*@", "//***@", self.url)}


def make_target(spec: dict, params: dict | None = None) -> Target:
    """spec comes from goldloop.yaml `target:` (or a named entry under `targets:`)."""
    params = {**(spec.get("params") or {}), **(params or {})}
    params = _load_param_files(params)
    kind = spec.get("kind", "python")
    if kind == "python":
        return PythonTarget(load_ref(spec["ref"]), params, spec["ref"])
    if kind == "langgraph":
        return LangGraphTarget(load_ref(spec["ref"]), params, spec["ref"],
                               spec.get("input_key"), spec.get("output_path"))
    if kind == "http":
        return HttpTarget(spec["url"], params, spec.get("method", "POST"), spec.get("headers"),
                          float(spec.get("timeout", 90)), spec.get("body"),
                          bool(spec.get("send_params")))
    raise ValueError(f"unknown target kind {kind!r}")


def _load_param_files(params: dict) -> dict:
    """`prompt_file: prompts/x.txt` also exposes `prompt` = the file's text, so
    a target reads params['prompt'] and the tuner can swap the text in memory."""
    out = dict(params)
    for k, v in params.items():
        if k.endswith("_file") and isinstance(v, str):
            # Fail loudly: a typo'd path silently falling back to the default
            # prompt makes an A/B test compare the default against itself.
            if not os.path.isfile(v):
                raise FileNotFoundError(f"param {k}={v!r}: no such file")
            out.setdefault(k[: -len("_file")], open(v).read())
    return out
