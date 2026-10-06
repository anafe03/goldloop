"""Use the local Claude Code CLI as the model, on the user's Claude subscription.

For running evals without an API key: each call is a headless `claude -p`
with the caller's own system prompt, no tools, no MCP servers and no user
settings, so it behaves like a plain model call. ANTHROPIC_API_KEY is removed
from the child's environment, so it can never fall back to billing an API key.

It plugs into code that uses LangChain's ChatAnthropic without editing that
code: `patch_chat_anthropic()` swaps the class in memory for the duration of a
run. `.invoke(messages)` and `.with_structured_output(Schema, include_raw=...)`
are supported, which is what the adopted projects use.

Slower than the API (a process per call, ~3s) and subject to subscription
limits; fine for golden sets of tens of cases.
"""
from __future__ import annotations

import contextlib
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

ALIASES = {"haiku": "haiku", "sonnet": "sonnet", "opus": "opus"}


def find_cli() -> str:
    cand = os.environ.get("GOLDLOOP_CLAUDE_BIN") or shutil.which("claude")
    if cand:
        return cand
    found = sorted(glob.glob(os.path.expanduser(
        "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude")))
    if found:
        return found[-1]
    raise FileNotFoundError("Claude Code CLI not found (set GOLDLOOP_CLAUDE_BIN)")


def alias_for(model: str | None) -> str:
    m = (model or "haiku").lower()
    return next((v for k, v in ALIASES.items() if k in m), "haiku")


def call(system: str, prompt: str, model: str = "haiku", timeout: float = 120) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    cmd = [find_cli(), "-p", prompt, "--model", alias_for(model), "--output-format", "json",
           "--max-turns", "1", "--system-prompt", system or "You are a helpful assistant.",
           "--tools", "", "--strict-mcp-config", "--setting-sources", ""]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd="/tmp")
    try:
        d = json.loads(p.stdout)
    except ValueError:
        raise RuntimeError(f"claude CLI failed ({p.returncode}): {(p.stderr or p.stdout)[:300]}")
    if d.get("is_error"):
        msg = str(d.get("result") or d)[:300]
        if re.search(r"limit|rate|overloaded|credit", msg, re.I):
            raise ConnectionError(f"Claude Code unavailable: {msg}")      # goldloop: outage, not a score
        raise RuntimeError(f"Claude Code error: {msg}")
    u = d.get("usage") or {}
    return {"text": d.get("result") or "", "model": next(iter(d.get("modelUsage") or {}), alias_for(model)),
            "input_tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                            + u.get("cache_creation_input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0)}


def _split(messages: list) -> tuple[str, str]:
    """LangChain messages or (role, text) tuples -> (system, transcript)."""
    system, turns = [], []
    def role_of(m):
        return m[0] if isinstance(m, tuple) else (m.type if isinstance(m, BaseMessage) else "human")
    n_turns = sum(1 for m in messages if role_of(m) != "system")
    for m in messages:
        if isinstance(m, tuple):
            role, text = m
        elif isinstance(m, BaseMessage):
            role, text = m.type, m.content
        else:
            role, text = "human", str(m)
        text = text if isinstance(text, str) else json.dumps(text)
        if role == "system":
            system.append(text)
        else:
            # a single user turn goes in as-is; a conversation keeps its speaker labels
            turns.append(text if n_turns == 1 else f"{role.upper()}: {text}")
    return "\n\n".join(system), "\n\n".join(turns)


class ClaudeCodeChat(BaseChatModel):
    model: str = "haiku"
    max_tokens: int | None = None
    timeout: float | None = None

    def __init__(self, model: str = "haiku", **kw: Any):
        super().__init__(model=model, **{k: v for k, v in kw.items() if k in ("max_tokens", "timeout")})

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        system, prompt = _split(messages)
        r = call(system, prompt, self.model)
        msg = AIMessage(content=r["text"], response_metadata={"model_name": r["model"]},
                        usage_metadata={"input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                                        "total_tokens": r["input_tokens"] + r["output_tokens"]})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def with_structured_output(self, schema, include_raw: bool = False, **kw):
        """Ask for JSON matching the schema, then validate it with the schema."""
        js = json.dumps(schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema)

        def run(messages):
            msgs = list(messages) if not isinstance(messages, str) else [("human", messages)]
            msgs = msgs + [("system", f"Return ONLY a JSON object that matches this JSON schema, no prose:\n{js}")]
            raw = self.invoke(msgs)
            text = raw.content if isinstance(raw.content, str) else str(raw.content)
            m = re.search(r"\{.*\}", text, re.S)
            parsed, err = None, None
            try:
                data = json.loads(m.group(0)) if m else {}
                parsed = schema.model_validate(data) if hasattr(schema, "model_validate") else data
            except Exception as e:  # noqa: BLE001
                err = e
            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": err}
            if err:
                raise err
            return parsed
        return RunnableLambda(run)


@contextlib.contextmanager
def patch_chat_anthropic(model: str | None = None):
    """Make `from langchain_anthropic import ChatAnthropic` return ClaudeCodeChat
    for the duration of the block. The adopted repo's code is not edited."""
    import langchain_anthropic
    orig = langchain_anthropic.ChatAnthropic

    def factory(*a, model_name=None, **kw):
        return ClaudeCodeChat(model=model or kw.pop("model", None) or model_name or "haiku",
                              **{k: v for k, v in kw.items() if k in ("max_tokens", "timeout")})
    langchain_anthropic.ChatAnthropic = factory
    try:
        yield
    finally:
        langchain_anthropic.ChatAnthropic = orig
