"""The Claude Code provider, without calling the CLI: message flattening,
structured output parsing, and that the API key never reaches the child."""
import json

import pytest
from pydantic import BaseModel

import goldloop.providers.claude_code as cc


class Lane(BaseModel):
    intent: str


def test_structured_output_parses_and_validates(monkeypatch):
    seen = {}

    def fake_call(system, prompt, model="haiku", timeout=120):
        seen.update(system=system, prompt=prompt, model=model)
        return {"text": 'Sure!\n```json\n{"intent": "flights"}\n```', "model": "claude-haiku-4-5-x",
                "input_tokens": 10, "output_tokens": 5}
    monkeypatch.setattr(cc, "call", fake_call)
    r = cc.ClaudeCodeChat("claude-haiku-4-5").with_structured_output(Lane, include_raw=True).invoke(
        [("system", "Pick a lane."), ("human", "book a flight")])
    assert r["parsed"] == Lane(intent="flights") and r["parsing_error"] is None
    assert "Pick a lane." in seen["system"] and "JSON schema" in seen["system"] and seen["prompt"] == "book a flight"
    assert r["raw"].usage_metadata["input_tokens"] == 10


def test_bad_json_is_a_parsing_error_not_a_crash(monkeypatch):
    monkeypatch.setattr(cc, "call", lambda *a, **k: {"text": "no idea", "model": "m", "input_tokens": 1, "output_tokens": 1})
    r = cc.ClaudeCodeChat().with_structured_output(Lane, include_raw=True).invoke([("human", "x")])
    assert r["parsed"] is None and r["parsing_error"] is not None


def test_api_key_is_stripped_from_the_child(monkeypatch):
    captured = {}

    class P:
        returncode, stderr = 0, ""
        stdout = json.dumps({"result": "ok", "usage": {}, "modelUsage": {"claude-haiku-4-5": {}}})

    def fake_run(cmd, **kw):
        captured["env"] = kw["env"]
        captured["cmd"] = cmd
        return P()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-be-used")
    monkeypatch.setattr(cc, "find_cli", lambda: "claude")
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    cc.call("sys", "hi", "sonnet")
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert "--tools" in captured["cmd"] and "--strict-mcp-config" in captured["cmd"]


def test_rate_limit_is_an_outage(monkeypatch):
    class P:
        returncode, stderr = 1, ""
        stdout = json.dumps({"is_error": True, "result": "Claude usage limit reached"})
    monkeypatch.setattr(cc, "find_cli", lambda: "claude")
    monkeypatch.setattr(cc.subprocess, "run", lambda cmd, **kw: P())
    from goldloop.targets import is_infra_error
    with pytest.raises(ConnectionError) as e:
        cc.call("s", "p")
    assert is_infra_error(e.value)
