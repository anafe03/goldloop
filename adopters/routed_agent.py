"""Shared adapter for agents built the same way: a LangGraph router that picks a
lane, deterministic workflows that do the work, and a gate in front of anything
that spends money. Meridian (health insurance) and TripTalk (travel) both are.

Reads the repo from outside (its root from an env var), never writes to it,
and translates its own golden.yaml into goldloop cases:

    expected_intent    -> equals intent                (the model's judgment)
    expect_citation    -> contains citations           (retrieval)
    expect_staged      -> equals pending_action.type   critical: the write gate
    expect_over_cap / expect_over_budget -> equals flag critical: diverts to a person
    expect_prior_auth / expect_active / expect_found   critical: a lookup must not be wrong

params:
    classifier   keyword | llm
    model        claude-code[:haiku|sonnet|opus]  run the LLM path on the Claude Code
                 subscription (no API key), or leave unset for the repo's own setup
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


class RoutedAgent:
    def __init__(self, root: Path, package: str):
        self.root, self.package = root.resolve(), package
        if not (self.root / "data" / "golden.yaml").is_file():
            raise SystemExit(f"{self.root}: no data/golden.yaml (set the root env var)")
        sys.path.insert(0, str(self.root / "src"))
        self._graph = None

    def cases(self) -> list[dict]:
        out = []
        for c in yaml.safe_load((self.root / "data" / "golden.yaml").read_text()):
            checks = [{"type": "equals", "path": "intent", "value": c["expected_intent"], "name": "right lane"}]
            tags = ["routing"]
            if "expect_citation" in c:
                checks.append({"type": "contains", "path": "citations", "value": c["expect_citation"], "name": "cites the right doc"})
                tags.append("grounding")
            if "expect_staged" in c:
                checks.append({"type": "equals", "path": "staged", "value": c["expect_staged"], "critical": True,
                               "name": "staged for approval, not executed"})
                tags.append("gate")
            for k in ("expect_over_cap", "expect_over_budget"):
                if k in c:
                    checks.append({"type": "equals", "path": "over_limit", "value": bool(c[k]), "critical": True,
                                   "name": "large amount goes to a person"})
                    tags.append("gate")
            for k, path in (("expect_prior_auth", "tool.prior_auth_required"), ("expect_active", "tool.active")):
                if k in c:
                    checks.append({"type": "equals", "path": path, "value": c[k], "critical": True, "name": "lookup is right"})
                    tags.append("determination")
            if c.get("expect_found"):
                checks.append({"type": "equals", "path": "tool.found", "value": True, "critical": True, "name": "member found"})
                tags.append("determination")
            out.append({"id": c["id"], "input": {k: c[k] for k in ("query", "member_id", "plan") if k in c},
                        "checks": checks, "tags": tags})
        return out

    def graph(self):
        if self._graph is None:
            mod = __import__(f"{self.package}.graph", fromlist=["compile_graph"])
            self._graph = mod.compile_graph()
        return self._graph

    def run(self, inp: dict, params: dict) -> dict:
        state = {"raw_query": inp["query"], "classifier": params.get("classifier", "keyword")}
        if "member_id" in inp or "plan" in inp:              # Meridian's session fields
            state.update({"plan": inp.get("plan", "GOLD-PPO"), "member_id": inp.get("member_id", ""),
                          "verified": bool(inp.get("member_id"))})
        spec = str(params.get("model") or "")
        if spec.startswith("claude-code"):
            from goldloop.providers.claude_code import patch_chat_anthropic
            os.environ.setdefault("ANTHROPIC_API_KEY", "routed-to-claude-code")   # only flips the repo's has_anthropic(); never sent
            with patch_chat_anthropic(spec.partition(":")[2] or "haiku"):
                s = self.graph().invoke(state)
        else:
            s = self.graph().invoke(state)
        return {"intent": s.get("intent"), "citations": s.get("citations") or [],
                "staged": (s.get("pending_action") or {}).get("type"),
                "over_limit": bool(s.get("over_cap") or s.get("over_budget")),
                "tool": s.get("tool_result") or {}, "answer": str(s.get("answer") or s.get("response") or "")[:500]}
