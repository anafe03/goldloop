"""goldloop target: the bakery assistant, with a given set of knowledge items.

params:
  model       rules | <provider:model>
  knowledge   list of knowledge items, or a path to a JSON file of them
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from goldloop import Result               # noqa: E402
import assistant as A                      # noqa: E402

_models: dict = {}


def _model(spec: str):
    if spec not in _models:
        from langchain.chat_models import init_chat_model
        _models[spec] = init_chat_model(spec, max_tokens=400)
    return _models[spec]


def knowledge_from(params: dict) -> list[dict]:
    k = params.get("knowledge") or []
    if isinstance(k, str):
        k = json.loads(Path(k).read_text()) if Path(k).exists() else []
    return list(k)


def run(inp: dict, params: dict) -> Result:
    log: list = []
    knowledge = knowledge_from(params)
    spec = params.get("model", "rules")
    out = A.rules_reply(inp["message"], knowledge, log) if spec == "rules" \
        else A.llm_reply(_model(spec), inp["message"], knowledge, log)
    out["reply"] = A.guard(out["reply"], log)
    for kid in out["used"]:
        log.append({"name": "knowledge.used", "attrs": {"id": kid}})
    return Result(output=out, events=log, trajectory=sorted(out["used"]))
