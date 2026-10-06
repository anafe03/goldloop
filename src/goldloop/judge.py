"""LLM-as-judge, for outputs a string check can't grade (a drafted email, a summary).

A judge is a model, so it is wrong sometimes. Calibrate it against human labels
before trusting it in a gate:

    goldloop calibrate judge_labels.jsonl --model openai:gpt-4o-mini

and only gate on rubrics where agreement is high. The judge never runs on the
production path, which is why it is allowed to be a model at all.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .checks import Observation, _text, resolve
from .targets import Unavailable, is_infra_error

JUDGE_SYS = """You grade one output of an AI system against a rubric.
Be strict and literal: score what the output does, not what it meant to do.
Return ONLY JSON: {"score": <number 0 to 1>, "reason": "<one sentence>"}"""

_cache: dict = {}


def _model(name: str):
    if name not in _cache:
        from langchain.chat_models import init_chat_model
        # No temperature: current Claude models reject sampling params. Judge
        # variance is what `goldloop calibrate` and repeats are for.
        _cache[name] = init_chat_model(name, max_tokens=2000)
    return _cache[name]


def grade(output_text: str, rubric: str, model: str, context: str = "") -> tuple[float, str]:
    msg = f"RUBRIC:\n{rubric}\n\n" + (f"CONTEXT:\n{context}\n\n" if context else "") + f"OUTPUT:\n{output_text}"
    try:
        raw = _model(model).invoke([("system", JUDGE_SYS), ("human", msg)]).content
    except Exception as e:
        if is_infra_error(e):
            raise Unavailable(f"judge {model}: {e}") from e
        raise
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") for b in raw if isinstance(b, dict))
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0)) if m else {}
    return max(0.0, min(1.0, float(data.get("score", 0)))), str(data.get("reason", ""))[:200]


def judge_check(obs: Observation, c: dict) -> tuple[float, str]:
    model = c.get("model") or os.environ.get("GOLDLOOP_JUDGE_MODEL")
    if not model:
        raise ValueError("judge check needs `model:` or GOLDLOOP_JUDGE_MODEL")
    score, reason = grade(_text(resolve(obs.output, c.get("path"))), c["rubric"], model,
                          c.get("context", ""))
    return score, f"judge: {reason}"


def calibrate(labels: Path, model: str, threshold: float = 0.5) -> dict:
    """labels.jsonl: {"output": str, "rubric": str, "human": 0..1 or true/false}.
    Reports raw agreement and Cohen's kappa on pass/fail at `threshold`."""
    rows = [json.loads(l) for l in labels.read_text().splitlines() if l.strip()]
    tp = tn = fp = fn = 0
    disagreements = []
    for r in rows:
        human = float(r["human"]) >= threshold
        s, why = grade(r["output"], r["rubric"], model, r.get("context", ""))
        judge = s >= threshold
        tp += human and judge
        tn += (not human) and (not judge)
        fp += (not human) and judge
        fn += human and (not judge)
        if human != judge:
            disagreements.append({"output": r["output"][:160], "human": human, "judge": s, "why": why})
    n = len(rows)
    agree = (tp + tn) / n if n else 0
    pe = (((tp + fp) * (tp + fn)) + ((tn + fn) * (tn + fp))) / (n * n) if n else 0
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": agree, "kappa": kappa,
            "judge_too_lenient": fp, "judge_too_strict": fn, "disagreements": disagreements}
