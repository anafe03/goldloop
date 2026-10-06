"""goldloop.yaml: one file that says what to test, against what, and what may ship.

    suite: evals/cases
    target:            {kind: python, ref: my_agent.eval:run, params: {model: openai:gpt-4o-mini}}
    targets:           # optional named alternates (parity, matrix)
      live:            {kind: http, url: "https://api.example.com/extract"}
    threshold: 0.8     # a run of a case passes at this score
    repeat: 3
    concurrency: 4
    gate:              {min_mean: 0.8, max_flaky: 0, max_regressions: 0}
    pricing:           {gpt-4o-mini: {input: 0.15, output: 0.60}}   # $ per 1M tokens
    tune:              {knob: prompt, optimizer_model: anthropic:claude-opus-5-5}
"""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULTS = {"suite": "evals/cases", "threshold": 0.8, "repeat": 1, "concurrency": 4,
            "gate": {}, "pricing": {}, "tune": {}, "targets": {}, "principles": {}}


def load_config(path: str | Path = "goldloop.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (run `goldloop init`)")
    cfg = {**DEFAULTS, **(yaml.safe_load(p.read_text()) or {})}
    if "target" not in cfg:
        raise ValueError(f"{p}: no `target:`")
    return cfg


def target_spec(cfg: dict, name: str | None) -> dict:
    if not name or name == "default":
        return cfg["target"]
    if name not in cfg["targets"]:
        raise KeyError(f"no target named {name!r} (have: default, {', '.join(cfg['targets'])})")
    return cfg["targets"][name]
