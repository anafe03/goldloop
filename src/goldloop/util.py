from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_ref(ref: str) -> Any:
    """'package.module:attr' -> the object. The cwd is importable, so a repo
    can point at its own code without installing it."""
    if ":" not in ref:
        raise ValueError(f"ref must look like 'module:attr', got {ref!r}")
    mod, attr = ref.split(":", 1)
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def short_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def git_state(cwd: str | Path = ".") -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()
    try:
        sha = run("rev-parse", "--short", "HEAD")
        dirty = bool(run("status", "--porcelain"))
        return {"sha": sha or None, "dirty": dirty}
    except FileNotFoundError:
        return {"sha": None, "dirty": None}


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
