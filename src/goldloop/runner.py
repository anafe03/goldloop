"""Run a suite against a target: every case, `repeat` times, in parallel."""
from __future__ import annotations

import json
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from .cases import Case, suite_fingerprint
from .checks import Observation, run_check, score_case
from .metrics import aggregate, case_summary
from .targets import Target, Unavailable
from .util import git_state, short_hash


def _jsonable(x: Any, limit: int = 4000) -> Any:
    """Keep a readable copy of the output in the run file, bounded in size."""
    try:
        s = json.dumps(x, default=lambda o: getattr(o, "content", None) or repr(o))
    except Exception:
        s = json.dumps(repr(x))
    if len(s) > limit:
        return {"_truncated": s[:limit]}
    return json.loads(s)


def run_one(target: Target, case: Case, threshold: float) -> dict:
    try:
        obs = target.call(case.input)
    except Unavailable as e:
        return {"status": "unavailable", "score": None, "passed": None, "error": str(e)[:300],
                "latency_ms": 0.0, "usage": {}}
    except Exception as e:
        # The target crashed on this input. That IS a quality result: score 0.
        tb = traceback.format_exception_only(type(e), e)[-1].strip()
        return {"status": "error", "score": 0.0, "passed": False, "error": tb[:300],
                "latency_ms": 0.0, "usage": {}, "checks": []}
    results = []
    for c in case.checks:
        try:
            results.append(run_check(obs, c))
        except Unavailable as e:          # an LLM judge that couldn't be reached
            return {"status": "unavailable", "score": None, "passed": None,
                    "error": f"check {c['type']}: {e}"[:300], "latency_ms": obs.latency_ms,
                    "usage": obs.usage}
    s = score_case(results)
    return {"status": "ok", "score": s, "passed": s >= threshold, "checks": results,
            "latency_ms": obs.latency_ms, "usage": obs.usage,
            "trajectory": obs.trajectory, "output": _jsonable(obs.output),
            "events": _jsonable(obs.events, limit=20000)}


def run_suite(cases: list[Case], target: Target, *, repeat: int = 1, threshold: float = 0.8,
              concurrency: int = 4, params: dict | None = None, pricing: dict | None = None,
              label: str = "", suite_path: str = "",
              progress: Callable[[str, dict], None] | None = None) -> dict:
    jobs = [(c, i) for i in range(repeat) for c in cases]
    started = time.time()

    def work(job):
        case, _ = job
        r = run_one(target, case, threshold)
        if progress:
            progress(case.id, r)
        return case.id, r

    out: dict[str, dict] = {c.id: {"tags": c.tags, "source": c.source, "fp": c.fingerprint(), "runs": []} for c in cases}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for cid, r in pool.map(work, jobs):
            out[cid]["runs"].append(r)
    for c in out.values():
        c["summary"] = case_summary(c["runs"])

    params = params or {}
    model = str(params.get("model", ""))
    return {
        "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label,
        "git": git_state(),
        "suite": {"path": suite_path, "fingerprint": suite_fingerprint(cases), "n": len(cases)},
        "target": {**target.describe(), "params": _param_view(params), "params_hash": short_hash(params)},
        "config": {"repeat": repeat, "threshold": threshold},
        "duration_s": round(time.time() - started, 2),
        "cases": out,
        "metrics": aggregate(out, pricing, model),
    }


def _param_view(params: dict) -> dict:
    """Long values (prompt text) are stored as a hash, not inline."""
    view = {}
    for k, v in params.items():
        if isinstance(v, str) and len(v) > 120:
            view[k] = f"<{len(v)} chars sha:{short_hash(v)}>"
        else:
            view[k] = v
    return view
