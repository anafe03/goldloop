"""Runs on disk. One JSON file per run under .goldloop/runs/.

The baseline is a separate small file you commit (.goldloop/baseline.json),
so CI compares a change against the run the team agreed was good, not against
whatever happened to run last.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(".goldloop")


def runs_dir() -> Path:
    d = ROOT / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(run: dict) -> Path:
    p = runs_dir() / f"{run['id']}.json"
    p.write_text(json.dumps(run, indent=1, default=str))
    return p


def list_runs() -> list[Path]:
    return sorted(runs_dir().glob("*.json"))


def load_run(ref: str) -> dict:
    """ref: a path, a run id (or unique prefix), 'latest', 'previous' or 'baseline'."""
    if ref == "baseline":
        b = ROOT / "baseline.json"
        if not b.exists():
            raise FileNotFoundError("no baseline set (goldloop baseline <run>)")
        return json.loads(b.read_text())
    p = Path(ref)
    if p.suffix == ".json" and p.exists():
        return json.loads(p.read_text())
    runs = list_runs()
    if ref in ("latest", "previous"):
        need = 1 if ref == "latest" else 2
        if len(runs) < need:
            raise FileNotFoundError(f"not enough runs for {ref!r}")
        return json.loads(runs[-need].read_text())
    hits = [r for r in runs if r.stem.startswith(ref) or ref in r.stem]
    if len(hits) != 1:
        raise FileNotFoundError(f"run {ref!r}: {len(hits)} matches")
    return json.loads(hits[0].read_text())


def set_baseline(run: dict) -> Path:
    """The baseline keeps per-case summaries and metrics, not raw outputs, so it
    stays small enough to review in a PR."""
    slim = {k: v for k, v in run.items() if k != "cases"}
    slim["cases"] = {cid: {"tags": c["tags"], "source": c.get("source"), "fp": c.get("fp"), "summary": c["summary"],
                           "runs": []} for cid, c in run["cases"].items()}
    p = ROOT / "baseline.json"
    ROOT.mkdir(exist_ok=True)
    p.write_text(json.dumps(slim, indent=1, default=str))
    return p
