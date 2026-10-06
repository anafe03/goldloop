"""TM GO on goldloop: the reply-extraction golden set, graded by the generic harness.

Runs from OUTSIDE the TM GO repo and never writes to it: TMGO_ROOT points at a
checkout (default ../../../toursync-deploy), bytecode writing is off, runs are
saved here. Nothing is copied. Cases come straight from run_evals.GOLDEN and the target is
reply_prompt_eval.run_case, the offline mirror of production's process_reply
(same prompt, same deterministic triage, same grounding and validation). This
file only translates between the two shapes.

    goldloop run --repeat 3
    goldloop matrix model=openai:gpt-4o-mini,anthropic:claude-haiku-4-5
    goldloop tune --holdout 0.3
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import sys
import threading
from pathlib import Path

sys.dont_write_bytecode = True     # no __pycache__ inside the TM GO checkout
ROOT = Path(os.environ.get("TMGO_ROOT") or Path(__file__).resolve().parents[3] / "toursync-deploy").resolve()
if not (ROOT / "scripts" / "reply_prompt_eval.py").is_file():
    raise SystemExit(f"TM GO checkout not found at {ROOT} (set TMGO_ROOT)")
sys.path.insert(0, str(ROOT / "scripts"))

import reply_prompt_eval as rpe   # noqa: E402  (loads .env, extracts prod code by AST)


def _missing_names() -> list[str]:
    """Module-level names in api_routes.py that the extracted functions use but
    reply_prompt_eval didn't extract. Found 2026-10-06: _SHAPE_RULES and
    _recover, missing since api_routes 25d0808 (2026-09-24), so TM GO's own
    offline harness scores ~44/61 cases as crashes."""
    tree = ast.parse((ROOT / "api_routes.py").read_text())
    top = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    top |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    have, missing = set(rpe.api), set()
    frontier = [n for n in tree.body if getattr(n, "name", None) in have]
    while frontier:
        node = frontier.pop()
        for name in {x.id for x in ast.walk(node) if isinstance(x, ast.Name)} & top:
            if name not in have and name not in missing:
                missing.add(name)
                frontier += [n for n in tree.body if getattr(n, "name", None) == name or
                             (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets))]
    return sorted(missing)


MISSING = _missing_names()
if MISSING and os.environ.get("TMGO_PATCH_MISSING", "1") == "1":
    # In memory only: the extracted functions share rpe.api as their globals,
    # so adding the names there makes them resolve. TM GO's files are untouched.
    rpe.api.update(rpe._extract(ROOT / "api_routes.py", set(MISSING)))
    print(f"[tmgo adapter] patched in memory, missing from reply_prompt_eval.py: {', '.join(MISSING)} "
          f"(TMGO_PATCH_MISSING=0 to reproduce the crash)", file=sys.stderr)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def case_id(c: dict) -> str:
    # Venue names repeat (two Bowery Ballroom cases), so the id carries a short
    # hash of the thread: stable across reorderings, new if the thread changes.
    h = hashlib.sha256(repr(c["thread"]).encode()).hexdigest()[:4]
    return f"{_slug(c['venue'])}-{h}"


def to_goldloop(c: dict) -> dict:
    checks = [{"type": "contains", "path": f"updates.{k}", "value": v} for k, v in c["expect"].items()]
    checks += [{"type": "absent", "path": f"updates.{k}"} for k in c.get("forbid", ())]
    if c.get("expect_kind"):
        # run_evals zeroes the case on a wrong kind: the triage classifier is a
        # safety gate (a 'problem' email must never auto-fill the sheet).
        checks.append({"type": "equals", "path": "kind", "value": c["expect_kind"], "critical": True})
    if c.get("expect_redirect"):
        # run_evals caps a missed redirect at 0.5, which fails at threshold 0.8.
        # goldloop fails it outright: same pass/fail, lower mean on that case.
        checks.append({"type": "contains", "path": "redirect_to", "value": c["expect_redirect"],
                       "critical": True})
    tags = ["triage"] if c.get("expect_kind") else ["extraction"]
    if c.get("forbid"):
        tags.append("negative-control")
    return {"id": case_id(c), "input": {"venue": c["venue"], "fields": c["fields"], "thread": c["thread"]},
            "checks": checks, "tags": tags, "source": "authored"}


def cases() -> list[dict]:
    return [to_goldloop(c) for c in rpe.GOLDEN]


_models: dict = {}
_lock = threading.Lock()


def _model(name: str):
    with _lock:
        if name not in _models:
            from langchain.chat_models import init_chat_model
            _models[name] = init_chat_model(name, temperature=0, max_tokens=800)
        return _models[name]


def _prompt_fn(template: str):
    """Same template -> prompt mapping prompt_tuner.py uses."""
    spec = rpe.api["_REPLY_FIELD_SPEC"]

    def fn(want=None):
        want_line = f"We're specifically trying to fill: {', '.join(want)}.\n" if want else ""
        return template.replace("{FIELD_SPEC}", spec).replace("{WANT_LINE}", want_line)
    return fn


def champion_template() -> str:
    from prompt_tuner import champion_template as ct
    return ct()


def run(inp: dict, params: dict) -> dict:
    model = _model(params.get("model", "openai:gpt-4o-mini"))
    prompt_fn = _prompt_fn(params["prompt"]) if params.get("prompt") else None
    case = {**inp, "expect": {}}          # run_case scores too; we ignore its score
    _, got = rpe.run_case(model, case, prompt_fn=prompt_fn)
    kind = got.pop("kind", "")
    redirect = got.pop("redirect_to", "")
    return {"kind": kind, "redirect_to": redirect, "updates": got}


def write_champion(path: str = "prompts/reply_extract.txt") -> None:
    """Materialise the live production prompt as a template file the tuner can
    vary. Written HERE, beside this adapter, never into the TM GO checkout."""
    p = Path(__file__).resolve().parent / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(champion_template())
    print(f"wrote {p}")


if __name__ == "__main__":
    write_champion()
