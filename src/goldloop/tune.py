"""Current prompt vs. challenger: propose a change, prove it helped, or keep what we have.

Per round:
  1. Score the champion (the current text of one param, e.g. the prompt).
  2. Hand an optimizer the champion and its failures: input, expected, got.
  3. It proposes K rewrites, each with a one-line rationale.
  4. Every candidate takes the SAME exam: same cases, same repeats, same checks.
  5. A candidate wins only if it gains >= min_gain on mean, regresses no case
     the champion passed, adds no flakiness, and (with a holdout) holds up on
     cases the optimizer never saw.

It never edits your code. It writes the proposal, a diff and the evidence; a
human reads the diff and commits. A loop that rewrites its own instructions
without review is the failure mode, not the feature.

The holdout matters most. An optimizer shown its failures can memorize them
("if the venue is Lincoln Hall, ..."). Train-set gains then look like
progress and are not.
"""
from __future__ import annotations

import difflib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .cases import Case, split
from .compare import compare
from .runner import run_suite
from .targets import make_target

Optimizer = Callable[[str, list[dict], dict], list[dict]]

OPTIMIZER_SYS = """You improve one text parameter (usually a system prompt) of an AI system.
You get the current text and the test cases it FAILS, with what was expected and what
it produced. Propose rewrites that fix those failures WITHOUT breaking what already works.

Rules for every rewrite:
- Keep every placeholder in curly braces exactly as written, once each: {placeholders}
- Keep the output contract the text describes unchanged.
- Prefer a precise, GENERAL rule over naming a specific case, person, venue or value.
  A rule that only fixes one example will be rejected on held-out cases.
- Be concise; every line must earn its place.

Return ONLY JSON: {{"candidates": [{{"rationale": "<one sentence>", "text": "<the full new text>"}}]}}
Return exactly {k} candidates, each trying a different idea."""


def placeholders(text: str) -> set[str]:
    return set(re.findall(r"(?<!\{)\{[A-Za-z_][A-Za-z0-9_]*\}(?!\})", text))


def llm_optimizer(model: str) -> Optimizer:
    def propose(champion: str, failures: list[dict], ctx: dict) -> list[dict]:
        from langchain.chat_models import init_chat_model
        # No sampling params: current Claude models reject temperature (400), and
        # candidate diversity comes from asking for k different ideas anyway.
        llm = init_chat_model(model, max_tokens=16000)
        ph = ", ".join(sorted(placeholders(champion))) or "(none)"
        sys = OPTIMIZER_SYS.format(placeholders=ph, k=ctx["k"])
        body = {"current_text": champion, "failing_cases": failures[:20],
                "previous_attempts": ctx.get("history", [])[-6:]}
        raw = llm.invoke([("system", sys), ("human", json.dumps(body, indent=1, default=str))]).content
        if isinstance(raw, list):   # content blocks
            raw = "".join(b.get("text", "") for b in raw if isinstance(b, dict))
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
        return [c for c in data.get("candidates", []) if c.get("text")]
    return propose


def failures_of(run: dict, cases: list[Case]) -> list[dict]:
    by_id = {c.id: c for c in cases}
    out = []
    for cid, c in run["cases"].items():
        if c["summary"].get("passed_all"):
            continue
        last = next((r for r in reversed(c["runs"]) if r["status"] == "ok"), None)
        out.append({"id": cid, "input": by_id[cid].input,
                    "failed_checks": [ck["detail"] for ck in (last or {}).get("checks", []) if ck["score"] < 1],
                    "got": (last or {}).get("output"),
                    "error": next((r.get("error") for r in c["runs"] if r.get("error")), None)})
    return out


def _accept(base: dict, cand: dict, min_gain: float) -> tuple[bool, str, dict]:
    cmp = compare(base, cand)
    mb, mc = base["metrics"], cand["metrics"]
    if not mc["complete"]:
        return False, "incomplete (unavailable runs)", cmp
    gain = (mc["mean"] or 0) - (mb["mean"] or 0)
    if cmp["regressions"]:
        return False, f"regressed {cmp['regressions']}", cmp
    if cmp["newly_flaky"]:
        return False, f"newly flaky {cmp['newly_flaky']}", cmp
    if gain < min_gain:
        return False, f"gain {gain:+.3f} < {min_gain}", cmp
    return True, f"gain {gain:+.3f}", cmp


def tune(cases: list[Case], target_spec: dict, *, knob: str, optimizer: Optimizer,
         base_params: dict | None = None, iterations: int = 2, k: int = 3, repeat: int = 2,
         min_gain: float = 0.01, holdout: float = 0.0, threshold: float = 0.8,
         concurrency: int = 4, pricing: dict | None = None, out_dir: Path = Path(".goldloop/tune"),
         log: Callable[[str], None] = print) -> dict:
    base_params = dict(base_params or {})
    spec_params = {**(target_spec.get("params") or {}), **base_params}
    # Resolve <knob>_file -> text once, via the same loader targets use.
    resolved = make_target(target_spec, base_params).params
    if knob not in resolved:
        raise KeyError(f"param {knob!r} not set (set params.{knob} or params.{knob}_file)")
    champion = resolved[knob]
    required = placeholders(champion)
    train, held = split(cases, holdout)
    log(f"tune · knob={knob} · train {len(train)} · holdout {len(held)} · repeat {repeat}")

    def score(text: str, subset: list[Case], label: str) -> dict:
        t = make_target(target_spec, {**base_params, knob: text})
        return run_suite(subset, t, repeat=repeat, threshold=threshold, concurrency=concurrency,
                         params={**spec_params, knob: text}, pricing=pricing, label=label)

    champ_run = score(champion, train, "champion")
    log(f"  champion  mean {champ_run['metrics']['mean']:.1%}  pass^k {champ_run['metrics']['pass_all']:.1%}")
    rounds, history, promoted_any = [], [], False
    start_text = champion

    for it in range(1, iterations + 1):
        fails = failures_of(champ_run, train)
        if not fails:
            log("  champion passes every train case; nothing to learn from")
            break
        proposals = optimizer(champion, fails, {"k": k, "history": history, "knob": knob})
        rnd = {"iteration": it, "champion_mean": champ_run["metrics"]["mean"], "candidates": []}
        best = None
        for i, p in enumerate(proposals[:k]):
            entry = {"rationale": p.get("rationale", ""), "text": p["text"]}
            missing = required - placeholders(p["text"])
            if missing:
                entry.update(accepted=False, reason=f"dropped placeholders {sorted(missing)}")
                rnd["candidates"].append(entry)
                log(f"  [{it}.{i}] rejected without scoring: {entry['reason']}")
                continue
            run = score(p["text"], train, f"candidate {it}.{i}")
            ok, why, cmp = _accept(champ_run, run, min_gain)
            entry.update(mean=run["metrics"]["mean"], pass_all=run["metrics"]["pass_all"],
                         flaky=run["metrics"]["flaky"], fixes=cmp["fixes"],
                         regressions=cmp["regressions"], accepted=ok, reason=why)
            entry["significance"] = cmp["verdict"]
            log(f"  [{it}.{i}] mean {run['metrics']['mean']:.1%}  {why}  ({cmp['verdict']})  · {entry['rationale'][:60]}")
            if ok and (best is None or run["metrics"]["mean"] > best[1]["metrics"]["mean"]):
                best = (entry, run)
            rnd["candidates"].append(entry)
            history.append({"rationale": entry["rationale"], "result": why})

        if best and held:
            h_champ = score(champion, held, "holdout champion")
            h_cand = score(best[0]["text"], held, "holdout candidate")
            # Must not regress a case AND must not lose mean. (First version only
            # checked regressions, and promoted a rewrite whose holdout mean fell
            # 75% -> 67%: per-case scores dropped without flipping pass/fail.)
            ok, why, cmp = _accept(h_champ, h_cand, min_gain=0.0)
            best[0]["holdout"] = {"champion_mean": h_champ["metrics"]["mean"],
                                  "candidate_mean": h_cand["metrics"]["mean"], "ok": ok, "reason": why}
            log(f"  holdout   champion {h_champ['metrics']['mean']:.1%} → candidate "
                f"{h_cand['metrics']['mean']:.1%}  {'ok' if ok else 'REJECTED (overfit): ' + why}")
            if not ok:
                best[0]["accepted"] = False
                best = None

        if best:
            champion, champ_run = best[0]["text"], best[1]
            rnd["promoted"] = best[0]["rationale"]
            promoted_any = True
            log(f"  → promoted: {best[0]['rationale'][:90]}")
        else:
            rnd["promoted"] = None
            log("  → champion stands")
        rounds.append(rnd)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"at": stamp, "knob": knob, "train": [c.id for c in train], "holdout": [c.id for c in held],
              "repeat": repeat, "min_gain": min_gain, "start_mean": rounds[0]["champion_mean"] if rounds else champ_run["metrics"]["mean"],
              "final_mean": champ_run["metrics"]["mean"], "promoted": promoted_any, "rounds": rounds}
    hist = out_dir / "history.json"
    all_hist = json.loads(hist.read_text()) if hist.exists() else []
    hist.write_text(json.dumps(all_hist + [record], indent=1, default=str))
    if promoted_any:
        (out_dir / f"proposed_{knob}.txt").write_text(champion)
        diff = "".join(difflib.unified_diff(start_text.splitlines(True), champion.splitlines(True),
                                            f"{knob} (current)", f"{knob} (proposed)"))
        (out_dir / f"proposed_{knob}.diff").write_text(diff)
        log(f"\nproposal → {out_dir}/proposed_{knob}.txt  (review the .diff, then commit it yourself)")
    return record
