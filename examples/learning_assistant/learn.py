#!/usr/bin/env python3
"""The self-improvement loop, with proof: learn from each correction, keep it only if it helps.

For each of Rosa's corrections, in order:
  1. A learner turns it into a candidate knowledge item (memory or skill).
  2. goldloop runs every task twice: with the current knowledge, and with the
     current knowledge plus the candidate. The tasks are new messages, none
     seen in any correction.
  3. Keep the candidate only if it fixes at least one task, breaks none, and
     triggers no guardrail. Otherwise reject it and say why.
Every kept item records the correction it came from, so it can be traced or
removed later.

For comparison it also runs "learn everything": every candidate kept, no gate.

    python learn.py                      # offline stand-in models, free
    python learn.py --model openai:gpt-4o-mini --repeat 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from goldloop import store                       # noqa: E402
from goldloop.cases import load_suite            # noqa: E402
from goldloop.compare import compare             # noqa: E402
from goldloop.runner import run_suite            # noqa: E402
from goldloop.targets import PythonTarget        # noqa: E402

import assistant as A                            # noqa: E402
import eval_target                               # noqa: E402


def score(cases, knowledge, model, repeat, label=None, save=False):
    t = PythonTarget(eval_target.run, {"model": model, "knowledge": knowledge})
    run = run_suite(cases, t, repeat=repeat, threshold=1.0, concurrency=4,
                    params={"model": model, "knowledge_items": len(knowledge)}, label=label or "")
    if save:
        store.save_run(run)
    return run


def violations(run) -> list[str]:
    return sorted({cid for cid, c in run["cases"].items() for r in c["runs"]
                   for ck in r.get("checks") or [] if ck.get("critical") and ck["score"] < 1})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rules")
    ap.add_argument("--learner", default=None, help="model for the learner (default: same as --model)")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    import os
    os.chdir(HERE)                                # runs are saved beside this example
    cases = load_suite(HERE / "cases")
    corrections = yaml.safe_load((HERE / "corrections.yaml").read_text())
    learner_spec = args.learner or args.model
    learn = A.rules_learn if learner_spec == "rules" else (
        lambda c: A.llm_learn(eval_target._model(learner_spec), c))

    base = score(cases, [], args.model, args.repeat, "before learning", save=True)
    print(f"before learning: {base['metrics']['pass_all_k']}/{len(cases)} tasks pass every run\n")

    knowledge, current, log = [], base, []
    for c in corrections:
        cand = learn(c)
        trial = score(cases, knowledge + [cand], args.model, args.repeat)
        cmp = compare(current, trial)
        bad = [v for v in violations(trial) if v not in violations(current)]
        if bad:
            ok, why = False, f"breaks a must-hold rule: {', '.join(bad)}"
        elif cmp["regressions"]:
            ok, why = False, f"breaks {', '.join(cmp['regressions'])}"
        elif not cmp["fixes"]:
            ok, why = False, "helps no task it hasn't seen"
        else:
            ok, why = True, f"fixes {', '.join(cmp['fixes'])}"
        log.append({"correction": c["id"], "item": cand, "kept": ok, "why": why,
                    "fixes": cmp["fixes"], "breaks": cmp["regressions"], "violations": bad})
        print(f"  {c['id']}  {'KEEP  ' if ok else 'REJECT'}  {cand['kind']:<6} \"{cand['says'][:58]}\"\n"
              f"         {why}")
        if ok:
            knowledge.append(cand)
            current = trial

    final = score(cases, knowledge, args.model, args.repeat, "after gated learning", save=True)
    naive_k = [learn(c) for c in corrections]
    naive = score(cases, naive_k, args.model, args.repeat, "learn everything (no gate)", save=True)
    print(f"\nafter gated learning:      {final['metrics']['pass_all_k']}/{len(cases)} pass every run · "
          f"must-hold rules broken: {len(violations(final))}")
    print(f"learn everything, no gate: {naive['metrics']['pass_all_k']}/{len(cases)} pass every run · "
          f"must-hold rules broken: {len(violations(naive))} ({', '.join(violations(naive)) or 'none'})")
    out = HERE / ".goldloop"
    out.mkdir(exist_ok=True)
    (out / "knowledge.json").write_text(json.dumps(knowledge, indent=1))
    (out / "learning_log.json").write_text(json.dumps({
        "model": args.model, "learner": learner_spec, "repeat": args.repeat,
        "before": base["metrics"]["pass_all_k"], "after": final["metrics"]["pass_all_k"],
        "naive": naive["metrics"]["pass_all_k"], "naive_violations": violations(naive),
        "n_tasks": len(cases), "decisions": log}, indent=1))


if __name__ == "__main__":
    main()
