"""goldloop CLI.

  goldloop init                     scaffold goldloop.yaml + an example case
  goldloop lint                     cases that could pass for the wrong reason
  goldloop run                      score the suite, save the run
  goldloop gate                     run (or take --run) and fail on policy / regressions vs baseline
  goldloop baseline latest          make a run the baseline (commit .goldloop/baseline.json)
  goldloop compare previous latest  case-level diff + McNemar
  goldloop runs                     history
  goldloop report                   HTML page: latest run + trend
  goldloop promote feedback.jsonl   production feedback -> candidate cases
  goldloop accept                   move reviewed candidates into the suite
  goldloop tune                     current prompt vs. challenger, with optional holdout
  goldloop matrix --param model=a,b same suite across param values (models, temps)
  goldloop parity offline live      do two targets agree on the same cases?
  goldloop calibrate labels.jsonl   LLM judge vs human labels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import report, store
from .cases import load_suite
from .compare import compare, gate
from .config import load_config, target_spec
from .runner import run_suite
from .targets import make_target


def _params(pairs: list[str] | None) -> dict:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k] = yaml.safe_load(v) if v else ""
    return out


def _cases(cfg, args):
    cases = load_suite(cfg["suite"])
    if getattr(args, "tag", None):
        cases = [c for c in cases if set(args.tag) & set(c.tags)]
    if getattr(args, "case", None):
        cases = [c for c in cases if c.id in args.case]
    if not cases:
        sys.exit("no cases match the filter")
    return cases


def _progress(quiet: bool):
    if quiet or not sys.stderr.isatty():
        return None
    marks = {"ok": None, "error": "E", "unavailable": "?"}

    def p(cid, r):
        m = marks[r["status"]] or ("." if r["passed"] else "x")
        sys.stderr.write(m)
        sys.stderr.flush()
    return p


def _do_run(cfg, args, target_name=None, extra_params=None, label=None, save=True):
    spec = target_spec(cfg, target_name or getattr(args, "target", None))
    params = {**_params(getattr(args, "param", None)), **(extra_params or {})}
    target = make_target(spec, params)
    cases = _cases(cfg, args)
    run = run_suite(cases, target, repeat=getattr(args, "repeat", None) or cfg["repeat"],
                    threshold=cfg["threshold"], concurrency=cfg["concurrency"],
                    params={**(spec.get("params") or {}), **params}, pricing=cfg["pricing"],
                    label=label or getattr(args, "label", "") or "", suite_path=str(cfg["suite"]),
                    progress=_progress(getattr(args, "quiet", False)))
    if sys.stderr.isatty():
        sys.stderr.write("\n")
    if save:
        p = store.save_run(run)
        run["_path"] = str(p)
    return run


def cmd_init(args):
    if Path("goldloop.yaml").exists():
        sys.exit("goldloop.yaml already exists")
    Path("goldloop.yaml").write_text(_INIT_CONFIG)
    Path("evals/cases").mkdir(parents=True, exist_ok=True)
    Path("evals/cases/example.yaml").write_text(_INIT_CASE)
    gi = Path(".gitignore")
    lines = gi.read_text().splitlines() if gi.exists() else []
    for entry in (".goldloop/runs/", ".goldloop/tune/"):
        if entry not in lines:
            lines.append(entry)
    gi.write_text("\n".join(lines) + "\n")
    print("wrote goldloop.yaml, evals/cases/example.yaml; .goldloop/runs/ is gitignored, "
          ".goldloop/baseline.json is meant to be committed")


def cmd_lint(args):
    from .cases import lint
    cfg = load_config(args.config)
    problems = lint(load_suite(cfg["suite"]))
    for p in problems:
        print(f"  ⚠ {p}")
    print("lint ✓" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


def cmd_run(args):
    cfg = load_config(args.config)
    run = _do_run(cfg, args)
    print(report.run_text(run, args.verbose))
    print(f"saved {run['_path']}")
    if args.json:
        print(json.dumps(run["metrics"], indent=1))


def cmd_gate(args):
    cfg = load_config(args.config)
    run = store.load_run(args.run) if args.run else _do_run(cfg, args, label="gate")
    if not args.run:
        print(report.run_text(run))
    try:
        base = store.load_run(args.baseline) if args.baseline != "none" else None
    except FileNotFoundError:
        base = None
        print("(no baseline yet: gating on absolute policy only)")
    ok, reasons, cmp = gate(run, base, cfg["gate"])
    if cmp:
        print("\n" + report.compare_text(cmp))
    print("\n" + ("PASS ✓" if ok else "FAIL ✗"))
    for r in reasons:
        print(f"  ⛔ {r}")
    if args.markdown:
        md = report.compare_markdown(cmp, run, ok, reasons) if cmp else \
            f"### goldloop {'✅' if ok else '❌'}\n" + "\n".join(f"- {r}" for r in reasons)
        Path(args.markdown).write_text(md)
    sys.exit(0 if ok else 1)


def cmd_baseline(args):
    run = store.load_run(args.run)
    p = store.set_baseline(run)
    print(f"baseline ← {run['id']}  (mean {report.pct(run['metrics']['mean'])}, "
          f"pass^k {report.pct(run['metrics']['pass_all'])})  → {p}  commit this file")


def cmd_compare(args):
    cmp = compare(store.load_run(args.base), store.load_run(args.cand))
    print(report.compare_text(cmp))


def cmd_runs(args):
    for p in store.list_runs()[-args.n:]:
        r = json.loads(p.read_text())
        m = r["metrics"]
        print(f"{r['id']}  {(r.get('git') or {}).get('sha') or '-':>8}  {r['suite']['n']:>3} cases  "
              f"mean {report.pct(m['mean']):>6}  pass^k {report.pct(m['pass_all']):>6}  "
              f"flaky {len(m['flaky'])}  {'' if m['complete'] else 'INCOMPLETE '}{r.get('label', '')}")


def cmd_report(args):
    runs = [json.loads(p.read_text()) for p in store.list_runs()]
    if not runs:
        sys.exit("no runs yet")
    print(report.html_report(runs, Path(args.out)))


def cmd_promote(args):
    from .promote import promote
    cfg = load_config(args.config)
    print(json.dumps(promote(args.feedback, load_suite(cfg["suite"]), Path(args.out), args.limit), indent=1))
    print("review each candidate: set `review: ok` (or `reject`), tighten checks, then `goldloop accept`")


def cmd_accept(args):
    from .promote import accept
    print(json.dumps(accept(Path(args.candidates), Path(args.into)), indent=1))


def cmd_tune(args):
    from .tune import llm_optimizer, tune
    from .util import load_ref
    cfg = load_config(args.config)
    t = cfg["tune"]
    if args.optimizer:
        opt = load_ref(args.optimizer)
    else:
        model = args.optimizer_model or t.get("optimizer_model")
        if not model:
            sys.exit("set tune.optimizer_model in goldloop.yaml or pass --optimizer-model")
        opt = llm_optimizer(model)
    tune(_cases(cfg, args), target_spec(cfg, args.target), knob=args.knob or t.get("knob", "prompt"),
         optimizer=opt, base_params=_params(args.param),
         iterations=args.iterations or t.get("iterations", 2), k=args.k or t.get("candidates", 3),
         repeat=args.repeat or t.get("repeat", 2), min_gain=args.min_gain if args.min_gain is not None
         else t.get("min_gain", 0.01), holdout=args.holdout if args.holdout is not None else t.get("holdout", 0.0),
         threshold=cfg["threshold"], concurrency=cfg["concurrency"], pricing=cfg["pricing"])


def cmd_matrix(args):
    cfg = load_config(args.config)
    key, _, vals = args.param_sweep.partition("=")
    rows = []
    for v in vals.split(","):
        run = _do_run(cfg, args, extra_params={key: yaml.safe_load(v)}, label=f"matrix {key}={v}")
        m = run["metrics"]
        rows.append((v, m))
        print(f"  {key}={v}: done")
    print(f"\n{key:<34} {'mean':>7} {'pass^k':>7} {'flaky':>5} {'p95':>8} {'$/case':>9}  note")
    for v, m in rows:
        note = "" if m["complete"] else f"UNAVAILABLE ×{m['unavailable_runs']} (not a score)"
        print(f"{v:<34} {report.pct(m['mean']):>7} {report.pct(m['pass_all']):>7} {len(m['flaky']):>5} "
              f"{m['latency_ms']['p95']:>6.0f}ms {report.money(m['cost_per_case_usd']):>9}  {note}")


def cmd_parity(args):
    cfg = load_config(args.config)
    a = _do_run(cfg, args, target_name=args.a, label=f"parity {args.a}")
    b = _do_run(cfg, args, target_name=args.b, label=f"parity {args.b}")
    diff = []
    for cid in a["cases"]:
        sa, sb = a["cases"][cid]["summary"], b["cases"][cid]["summary"]
        if "unavailable" in (sa["status"], sb["status"]):
            continue
        if sa["passed_all"] != sb["passed_all"] or abs((sa["mean"] or 0) - (sb["mean"] or 0)) > 1e-9:
            diff.append((cid, sa["mean"], sb["mean"]))
    print(f"{args.a}: mean {report.pct(a['metrics']['mean'])}   {args.b}: mean {report.pct(b['metrics']['mean'])}")
    for cid, x, y in diff:
        print(f"  ≠ {cid}: {args.a} {x:.0%}  {args.b} {y:.0%}")
    print("parity ✓" if not diff else f"parity ✗: {len(diff)} case(s) disagree; one harness has drifted")
    sys.exit(0 if not diff else 1)


def cmd_calibrate(args):
    from .judge import calibrate
    r = calibrate(Path(args.labels), args.model, args.threshold)
    print(f"n={r['n']}  agreement {r['agreement']:.0%}  kappa {r['kappa']:.2f}  "
          f"too lenient {r['judge_too_lenient']}  too strict {r['judge_too_strict']}")
    for d in r["disagreements"]:
        print(f"  human={'pass' if d['human'] else 'fail'} judge={d['judge']:.2f}  {d['output']!r}  · {d['why']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="goldloop", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="goldloop.yaml")
    ap.add_argument("--env-file", help="load KEY=VALUE lines (API keys) without overriding the environment")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def runopts(p):
        p.add_argument("--target", help="named target from `targets:` (default: `target:`)")
        p.add_argument("--param", action="append", metavar="K=V", help="override a target param")
        p.add_argument("--repeat", type=int)
        p.add_argument("--tag", action="append", help="only cases with this tag")
        p.add_argument("--case", action="append", help="only this case id")
        p.add_argument("--label", default="")
        p.add_argument("-q", "--quiet", action="store_true")

    sub.add_parser("init").set_defaults(fn=cmd_init)
    sub.add_parser("lint").set_defaults(fn=cmd_lint)
    p = sub.add_parser("run"); runopts(p)
    p.add_argument("-v", "--verbose", action="store_true"); p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("gate"); runopts(p)
    p.add_argument("--run", help="gate an existing run instead of running")
    p.add_argument("--baseline", default="baseline", help="run ref, or 'none'")
    p.add_argument("--markdown", help="write a PR-comment summary here")
    p.set_defaults(fn=cmd_gate)
    p = sub.add_parser("baseline"); p.add_argument("run", nargs="?", default="latest"); p.set_defaults(fn=cmd_baseline)
    p = sub.add_parser("compare"); p.add_argument("base", nargs="?", default="baseline")
    p.add_argument("cand", nargs="?", default="latest"); p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("runs"); p.add_argument("-n", type=int, default=20); p.set_defaults(fn=cmd_runs)
    p = sub.add_parser("report"); p.add_argument("--out", default=".goldloop/report.html"); p.set_defaults(fn=cmd_report)
    p = sub.add_parser("promote"); p.add_argument("feedback", help=".jsonl or module:fn")
    p.add_argument("--out", default="evals/candidates.yaml"); p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_promote)
    p = sub.add_parser("accept"); p.add_argument("--candidates", default="evals/candidates.yaml")
    p.add_argument("--into", default="evals/cases/promoted.yaml"); p.set_defaults(fn=cmd_accept)
    p = sub.add_parser("tune"); runopts(p)
    p.add_argument("--knob"); p.add_argument("--iterations", type=int); p.add_argument("--k", type=int)
    p.add_argument("--min-gain", type=float); p.add_argument("--holdout", type=float)
    p.add_argument("--optimizer-model"); p.add_argument("--optimizer", help="module:fn custom optimizer")
    p.set_defaults(fn=cmd_tune)
    p = sub.add_parser("matrix"); runopts(p)
    p.add_argument("param_sweep", metavar="KEY=v1,v2,..."); p.set_defaults(fn=cmd_matrix)
    p = sub.add_parser("parity"); runopts(p); p.add_argument("a"); p.add_argument("b"); p.set_defaults(fn=cmd_parity)
    p = sub.add_parser("calibrate"); p.add_argument("labels"); p.add_argument("--model", required=True)
    p.add_argument("--threshold", type=float, default=0.5); p.set_defaults(fn=cmd_calibrate)

    args = ap.parse_args(argv)
    _tracing_off_unless_asked()
    if args.env_file:
        n = load_env_file(args.env_file)
        print(f"(loaded {n} variable(s) from {args.env_file})", file=sys.stderr)
    args.fn(args)


def _tracing_off_unless_asked() -> None:
    """Eval runs make hundreds of model calls. A repo's .env often turns on
    third-party tracing (LangSmith), which would ship every test input to
    that service and burn its quota, without anyone deciding to. Off unless
    GOLDLOOP_TRACING=1. Set BEFORE --env-file loads, so the file can't flip it."""
    import os
    if os.environ.get("GOLDLOOP_TRACING") != "1":
        for k in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING"):
            os.environ[k] = "false"


def load_env_file(path: str) -> int:
    import os
    n = 0
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


_INIT_CONFIG = """# goldloop: what to test, against what, and what may ship.
suite: evals/cases
target:
  kind: python               # python | langgraph | http
  ref: my_agent.eval:run      # fn(input) or fn(input, params)
  params:
    model: openai:gpt-4o-mini
threshold: 0.8                # one run of a case passes at this score
repeat: 3                     # >1 measures stability (pass^k), not just accuracy
concurrency: 4
gate:
  min_mean: 0.8
  max_flaky: 0
  max_regressions: 0          # vs .goldloop/baseline.json
pricing: {}                   # e.g. {gpt-4o-mini: {input: 0.15, output: 0.60}}  $ per 1M tokens
tune:
  knob: prompt                # the param the tuner rewrites (params.prompt or params.prompt_file)
  optimizer_model: anthropic:claude-opus-5-5
  holdout: 0.3
"""

_INIT_CASE = """- id: example-refund-request
  tags: [routing]
  input: {message: "I was charged twice for my October order, please refund one."}
  checks:
    - {type: equals, path: route, value: billing, critical: true}   # wrong lane fails the case
    - {type: contains, path: reply, value: refund}
    - {type: absent, path: refund_issued}                          # negative control: never auto-refund
"""

if __name__ == "__main__":
    main()
