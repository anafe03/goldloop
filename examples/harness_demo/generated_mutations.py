#!/usr/bin/env python3
"""Mutations nobody chose: rewrite the harness mechanically, one change at a
time, and run the whole suite against each version.

mutation_check.py removes safeguards I picked, so it can only confirm what I
already thought to guard. This script makes small changes by rule instead:

    ==  <->  !=        <  <->  >=        >  <->  <=        in  <->  not in
    and <->  or        not x  ->  x      True <-> False    number n -> n + 1

A mutant the suite catches is "killed". One that survives either changes
nothing that matters (an equivalent mutant) or exposes a gap in the tests.
Survivors are listed so a person can tell which.

    python generated_mutations.py                 # policy, ledger, memory, inference
    python generated_mutations.py --files policy.py --limit 40
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWAP_CMP = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE, ast.GtE: ast.Lt,
            ast.Gt: ast.LtE, ast.LtE: ast.Gt, ast.In: ast.NotIn, ast.NotIn: ast.In}

RUNNER = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from goldloop.cases import load_suite
from goldloop.runner import run_suite
from goldloop.targets import PythonTarget
import eval_target
cases = load_suite(sys.argv[1] + "/cases")
params = {"model": "rules", "fallback": "rules", "idempotent": True, "spotlight": True}
run = run_suite(cases, PythonTarget(eval_target.run, params), threshold=1.0, concurrency=4)
print(json.dumps(sorted(c for c, v in run["cases"].items() if not v["summary"]["passed_all"])))
"""


def sites(tree: ast.AST):
    """Every place a rule applies, as (node, kind, description)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in SWAP_CMP:
                    yield node, ("cmp", i), f"{type(op).__name__} -> {SWAP_CMP[type(op)].__name__}"
        elif isinstance(node, ast.BoolOp):
            yield node, ("bool", 0), f"{type(node.op).__name__} -> {'Or' if isinstance(node.op, ast.And) else 'And'}"
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            yield node, ("not", 0), "drop `not`"
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            yield node, ("bool_const", 0), f"{node.value} -> {not node.value}"
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float) and not isinstance(node.value, bool):
            yield node, ("num", 0), f"{node.value} -> {node.value + 1}"


def mutate(source: str, index: int) -> tuple[str, str, int]:
    tree = ast.parse(source)
    targets = list(sites(tree))
    node, (kind, i), desc = targets[index]
    if kind == "cmp":
        node.ops[i] = SWAP_CMP[type(node.ops[i])]()
    elif kind == "bool":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif kind == "not":
        node.operand = ast.UnaryOp(op=ast.Not(), operand=node.operand)  # `not x` -> `not not x`, i.e. x
    elif kind == "bool_const":
        node.value = not node.value
    elif kind == "num":
        node.value = node.value + 1
    return ast.unparse(ast.fix_missing_locations(tree)), desc, getattr(node, "lineno", 0)


def run_suite_on(copy: Path) -> list[str] | None:
    p = subprocess.run([sys.executable, "-c", RUNNER, str(copy)], capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        return None                     # the mutant doesn't even import: counts as killed
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=["policy.py", "ledger.py", "memory.py", "inference.py"])
    ap.add_argument("--limit", type=int, default=0, help="max mutants per file (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=".goldloop/generated_mutations.json")
    args = ap.parse_args()

    jobs = []
    for name in args.files:
        src = (HERE / "harness" / name).read_text()
        n = len(list(sites(ast.parse(src))))
        idx = list(range(n))
        if args.limit and n > args.limit:
            step = n / args.limit
            idx = [int(k * step) for k in range(args.limit)]       # evenly spread, deterministic
        jobs += [(name, src, k) for k in idx]

    def one(job):
        name, src, k = job
        mutated, desc, line = mutate(src, k)
        with tempfile.TemporaryDirectory() as d:
            copy = Path(d) / "demo"
            shutil.copytree(HERE, copy, ignore=shutil.ignore_patterns(".goldloop", "__pycache__", "prompts"))
            (copy / "harness" / name).write_text(mutated)
            failing = run_suite_on(copy)
        killed = failing is None or len(failing) > 0
        return {"file": name, "line": line, "change": desc, "killed": killed,
                "caught_by": failing if failing else ([] if failing is not None else ["(does not import)"])}

    print(f"{len(jobs)} mutants across {', '.join(args.files)} …", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, jobs))
    by_file = {}
    for r in results:
        f = by_file.setdefault(r["file"], [0, 0])
        f[0] += r["killed"]
        f[1] += 1
    killed = sum(r["killed"] for r in results)
    print(f"\n{'file':<14} killed")
    for f, (k, n) in by_file.items():
        print(f"{f:<14} {k}/{n} = {k / n:.0%}")
    print(f"{'total':<14} {killed}/{len(results)} = {killed / len(results):.0%}")
    print("\nSurvivors (each is either harmless or a missing test):")
    for r in results:
        if not r["killed"]:
            print(f"  {r['file']}:{r['line']}  {r['change']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
