#!/usr/bin/env python3
"""Does goldloop grade TM GO cases exactly like run_evals.py does?

Offline, no model: for every golden case, build synthetic outputs (perfect,
empty, partial, hallucinating, wrong kind, missed redirect), score each with
run_evals' own logic and with goldloop's checks, and report every case where
pass/fail differs. A port that changes the verdict is a new exam, not a port.

    python check_parity.py        (from adopters/tmgo)
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from goldloop.cases import _expand                      # noqa: E402
from goldloop.checks import Observation, run_check, score_case   # noqa: E402

from adapter import rpe, to_goldloop                    # noqa: E402

THRESHOLD = 0.8


def tmgo_score(case, out):
    """llm_golden's grading, verbatim in behaviour."""
    s = rpe.score(out["updates"], case["expect"], case.get("forbid", ()))
    ek = case.get("expect_kind")
    if ek and str(out.get("kind") or "").lower() != ek:
        s = 0.0
    er = case.get("expect_redirect")
    if er and er.lower() not in str(out.get("redirect_to") or "").lower():
        s = min(s, 0.5)
    return s


def variants(case):
    perfect = {"kind": case.get("expect_kind") or "answer", "redirect_to": case.get("expect_redirect", ""),
               "updates": {k: f"it's {v}" for k, v in case["expect"].items()}}
    yield "perfect", perfect
    yield "empty", {"kind": "answer", "redirect_to": "", "updates": {}}
    if len(case["expect"]) > 1:
        k = next(iter(case["expect"]))
        yield "partial", {**perfect, "updates": {k: case["expect"][k]}}
    if case.get("forbid"):
        yield "hallucinates", {**perfect, "updates": {**perfect["updates"],
                                                      **{f: "made up" for f in case["forbid"]}}}
    if case.get("expect_kind"):
        yield "wrong-kind", {**perfect, "kind": "other" if case["expect_kind"] != "other" else "answer"}
    if case.get("expect_redirect"):
        yield "missed-redirect", {**perfect, "redirect_to": ""}


def main():
    disagree, n = [], 0
    for case in rpe.GOLDEN:
        gl = _expand(to_goldloop(case), "parity")
        for name, out in variants(case):
            n += 1
            a = tmgo_score(case, out)
            b = score_case([run_check(Observation(out), c) for c in gl.checks])
            if (a >= THRESHOLD) != (b >= THRESHOLD):
                disagree.append((gl.id, name, a, b))
    print(f"{len(rpe.GOLDEN)} cases, {n} synthetic outputs scored by both harnesses")
    for cid, name, a, b in disagree:
        print(f"  ≠ {cid} [{name}]: run_evals {a:.2f}  goldloop {b:.2f}")
    print("pass/fail parity ✓" if not disagree else f"✗ {len(disagree)} verdicts differ")
    sys.exit(1 if disagree else 0)


if __name__ == "__main__":
    main()
