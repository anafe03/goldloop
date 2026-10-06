import itertools
import json
import threading

import pytest
import yaml

from goldloop.cases import CaseError, _expand, load_suite, split, suite_fingerprint
from goldloop.checks import Observation, run_check, score_case
from goldloop.compare import compare, gate, mcnemar_p
from goldloop.metrics import wilson
from goldloop.runner import run_suite
from goldloop.targets import PythonTarget, Unavailable, is_infra_error


# ── cases ───────────────────────────────────────────────────────────────────

def test_shorthands_expand_to_checks():
    c = _expand({"id": "a", "input": {}, "expect": {"x.y": "8"}, "forbid": ["z"]}, "t")
    assert {"type": "contains", "path": "x.y", "value": "8"} in c.checks
    assert {"type": "absent", "path": "z"} in c.checks


def test_case_without_checks_is_rejected():
    # A case with no checks always passes: worse than no case.
    with pytest.raises(CaseError):
        _expand({"id": "a", "input": {}}, "t")


def test_duplicate_ids_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump([{"id": "x", "input": 1, "expect": {"": "1"}}]))
    (tmp_path / "b.jsonl").write_text(json.dumps({"id": "x", "input": 2, "expect": {"": "2"}}))
    with pytest.raises(CaseError, match="duplicate"):
        load_suite(tmp_path)


def test_split_is_deterministic_and_disjoint():
    cases = [_expand({"id": f"c{i}", "input": i, "expect": {"": str(i)}}, "t") for i in range(200)]
    a1, h1 = split(cases, 0.3)
    a2, h2 = split(cases, 0.3)
    assert [c.id for c in h1] == [c.id for c in h2]
    assert not {c.id for c in a1} & {c.id for c in h1}
    assert 40 < len(h1) < 80


def test_suite_fingerprint_changes_when_a_check_changes():
    a = [_expand({"id": "x", "input": 1, "expect": {"": "1"}}, "t")]
    b = [_expand({"id": "x", "input": 1, "expect": {"": "2"}}, "t")]
    assert suite_fingerprint(a) != suite_fingerprint(b)


# ── checks ──────────────────────────────────────────────────────────────────

def obs(out, traj=()):
    return Observation(out, list(traj))


def test_contains_is_case_insensitive_and_partial_for_lists():
    r = run_check(obs({"wifi": "VenueGuest"}), {"type": "contains", "path": "wifi", "value": "venueguest"})
    assert r["score"] == 1
    r = run_check(obs({"t": "doors 8"}), {"type": "contains", "path": "t", "value": ["8", "9:30"]})
    assert r["score"] == 0.5


def test_absent_is_a_negative_control():
    assert run_check(obs({"catering": ""}), {"type": "absent", "path": "catering"})["score"] == 1
    assert run_check(obs({}), {"type": "absent", "path": "catering"})["score"] == 1
    assert run_check(obs({"catering": "sandwiches"}), {"type": "absent", "path": "catering"})["score"] == 0


def test_critical_failure_zeroes_the_case():
    rs = [run_check(obs({"kind": "answer", "a": "1"}), {"type": "equals", "path": "kind", "value": "problem", "critical": True}),
          run_check(obs({"kind": "answer", "a": "1"}), {"type": "contains", "path": "a", "value": "1"})]
    assert score_case(rs) == 0.0


def test_one_of_accepts_ambiguous_answers_including_null():
    c = {"type": "one_of", "path": "u", "values": ["low", "medium"]}
    assert run_check(obs({"u": "Medium"}), c)["score"] == 1
    assert run_check(obs({"u": "high"}), c)["score"] == 0
    assert run_check(obs({"name": None}), {"type": "one_of", "path": "name", "values": [None, "Acme"]})["score"] == 1


def test_path_into_lists():
    assert run_check(obs({"m": [{"c": "a"}, {"c": "hello"}]}), {"type": "contains", "path": "m.-1.c", "value": "hell"})["score"] == 1


def test_trajectory_checks():
    t = ["route", "billing", "approve", "commit"]
    assert run_check(obs({}, t), {"type": "sequence", "nodes": ["route", "approve", "commit"]})["score"] == 1
    assert run_check(obs({}, t), {"type": "sequence", "nodes": ["commit", "approve"]})["score"] == 0
    assert run_check(obs({}, t), {"type": "not_visited", "node": "refund"})["score"] == 1


# ── runner + metrics ────────────────────────────────────────────────────────

def cases_of(*ids):
    return [_expand({"id": i, "input": i, "expect": {"": "ok"}}, "t") for i in ids]


def test_flaky_case_is_caught_by_repeats():
    flips = itertools.cycle(["ok", "no"])
    lock = threading.Lock()

    def fn(x):
        if x == "wobbly":
            with lock:
                return next(flips)
        return "ok"
    run = run_suite(cases_of("steady", "wobbly"), PythonTarget(fn, {}), repeat=4, concurrency=1)
    m = run["metrics"]
    assert m["flaky"] == ["wobbly"]
    assert m["pass_all"] == 0.5            # pass^k: wobbly never passes EVERY run
    assert m["mean"] == pytest.approx(0.75)


def test_unavailable_is_not_a_score():
    def fn(x):
        if x == "b":
            raise Unavailable("429")
        return "ok"
    run = run_suite(cases_of("a", "b"), PythonTarget(fn, {}), repeat=2)
    m = run["metrics"]
    assert m["mean"] == 1.0 and m["n_scored"] == 1
    assert not m["complete"] and m["unavailable_runs"] == 2


def test_target_crash_is_a_failure_not_unavailable():
    def fn(x):
        raise KeyError("updates")
    run = run_suite(cases_of("a"), PythonTarget(fn, {}))
    assert run["metrics"]["errors"] == ["a"] and run["metrics"]["mean"] == 0
    assert run["metrics"]["complete"]


def test_infra_errors_detected_by_name_and_status():
    class RateLimitError(Exception): ...
    class Boom(Exception):
        status_code = 503
    assert is_infra_error(RateLimitError()) and is_infra_error(Boom())
    assert not is_infra_error(ValueError("bad json"))


def test_target_receives_params_when_it_takes_two_args():
    seen = {}

    def fn(x, params):
        seen.update(params)
        return "ok"
    run_suite(cases_of("a"), PythonTarget(fn, {"model": "m1"}))
    assert seen == {"model": "m1"}


def test_wilson_interval_is_honest_about_small_n():
    lo, hi = wilson(41, 42)
    assert lo < 0.88 and hi > 0.99


# ── compare + gate ──────────────────────────────────────────────────────────

def _run(results: dict, rid="r"):
    fn = lambda x: "ok" if results[x] else "no"
    run = run_suite(cases_of(*results), PythonTarget(fn, {}))
    run["id"] = rid
    return run


def test_mcnemar_exact():
    assert mcnemar_p(0, 1) == 1.0
    assert mcnemar_p(0, 6) == pytest.approx(0.03125)
    assert mcnemar_p(3, 3) == 1.0


def test_compare_finds_case_level_regressions_hidden_by_a_flat_mean():
    base = _run({"a": True, "b": False, "c": True}, "base")
    cand = _run({"a": False, "b": True, "c": True}, "cand")
    cmp = compare(base, cand)
    assert cmp["delta"]["mean"] == 0
    assert cmp["regressions"] == ["a"] and cmp["fixes"] == ["b"]
    assert cmp["verdict"].startswith("mixed")


def test_one_fixed_case_is_not_called_significant():
    base = _run({f"c{i}": i != 0 for i in range(42)}, "base")
    cand = _run({f"c{i}": True for i in range(42)}, "cand")
    cmp = compare(base, cand)
    assert cmp["fixes"] == ["c0"]
    assert "too few" in cmp["verdict"]


def test_compare_skips_cases_whose_checks_changed():
    base = _run({"a": True}, "base")
    cand = _run({"a": False}, "cand")
    cand["cases"]["a"]["fp"] = "different"
    cmp = compare(base, cand)
    assert cmp["changed_checks"] == ["a"] and not cmp["regressions"]


def test_gate_blocks_regression_and_incomplete_runs():
    base = _run({"a": True, "b": True}, "base")
    cand = _run({"a": True, "b": False}, "cand")
    ok, reasons, _ = gate(cand, base, {})
    assert not ok and "regressed" in reasons[0]
    cand2 = _run({"a": True, "b": True}, "cand2")
    cand2["metrics"]["complete"] = False
    cand2["metrics"]["unavailable_runs"] = 3
    ok, reasons, _ = gate(cand2, base, {})
    assert not ok and "incomplete" in reasons[0]


def test_gate_absolute_policy():
    run = _run({"a": True, "b": False}, "r")
    ok, reasons, _ = gate(run, None, {"min_mean": 0.8})
    assert not ok and "mean" in reasons[0]


def raw_cases():
    return [{"id": "from-code", "input": 1, "expect": {"": "1"}}]


def test_suite_can_come_from_a_function(tmp_path):
    (tmp_path / "more.yaml").write_text(yaml.safe_dump([{"id": "from-file", "input": 2, "expect": {"": "2"}}]))
    cases = load_suite(["tests.test_core:raw_cases", str(tmp_path)])
    assert [c.id for c in cases] == ["from-code", "from-file"]


def test_event_checks_grade_the_trace():
    evs = [{"name": "policy.decision", "attrs": {"tool": "issue_refund", "decision": "ask"}},
           {"name": "approval.granted", "attrs": {"tool": "issue_refund"}},
           {"name": "side_effect.commit", "attrs": {"tool": "issue_refund"}}]
    o = Observation({}, events=evs)
    assert run_check(o, {"type": "event_present", "name": "policy.decision", "match": {"decision": "ASK"}})["score"] == 1
    assert run_check(o, {"type": "event_absent", "name": "side_effect.commit", "match": {"tool": "delete_account"}})["score"] == 1
    assert run_check(o, {"type": "event_count", "name": "side_effect.commit", "value": 1})["score"] == 1
    assert run_check(o, {"type": "event_order", "steps": [{"name": "approval.granted"}, {"name": "side_effect.commit"}]})["score"] == 1
    assert run_check(o, {"type": "event_order", "steps": [{"name": "side_effect.commit"}, {"name": "approval.granted"}]})["score"] == 0


def test_lint_flags_vacuous_cases():
    from goldloop.cases import lint
    only_neg = _expand({"id": "n", "input": 1, "forbid": ["x"]}, "t")
    mixed = _expand({"id": "m", "input": 2, "expect": {"y": "1"}, "forbid": ["x"]}, "t")
    dupe = _expand({"id": "d", "input": 2, "expect": {"y": "2"}}, "t")
    problems = lint([only_neg, mixed, dupe])
    assert any(p.startswith("n:") for p in problems) and any("same input as m" in p for p in problems)
    assert not any(p.startswith("m:") for p in problems)


def test_empty_account_400_is_an_outage_not_a_score():
    class AnthropicInvalidRequestError(Exception):
        status_code = 400
    e = AnthropicInvalidRequestError("Error code: 400 - Your credit balance is too low to access the Anthropic API.")
    assert is_infra_error(e)
    assert not is_infra_error(type("BadRequestError", (Exception,), {"status_code": 400})("max_tokens too large"))


def test_missing_param_file_fails_loudly():
    from goldloop.targets import _load_param_files
    with pytest.raises(FileNotFoundError):
        _load_param_files({"prompt_file": "prompts/typo.txt"})
