"""The improvement loop: promotion, tuning, LangGraph trajectories, CLI."""
import json
import os
import textwrap
from typing import TypedDict

import pytest
import yaml

from goldloop.cases import _expand, load_suite
from goldloop.promote import accept, promote
from goldloop.targets import LangGraphTarget
from goldloop.tune import tune

# A toy "model" whose behaviour is fully decided by the prompt text, so the
# tuner can be tested without an LLM. Rules in the prompt switch behaviours on.

def toy_extract(inp, params):
    p = params["prompt"]
    out = {}
    text = inp["text"].lower()
    if "doors" in text:
        out["doors"] = text.split("doors")[1].split()[0] if "LATEST WINS" not in p else text.split("doors")[-1].split()[0]
    if "food" in text or "catering" in text:
        if "no food" in text:
            out["catering"] = "none" if "NO IS AN ANSWER" in p else ""
        else:
            out["catering"] = "provided"
    if "FILL EVERYTHING" in p:
        out.setdefault("parking", "lot")      # a rule that 'fixes' by hallucinating
    return out


def case(cid, text, **kw):
    return _expand({"id": cid, "input": {"text": text}, **kw}, "t")


CASES = [
    case("plain", "doors 7 tonight", expect={"doors": "7"}),
    case("correction", "doors 7 ... sorry, doors 8", expect={"doors": "8"}),
    case("explicit-no", "there will be no food", expect={"catering": "none"}),
    case("no-parking-mentioned", "doors 9", expect={"doors": "9"}, forbid=["parking"]),
]
SPEC = {"kind": "python", "ref": "tests.test_loop:toy_extract", "params": {"prompt": "Extract fields. {FIELDS}"}}


def scripted(*proposals):
    """An optimizer that returns pre-written candidates, one batch per round."""
    batches = iter(proposals)

    def opt(champion, failures, ctx):
        opt.seen_failures.append([f["id"] for f in failures])
        return next(batches, [])
    opt.seen_failures = []
    return opt


def test_tuner_promotes_a_real_fix(tmp_path):
    opt = scripted([{"rationale": "latest statement wins", "text": "Extract fields. {FIELDS} LATEST WINS"}],
                   [{"rationale": "explicit no", "text": "Extract fields. {FIELDS} LATEST WINS NO IS AN ANSWER"}])
    rec = tune(CASES, SPEC, knob="prompt", optimizer=opt, iterations=2, k=1, repeat=1,
               out_dir=tmp_path, log=lambda *_: None)
    assert opt.seen_failures[0] == ["correction", "explicit-no"]
    assert rec["promoted"] and rec["final_mean"] == 1.0
    assert "NO IS AN ANSWER" in (tmp_path / "proposed_prompt.txt").read_text()
    assert "+Extract fields. {FIELDS} LATEST WINS NO IS AN ANSWER" in (tmp_path / "proposed_prompt.diff").read_text()


def test_tuner_rejects_hallucinating_fix_via_negative_control(tmp_path):
    opt = scripted([{"rationale": "fill everything", "text": "Extract fields. {FIELDS} LATEST WINS FILL EVERYTHING"}])
    rec = tune(CASES, SPEC, knob="prompt", optimizer=opt, iterations=1, k=1, repeat=1,
               out_dir=tmp_path, log=lambda *_: None)
    c = rec["rounds"][0]["candidates"][0]
    assert not c["accepted"] and "no-parking-mentioned" in c["reason"]
    assert not (tmp_path / "proposed_prompt.txt").exists()


def test_tuner_rejects_candidate_that_drops_a_placeholder(tmp_path):
    opt = scripted([{"rationale": "rewrite", "text": "Extract fields. LATEST WINS"}])
    rec = tune(CASES, SPEC, knob="prompt", optimizer=opt, iterations=1, k=1, repeat=1,
               out_dir=tmp_path, log=lambda *_: None)
    assert "placeholders" in rec["rounds"][0]["candidates"][0]["reason"]


def test_tuner_never_touches_the_source_file(tmp_path):
    pf = tmp_path / "prompt.txt"
    pf.write_text("Extract fields. {FIELDS}")
    spec = {**SPEC, "params": {"prompt_file": str(pf)}}
    opt = scripted([{"rationale": "x", "text": "Extract fields. {FIELDS} LATEST WINS"}])
    tune(CASES, spec, knob="prompt", optimizer=opt, iterations=1, k=1, repeat=1,
         out_dir=tmp_path / "t", log=lambda *_: None)
    assert pf.read_text() == "Extract fields. {FIELDS}"


# ── promotion ───────────────────────────────────────────────────────────────

def test_promote_shapes_feedback_and_dedupes(tmp_path):
    fb = tmp_path / "fb.jsonl"
    rows = [{"input": {"text": "no food"}, "output": {}, "corrections": {"catering": "none"}, "verdict": "edited"},
            {"input": {"text": "doors 9"}, "output": {"doors": "9", "parking": "lot"},
             "corrections": {"parking": None}, "verdict": "edited"},
            {"input": {"text": "no food"}, "corrections": {"catering": "none"}},       # dupe
            {"input": {"text": "doors 7 tonight"}, "corrections": {"doors": "7"}}]     # already in suite
    fb.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "cand.yaml"
    res = promote(str(fb), [CASES[0]], out)
    assert res["new"] == 2 and res["duplicates_skipped"] == 2
    cands = yaml.safe_load(out.read_text())
    assert cands[1]["checks"] == [{"type": "absent", "path": "parking"}]   # hallucination -> negative control
    assert all(c["review"] == "pending" for c in cands)


def test_accept_moves_only_reviewed_cases(tmp_path):
    cand = tmp_path / "cand.yaml"
    cand.write_text(yaml.safe_dump([
        {"id": "p1", "review": "ok", "was": "{}", "input": {"text": "x"}, "checks": [{"type": "absent", "path": "a"}]},
        {"id": "p2", "review": "pending", "input": {"text": "y"}, "checks": [{"type": "absent", "path": "a"}]},
        {"id": "p3", "review": "reject", "input": {"text": "z"}, "checks": []}]))
    suite = tmp_path / "cases" / "promoted.yaml"
    res = accept(cand, suite)
    assert res == {"accepted": 1, "rejected": 1, "still_pending": 1, "suite_file": str(suite)}
    loaded = load_suite(suite)
    assert [c.id for c in loaded] == ["p1"] and "review" not in yaml.safe_load(suite.read_text())[0]


# ── LangGraph ───────────────────────────────────────────────────────────────

def test_langgraph_target_records_the_path_taken():
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict, total=False):
        message: str
        lane: str
        reply: str

    g = StateGraph(S)
    g.add_node("route", lambda s: {"lane": "billing" if "charge" in s["message"] else "general"})
    g.add_node("billing", lambda s: {"reply": "refund staged for approval"})
    g.add_node("general", lambda s: {"reply": "hi"})
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: s["lane"], {"billing": "billing", "general": "general"})
    g.add_edge("billing", END)
    g.add_edge("general", END)
    t = LangGraphTarget(g.compile(), {})
    o = t.call({"message": "double charge"})
    assert o.trajectory == ["route", "billing"]
    assert o.output["reply"] == "refund staged for approval"


# ── CLI end to end ──────────────────────────────────────────────────────────

def test_cli_run_baseline_gate(tmp_path, monkeypatch, capsys):
    from goldloop.cli import main
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.py").write_text(textwrap.dedent("""
        import os
        def run(inp):
            return {"route": "billing", "reply": "we will refund" if not os.environ.get("BREAK") else "no"}
    """))
    (tmp_path / "goldloop.yaml").write_text("suite: cases\ntarget: {kind: python, ref: 'agent:run'}\n")
    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "c.yaml").write_text(yaml.safe_dump([
        {"id": "refund", "input": {"m": "x"}, "checks": [
            {"type": "equals", "path": "route", "value": "billing", "critical": True},
            {"type": "contains", "path": "reply", "value": "refund"}]}]))
    main(["run", "-q"])
    main(["baseline", "latest"])
    with pytest.raises(SystemExit) as e:
        main(["gate", "-q"])
    assert e.value.code == 0
    monkeypatch.setenv("BREAK", "1")
    with pytest.raises(SystemExit) as e:
        main(["gate", "-q", "--markdown", "pr.md"])
    assert e.value.code == 1
    assert "regressed" in (tmp_path / "pr.md").read_text()
    main(["report", "--out", "r.html"])
    assert "goldloop report" in (tmp_path / "r.html").read_text()


HELD: set = set()


def holdout_extract(inp, params):
    """Base prompt: a and c right, b wrong (0.67). Candidate 'TRAIN': perfect on
    train cases, but on held-out cases it loses c (0.33): mean drops. At
    threshold 0.8 the champion fails holdout too, so no case flips: only the mean drops."""
    over = "TRAIN" in params["prompt"]
    if not over:
        return {"a": "yes", "b": "", "c": "yes"}
    if inp["id"] in HELD:
        return {"a": "yes", "b": "", "c": ""}
    return {"a": "yes", "b": "yes", "c": "yes"}


def test_holdout_mean_drop_rejects_even_without_case_regressions(tmp_path):
    from goldloop.cases import split
    cases = [_expand({"id": f"c{i}", "input": {"id": f"c{i}"}, "expect": {"a": "yes", "b": "yes", "c": "yes"}}, "t")
             for i in range(30)]
    HELD.clear()
    HELD.update(c.id for c in split(cases, 0.3)[1])
    assert HELD
    spec = {"kind": "python", "ref": "tests.test_loop:holdout_extract", "params": {"prompt": "base"}}
    opt = scripted([{"rationale": "overfit", "text": "base TRAIN"}])
    rec = tune(cases, spec, knob="prompt", optimizer=opt, iterations=1, k=1, repeat=1, holdout=0.3,
               threshold=0.8, out_dir=tmp_path, log=lambda *_: None)
    cand = rec["rounds"][0]["candidates"][0]
    assert cand["regressions"] == [] and cand["mean"] == 1.0          # looked great on train
    assert cand["holdout"]["candidate_mean"] < cand["holdout"]["champion_mean"]
    assert not rec["promoted"]


def test_cli_turns_third_party_tracing_off(tmp_path, monkeypatch):
    import os
    from goldloop.cli import main
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOLDLOOP_TRACING", raising=False)
    (tmp_path / ".env").write_text("LANGCHAIN_TRACING_V2=true\nSOME_KEY=x\n")
    main(["--env-file", ".env", "runs"])          # any command will do; we only check the environment
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false" and os.environ["SOME_KEY"] == "x"


def test_drift_flags_a_silent_model_swap(tmp_path, monkeypatch, capsys):
    from goldloop.cli import main
    monkeypatch.chdir(tmp_path)
    (tmp_path / "drift_agent.py").write_text(textwrap.dedent("""
        import os
        from goldloop import Result
        def run(inp):
            m = os.environ.get("SERVED", "model-2024-07-18")
            return Result({"a": "ok"}, usage={"input_tokens": 5, "output_tokens": 1,
                          "by_model": {m: {"input_tokens": 5, "output_tokens": 1}}})
    """))
    (tmp_path / "goldloop.yaml").write_text("suite: cases\ntarget: {kind: python, ref: 'drift_agent:run'}\n")
    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "c.yaml").write_text(yaml.safe_dump([{"id": "x", "input": 1, "expect": {"a": "ok"}}]))
    main(["run", "-q"]); main(["baseline", "latest"])
    with pytest.raises(SystemExit) as e:
        main(["drift", "-q"])
    assert e.value.code == 0
    monkeypatch.setenv("SERVED", "model-2026-09-30")
    with pytest.raises(SystemExit) as e:
        main(["drift", "-q"])
    assert e.value.code == 1 and "provider model changed" in capsys.readouterr().out
