"""The learning loop keeps corrections that help on unseen tasks and rejects
the two that would do harm: a one-off that overgeneralizes, and one that
breaks the discount rule."""
import json
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent / "examples" / "learning_assistant"


def test_gated_learning_keeps_the_good_and_rejects_the_harmful(tmp_path):
    subprocess.run([sys.executable, "learn.py"], cwd=DEMO, check=True, capture_output=True)
    log = json.loads((DEMO / ".goldloop" / "learning_log.json").read_text())
    kept = {d["correction"] for d in log["decisions"] if d["kept"]}
    rejected = {d["correction"]: d["why"] for d in log["decisions"] if not d["kept"]}
    assert kept == {"c1", "c2", "c3", "c4", "c7"}
    assert set(rejected) == {"c5", "c6"}
    assert "order-ready-today" in rejected["c5"] and "complaint-discount-capped" in rejected["c6"]
    assert log["after"] == log["n_tasks"] and log["before"] < log["after"]
    assert sorted(log["naive_violations"]) == ["complaint-discount-capped", "order-ready-today"]
    # every kept item can be traced to the correction it came from
    assert all(d["item"]["source"] == d["correction"] for d in log["decisions"])
