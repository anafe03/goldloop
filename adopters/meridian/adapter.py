"""Meridian (health-insurance support agent) on goldloop, read from outside (MERIDIAN_ROOT)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from routed_agent import RoutedAgent  # noqa: E402

A = RoutedAgent(Path(os.environ.get("MERIDIAN_ROOT") or Path(__file__).resolve().parents[3] / "meridian-agent"), "triage")
cases, run = A.cases, A.run
