"""TripTalk (travel concierge agent) on goldloop, read from outside (TRIPTALK_ROOT)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from routed_agent import RoutedAgent  # noqa: E402

A = RoutedAgent(Path(os.environ.get("TRIPTALK_ROOT") or Path(__file__).resolve().parents[3] / "triptalk"), "triptalk")
cases, run = A.cases, A.run
