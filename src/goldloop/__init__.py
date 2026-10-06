"""goldloop: golden-set evals for LLM agents, with a closed improvement loop."""
from .cases import Case, load_suite
from .checks import Observation, register
from .compare import compare, gate
from .runner import run_suite
from .targets import Result, Unavailable, make_target

__all__ = ["Case", "load_suite", "Observation", "register", "compare", "gate", "run_suite",
           "Result", "Unavailable", "make_target"]
__version__ = "0.1.0"
