"""Phase 5: turn Verdict from a one-shot checker into a scorecard — point
at a folder of tasks, run every agent/config across all of them, rank the
result. See DESIGN.md's Phase 5 section for the task/acceptance-criteria
format and the suite runner's design.
"""

from __future__ import annotations

from verdict.suite.executor import Executor, LocalProcessPoolExecutor, SerialExecutor
from verdict.suite.loader import SuiteLoadError, SuiteTask, load_suite
from verdict.suite.runner import BenchConfig, run_suite

__all__ = [
    "SuiteLoadError",
    "SuiteTask",
    "load_suite",
    "BenchConfig",
    "run_suite",
    "Executor",
    "SerialExecutor",
    "LocalProcessPoolExecutor",
]
