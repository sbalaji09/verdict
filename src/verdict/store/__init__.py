"""Phase 17: persist runs so Verdict is something a team returns to over
time, not just a per-invocation report. See `base.py`'s module docstring
for the `Store` Protocol and what actually gets persisted, and
`regression.py` for how persisted history feeds Phase 7's two-proportion
z-test to flag a regression against a historical baseline.
"""

from __future__ import annotations

from verdict.store.base import RunRecord, Store, TaskOutcome
from verdict.store.regression import DEFAULT_BASELINE_WINDOW, TaskRegression, detect_regressions
from verdict.store.sqlite_store import SQLiteStore

__all__ = [
    "Store",
    "TaskOutcome",
    "RunRecord",
    "SQLiteStore",
    "TaskRegression",
    "detect_regressions",
    "DEFAULT_BASELINE_WINDOW",
]
