"""The Adapter interface every coding agent plugs into.

An Adapter's only job is: given a task description and a path to an
isolated worktree, make the agent do the work, and report back what changed
and what it cost. It must NOT decide whether the work is correct — that's
the gates' job. Keeping "did the work" and "was it good" as separate
components is what lets Verdict swap agents without touching verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from verdict.schema import AttemptResult


class Adapter(Protocol):
    """Implement this to plug a new coding agent into Verdict."""

    name: str

    def run(self, task: str, worktree: Path) -> AttemptResult:
        """Drive the agent on `task` inside `worktree`, then return its diff
        and token/cost accounting. Must not raise on the agent merely
        failing the task — only on the adapter itself being unable to run.
        """
        ...
