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

from verdict.sandbox import Sandbox
from verdict.schema import AttemptResult


class AdapterError(RuntimeError):
    """Raised when an adapter itself couldn't run — the CLI binary is
    missing, crashed, or timed out — never for the agent merely failing
    the task (see `Adapter.run`'s docstring below). Every per-agent error
    class (`ClaudeCodeAdapterError`, `CursorAdapterError`, ...) subclasses
    this so `runner.py` can catch "the adapter itself is broken" as one
    thing, without hardcoding a per-adapter list that grows every time a
    new agent is added. Phase 11 routes this to `VerdictStatus.ERROR` —
    infra/tooling noise, excluded from pass-rate accounting — exactly like
    a `SandboxError`, since from the grader's point of view both mean the
    same thing: nothing about the agent's actual work was evaluated.
    """


class Adapter(Protocol):
    """Implement this to plug a new coding agent into Verdict."""

    name: str

    def run(self, task: str, worktree: Path, sandbox: Sandbox | None = None) -> AttemptResult:
        """Drive the agent on `task` inside `worktree`, then return its diff
        and token/cost accounting. Must not raise on the agent merely
        failing the task — only on the adapter itself being unable to run.

        `sandbox` is how the agent's own CLI actually executes (Phase 8) —
        None only from callers that haven't been threaded through
        explicitly (falls back to an unsafe local execution — see
        sandbox/config.py's `fallback_sandbox`); `runner.py`'s real entry
        points always pass one.
        """
        ...
