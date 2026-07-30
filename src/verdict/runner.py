"""Orchestrates one end-to-end grading run: isolate a worktree, let the
adapter do the work, run the gates, assemble a Verdict.
"""

from __future__ import annotations

from pathlib import Path

from verdict.adapters import Adapter
from verdict.gates.test_gate import run_test_gate
from verdict.schema import Verdict
from verdict.worktree import isolated_worktree


def run(
    task: str,
    repo: Path,
    adapter: Adapter,
    test_command: str | None = None,
) -> Verdict:
    """Run `adapter` on `task` inside a throwaway worktree of `repo`, grade
    the result against the test gate, and return the Verdict. The worktree
    (and its branch) is always torn down before this returns, win or lose.
    """
    with isolated_worktree(repo) as worktree:
        attempt = adapter.run(task, worktree.path)
        signal = run_test_gate(worktree.path, command=test_command)

    return Verdict(
        task=task,
        agent=adapter.name,
        repo=str(repo),
        attempt=attempt,
        signals=[signal],
    )
