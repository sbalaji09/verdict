"""Orchestrates one end-to-end grading run: isolate a worktree, let the
adapter do the work, run the gates, assemble a Verdict.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from verdict.adapters import Adapter
from verdict.config import load_config
from verdict.gates.registry import run_all_gates
from verdict.schema import Verdict
from verdict.worktree import isolated_worktree

# Dependency directories that live outside git (installed, not committed —
# correctly so) but that build/typecheck/lint gates need present to run at
# all. `git worktree` only checks out tracked files, so a fresh worktree
# never has these; without this, an npm-based repo would report every gate
# N/A (or a misleading FAIL from a missing binary) on every single run.
# Copied rather than symlinked so the agent can freely reinstall/mutate
# without ever touching the source repo's copy — same isolation guarantee
# worktree.py already gives tracked files, extended to this untracked one.
_VENDORED_DEPENDENCY_DIRS = ("node_modules",)


def _copy_vendored_dependencies(repo: Path, worktree: Path) -> None:
    for name in _VENDORED_DEPENDENCY_DIRS:
        source = repo / name
        if source.is_dir() and not (worktree / name).exists():
            shutil.copytree(source, worktree / name, symlinks=True)


def run(
    task: str,
    repo: Path,
    adapter: Adapter,
) -> Verdict:
    """Run `adapter` on `task` inside a throwaway worktree of `repo`, grade
    the result against every applicable gate (test/typecheck/build/lint),
    and return the Verdict. The worktree (and its branch) is always torn
    down before this returns, win or lose.
    """
    with isolated_worktree(repo) as worktree:
        _copy_vendored_dependencies(repo, worktree.path)
        attempt = adapter.run(task, worktree.path)
        config = load_config(worktree.path)
        signals = run_all_gates(worktree.path, config)

    return Verdict(
        task=task,
        agent=adapter.name,
        repo=str(repo),
        attempt=attempt,
        signals=signals,
    )
