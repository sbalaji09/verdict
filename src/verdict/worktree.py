"""Isolated git worktrees: give an agent a real checkout without ever
touching the user's actual working tree.

We use `git worktree` rather than `git clone` because a worktree shares the
repo's object store (no copying gigabytes of history) while still giving the
agent a fully independent directory and index to mutate. Each attempt gets
its own throwaway branch off the current HEAD, and both the branch and the
directory are removed on cleanup, success or failure.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class WorktreeError(RuntimeError):
    """Raised when git worktree setup or teardown fails."""


@dataclass
class Worktree:
    path: Path
    branch: str


def _run_git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise WorktreeError(
            f"git {' '.join(args)} failed in {cwd}:\n{result.stderr.strip()}"
        )
    return result.stdout


def _assert_is_git_repo(repo: Path) -> None:
    if not repo.exists():
        raise WorktreeError(f"repo path does not exist: {repo}")
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise WorktreeError(f"not a git repository: {repo}")


def diff_against_base(worktree_path: Path) -> tuple[str, list[str]]:
    """Capture everything an agent changed in `worktree_path` relative to
    the commit it started from — staged, unstaged, and committed on the
    throwaway branch alike — as a unified diff plus a file list.
    """
    _run_git("add", "-A", cwd=worktree_path)
    diff = _run_git("diff", "--cached", "HEAD", cwd=worktree_path)
    files_raw = _run_git(
        "diff", "--cached", "--name-only", "HEAD", cwd=worktree_path
    )
    files = [f for f in files_raw.splitlines() if f]
    return diff, files


@contextmanager
def isolated_worktree(repo: Path) -> Iterator[Worktree]:
    """Check out `repo`'s HEAD into a fresh temp directory on a throwaway
    branch, yield it, then remove both — regardless of what happens inside.
    """
    repo = repo.resolve()
    _assert_is_git_repo(repo)

    run_id = uuid.uuid4().hex[:8]
    branch = f"verdict/{run_id}"
    parent_dir = Path(tempfile.mkdtemp(prefix="verdict-worktree-"))
    worktree_path = parent_dir / "worktree"

    _run_git(
        "worktree", "add", "-b", branch, str(worktree_path), "HEAD",
        cwd=repo,
    )

    try:
        yield Worktree(path=worktree_path, branch=branch)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(parent_dir, ignore_errors=True)
