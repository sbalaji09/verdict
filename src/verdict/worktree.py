"""Isolated git worktrees: give an agent a real checkout without ever
touching the user's actual working tree.

We use `git worktree` rather than `git clone` because a worktree shares the
repo's object store (no copying gigabytes of history) while still giving the
agent a fully independent directory and index to mutate. Each attempt gets
its own throwaway branch off the current HEAD, and both the branch and the
directory are removed on cleanup, success or failure.

This module also carries the small git plumbing helpers Phase 2's
attribution engine needs (commit ladders, detached scratch checkouts,
cherry-picking a single path's content from another commit) — they're
generic git operations, not attribution-specific logic, so they live next
to the isolation primitive they're built on rather than in `attribution/`.
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


# Dependency directories that live outside git (installed, not committed —
# correctly so) but that build/typecheck/lint gates — and, since Phase 4,
# the frontend dev server — need present to run at all. `git worktree` only
# checks out tracked files, so a fresh worktree never has these. Copied
# rather than symlinked so an agent (or a dev server it starts) can freely
# reinstall/mutate without ever touching the source repo's copy.
VENDORED_DEPENDENCY_DIRS = ("node_modules",)


def copy_vendored_dependencies(repo: Path, worktree_path: Path) -> None:
    for name in VENDORED_DEPENDENCY_DIRS:
        source = repo / name
        if source.is_dir() and not (worktree_path / name).exists():
            shutil.copytree(source, worktree_path / name, symlinks=True)


@dataclass
class Worktree:
    path: Path
    branch: str
    base_commit: str
    """The commit the worktree branched from — i.e. the repo's state
    *before* the agent touched anything. Attribution needs this to check
    whether a failure already existed prior to the attempt.
    """


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


def run_git(*args: str, cwd: Path) -> str:
    """Public escape hatch for git plumbing this module doesn't already
    wrap (e.g. `git bisect start/bad/good`) — same error handling as the
    internal helper, exposed for `attribution/` to build on.
    """
    return _run_git(*args, cwd=cwd)


def rev_parse(repo: Path, ref: str) -> str:
    return _run_git("rev-parse", ref, cwd=repo).strip()


def diff_against_base(worktree_path: Path, base_commit: str) -> tuple[str, list[str]]:
    """Capture everything an agent changed in `worktree_path` relative to
    `base_commit` — staged, unstaged, and committed on the throwaway branch
    alike — as a unified diff plus a file list.

    Diffs against the recorded base commit, not `HEAD`: if the agent made
    real commits during its run, `HEAD` has moved to point at them, and a
    naive `diff --cached HEAD` would only see trailing uncommitted scraps
    and silently miss everything already committed.
    """
    _run_git("add", "-A", cwd=worktree_path)
    diff = _run_git("diff", "--cached", base_commit, cwd=worktree_path)
    files_raw = _run_git("diff", "--cached", "--name-only", base_commit, cwd=worktree_path)
    files = [f for f in files_raw.splitlines() if f]
    return diff, files


def diff_between(repo: Path, base: str, final: str) -> tuple[str, list[str]]:
    """A read-only counterpart to `diff_against_base`, for Phase 6's
    merge-gate mode: grading a repo that's already checked out at `final`
    (a real commit — a PR's tip — not an agent's possibly-uncommitted
    edits), so there's nothing to stage. Unlike `diff_against_base`, this
    never runs `git add -A` — mutating the index of a repo Verdict didn't
    create (a CI runner's own checkout, not a throwaway worktree) would be
    a real, unwanted side effect for a mode whose whole point is to grade
    the repo *as found*.
    """
    diff = _run_git("diff", base, final, cwd=repo)
    files_raw = _run_git("diff", "--name-only", base, final, cwd=repo)
    files = [f for f in files_raw.splitlines() if f]
    return diff, files


def commit_all(worktree_path: Path, message: str) -> str:
    """Stage and commit everything currently in the worktree (including
    uncommitted agent edits), then return the resulting commit SHA. A no-op
    (returns the current HEAD SHA unchanged) if there's nothing to commit —
    e.g. the agent already committed everything itself.
    """
    _run_git("add", "-A", cwd=worktree_path)
    status = _run_git("status", "--porcelain", cwd=worktree_path)
    if status.strip():
        _run_git("commit", "-q", "-m", message, cwd=worktree_path)
    return rev_parse(worktree_path, "HEAD")


def commits_between(repo: Path, base: str, final: str) -> list[str]:
    """Real commits the agent made, oldest first. Empty if `final` has no
    commits beyond `base` (e.g. everything was captured in one
    `commit_all` call with nothing pre-committed by the agent).
    """
    if base == final:
        return []
    out = _run_git("rev-list", "--reverse", f"{base}..{final}", cwd=repo)
    return [line for line in out.splitlines() if line]


def parent_commit(repo: Path, sha: str) -> str:
    return rev_parse(repo, f"{sha}^")


def changed_files(repo: Path, a: str, b: str) -> list[str]:
    out = _run_git("diff", "--name-only", a, b, cwd=repo)
    return [line for line in out.splitlines() if line]


def file_content_at(repo: Path, ref: str, path: str) -> str | None:
    """`path`'s content at `ref`, or `None` if it doesn't exist there —
    Phase 12's integrity check reads a test file's before/after content
    this way (`git show <ref>:<path>`) rather than checking anything out,
    since it only needs to compare text, never to actually run either
    version.
    """
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def checkout_path_from(worktree_path: Path, commit: str, path: str) -> bool:
    """Overwrite `path` in `worktree_path` with its content at `commit`,
    handling the case where `path` doesn't exist at `commit` (i.e. it was
    deleted there) by removing it instead. Returns True if the resulting
    working tree differs from before the call (so callers can skip an
    empty commit).
    """
    exists_at_commit = (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{path}"],
            cwd=worktree_path, capture_output=True,
        ).returncode == 0
    )
    if exists_at_commit:
        _run_git("checkout", commit, "--", path, cwd=worktree_path)
    elif (worktree_path / path).exists():
        _run_git("rm", "-q", "-f", path, cwd=worktree_path)
    else:
        return False
    return True


@contextmanager
def isolated_worktree(repo: Path, ref: str = "HEAD") -> Iterator[Worktree]:
    """Check out `repo` at `ref` into a fresh temp directory on a throwaway
    branch, yield it, then remove both — regardless of what happens inside.
    """
    repo = repo.resolve()
    _assert_is_git_repo(repo)

    base_commit = rev_parse(repo, ref)
    run_id = uuid.uuid4().hex[:8]
    branch = f"verdict/{run_id}"
    parent_dir = Path(tempfile.mkdtemp(prefix="verdict-worktree-"))
    worktree_path = parent_dir / "worktree"

    _run_git(
        "worktree", "add", "-b", branch, str(worktree_path), base_commit,
        cwd=repo,
    )

    try:
        yield Worktree(path=worktree_path, branch=branch, base_commit=base_commit)
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


@contextmanager
def scratch_worktree(repo: Path, ref: str) -> Iterator[Path]:
    """A detached-HEAD worktree at `ref`, with no throwaway branch of its
    own. Used for attribution's disposable bisection checkouts — detached
    so multiple scratch worktrees can each explore commits freely (moving
    their own detached HEAD around, as `git bisect` does) without
    colliding on branch names or on `isolated_worktree`'s branch, which may
    already be checked out elsewhere for the same run.
    """
    repo = repo.resolve()
    parent_dir = Path(tempfile.mkdtemp(prefix="verdict-scratch-"))
    worktree_path = parent_dir / "scratch"

    _run_git("worktree", "add", "--detach", str(worktree_path), ref, cwd=repo)

    try:
        yield worktree_path
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(parent_dir, ignore_errors=True)
