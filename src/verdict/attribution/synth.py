"""Builds the synthetic per-file commit ladder level-2 bisection runs over.

Real agent commits (if any) give level-1 bisection a coarse answer: which
*commit* introduced the failure. That commit might touch several files at
once, so level 2 needs its own ladder to get down to one file. Rather than
hand-applying that commit's unified diff hunk by hunk (fragile — context
lines can fail to match), each synthetic commit just pulls one file's
*final* content straight from the culprit commit via `git checkout
<culprit> -- <path>` and commits it. Cumulative, so the last synthetic
commit's tree is byte-identical to the culprit commit's tree.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from verdict.worktree import checkout_path_from, rev_parse, run_git, scratch_worktree


@contextmanager
def build_synthetic_ladder(
    repo: Path, parent_sha: str, culprit_sha: str, files: list[str]
) -> Iterator[tuple[str, dict[str, str]]]:
    """Yields `(final_synthetic_sha, {commit_sha: file_path})` while the
    scratch worktree holding these commits is still open — level-2 bisect
    must run inside this `with` block, since the synthetic commits only
    stay reachable (referenced by this worktree's detached HEAD) while
    it's alive.
    """
    with scratch_worktree(repo, parent_sha) as wt:
        sha_to_file: dict[str, str] = {}
        final_sha = parent_sha
        for path in sorted(files):
            # checkout_path_from already stages its change (`git checkout
            # <commit> -- <path>` or `git rm`), so no separate `git add`.
            if not checkout_path_from(wt, culprit_sha, path):
                continue
            run_git("commit", "-q", "-m", f"verdict-synthetic: {path}", cwd=wt)
            final_sha = rev_parse(wt, "HEAD")
            sha_to_file[final_sha] = path
        yield final_sha, sha_to_file
