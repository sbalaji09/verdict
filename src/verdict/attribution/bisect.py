"""The one bisection primitive both levels of attribution build on: given
a known-good commit, a known-bad commit, and a specific failure to check
for, run real `git bisect` between them and return the first commit where
that failure starts reproducing.

Both "which commit introduced this" (real agent commits) and "which file
within that commit introduced this" (synthetic per-file commits) call this
same function — they only differ in what commit graph they hand it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from verdict.worktree import WorktreeError, scratch_worktree

BISECT_TIMEOUT_SECONDS = 600

# What git prints when bisection converges cleanly. NOT the same as reading
# HEAD after the run: git only re-checks-out a commit when it needs to
# *test* it, so if the bisection concludes without needing to re-visit the
# already-known-bad commit (a common case with few candidates), HEAD is
# left at the last commit that was actually tested — which can be a GOOD
# one. The SHA has to come from this line, not from the final working tree
# state. (Found by testing against a real 3-commit fixture — the "obvious"
# `git rev-parse HEAD` approach silently returned the wrong commit.)
_FIRST_BAD_RE = re.compile(r"^([0-9a-f]{7,40}) is the first bad commit$", re.MULTILINE)


def run_bisect(
    repo: Path, good_sha: str, bad_sha: str, gate: str, identity: str | None
) -> str | None:
    """Returns the first-bad commit SHA, or None if bisection couldn't
    reach a clean answer (e.g. every intermediate state was untestable).
    """
    try:
        with scratch_worktree(repo, bad_sha) as wt:
            subprocess.run(["git", "bisect", "start"], cwd=wt, capture_output=True, text=True, check=True)
            try:
                subprocess.run(
                    ["git", "bisect", "bad", bad_sha], cwd=wt, capture_output=True, text=True, check=True
                )
                subprocess.run(
                    ["git", "bisect", "good", good_sha], cwd=wt, capture_output=True, text=True, check=True
                )

                check_cmd = [sys.executable, "-m", "verdict.attribution.bisect_cli", gate]
                if identity:
                    check_cmd.append(identity)

                result = subprocess.run(
                    ["git", "bisect", "run", *check_cmd],
                    cwd=wt,
                    capture_output=True,
                    text=True,
                    timeout=BISECT_TIMEOUT_SECONDS,
                )
                match = _FIRST_BAD_RE.search(result.stdout)
                return match.group(1) if match else None
            finally:
                subprocess.run(["git", "bisect", "reset"], cwd=wt, capture_output=True, text=True)
    except (WorktreeError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return None
