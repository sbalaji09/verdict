"""The one bisection primitive both levels of attribution build on: given
a known-good commit, a known-bad commit, and a specific failure to check
for, find the first commit where that failure starts reproducing.

Both "which commit introduced this" (real agent commits) and "which file
within that commit introduced this" (synthetic per-file commits) call this
same function — they only differ in what commit graph they hand it.

Phase 8 rewrote this to drive its own binary search over `commits_between`
rather than shelling out to `git bisect run <cmd>`. The old approach had
`git bisect run` repeatedly check out a candidate commit and re-invoke
`bisect_cli.py` as a *fresh host subprocess* at each one — a nested
subprocess-of-a-subprocess Verdict couldn't route through a `Sandbox`
individually. Driving the search here means every candidate commit's check
is one `Sandbox.exec()` call this module controls directly. The search
strategy (plain binary search, with a bounded forward nudge past
untestable/SKIP commits — mirroring `git bisect skip`'s own heuristic) is
new; the *result* for any candidate set with no SKIPs is identical to what
`git bisect` would have found, since both converge on the same first commit
where the target failure starts reproducing.
"""

from __future__ import annotations

from pathlib import Path

from verdict.attribution.reproduce import Reproduction, check_reproduces
from verdict.sandbox.config import SandboxConfig, create_sandbox
from verdict.worktree import WorktreeError, commits_between, scratch_worktree

# How many commits past a SKIP a search step will probe, looking for a
# testable one to resume the binary search from. Bounded so a long run of
# genuinely-untestable commits gives up (returns None) rather than
# degrading into an O(n) scan.
_MAX_SKIP_NUDGE = 8


def _check_at(
    repo: Path, sha: str, gate: str, identity: str | None, sandbox_config: SandboxConfig
) -> Reproduction:
    try:
        with scratch_worktree(repo, sha) as wt, create_sandbox(wt, sandbox_config) as sandbox:
            return check_reproduces(gate, identity, wt, sandbox=sandbox)
    except WorktreeError:
        return Reproduction.SKIP


def run_bisect(
    repo: Path,
    good_sha: str,
    bad_sha: str,
    gate: str,
    identity: str | None,
    sandbox_config: SandboxConfig | None = None,
) -> str | None:
    """Returns the first-bad commit SHA, or None if bisection couldn't
    reach a clean answer (e.g. every intermediate state was untestable).

    Callers are expected to already know `good_sha` reproduces GOOD and
    `bad_sha` reproduces BAD — this function doesn't re-verify either
    endpoint, matching the contract the old `git bisect start`/`bad`/`good`
    sequence had.
    """
    sandbox_config = sandbox_config or SandboxConfig()
    candidates = commits_between(repo, good_sha, bad_sha)
    if not candidates:
        return None

    lo, hi = -1, len(candidates) - 1  # candidates[hi] == bad_sha, known BAD
    while hi - lo > 1:
        mid = (lo + hi) // 2
        outcome = _check_at(repo, candidates[mid], gate, identity, sandbox_config)
        if outcome is Reproduction.BAD:
            hi = mid
            continue
        if outcome is Reproduction.GOOD:
            lo = mid
            continue

        # SKIP: probe forward (toward the known-bad end) for a testable
        # commit to resume the search from, same direction git bisect's
        # own skip-and-retry defaults to.
        resolved = False
        for probe in range(mid + 1, min(hi, mid + 1 + _MAX_SKIP_NUDGE)):
            probe_outcome = _check_at(repo, candidates[probe], gate, identity, sandbox_config)
            if probe_outcome is Reproduction.BAD:
                hi = probe
                resolved = True
                break
            if probe_outcome is Reproduction.GOOD:
                lo = probe
                resolved = True
                break
        if not resolved:
            return None

    return candidates[hi]
