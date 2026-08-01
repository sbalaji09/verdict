"""Answers one question: at the *current* worktree state, does a specific
already-known failure still reproduce?

This is the check `git bisect run` calls at every candidate commit. It
deliberately re-runs the real gate machinery from `gates/registry.py`
rather than re-implementing tool invocation — the exact same autodetection
and parsing Phase 1 already built and tested is what decides, at each
bisection step, whether "typecheck error TS2322 in calculator.ts" is still
present. No new parsing logic, no new commands — just re-asking the
existing question at a different point in history.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from verdict.config import load_config
from verdict.gates.registry import resolve_gate
from verdict.sandbox import Sandbox
from verdict.schema import GateStatus


class Reproduction(str, Enum):
    GOOD = "good"   # the target failure does NOT reproduce here
    BAD = "bad"     # the target failure DOES reproduce here
    SKIP = "skip"   # untestable state — don't let bisection draw a conclusion


def check_reproduces(
    gate: str, target_identity: str | None, worktree: Path, sandbox: Sandbox | None = None
) -> Reproduction:
    """`target_identity` is a `FailureLocation.identity`, or None to check
    the gate as a whole (used for `build`, and for any gate whose original
    failure came from a `verdict.yml` override with no structured detail).
    """
    try:
        config = load_config(worktree)
        signal = resolve_gate(gate, worktree, config, sandbox=sandbox)
    except Exception:
        # Anything we didn't anticipate (permissions, a tool crashing in a
        # way exec_command's own guards don't cover) — treat as untestable
        # rather than guessing which way it should count.
        return Reproduction.SKIP

    if signal.status is GateStatus.NA:
        # this gate's stack isn't even present at this state — can't tell
        return Reproduction.SKIP

    if target_identity is None:
        return Reproduction.BAD if signal.status is GateStatus.FAIL else Reproduction.GOOD

    if signal.status is GateStatus.FAIL and not signal.failures:
        # failed, but with no structured failure list to check identity
        # against (e.g. a verdict.yml override) — can't confirm it's the
        # *same* failure, only that *something* failed.
        return Reproduction.SKIP

    ids = {f.identity for f in signal.failures}
    return Reproduction.BAD if target_identity in ids else Reproduction.GOOD
