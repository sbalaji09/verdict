"""Ties gate names to their candidate tool runners and resolves each gate
for a given worktree: verdict.yml override, if any, otherwise the first
applicable autodetected tool, otherwise N/A.
"""

from __future__ import annotations

import time
from pathlib import Path

from verdict.config import VerdictConfig
from verdict.gates import build, lint, test, typecheck
from verdict.gates.base import DEFAULT_TIMEOUT_SECONDS, ToolRunner, not_applicable, raw_signal
from verdict.sandbox import Sandbox
from verdict.schema import Signal

GATE_RUNNERS: dict[str, list[ToolRunner]] = {
    "test": test.RUNNERS,
    "typecheck": typecheck.RUNNERS,
    "build": build.RUNNERS,
    "lint": lint.RUNNERS,
}


def resolve_gate(
    gate: str,
    worktree: Path,
    config: VerdictConfig,
    sandbox: Sandbox | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Signal:
    override = config.override_for(gate)
    if override:
        return raw_signal(gate, override, worktree, sandbox=sandbox, timeout_seconds=timeout_seconds)

    for runner in GATE_RUNNERS[gate]:
        if runner.applicable(worktree):
            return runner.run(worktree, sandbox=sandbox, timeout_seconds=timeout_seconds)

    return not_applicable(gate)


def run_all_gates(
    worktree: Path,
    config: VerdictConfig,
    sandbox: Sandbox | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    deadline: float | None = None,
) -> tuple[list[Signal], bool]:
    """Runs every gate in order, unless `deadline` (a `time.monotonic()`
    timestamp — Phase 9's global per-attempt budget) is reached first, in
    which case remaining gates are skipped entirely rather than attempted
    and returns `(signals_so_far, True)`. Skipped gates are simply absent
    from the returned list — never reported as FAIL (that would blame the
    agent for something Verdict itself didn't get to check) and never NA
    (NA means "this repo has no such stack," which isn't what happened).
    """
    signals: list[Signal] = []
    for gate in GATE_RUNNERS:
        if deadline is not None and time.monotonic() >= deadline:
            return signals, True
        signals.append(resolve_gate(gate, worktree, config, sandbox=sandbox, timeout_seconds=timeout_seconds))
    return signals, False
