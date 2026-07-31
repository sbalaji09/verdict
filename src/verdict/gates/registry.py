"""Ties gate names to their candidate tool runners and resolves each gate
for a given worktree: verdict.yml override, if any, otherwise the first
applicable autodetected tool, otherwise N/A.
"""

from __future__ import annotations

from pathlib import Path

from verdict.config import VerdictConfig
from verdict.gates import build, lint, test, typecheck
from verdict.gates.base import ToolRunner, not_applicable, raw_signal
from verdict.schema import Signal

GATE_RUNNERS: dict[str, list[ToolRunner]] = {
    "test": test.RUNNERS,
    "typecheck": typecheck.RUNNERS,
    "build": build.RUNNERS,
    "lint": lint.RUNNERS,
}


def resolve_gate(gate: str, worktree: Path, config: VerdictConfig) -> Signal:
    override = config.override_for(gate)
    if override:
        return raw_signal(gate, override, worktree)

    for runner in GATE_RUNNERS[gate]:
        if runner.applicable(worktree):
            return runner.run(worktree)

    return not_applicable(gate)


def run_all_gates(worktree: Path, config: VerdictConfig) -> list[Signal]:
    return [resolve_gate(gate, worktree, config) for gate in GATE_RUNNERS]
