"""Shared contract and execution helpers for gate tool runners.

A "gate" (test/typecheck/build/lint) is a category; a ToolRunner is one
concrete way to satisfy it (pytest satisfies "test", so does jest). Keeping
these separate is what lets `registry.py` try several tools per gate in
priority order and stop at the first one that actually applies to the repo
being graded.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol

from verdict.schema import GateStatus, Provenance, Signal

DEFAULT_TIMEOUT_SECONDS = 600


class ToolRunner(Protocol):
    """One tool (pytest, tsc, eslint, ...) that can satisfy a gate."""

    tool: str
    gate: str

    def applicable(self, worktree: Path) -> bool:
        """Does this repo look like it uses this tool? Checked against the
        worktree (the agent's checked-out copy), not the source repo — an
        agent could in principle add a stack the original repo lacked.
        """
        ...

    def run(self, worktree: Path) -> Signal:
        """Execute the tool and return a structured PROVEN Signal. Only
        called after `applicable` returned True, and only when no
        verdict.yml override exists for this gate — this method controls
        its own command precisely so it can request structured output
        (--output-format=json and friends).
        """
        ...


def exec_command(
    args: list[str],
    cwd: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run an argv list (never shell=True — we built these ourselves, no
    need to accept the injection surface of a shell).
    """
    try:
        return subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout=str(exc.stdout or ""),
            stderr=f"timed out after {timeout_seconds}s",
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(
            args=args, returncode=127, stdout="", stderr=str(exc)
        )


def tail(text: str, lines: int = 15) -> str:
    text = text.strip()
    if not text:
        return "(no output)"
    return "\n".join(text.splitlines()[-lines:])


def raw_signal(
    gate: str,
    command: str,
    worktree: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Signal:
    """Run an arbitrary shell command (e.g. a verdict.yml override) and
    grade it purely by exit code. No structured parsing: we don't control
    the invocation, so we can't guarantee a flag that produces machine-
    readable output. Exit code is still real, executed fact — it's the
    detail that's coarser here, not the trust level.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return Signal(
            name=gate,
            provenance=Provenance.PROVEN,
            status=GateStatus.FAIL,
            detail=f"timed out after {timeout_seconds}s",
            command=command,
        )

    output = tail(result.stdout + result.stderr)
    return Signal(
        name=gate,
        provenance=Provenance.PROVEN,
        status=GateStatus.PASS if result.returncode == 0 else GateStatus.FAIL,
        detail=output,
        command=command,
        exit_code=result.returncode,
    )


def not_applicable(gate: str) -> Signal:
    return Signal(
        name=gate,
        provenance=Provenance.PROVEN,
        status=GateStatus.NA,
        detail=f"no {gate} stack detected in this repo",
        command=None,
    )
