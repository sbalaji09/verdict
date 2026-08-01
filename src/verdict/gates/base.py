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

from verdict.sandbox import Sandbox
from verdict.sandbox.config import fallback_sandbox
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

    def run(
        self,
        worktree: Path,
        sandbox: Sandbox | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        env: dict[str, str] | None = None,
    ) -> Signal:
        """Execute the tool and return a structured PROVEN Signal. Only
        called after `applicable` returned True, and only when no
        verdict.yml override exists for this gate — this method controls
        its own command precisely so it can request structured output
        (--output-format=json and friends). `sandbox` is None only from
        call sites that haven't been threaded through explicitly (tests);
        real runs always pass one — see registry.py.

        `env` (Phase 10) carries the version-pin overlay
        (`sandbox/versions.py`) — a resolved `PATH`/`PYENV_VERSION` for a
        repo that pins a language version via `.python-version`/`.nvmrc`.
        Empty/`None` for the common unpinned case, in which case the gate
        runs against whatever the sandbox image defaults to, unchanged
        from before this phase.

        A hang here (`timeout_seconds` elapses) is graded exactly like any
        other nonzero exit: a real PROVEN FAIL, never a special status —
        see DESIGN.md's Phase 9 section for why an agent-introduced
        infinite loop is treated as the agent's own defect, not
        infrastructure. `exec_command` labels the FAIL's detail text with
        "timed out after Ns" so the report reads honestly, but the
        GateStatus itself needs no separate handling: `timeout_seconds`
        elapsing produces exit code 124, which every parser below already
        treats as a failure.
        """
        ...


def exec_command(
    args: list[str],
    cwd: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    sandbox: Sandbox | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an argv list inside `sandbox` (never a shell string — we built
    these ourselves, no need to accept the injection surface of a shell).
    Returns a `subprocess.CompletedProcess`-shaped result so the many
    parsers built against that shape (junit/jest/go-test parsing, etc.)
    don't need to change — only the execution underneath does.
    """
    result = (sandbox or fallback_sandbox()).exec(args, cwd=cwd, timeout_seconds=timeout_seconds, env=env)
    stderr = result.stderr
    if result.timed_out:
        # Folded into stderr (not a separate field) so every existing
        # parser's own fallback-to-tail(stdout+stderr) path already shows
        # this without each gate file needing its own timeout branch.
        stderr = (stderr + f"\ntimed out after {timeout_seconds}s").strip()
    return subprocess.CompletedProcess(
        args=args, returncode=result.exit_code, stdout=result.stdout, stderr=stderr
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
    sandbox: Sandbox | None = None,
    env: dict[str, str] | None = None,
) -> Signal:
    """Run an arbitrary shell command (e.g. a verdict.yml override) and
    grade it purely by exit code. No structured parsing: we don't control
    the invocation, so we can't guarantee a flag that produces machine-
    readable output. Exit code is still real, executed fact — it's the
    detail that's coarser here, not the trust level.

    `command` is a raw shell string sourced from the repo being graded —
    the shell interpretation (`sh -c`) happens *inside* the sandbox, never
    via host `shell=True`, which is what made this the one intentional
    shell-string exception in gates/base.py before Phase 8.
    """
    result = (sandbox or fallback_sandbox()).exec(
        ["sh", "-c", command], cwd=worktree, timeout_seconds=timeout_seconds, env=env
    )
    if result.timed_out:
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
        status=GateStatus.PASS if result.exit_code == 0 else GateStatus.FAIL,
        detail=output,
        command=command,
        exit_code=result.exit_code,
    )


def not_applicable(gate: str) -> Signal:
    return Signal(
        name=gate,
        provenance=Provenance.PROVEN,
        status=GateStatus.NA,
        detail=f"no {gate} stack detected in this repo",
        command=None,
    )
