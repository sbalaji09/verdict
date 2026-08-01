"""The `typecheck` gate: tsc for TypeScript repos, mypy for Python ones.

tsc has no native machine-readable output format, so we parse its default
`file(line,col): error TSxxxx: message` line shape. mypy's `--output json`
(available since mypy 2.x) prints one JSON object per diagnostic line, so
that path is exact rather than regex-based.

Each error's `identity` is `"{file}:{code}"` — deliberately *not*
`"{file}:{line}:{code}"`. Phase 2's attribution re-runs this gate at other
commits to see whether a specific error still reproduces, and at an
intermediate bisection state the surrounding code has shifted, so the same
root-cause error can land on a different line. Matching on file+code is the
right granularity for "is this still the same problem," even though it
means two genuinely distinct errors of the same code in the same file
would collide — an accepted, documented imprecision, not an oversight.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from verdict.gates.base import DEFAULT_TIMEOUT_SECONDS, ToolRunner, exec_command, tail
from verdict.sandbox import Sandbox
from verdict.schema import FailureLocation, GateStatus, Provenance, Signal

_TSC_ERROR_RE = re.compile(
    r"^(?P<file>[^(]+)\((?P<line>\d+),(?P<col>\d+)\): error (?P<code>TS\d+): (?P<msg>.+)$"
)


def _parse_tsc(result: subprocess.CompletedProcess[str]) -> tuple[str, GateStatus, list[FailureLocation]]:
    matches = [m for m in (_TSC_ERROR_RE.match(line) for line in result.stdout.splitlines()) if m]
    if not matches and result.returncode != 0:
        return tail(result.stdout + result.stderr), GateStatus.FAIL, []

    failures = [
        FailureLocation(
            identity=f"{m.group('file')}:{m.group('code')}",
            file=m.group("file"),
            line=int(m.group("line")),
            code=m.group("code"),
            message=m.group("msg"),
        )
        for m in matches
    ]

    detail = f"{len(failures)} error(s)"
    if failures:
        detail += "\n" + "\n".join(f"{f.file}:{f.line} {f.code}: {f.message}" for f in failures[:10])
    status = GateStatus.PASS if not failures else GateStatus.FAIL
    return detail, status, failures


class TscRunner:
    tool = "tsc"
    gate = "typecheck"

    def applicable(self, worktree: Path) -> bool:
        return (worktree / "tsconfig.json").exists() and (
            worktree / "node_modules" / ".bin" / "tsc"
        ).exists()

    def run(
        self, worktree: Path, sandbox: Sandbox | None = None, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    ) -> Signal:
        binary = str(worktree / "node_modules" / ".bin" / "tsc")
        command = [binary, "--noEmit", "--pretty", "false"]
        result = exec_command(command, cwd=worktree, sandbox=sandbox, timeout_seconds=timeout_seconds)
        detail, status, failures = _parse_tsc(result)
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
            failures=failures,
        )


def _parse_mypy(result: subprocess.CompletedProcess[str]) -> tuple[str, GateStatus, list[FailureLocation]]:
    diagnostics = []
    for line in result.stdout.splitlines():
        try:
            diagnostics.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    errors = [d for d in diagnostics if d.get("severity") == "error"]

    if not diagnostics and result.returncode not in (0, 1):
        return tail(result.stdout + result.stderr), GateStatus.FAIL, []

    failures = [
        FailureLocation(
            identity=f"{e.get('file')}:{e.get('code')}",
            file=e.get("file"),
            line=e.get("line"),
            code=e.get("code"),
            message=e.get("message", ""),
        )
        for e in errors
    ]

    detail = f"{len(failures)} error(s)"
    if failures:
        detail += "\n" + "\n".join(f"{f.file}:{f.line} {f.code}: {f.message}" for f in failures[:10])
    status = GateStatus.PASS if not failures else GateStatus.FAIL
    return detail, status, failures


class MypyRunner:
    tool = "mypy"
    gate = "typecheck"

    def applicable(self, worktree: Path) -> bool:
        if (worktree / "mypy.ini").exists():
            return True
        pyproject = worktree / "pyproject.toml"
        return pyproject.exists() and "[tool.mypy" in pyproject.read_text(errors="ignore")

    def run(
        self, worktree: Path, sandbox: Sandbox | None = None, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    ) -> Signal:
        command = ["mypy", "--output", "json", "."]
        result = exec_command(command, cwd=worktree, sandbox=sandbox, timeout_seconds=timeout_seconds)
        detail, status, failures = _parse_mypy(result)
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
            failures=failures,
        )


RUNNERS: list[ToolRunner] = [TscRunner(), MypyRunner()]
