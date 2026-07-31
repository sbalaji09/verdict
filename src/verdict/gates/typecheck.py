"""The `typecheck` gate: tsc for TypeScript repos, mypy for Python ones.

tsc has no native machine-readable output format, so we parse its default
`file(line,col): error TSxxxx: message` line shape. mypy's `--output json`
(available since mypy 2.x) prints one JSON object per diagnostic line, so
that path is exact rather than regex-based.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from verdict.gates.base import ToolRunner, exec_command, tail
from verdict.schema import GateStatus, Provenance, Signal

_TSC_ERROR_RE = re.compile(
    r"^(?P<file>[^(]+)\((?P<line>\d+),(?P<col>\d+)\): error (?P<code>TS\d+): (?P<msg>.+)$"
)


def _parse_tsc(result: subprocess.CompletedProcess[str]) -> tuple[str, GateStatus]:
    errors = [
        m.groupdict() for m in (_TSC_ERROR_RE.match(line) for line in result.stdout.splitlines()) if m
    ]
    if not errors and result.returncode != 0:
        return tail(result.stdout + result.stderr), GateStatus.FAIL

    detail = f"{len(errors)} error(s)"
    if errors:
        detail += "\n" + "\n".join(
            f"{e['file']}:{e['line']} {e['code']}: {e['msg']}" for e in errors[:10]
        )
    status = GateStatus.PASS if not errors else GateStatus.FAIL
    return detail, status


class TscRunner:
    tool = "tsc"
    gate = "typecheck"

    def applicable(self, worktree: Path) -> bool:
        return (worktree / "tsconfig.json").exists() and (
            worktree / "node_modules" / ".bin" / "tsc"
        ).exists()

    def run(self, worktree: Path) -> Signal:
        binary = str(worktree / "node_modules" / ".bin" / "tsc")
        command = [binary, "--noEmit", "--pretty", "false"]
        result = exec_command(command, cwd=worktree)
        detail, status = _parse_tsc(result)
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
        )


def _parse_mypy(result: subprocess.CompletedProcess[str]) -> tuple[str, GateStatus]:
    diagnostics = []
    for line in result.stdout.splitlines():
        try:
            diagnostics.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    errors = [d for d in diagnostics if d.get("severity") == "error"]

    if not diagnostics and result.returncode not in (0, 1):
        return tail(result.stdout + result.stderr), GateStatus.FAIL

    detail = f"{len(errors)} error(s)"
    if errors:
        detail += "\n" + "\n".join(
            f"{e.get('file')}:{e.get('line')} {e.get('code')}: {e.get('message')}"
            for e in errors[:10]
        )
    status = GateStatus.PASS if not errors else GateStatus.FAIL
    return detail, status


class MypyRunner:
    tool = "mypy"
    gate = "typecheck"

    def applicable(self, worktree: Path) -> bool:
        if (worktree / "mypy.ini").exists():
            return True
        pyproject = worktree / "pyproject.toml"
        return pyproject.exists() and "[tool.mypy" in pyproject.read_text(errors="ignore")

    def run(self, worktree: Path) -> Signal:
        command = ["mypy", "--output", "json", "."]
        result = exec_command(command, cwd=worktree)
        detail, status = _parse_mypy(result)
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
        )


RUNNERS: list[ToolRunner] = [TscRunner(), MypyRunner()]
