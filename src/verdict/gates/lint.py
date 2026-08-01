"""The `lint` gate: eslint for JS/TS repos, ruff for Python ones.

Both tools have native JSON reporters, so both paths get exact, parsed
diagnostic counts rather than scraped text. As with typecheck, each
violation's `identity` is `"{file}:{code}"`, not including the line number
— see typecheck.py's docstring for why.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from verdict.gates.base import ToolRunner, exec_command, tail
from verdict.sandbox import Sandbox
from verdict.schema import FailureLocation, GateStatus, Provenance, Signal


def _parse_eslint(result: subprocess.CompletedProcess[str]) -> tuple[str, GateStatus, list[FailureLocation]]:
    try:
        files = json.loads(result.stdout)
    except json.JSONDecodeError:
        status = GateStatus.PASS if result.returncode == 0 else GateStatus.FAIL
        return tail(result.stdout + result.stderr), status, []

    failures = [
        FailureLocation(
            identity=f"{f['filePath']}:{m.get('ruleId')}",
            file=f["filePath"],
            line=m.get("line"),
            code=m.get("ruleId"),
            message=m.get("message", ""),
        )
        for f in files
        for m in f.get("messages", [])
        if m.get("severity") == 2
    ]

    detail = f"{len(failures)} error(s)"
    if failures:
        detail += "\n" + "\n".join(f"{f.file}:{f.line} {f.code}: {f.message}" for f in failures[:10])
    status = GateStatus.FAIL if failures else GateStatus.PASS
    return detail, status, failures


class EslintRunner:
    tool = "eslint"
    gate = "lint"

    def applicable(self, worktree: Path) -> bool:
        has_binary = (worktree / "node_modules" / ".bin" / "eslint").exists()
        if not has_binary:
            return False
        eslint_config_names = (
            ".eslintrc",
            ".eslintrc.js",
            ".eslintrc.json",
            ".eslintrc.yml",
            "eslint.config.js",
            "eslint.config.mjs",
        )
        if any((worktree / name).exists() for name in eslint_config_names):
            return True
        package_json = worktree / "package.json"
        if package_json.exists():
            try:
                data = json.loads(package_json.read_text())
            except json.JSONDecodeError:
                data = {}
            if "eslintConfig" in data:
                return True
        return False

    def run(self, worktree: Path, sandbox: Sandbox | None = None) -> Signal:
        binary = str(worktree / "node_modules" / ".bin" / "eslint")
        command = [binary, ".", "-f", "json"]
        result = exec_command(command, cwd=worktree, sandbox=sandbox)
        detail, status, failures = _parse_eslint(result)
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
            failures=failures,
        )


def _parse_ruff(result: subprocess.CompletedProcess[str]) -> tuple[str, GateStatus, list[FailureLocation]]:
    try:
        violations = json.loads(result.stdout)
    except json.JSONDecodeError:
        status = GateStatus.PASS if result.returncode == 0 else GateStatus.FAIL
        return tail(result.stdout + result.stderr), status, []

    failures = [
        FailureLocation(
            identity=f"{v.get('filename')}:{v.get('code')}",
            file=v.get("filename"),
            line=v.get("location", {}).get("row"),
            code=v.get("code"),
            message=v.get("message", ""),
        )
        for v in violations
    ]

    detail = f"{len(failures)} violation(s)"
    if failures:
        detail += "\n" + "\n".join(f"{f.file}:{f.line} {f.code}: {f.message}" for f in failures[:10])
    status = GateStatus.FAIL if failures else GateStatus.PASS
    return detail, status, failures


class RuffRunner:
    tool = "ruff"
    gate = "lint"

    def applicable(self, worktree: Path) -> bool:
        if (worktree / "ruff.toml").exists() or (worktree / ".ruff.toml").exists():
            return True
        pyproject = worktree / "pyproject.toml"
        return pyproject.exists() and "[tool.ruff" in pyproject.read_text(errors="ignore")

    def run(self, worktree: Path, sandbox: Sandbox | None = None) -> Signal:
        command = ["ruff", "check", "--output-format=json", "."]
        result = exec_command(command, cwd=worktree, sandbox=sandbox)
        detail, status, failures = _parse_ruff(result)
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
            failures=failures,
        )


RUNNERS: list[ToolRunner] = [EslintRunner(), RuffRunner()]
