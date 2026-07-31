"""The `lint` gate: eslint for JS/TS repos, ruff for Python ones.

Both tools have native JSON reporters, so both paths get exact, parsed
diagnostic counts rather than scraped text.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from verdict.gates.base import ToolRunner, exec_command, tail
from verdict.schema import GateStatus, Provenance, Signal


def _parse_eslint(result: subprocess.CompletedProcess[str]) -> tuple[str, GateStatus]:
    try:
        files = json.loads(result.stdout)
    except json.JSONDecodeError:
        return tail(result.stdout + result.stderr), (
            GateStatus.PASS if result.returncode == 0 else GateStatus.FAIL
        )

    error_count = sum(f.get("errorCount", 0) for f in files)
    detail = f"{error_count} error(s)"
    messages = [
        f"{f['filePath']}:{m.get('line')} {m.get('ruleId')}: {m.get('message')}"
        for f in files
        for m in f.get("messages", [])
        if m.get("severity") == 2
    ]
    if messages:
        detail += "\n" + "\n".join(messages[:10])
    status = GateStatus.FAIL if error_count else GateStatus.PASS
    return detail, status


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

    def run(self, worktree: Path) -> Signal:
        binary = str(worktree / "node_modules" / ".bin" / "eslint")
        command = [binary, ".", "-f", "json"]
        result = exec_command(command, cwd=worktree)
        detail, status = _parse_eslint(result)
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
        )


def _parse_ruff(result: subprocess.CompletedProcess[str]) -> tuple[str, GateStatus]:
    try:
        violations = json.loads(result.stdout)
    except json.JSONDecodeError:
        return tail(result.stdout + result.stderr), (
            GateStatus.PASS if result.returncode == 0 else GateStatus.FAIL
        )

    detail = f"{len(violations)} violation(s)"
    if violations:
        detail += "\n" + "\n".join(
            f"{v.get('filename')}:{v.get('location', {}).get('row')} "
            f"{v.get('code')}: {v.get('message')}"
            for v in violations[:10]
        )
    status = GateStatus.FAIL if violations else GateStatus.PASS
    return detail, status


class RuffRunner:
    tool = "ruff"
    gate = "lint"

    def applicable(self, worktree: Path) -> bool:
        if (worktree / "ruff.toml").exists() or (worktree / ".ruff.toml").exists():
            return True
        pyproject = worktree / "pyproject.toml"
        return pyproject.exists() and "[tool.ruff" in pyproject.read_text(errors="ignore")

    def run(self, worktree: Path) -> Signal:
        command = ["ruff", "check", "--output-format=json", "."]
        result = exec_command(command, cwd=worktree)
        detail, status = _parse_ruff(result)
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
        )


RUNNERS: list[ToolRunner] = [EslintRunner(), RuffRunner()]
