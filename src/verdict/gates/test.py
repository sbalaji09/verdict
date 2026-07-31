"""The `test` gate: pytest, jest, or go test, tried in that order.

Each runner asks its tool for a structured report (junit XML, jest's own
--json, go test's -json event stream) rather than scraping human-readable
stdout, so the parsed counts are exact rather than regex-guessed.
"""

from __future__ import annotations

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from verdict.gates.base import ToolRunner, exec_command, tail
from verdict.schema import GateStatus, Provenance, Signal


class PytestRunner:
    tool = "pytest"
    gate = "test"

    def applicable(self, worktree: Path) -> bool:
        if (worktree / "pytest.ini").exists() or (worktree / "setup.cfg").exists():
            return True
        pyproject = worktree / "pyproject.toml"
        if pyproject.exists() and "[tool.pytest" in pyproject.read_text(errors="ignore"):
            return True
        tests_dir = worktree / "tests"
        return tests_dir.is_dir() and any(tests_dir.glob("test_*.py"))

    def run(self, worktree: Path) -> Signal:
        fd, report_path = tempfile.mkstemp(suffix=".xml")
        os.close(fd)
        command = ["pytest", "-q", f"--junitxml={report_path}"]
        try:
            result = exec_command(command, cwd=worktree)
            detail, status = _parse_junit(report_path, fallback=result.stdout + result.stderr)
        finally:
            Path(report_path).unlink(missing_ok=True)

        if status is None:
            status = GateStatus.PASS if result.returncode == 0 else GateStatus.FAIL
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
        )


def _parse_junit(report_path: str, fallback: str) -> tuple[str, GateStatus | None]:
    try:
        root = ET.parse(report_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return tail(fallback), None

    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return tail(fallback), None

    total = int(suite.get("tests", 0))
    failures = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    passed = total - failures - errors - skipped

    failed_names = [
        tc.get("classname", "") + "::" + tc.get("name", "")
        for tc in suite.findall("testcase")
        if tc.find("failure") is not None or tc.find("error") is not None
    ]

    detail = f"{passed} passed, {failures} failed, {errors} errors, {skipped} skipped"
    if failed_names:
        detail += "\nfailed: " + ", ".join(failed_names[:10])

    status = GateStatus.FAIL if (failures or errors) else GateStatus.PASS
    return detail, status


class JestRunner:
    tool = "jest"
    gate = "test"

    def applicable(self, worktree: Path) -> bool:
        if any((worktree / f"jest.config.{ext}").exists() for ext in ("js", "ts", "json", "mjs", "cjs")):
            return True
        package_json = worktree / "package.json"
        if not package_json.exists():
            return False
        data = _load_json(package_json)
        dependencies = data.get("dependencies") or {}
        dev_dependencies = data.get("devDependencies") or {}
        if not isinstance(dependencies, dict) or not isinstance(dev_dependencies, dict):
            return False
        return "jest" in dependencies or "jest" in dev_dependencies

    def run(self, worktree: Path) -> Signal:
        fd, report_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        command = ["npx", "--no-install", "jest", "--json", f"--outputFile={report_path}"]
        try:
            result = exec_command(command, cwd=worktree)
            detail, status = _parse_jest(report_path, fallback=result.stdout + result.stderr)
        finally:
            Path(report_path).unlink(missing_ok=True)

        if status is None:
            status = GateStatus.PASS if result.returncode == 0 else GateStatus.FAIL
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
        )


def _parse_jest(report_path: str, fallback: str) -> tuple[str, GateStatus | None]:
    try:
        data = json.loads(Path(report_path).read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return tail(fallback), None

    passed = data.get("numPassedTests", 0)
    failed = data.get("numFailedTests", 0)
    total = data.get("numTotalTests", 0)
    detail = f"{passed}/{total} passed, {failed} failed"
    status = GateStatus.PASS if data.get("success") else GateStatus.FAIL
    return detail, status


class GoTestRunner:
    tool = "go-test"
    gate = "test"

    def applicable(self, worktree: Path) -> bool:
        return (worktree / "go.mod").exists()

    def run(self, worktree: Path) -> Signal:
        command = ["go", "test", "-json", "./..."]
        result = exec_command(command, cwd=worktree)
        detail, status = _parse_go_test(result.stdout, fallback=result.stdout + result.stderr)

        if status is None:
            status = GateStatus.PASS if result.returncode == 0 else GateStatus.FAIL
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
        )


def _parse_go_test(stdout: str, fallback: str) -> tuple[str, GateStatus | None]:
    passed = failed = 0
    failed_names: list[str] = []
    saw_any = False
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("Action") not in ("pass", "fail") or "Test" not in event:
            continue
        saw_any = True
        if event["Action"] == "pass":
            passed += 1
        else:
            failed += 1
            failed_names.append(f"{event.get('Package', '')}.{event['Test']}")

    if not saw_any:
        return tail(fallback), None

    detail = f"{passed} passed, {failed} failed"
    if failed_names:
        detail += "\nfailed: " + ", ".join(failed_names[:10])
    return detail, (GateStatus.FAIL if failed else GateStatus.PASS)


def _load_json(path: Path) -> dict[str, object]:
    try:
        result: dict[str, object] = json.loads(path.read_text())
        return result
    except json.JSONDecodeError:
        return {}


RUNNERS: list[ToolRunner] = [PytestRunner(), JestRunner(), GoTestRunner()]
