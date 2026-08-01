"""The `test` gate: pytest, jest, or go test, tried in that order.

Each runner asks its tool for a structured report (junit XML, jest's own
--json, go test's -json event stream) rather than scraping human-readable
stdout, so the parsed counts are exact rather than regex-guessed, and each
individual failure comes back as a `FailureLocation` with a stable
`identity` — Phase 2's attribution re-runs a single test by this identity
to check whether it reproduces at other commits, so it has to be something
`pytest <identity>` can actually be given back, not just a display label.
"""

from __future__ import annotations

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from verdict.gates.base import DEFAULT_TIMEOUT_SECONDS, ToolRunner, exec_command, tail
from verdict.sandbox import Sandbox
from verdict.schema import FailureLocation, GateStatus, Provenance, Signal


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

    def run(
        self,
        worktree: Path,
        sandbox: Sandbox | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        env: dict[str, str] | None = None,
    ) -> Signal:
        fd, report_path = tempfile.mkstemp(suffix=".xml")
        os.close(fd)
        command = ["pytest", "-q", f"--junitxml={report_path}"]
        try:
            result = exec_command(
                command, cwd=worktree, sandbox=sandbox, timeout_seconds=timeout_seconds, env=env
            )
            detail, status, failures = _parse_junit(report_path, fallback=result.stdout + result.stderr)
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
            failures=failures,
        )


def _node_id(classname: str, name: str) -> str:
    """Reconstruct a `pytest <path>::<name>`-runnable node id from junit's
    `classname` (a dotted module path, e.g. "tests.test_x"). Doesn't handle
    every possible rootdir/conftest layout, but covers the common
    unpackaged-tests-directory case our autodetection already targets.

    A collection error (e.g. an ImportError at module load) is a special
    case: pytest reports it with `classname=""` and `name=<module>` — there
    was no specific test function to select, since the module never even
    finished importing. The re-runnable id there is just the file itself,
    not `file.py::name` (which isn't a valid node id and wouldn't
    reproduce the same failure mode during bisection).
    """
    if not classname:
        return f"{name}.py"
    path = classname.replace(".", "/") + ".py"
    return f"{path}::{name}"


def _parse_junit(
    report_path: str, fallback: str
) -> tuple[str, GateStatus | None, list[FailureLocation]]:
    try:
        root = ET.parse(report_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return tail(fallback), None, []

    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return tail(fallback), None, []

    total = int(suite.get("tests", 0))
    failures_count = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    passed = total - failures_count - errors - skipped

    failures: list[FailureLocation] = []
    for tc in suite.findall("testcase"):
        failure_el = tc.find("failure")
        error_el = tc.find("error")
        if failure_el is None and error_el is None:
            continue
        node_id = _node_id(tc.get("classname", ""), tc.get("name", ""))
        detail_el = failure_el if failure_el is not None else error_el
        message = detail_el.get("message", "") if detail_el is not None else ""
        failures.append(FailureLocation(identity=node_id, file=None, message=message or ""))

    detail = f"{passed} passed, {failures_count} failed, {errors} errors, {skipped} skipped"
    if failures:
        detail += "\nfailed: " + ", ".join(f.identity for f in failures[:10])

    status = GateStatus.FAIL if (failures_count or errors) else GateStatus.PASS
    return detail, status, failures


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

    def run(
        self,
        worktree: Path,
        sandbox: Sandbox | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        env: dict[str, str] | None = None,
    ) -> Signal:
        fd, report_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        command = ["npx", "--no-install", "jest", "--json", f"--outputFile={report_path}"]
        try:
            result = exec_command(
                command, cwd=worktree, sandbox=sandbox, timeout_seconds=timeout_seconds, env=env
            )
            detail, status, failures = _parse_jest(report_path, fallback=result.stdout + result.stderr)
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
            failures=failures,
        )


def _parse_jest(
    report_path: str, fallback: str
) -> tuple[str, GateStatus | None, list[FailureLocation]]:
    try:
        data = json.loads(Path(report_path).read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return tail(fallback), None, []

    passed = data.get("numPassedTests", 0)
    failed = data.get("numFailedTests", 0)
    total = data.get("numTotalTests", 0)
    detail = f"{passed}/{total} passed, {failed} failed"

    failures: list[FailureLocation] = []
    for result in data.get("testResults", []):
        for assertion in result.get("assertionResults", []):
            if assertion.get("status") != "failed":
                continue
            full_name = assertion.get("fullName") or assertion.get("title", "")
            failures.append(
                FailureLocation(
                    identity=f"{result.get('name', '')}::{full_name}",
                    file=result.get("name"),
                    message="\n".join(assertion.get("failureMessages", [])),
                )
            )

    status = GateStatus.PASS if data.get("success") else GateStatus.FAIL
    return detail, status, failures


class GoTestRunner:
    tool = "go-test"
    gate = "test"

    def applicable(self, worktree: Path) -> bool:
        return (worktree / "go.mod").exists()

    def run(
        self,
        worktree: Path,
        sandbox: Sandbox | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        env: dict[str, str] | None = None,
    ) -> Signal:
        command = ["go", "test", "-json", "./..."]
        result = exec_command(
            command, cwd=worktree, sandbox=sandbox, timeout_seconds=timeout_seconds, env=env
        )
        detail, status, failures = _parse_go_test(result.stdout, fallback=result.stdout + result.stderr)

        if status is None:
            status = GateStatus.PASS if result.returncode == 0 else GateStatus.FAIL
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
            failures=failures,
        )


def _parse_go_test(
    stdout: str, fallback: str
) -> tuple[str, GateStatus | None, list[FailureLocation]]:
    passed = failed = 0
    failures: list[FailureLocation] = []
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
            package = event.get("Package", "")
            failures.append(
                FailureLocation(identity=f"{package}.{event['Test']}", file=package)
            )

    if not saw_any:
        return tail(fallback), None, []

    detail = f"{passed} passed, {failed} failed"
    if failures:
        detail += "\nfailed: " + ", ".join(f.identity for f in failures[:10])
    return detail, (GateStatus.FAIL if failed else GateStatus.PASS), failures


def _load_json(path: Path) -> dict[str, object]:
    try:
        result: dict[str, object] = json.loads(path.read_text())
        return result
    except json.JSONDecodeError:
        return {}


RUNNERS: list[ToolRunner] = [PytestRunner(), JestRunner(), GoTestRunner()]
