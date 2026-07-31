"""Unit tests for each gate's output parser, decoupled from actually having
pytest/jest/go/tsc/eslint/ruff installed — we feed each parser the exact
shape its tool emits and assert on the parsed Signal, including the
structured `failures` list Phase 2's attribution consumes directly.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from verdict.gates.lint import _parse_eslint, _parse_ruff
from verdict.gates.test import _parse_go_test, _parse_jest, _parse_junit
from verdict.gates.typecheck import _TSC_ERROR_RE, _parse_mypy, _parse_tsc
from verdict.schema import GateStatus


def _proc(stdout: str, returncode: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)

# --- pytest (junit xml) ------------------------------------------------

_JUNIT_ALL_PASS = """<?xml version="1.0"?>
<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0">
  <testcase classname="tests.test_x" name="test_a"/>
  <testcase classname="tests.test_x" name="test_b"/>
</testsuite></testsuites>"""

_JUNIT_ONE_FAIL = """<?xml version="1.0"?>
<testsuites><testsuite tests="2" failures="1" errors="0" skipped="0">
  <testcase classname="tests.test_x" name="test_a"/>
  <testcase classname="tests.test_x" name="test_b"><failure message="boom">trace</failure></testcase>
</testsuite></testsuites>"""


def _write(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def test_parse_junit_all_passing(tmp_path: Path) -> None:
    path = _write(tmp_path, "r.xml", _JUNIT_ALL_PASS)
    detail, status, failures = _parse_junit(path, fallback="")
    assert status is GateStatus.PASS
    assert "2 passed, 0 failed" in detail
    assert failures == []


def test_parse_junit_with_failure(tmp_path: Path) -> None:
    path = _write(tmp_path, "r.xml", _JUNIT_ONE_FAIL)
    detail, status, failures = _parse_junit(path, fallback="")
    assert status is GateStatus.FAIL
    assert "1 passed, 1 failed" in detail
    assert "test_b" in detail
    assert len(failures) == 1
    assert failures[0].identity == "tests/test_x.py::test_b"
    assert failures[0].message == "boom"


def test_parse_junit_missing_file_falls_back(tmp_path: Path) -> None:
    detail, status, failures = _parse_junit(str(tmp_path / "missing.xml"), fallback="raw output here")
    assert status is None
    assert "raw output here" in detail
    assert failures == []


# --- jest ----------------------------------------------------------------

def test_parse_jest_success(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "r.json",
        json.dumps({"success": True, "numPassedTests": 3, "numFailedTests": 0, "numTotalTests": 3}),
    )
    detail, status, failures = _parse_jest(path, fallback="")
    assert status is GateStatus.PASS
    assert "3/3 passed" in detail
    assert failures == []


def test_parse_jest_failure(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "r.json",
        json.dumps({
            "success": False, "numPassedTests": 1, "numFailedTests": 1, "numTotalTests": 2,
            "testResults": [{
                "name": "calculator.test.js",
                "assertionResults": [{
                    "status": "failed", "fullName": "add works",
                    "failureMessages": ["expected 5 got -1"],
                }],
            }],
        }),
    )
    detail, status, failures = _parse_jest(path, fallback="")
    assert status is GateStatus.FAIL
    assert "1 failed" in detail
    assert len(failures) == 1
    assert failures[0].identity == "calculator.test.js::add works"


# --- go test ---------------------------------------------------------------

_GO_STREAM_PASS = "\n".join([
    json.dumps({"Action": "run", "Package": "pkg", "Test": "TestAdd"}),
    json.dumps({"Action": "pass", "Package": "pkg", "Test": "TestAdd"}),
])

_GO_STREAM_FAIL = "\n".join([
    json.dumps({"Action": "run", "Package": "pkg", "Test": "TestAdd"}),
    json.dumps({"Action": "fail", "Package": "pkg", "Test": "TestAdd"}),
])


def test_parse_go_test_pass() -> None:
    detail, status, failures = _parse_go_test(_GO_STREAM_PASS, fallback="")
    assert status is GateStatus.PASS
    assert "1 passed, 0 failed" in detail
    assert failures == []


def test_parse_go_test_fail() -> None:
    detail, status, failures = _parse_go_test(_GO_STREAM_FAIL, fallback="")
    assert status is GateStatus.FAIL
    assert "pkg.TestAdd" in detail
    assert failures[0].identity == "pkg.TestAdd"


def test_parse_go_test_no_events_falls_back() -> None:
    detail, status, failures = _parse_go_test("not json at all", fallback="build failed: syntax error")
    assert status is None
    assert "syntax error" in detail
    assert failures == []


# --- tsc error regex ---------------------------------------------------------

def test_tsc_error_regex_matches_default_output() -> None:
    line = "src/calculator.ts(2,10): error TS2322: Type 'string' is not assignable to type 'number'."
    m = _TSC_ERROR_RE.match(line)
    assert m is not None
    assert m.group("file") == "src/calculator.ts"
    assert m.group("line") == "2"
    assert m.group("code") == "TS2322"


def test_tsc_error_regex_ignores_non_error_lines() -> None:
    assert _TSC_ERROR_RE.match("Found 0 errors. Watching for file changes.") is None


def test_parse_tsc_clean_run() -> None:
    detail, status, failures = _parse_tsc(_proc(stdout="", returncode=0))
    assert status is GateStatus.PASS
    assert "0 error(s)" in detail
    assert failures == []


def test_parse_tsc_with_errors() -> None:
    stdout = "src/calculator.ts(2,10): error TS2322: Type 'string' is not assignable to type 'number'."
    detail, status, failures = _parse_tsc(_proc(stdout=stdout, returncode=1))
    assert status is GateStatus.FAIL
    assert "1 error(s)" in detail
    assert "TS2322" in detail
    assert failures[0].identity == "src/calculator.ts:TS2322"


# --- mypy --------------------------------------------------------------------

def test_parse_mypy_clean_run() -> None:
    detail, status, failures = _parse_mypy(_proc(stdout="", returncode=0))
    assert status is GateStatus.PASS
    assert failures == []


def test_parse_mypy_with_errors() -> None:
    line = json.dumps(
        {"file": "calculator.py", "line": 3, "code": "assignment", "message": "bad", "severity": "error"}
    )
    detail, status, failures = _parse_mypy(_proc(stdout=line, returncode=1))
    assert status is GateStatus.FAIL
    assert "1 error(s)" in detail
    assert "assignment" in detail
    assert failures[0].identity == "calculator.py:assignment"


# --- eslint --------------------------------------------------------------------

def test_parse_eslint_clean_run() -> None:
    payload = json.dumps([{"filePath": "a.js", "errorCount": 0, "messages": []}])
    detail, status, failures = _parse_eslint(_proc(stdout=payload, returncode=0))
    assert status is GateStatus.PASS
    assert "0 error(s)" in detail
    assert failures == []


def test_parse_eslint_with_errors() -> None:
    payload = json.dumps([
        {
            "filePath": "a.js",
            "errorCount": 1,
            "messages": [{"severity": 2, "line": 4, "ruleId": "no-unused-vars", "message": "x unused"}],
        }
    ])
    detail, status, failures = _parse_eslint(_proc(stdout=payload, returncode=1))
    assert status is GateStatus.FAIL
    assert "1 error(s)" in detail
    assert "no-unused-vars" in detail
    assert failures[0].identity == "a.js:no-unused-vars"


# --- ruff --------------------------------------------------------------------

def test_parse_ruff_clean_run() -> None:
    detail, status, failures = _parse_ruff(_proc(stdout="[]", returncode=0))
    assert status is GateStatus.PASS
    assert "0 violation(s)" in detail
    assert failures == []


def test_parse_ruff_with_violations() -> None:
    payload = json.dumps([
        {"filename": "a.py", "code": "F401", "message": "unused import", "location": {"row": 1}}
    ])
    detail, status, failures = _parse_ruff(_proc(stdout=payload, returncode=1))
    assert status is GateStatus.FAIL
    assert "1 violation(s)" in detail
    assert "F401" in detail
    assert failures[0].identity == "a.py:F401"
