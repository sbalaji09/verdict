"""Unit tests for the `Sandbox` contract via `LocalSandbox` — fast, no
Docker required. Docker-specific containment behavior (network/env/
filesystem isolation) is exercised separately in
`test_sandbox_docker_adversarial.py`, gated behind a reachable daemon.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from verdict.sandbox.local import LocalSandbox


@pytest.fixture
def sandbox() -> LocalSandbox:
    return LocalSandbox()


def test_exec_returns_stdout_stderr_and_exit_code(sandbox: LocalSandbox, tmp_path: Path) -> None:
    result = sandbox.exec(["sh", "-c", "echo out; echo err >&2; exit 3"], cwd=tmp_path)
    assert result.exit_code == 3
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"
    assert not result.timed_out


def test_exec_runs_in_the_given_cwd(sandbox: LocalSandbox, tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("hello")
    result = sandbox.exec(["cat", "marker.txt"], cwd=tmp_path)
    assert result.stdout == "hello"


def test_exec_does_not_inherit_the_host_environment(
    sandbox: LocalSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERDICT_TEST_SECRET", "leaked-value")
    result = sandbox.exec(["sh", "-c", 'echo "${VERDICT_TEST_SECRET:-UNSET}"'], cwd=tmp_path)
    assert result.stdout.strip() == "UNSET"


def test_exec_env_only_carries_what_is_explicitly_passed(sandbox: LocalSandbox, tmp_path: Path) -> None:
    result = sandbox.exec(
        ["sh", "-c", 'echo "$FOO"'], cwd=tmp_path, env={"FOO": "bar"}
    )
    assert result.stdout.strip() == "bar"


def test_exec_times_out(sandbox: LocalSandbox, tmp_path: Path) -> None:
    result = sandbox.exec(["sleep", "5"], cwd=tmp_path, timeout_seconds=1)
    assert result.timed_out
    assert result.killed_reason == "timeout"
    assert result.exit_code == 124


def test_exec_missing_binary_returns_127(sandbox: LocalSandbox, tmp_path: Path) -> None:
    result = sandbox.exec(["verdict-definitely-not-a-real-binary"], cwd=tmp_path)
    assert result.exit_code == 127


def test_exec_background_can_be_terminated(sandbox: LocalSandbox, tmp_path: Path) -> None:
    handle = sandbox.exec_background(["sh", "-c", "echo started; sleep 30"], cwd=tmp_path)
    try:
        deadline = time.monotonic() + 5
        while "started" not in handle.read_output() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert handle.is_alive()
    finally:
        handle.terminate(grace_seconds=2)
    assert not handle.is_alive()
