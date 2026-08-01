"""OpenHandsAdapter implements the exact same Adapter Protocol as every
other adapter — these tests fake `subprocess.run` to exercise its (honestly
minimal) output handling without depending on a real `openhands` binary.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from verdict.adapters.openhands import OpenHandsAdapter, OpenHandsAdapterError


def _fake_result(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["openhands"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_run_reports_no_usage_or_cost_but_keeps_raw_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_command: list[str] = []

    def fake_run(command, cwd, capture_output, text, timeout):
        captured_command.extend(command)
        assert cwd == tmp_path
        return _fake_result(stdout="agent finished the task")

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = OpenHandsAdapter()
    result = adapter.run("fix the bug", tmp_path)

    assert result.tokens_input == 0
    assert result.tokens_output == 0
    assert result.cost_usd is None
    assert result.raw_output == "agent finished the task"
    assert captured_command == ["openhands", "run", "--task", "fix the bug", "--no-auto-continue"]


def test_run_raises_when_cli_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = OpenHandsAdapter()

    with pytest.raises(OpenHandsAdapterError, match="openhands"):
        adapter.run("task", tmp_path)


def test_run_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_result(stdout="", stderr="boom", returncode=1)
    )
    adapter = OpenHandsAdapter()

    with pytest.raises(OpenHandsAdapterError, match="boom"):
        adapter.run("task", tmp_path)


def test_run_raises_on_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="openhands", timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = OpenHandsAdapter(timeout_seconds=1)

    with pytest.raises(OpenHandsAdapterError, match="did not finish"):
        adapter.run("task", tmp_path)
