"""AiderAdapter implements the exact same Adapter Protocol as every other
adapter — these tests fake `subprocess.run` to exercise its regex-based
summary-line parsing (Aider has no JSON output mode) without depending on
a real `aider` binary.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from verdict.adapters.aider import AiderAdapter, AiderAdapterError


def _fake_result(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(args=["aider"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_parses_tokens_and_session_cost_from_summary_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdout = (
        "Applying edits...\n"
        "Tokens: 2.3k sent, 456 received. Cost: $0.01 message, $0.03 session.\n"
    )
    captured_command: list[str] = []

    def fake_run(command, cwd, capture_output, text, timeout):
        captured_command.extend(command)
        return _fake_result(stdout=stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = AiderAdapter()
    result = adapter.run("fix the bug", tmp_path)

    assert result.tokens_input == 2300
    assert result.tokens_output == 456
    assert result.cost_usd == pytest.approx(0.03)
    assert captured_command == ["aider", "--message", "fix the bug", "--yes-always"]


def test_run_falls_back_gracefully_when_summary_line_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_result(stdout="no summary here", returncode=0)
    )
    adapter = AiderAdapter()
    result = adapter.run("task", tmp_path)

    assert result.tokens_input == 0
    assert result.tokens_output == 0
    assert result.cost_usd is None


def test_run_raises_when_cli_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = AiderAdapter()

    with pytest.raises(AiderAdapterError, match="aider"):
        adapter.run("task", tmp_path)


def test_run_raises_on_nonzero_exit_with_no_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_result(stdout="", stderr="boom", returncode=1)
    )
    adapter = AiderAdapter()

    with pytest.raises(AiderAdapterError, match="boom"):
        adapter.run("task", tmp_path)


def test_run_raises_on_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="aider", timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = AiderAdapter(timeout_seconds=1)

    with pytest.raises(AiderAdapterError, match="did not finish"):
        adapter.run("task", tmp_path)
