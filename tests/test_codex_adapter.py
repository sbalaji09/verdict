"""CodexAdapter implements the exact same Adapter Protocol as every other
adapter — these tests fake `subprocess.run` to exercise its JSONL-event
parsing without depending on a real `codex` binary.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from verdict.adapters.codex import CodexAdapter, CodexAdapterError


def _fake_result(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(args=["codex"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_parses_last_usage_event_and_last_text_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = [
        {"type": "token_count", "usage": {"input_tokens": 50, "output_tokens": 10}},
        {"type": "message", "text": "working on it"},
        {"type": "token_count", "usage": {"input_tokens": 120, "output_tokens": 45}},
        {"type": "message", "text": "done, fixed the bug"},
    ]
    stdout = "\n".join(json.dumps(e) for e in events)
    captured_command: list[str] = []

    def fake_run(command, cwd, capture_output, text, timeout):
        captured_command.extend(command)
        return _fake_result(stdout=stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = CodexAdapter()
    result = adapter.run("fix the bug", tmp_path)

    assert result.tokens_input == 120
    assert result.tokens_output == 45
    assert result.cost_usd is None
    assert result.raw_output == "done, fixed the bug"
    assert captured_command[0] == "codex"
    assert "fix the bug" in captured_command


def test_run_skips_unparseable_lines_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    usage_line = json.dumps({"usage": {"input_tokens": 5, "output_tokens": 2}})
    stdout = f"not json\n{usage_line}\n{{also not json"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_result(stdout=stdout))

    adapter = CodexAdapter()
    result = adapter.run("task", tmp_path)

    assert result.tokens_input == 5
    assert result.tokens_output == 2


def test_run_raises_when_cli_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = CodexAdapter()

    with pytest.raises(CodexAdapterError, match="codex"):
        adapter.run("task", tmp_path)


def test_run_raises_on_nonzero_exit_with_no_events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_result(stdout="", stderr="boom", returncode=1)
    )
    adapter = CodexAdapter()

    with pytest.raises(CodexAdapterError, match="boom"):
        adapter.run("task", tmp_path)


def test_run_raises_on_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="codex", timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = CodexAdapter(timeout_seconds=1)

    with pytest.raises(CodexAdapterError, match="did not finish"):
        adapter.run("task", tmp_path)
