"""CursorAdapter's contract is the same Adapter Protocol every adapter
implements (`name` + `run(task, worktree) -> AttemptResult`) — these tests
exercise its subprocess/JSON-parsing logic by faking `subprocess.run`,
never a real `cursor-agent` binary (which isn't installed in CI or this
sandbox). This is the same testing shape a real integration would need for
every CLI-subprocess adapter: verify Verdict's side of the contract
without depending on the external tool actually being present.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from verdict.adapters.cursor import CursorAdapter, CursorAdapterError


def _fake_result(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["cursor-agent"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_run_parses_usage_and_cost_from_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = {
        "result": "done",
        "usage": {"input_tokens": 100, "output_tokens": 40},
        "total_cost_usd": 0.05,
    }
    captured_command: list[str] = []

    def fake_run(command, cwd, capture_output, text, timeout):
        captured_command.extend(command)
        assert cwd == tmp_path
        return _fake_result(stdout=json.dumps(payload))

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = CursorAdapter()
    result = adapter.run("fix the bug", tmp_path)

    assert result.tokens_input == 100
    assert result.tokens_output == 40
    assert result.cost_usd == 0.05
    assert result.raw_output == "done"
    assert captured_command[0] == "cursor-agent"
    assert "fix the bug" in captured_command


def test_run_falls_back_gracefully_on_non_json_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_result(stdout="not json", returncode=0)
    )
    adapter = CursorAdapter()
    result = adapter.run("task", tmp_path)

    assert result.tokens_input == 0
    assert result.tokens_output == 0
    assert result.cost_usd is None
    assert result.raw_output == "not json"


def test_run_raises_when_cli_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = CursorAdapter()

    with pytest.raises(CursorAdapterError, match="cursor-agent"):
        adapter.run("task", tmp_path)


def test_run_raises_on_nonzero_exit_with_no_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_result(stdout="", stderr="boom", returncode=1)
    )
    adapter = CursorAdapter()

    with pytest.raises(CursorAdapterError, match="boom"):
        adapter.run("task", tmp_path)


def test_run_raises_on_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="cursor-agent", timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = CursorAdapter(timeout_seconds=1)

    with pytest.raises(CursorAdapterError, match="did not finish"):
        adapter.run("task", tmp_path)
