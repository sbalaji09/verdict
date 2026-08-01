"""CodexAdapter implements the exact same Adapter Protocol as every other
adapter — these tests drive it through a `FakeSandbox` to exercise its
JSONL-event parsing without depending on a real `codex` binary or a real
sandbox backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.sandbox_fakes import FakeSandbox
from verdict.adapters.codex import CodexAdapter, CodexAdapterError
from verdict.sandbox.base import ExecResult


def test_run_parses_last_usage_event_and_last_text_event(tmp_path: Path) -> None:
    events = [
        {"type": "token_count", "usage": {"input_tokens": 50, "output_tokens": 10}},
        {"type": "message", "text": "working on it"},
        {"type": "token_count", "usage": {"input_tokens": 120, "output_tokens": 45}},
        {"type": "message", "text": "done, fixed the bug"},
    ]
    stdout = "\n".join(json.dumps(e) for e in events)
    sandbox = FakeSandbox(result=ExecResult(exit_code=0, stdout=stdout, stderr=""))

    adapter = CodexAdapter()
    result = adapter.run("fix the bug", tmp_path, sandbox=sandbox)

    assert result.tokens_input == 120
    assert result.tokens_output == 45
    assert result.cost_usd is None
    assert result.raw_output == "done, fixed the bug"
    command = sandbox.calls[0]["cmd"]
    assert command[0] == "codex"
    assert "fix the bug" in command


def test_run_skips_unparseable_lines_without_crashing(tmp_path: Path) -> None:
    usage_line = json.dumps({"usage": {"input_tokens": 5, "output_tokens": 2}})
    stdout = f"not json\n{usage_line}\n{{also not json"
    sandbox = FakeSandbox(result=ExecResult(exit_code=0, stdout=stdout, stderr=""))

    adapter = CodexAdapter()
    result = adapter.run("task", tmp_path, sandbox=sandbox)

    assert result.tokens_input == 5
    assert result.tokens_output == 2


def test_run_raises_when_cli_missing(tmp_path: Path) -> None:
    sandbox = FakeSandbox(result=ExecResult(exit_code=127, stdout="", stderr="no such file"))
    adapter = CodexAdapter()

    with pytest.raises(CodexAdapterError, match="codex"):
        adapter.run("task", tmp_path, sandbox=sandbox)


def test_run_raises_on_nonzero_exit_with_no_events(tmp_path: Path) -> None:
    sandbox = FakeSandbox(result=ExecResult(exit_code=1, stdout="", stderr="boom"))
    adapter = CodexAdapter()

    with pytest.raises(CodexAdapterError, match="boom"):
        adapter.run("task", tmp_path, sandbox=sandbox)


def test_run_raises_on_timeout(tmp_path: Path) -> None:
    sandbox = FakeSandbox(
        result=ExecResult(exit_code=124, stdout="", stderr="timed out", timed_out=True)
    )
    adapter = CodexAdapter(timeout_seconds=1)

    with pytest.raises(CodexAdapterError, match="did not finish"):
        adapter.run("task", tmp_path, sandbox=sandbox)
