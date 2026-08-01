"""AiderAdapter implements the exact same Adapter Protocol as every other
adapter — these tests drive it through a `FakeSandbox` to exercise its
regex-based summary-line parsing (Aider has no JSON output mode) without
depending on a real `aider` binary or a real sandbox backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.sandbox_fakes import FakeSandbox
from verdict.adapters.aider import AiderAdapter, AiderAdapterError
from verdict.sandbox.base import ExecResult


def test_run_parses_tokens_and_session_cost_from_summary_line(tmp_path: Path) -> None:
    stdout = (
        "Applying edits...\n"
        "Tokens: 2.3k sent, 456 received. Cost: $0.01 message, $0.03 session.\n"
    )
    sandbox = FakeSandbox(result=ExecResult(exit_code=0, stdout=stdout, stderr=""))

    adapter = AiderAdapter()
    result = adapter.run("fix the bug", tmp_path, sandbox=sandbox)

    assert result.tokens_input == 2300
    assert result.tokens_output == 456
    assert result.cost_usd == pytest.approx(0.03)
    assert sandbox.calls[0]["cmd"] == ["aider", "--message", "fix the bug", "--yes-always"]


def test_run_falls_back_gracefully_when_summary_line_is_missing(tmp_path: Path) -> None:
    sandbox = FakeSandbox(result=ExecResult(exit_code=0, stdout="no summary here", stderr=""))
    adapter = AiderAdapter()
    result = adapter.run("task", tmp_path, sandbox=sandbox)

    assert result.tokens_input == 0
    assert result.tokens_output == 0
    assert result.cost_usd is None


def test_run_raises_when_cli_missing(tmp_path: Path) -> None:
    sandbox = FakeSandbox(result=ExecResult(exit_code=127, stdout="", stderr="no such file"))
    adapter = AiderAdapter()

    with pytest.raises(AiderAdapterError, match="aider"):
        adapter.run("task", tmp_path, sandbox=sandbox)


def test_run_raises_on_nonzero_exit_with_no_summary(tmp_path: Path) -> None:
    sandbox = FakeSandbox(result=ExecResult(exit_code=1, stdout="", stderr="boom"))
    adapter = AiderAdapter()

    with pytest.raises(AiderAdapterError, match="boom"):
        adapter.run("task", tmp_path, sandbox=sandbox)


def test_run_raises_on_timeout(tmp_path: Path) -> None:
    sandbox = FakeSandbox(
        result=ExecResult(exit_code=124, stdout="", stderr="timed out", timed_out=True)
    )
    adapter = AiderAdapter(timeout_seconds=1)

    with pytest.raises(AiderAdapterError, match="did not finish"):
        adapter.run("task", tmp_path, sandbox=sandbox)
