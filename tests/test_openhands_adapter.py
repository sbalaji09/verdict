"""OpenHandsAdapter implements the exact same Adapter Protocol as every
other adapter — these tests drive it through a `FakeSandbox` to exercise
its (honestly minimal) output handling without depending on a real
`openhands` binary or a real sandbox backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.sandbox_fakes import FakeSandbox
from verdict.adapters.openhands import OpenHandsAdapter, OpenHandsAdapterError
from verdict.sandbox.base import ExecResult


def test_run_reports_no_usage_or_cost_but_keeps_raw_output(tmp_path: Path) -> None:
    sandbox = FakeSandbox(result=ExecResult(exit_code=0, stdout="agent finished the task", stderr=""))

    adapter = OpenHandsAdapter()
    result = adapter.run("fix the bug", tmp_path, sandbox=sandbox)

    assert result.tokens_input == 0
    assert result.tokens_output == 0
    assert result.cost_usd is None
    assert result.raw_output == "agent finished the task"
    call = sandbox.calls[0]
    assert call["cmd"] == ["openhands", "run", "--task", "fix the bug", "--no-auto-continue"]
    assert call["cwd"] == tmp_path


def test_run_raises_when_cli_missing(tmp_path: Path) -> None:
    sandbox = FakeSandbox(result=ExecResult(exit_code=127, stdout="", stderr="no such file"))
    adapter = OpenHandsAdapter()

    with pytest.raises(OpenHandsAdapterError, match="openhands"):
        adapter.run("task", tmp_path, sandbox=sandbox)


def test_run_raises_on_nonzero_exit(tmp_path: Path) -> None:
    sandbox = FakeSandbox(result=ExecResult(exit_code=1, stdout="", stderr="boom"))
    adapter = OpenHandsAdapter()

    with pytest.raises(OpenHandsAdapterError, match="boom"):
        adapter.run("task", tmp_path, sandbox=sandbox)


def test_run_raises_on_timeout(tmp_path: Path) -> None:
    sandbox = FakeSandbox(
        result=ExecResult(exit_code=124, stdout="", stderr="timed out", timed_out=True)
    )
    adapter = OpenHandsAdapter(timeout_seconds=1)

    with pytest.raises(OpenHandsAdapterError, match="did not finish"):
        adapter.run("task", tmp_path, sandbox=sandbox)
