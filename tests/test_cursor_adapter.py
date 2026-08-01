"""CursorAdapter's contract is the same Adapter Protocol every adapter
implements (`name` + `run(task, worktree, sandbox) -> AttemptResult`) —
these tests exercise its command-construction/JSON-parsing logic by
driving it through a `FakeSandbox`, never a real `cursor-agent` binary
(which isn't installed in CI or this sandbox). This is the same testing
shape a real integration would need for every CLI-subprocess adapter:
verify Verdict's side of the contract without depending on the external
tool actually being present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.sandbox_fakes import FakeSandbox
from verdict.adapters.cursor import CursorAdapter, CursorAdapterError
from verdict.sandbox.base import ExecResult


def test_run_parses_usage_and_cost_from_json(tmp_path: Path) -> None:
    payload = {
        "result": "done",
        "usage": {"input_tokens": 100, "output_tokens": 40},
        "total_cost_usd": 0.05,
    }
    sandbox = FakeSandbox(result=ExecResult(exit_code=0, stdout=json.dumps(payload), stderr=""))

    adapter = CursorAdapter()
    result = adapter.run("fix the bug", tmp_path, sandbox=sandbox)

    assert result.tokens_input == 100
    assert result.tokens_output == 40
    assert result.cost_usd == 0.05
    assert result.raw_output == "done"
    call = sandbox.calls[0]
    assert call["cmd"][0] == "cursor-agent"
    assert "fix the bug" in call["cmd"]
    assert call["cwd"] == tmp_path


def test_run_falls_back_gracefully_on_non_json_stdout(tmp_path: Path) -> None:
    sandbox = FakeSandbox(result=ExecResult(exit_code=0, stdout="not json", stderr=""))
    adapter = CursorAdapter()
    result = adapter.run("task", tmp_path, sandbox=sandbox)

    assert result.tokens_input == 0
    assert result.tokens_output == 0
    assert result.cost_usd is None
    assert result.raw_output == "not json"


def test_run_raises_when_cli_missing(tmp_path: Path) -> None:
    sandbox = FakeSandbox(result=ExecResult(exit_code=127, stdout="", stderr="no such file"))
    adapter = CursorAdapter()

    with pytest.raises(CursorAdapterError, match="cursor-agent"):
        adapter.run("task", tmp_path, sandbox=sandbox)


def test_run_raises_on_nonzero_exit_with_no_payload(tmp_path: Path) -> None:
    sandbox = FakeSandbox(result=ExecResult(exit_code=1, stdout="", stderr="boom"))
    adapter = CursorAdapter()

    with pytest.raises(CursorAdapterError, match="boom"):
        adapter.run("task", tmp_path, sandbox=sandbox)


def test_run_raises_on_timeout(tmp_path: Path) -> None:
    sandbox = FakeSandbox(
        result=ExecResult(exit_code=124, stdout="", stderr="timed out", timed_out=True)
    )
    adapter = CursorAdapter(timeout_seconds=1)

    with pytest.raises(CursorAdapterError, match="did not finish"):
        adapter.run("task", tmp_path, sandbox=sandbox)
