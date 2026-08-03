"""Phase 18: retry-with-backoff for a 429/5xx from the provider a CLI
wraps. `test_call_with_backoff_*`/`test_looks_transient_*`/`test_backoff_
delay_*` exercise `adapters/backoff.py`'s primitives directly, with no
sleeping and no real subprocess. `test_all_adapters_retry_a_transient_
failure_and_then_succeed` proves the "per-adapter" half of the brief: all
five shipped adapters actually retry through `FakeSandbox` when the
canned result looks like a rate limit, then return a normal
`AttemptResult` once a later call succeeds — the same "an injected 429 is
retried" claim the brief asks for, checked against every adapter, not
just one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.sandbox_fakes import FakeSandbox
from verdict.adapters.aider import AiderAdapter
from verdict.adapters.backoff import (
    DEFAULT_MAX_RETRIES,
    backoff_delay,
    call_with_backoff,
    exec_result_is_transient,
    looks_transient,
)
from verdict.adapters.claude_code import ClaudeCodeAdapter, ClaudeCodeAdapterError
from verdict.adapters.codex import CodexAdapter
from verdict.adapters.cursor import CursorAdapter, CursorAdapterError
from verdict.adapters.openhands import OpenHandsAdapter
from verdict.sandbox.base import ExecResult

# --- looks_transient / exec_result_is_transient --------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Error: 429 Too Many Requests",
        "HTTP 503 Service Unavailable",
        "rate limit exceeded, please retry",
        "Rate Limited by upstream",
        "the model is currently overloaded",
        "502 Bad Gateway",
        "Gateway Timeout",
        "Internal Server Error",
    ],
)
def test_looks_transient_matches_known_rate_limit_and_server_error_shapes(text: str) -> None:
    assert looks_transient(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no such file or directory",
        "SyntaxError: invalid syntax",
        "test_calculator.py::test_add FAILED",
    ],
)
def test_looks_transient_does_not_flag_unrelated_failures(text: str) -> None:
    assert looks_transient(text) is False


def test_exec_result_is_transient_ignores_timeouts() -> None:
    result = ExecResult(exit_code=124, stdout="", stderr="429 rate limited", timed_out=True)
    assert exec_result_is_transient(result) is False


def test_exec_result_is_transient_ignores_success() -> None:
    result = ExecResult(exit_code=0, stdout="429 mentioned but irrelevant", stderr="")
    assert exec_result_is_transient(result) is False


def test_exec_result_is_transient_ignores_command_not_found() -> None:
    result = ExecResult(exit_code=127, stdout="", stderr="429 too many requests")
    assert exec_result_is_transient(result) is False


def test_exec_result_is_transient_flags_a_real_rate_limit() -> None:
    result = ExecResult(exit_code=1, stdout="", stderr="429 Too Many Requests")
    assert exec_result_is_transient(result) is True


def test_exec_result_is_transient_does_not_flag_an_ordinary_failure() -> None:
    result = ExecResult(exit_code=1, stdout="", stderr="invalid task description")
    assert exec_result_is_transient(result) is False


# --- backoff_delay ---------------------------------------------------


def test_backoff_delay_is_bounded_by_the_exponential_cap() -> None:
    for attempt in range(6):
        delay = backoff_delay(attempt, base_delay_seconds=1.0, max_delay_seconds=10.0, jitter=lambda: 1.0)
        assert 0.0 <= delay <= 10.0


def test_backoff_delay_grows_with_attempt_number_before_capping() -> None:
    d0 = backoff_delay(0, base_delay_seconds=1.0, max_delay_seconds=1000.0, jitter=lambda: 1.0)
    d1 = backoff_delay(1, base_delay_seconds=1.0, max_delay_seconds=1000.0, jitter=lambda: 1.0)
    d2 = backoff_delay(2, base_delay_seconds=1.0, max_delay_seconds=1000.0, jitter=lambda: 1.0)
    assert d0 == 1.0
    assert d1 == 2.0
    assert d2 == 4.0


def test_backoff_delay_scales_with_jitter_draw() -> None:
    assert backoff_delay(0, base_delay_seconds=2.0, max_delay_seconds=100.0, jitter=lambda: 0.5) == 1.0


# --- call_with_backoff -------------------------------------------------


def test_call_with_backoff_returns_immediately_on_first_success() -> None:
    calls = []

    def attempt() -> str:
        calls.append(1)
        return "ok"

    sleeps: list[float] = []
    result = call_with_backoff(attempt, is_transient=lambda r: False, sleep=sleeps.append)

    assert result == "ok"
    assert len(calls) == 1
    assert sleeps == []


def test_call_with_backoff_retries_transient_results_then_succeeds() -> None:
    results = iter(["fail", "fail", "ok"])
    sleeps: list[float] = []

    result = call_with_backoff(
        lambda: next(results),
        is_transient=lambda r: r == "fail",
        sleep=sleeps.append,
        jitter=lambda: 1.0,
    )

    assert result == "ok"
    assert len(sleeps) == 2  # slept once before each of the 2 retries


def test_call_with_backoff_uses_exponential_delays_between_retries() -> None:
    results = iter(["fail"] * 4 + ["ok"])
    sleeps: list[float] = []

    call_with_backoff(
        lambda: next(results),
        is_transient=lambda r: r == "fail",
        base_delay_seconds=1.0,
        max_delay_seconds=1000.0,
        sleep=sleeps.append,
        jitter=lambda: 1.0,
    )

    assert sleeps == [1.0, 2.0, 4.0, 8.0]


def test_call_with_backoff_gives_up_after_max_retries_and_returns_last_result() -> None:
    calls = []

    def attempt() -> str:
        calls.append(1)
        return "still-failing"

    sleeps: list[float] = []
    result = call_with_backoff(
        attempt, is_transient=lambda r: True, max_retries=3, sleep=sleeps.append, jitter=lambda: 0.0
    )

    assert result == "still-failing"
    assert len(calls) == 1 + 3  # first try + 3 retries, then gives up
    assert len(sleeps) == 3


def test_call_with_backoff_default_max_retries_matches_module_constant() -> None:
    calls = []

    def attempt() -> str:
        calls.append(1)
        return "fail"

    call_with_backoff(attempt, is_transient=lambda r: True, sleep=lambda s: None, jitter=lambda: 0.0)
    assert len(calls) == 1 + DEFAULT_MAX_RETRIES


# --- per-adapter integration: an injected 429 is retried ------------------

_CLAUDE_SUCCESS = json.dumps(
    {"result": "done", "usage": {"input_tokens": 10, "output_tokens": 5}, "total_cost_usd": 0.01}
)
_CODEX_SUCCESS = json.dumps({"usage": {"input_tokens": 10, "output_tokens": 5}, "text": "done"})
_AIDER_SUCCESS = "Tokens: 1.0k sent, 500 received. Cost: $0.01 message, $0.03 session."

_RATE_LIMITED = ExecResult(exit_code=1, stdout="", stderr="429 Too Many Requests")


@pytest.mark.parametrize(
    "adapter_factory,success_stdout,cmd0",
    [
        (ClaudeCodeAdapter, _CLAUDE_SUCCESS, "claude"),
        (CursorAdapter, _CLAUDE_SUCCESS, "cursor-agent"),
        (CodexAdapter, _CODEX_SUCCESS, "codex"),
        (AiderAdapter, _AIDER_SUCCESS, "aider"),
        (OpenHandsAdapter, "done", "openhands"),
    ],
)
def test_all_adapters_retry_a_transient_failure_and_then_succeed(
    adapter_factory, success_stdout: str, cmd0: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)  # no real waiting in tests
    sandbox = FakeSandbox(
        results=[
            _RATE_LIMITED,
            _RATE_LIMITED,
            ExecResult(exit_code=0, stdout=success_stdout, stderr=""),
        ]
    )
    adapter = adapter_factory()

    result = adapter.run("fix the bug", tmp_path, sandbox=sandbox)

    assert len(sandbox.calls) == 3
    assert sandbox.calls[0]["cmd"][0] == cmd0
    assert result.raw_output is not None


@pytest.mark.parametrize(
    "adapter_factory,error_cls",
    [
        (ClaudeCodeAdapter, ClaudeCodeAdapterError),
        (CursorAdapter, CursorAdapterError),
    ],
)
def test_adapter_still_raises_after_exhausting_backoff_retries(
    adapter_factory, error_cls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    # Always rate-limited — never recovers within the retry budget.
    sandbox = FakeSandbox(results=[_RATE_LIMITED] * (DEFAULT_MAX_RETRIES + 1))
    adapter = adapter_factory()

    with pytest.raises(error_cls):
        adapter.run("fix the bug", tmp_path, sandbox=sandbox)

    assert len(sandbox.calls) == DEFAULT_MAX_RETRIES + 1


def test_adapter_does_not_retry_a_non_transient_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    sandbox = FakeSandbox(result=ExecResult(exit_code=1, stdout="", stderr="invalid task"))
    adapter = CursorAdapter()

    with pytest.raises(CursorAdapterError):
        adapter.run("task", tmp_path, sandbox=sandbox)

    assert len(sandbox.calls) == 1  # no retry — this never looked transient
