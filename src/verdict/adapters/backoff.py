"""Shared retry-with-backoff for the one class of `Adapter` failure worth
retrying automatically, from inside the adapter itself, rather than
surfacing straight to Phase 11's infra-retry loop: a paid, rate-limited
API's 429, or a transient 5xx from the provider's own backend.

Every real adapter (Phase 5) shells out to a CLI rather than calling a
vendor API directly (see each adapter's own module docstring for why),
so Verdict never sees a raw HTTP status code — only whatever the CLI
printed on a failed run. `looks_transient` is a deliberately generic text
scan for the handful of ways a CLI is likely to surface a 429/5xx to its
own stdout/stderr (a literal status code, "rate limit", "too many
requests", "overloaded", "service unavailable", ...); every adapter wires
it into its own `run()` around its own `sandbox.exec()` call, since only
the adapter knows the shape of the command/output it's actually parsing
— this module supplies the shared mechanism, "per-adapter" is where it's
applied.

Why this lives inside the adapter rather than as a blanket retry in
`runner.py`'s existing `max_error_retries` loop: that loop already
retries any `AdapterError` a bounded number of times, but with NO backoff
and NO distinction between "the CLI binary isn't installed" (retrying is
pointless) and "the provider briefly rate-limited us" (retrying after a
short wait is exactly right). Retrying instantly against a live rate
limit just trips it again; backing off first is the entire point of this
module. Keeping the retry INSIDE the adapter also means it's invisible to
`TaskRun.attempts`/`attempt_count` — from `runner.py`'s point of view,
`adapter.run()` either takes a little longer than usual and succeeds, or
still fails after using its own retry budget and raises `AdapterError`
exactly as before, falling through to the existing infra-retry path
completely unchanged.

**Cost accounting stays exact through a retry.** `call_with_backoff`
returns only the LAST attempt's raw result — a rejected 429 call never
produced billable output for a CLI to report a `total_cost_usd`/usage
figure from in the first place, so there is nothing from a failed,
retried call for an adapter to accidentally sum into the eventual
successful one's numbers. Each adapter's own usage-parsing code (already
written to read `usage`/`total_cost_usd` from exactly one JSON payload)
needed no change to keep respecting "unknown cost stays `None`, never a
partial sum" — the invariant was never at risk, since retrying just means
calling `sandbox.exec` again, not accumulating anything across calls.
"""

from __future__ import annotations

import random
import re
import time
from typing import Callable, TypeVar

from verdict.sandbox import ExecResult

DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_DELAY_SECONDS = 2.0
DEFAULT_MAX_DELAY_SECONDS = 30.0

_TRANSIENT_RE = re.compile(
    r"\b(429|5\d{2}|rate.?limit(?:ed)?|too many requests|overloaded|"
    r"service unavailable|internal server error|bad gateway|gateway timeout)\b",
    re.IGNORECASE,
)


def looks_transient(text: str) -> bool:
    """A generic scan for the handful of ways a CLI is likely to print a
    429/5xx from the provider it wraps. Deliberately loose (a real 5xx
    status or a literal "rate limit" substring anywhere in stdout/stderr
    is enough) — the cost of a false positive is one extra backoff-and-
    retry cycle against a command that may have failed for some other
    reason anyway; the cost of a false negative is silently NOT retrying
    a transient failure, which is the worse mistake to make against a
    paid API a team is relying on for a real answer.
    """
    return bool(_TRANSIENT_RE.search(text))


def backoff_delay(
    attempt: int,
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS,
    jitter: Callable[[], float] | None = None,
) -> float:
    """Exponential backoff with full jitter (`attempt` is 0-indexed: the
    delay before the SECOND call overall). Full jitter (`jitter() * exp`,
    not `exp` plus/minus a small jitter term) is the standard
    recommendation for exactly this problem: several clients backing off
    on a fixed exponential schedule in lockstep would all retry at the
    same instant and re-trip the same rate limit; multiplying by a
    uniform `[0, 1)` draw spreads retries out instead of synchronizing
    them.
    """
    exp = min(max_delay_seconds, base_delay_seconds * (2**attempt))
    draw: float = jitter() if jitter is not None else random.random()
    return float(draw * exp)


def exec_result_is_transient(
    result: ExecResult, permanent_exit_codes: frozenset[int] = frozenset({127})
) -> bool:
    """The `is_transient` predicate every CLI-subprocess adapter passes to
    `call_with_backoff`: a timeout is never retried here (Phase 9's
    per-call timeout is already generous, and a hung process isn't a rate
    limit); exit code 0 is success, nothing to retry; `permanent_exit_
    codes` defaults to `{127}` ("command not found" — every adapter in
    this codebase raises its own `AdapterError` on 127, and retrying a
    missing binary would just fail exactly the same way `max_retries + 1`
    times for no benefit). Anything else falls through to a text scan of
    the combined stdout+stderr via `looks_transient`.
    """
    if result.timed_out or result.exit_code == 0 or result.exit_code in permanent_exit_codes:
        return False
    return looks_transient(result.stdout + result.stderr)


R = TypeVar("R")


def call_with_backoff(
    attempt_fn: Callable[[], R],
    is_transient: Callable[[R], bool],
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS,
    sleep: Callable[[float], None] | None = None,
    jitter: Callable[[], float] | None = None,
) -> R:
    """Calls `attempt_fn()` up to `max_retries + 1` times total. After
    each call, if `is_transient(result)` says the failure looks like a
    429/5xx, sleeps (exponential backoff + full jitter) and tries again;
    otherwise returns immediately — a non-transient failure (bad command,
    CLI not installed, a real bug) is not this function's job to retry,
    and `is_transient` returning False on the very first call means it
    returns straight through with zero added latency, same as before this
    wrapper existed.

    Always returns the LAST result, even if every retry was still
    transient — the caller (each adapter's own `run()`) decides what a
    still-failing result means (raise its own `AdapterError`), exactly as
    it already did before this wrapper existed; this function's only job
    is deciding *when to try again*, never what a result means.

    `sleep`/`jitter` are resolved to `time.sleep`/`random.random` inside
    the loop (not bound as default parameter values) specifically so a
    caller — or a test — can monkeypatch the real `time.sleep`/
    `random.random` and have it take effect here without this function
    needing its own injection points threaded through every adapter's
    constructor.
    """
    result = attempt_fn()
    for attempt in range(max_retries):
        if not is_transient(result):
            return result
        delay = backoff_delay(attempt, base_delay_seconds, max_delay_seconds, jitter)
        (sleep or time.sleep)(delay)
        result = attempt_fn()
    return result
