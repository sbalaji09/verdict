"""Phase 18: a hard, per-run cost ceiling that aborts a task's own retry
loop cleanly — every attempt already made is preserved exactly as it ran
(no results lost), and the abort itself is marked `VerdictStatus.ERROR`
so it ties into Phase 11's existing "not an agent failure" accounting:
`ConfigResult.pass_rate` already excludes ERROR from both sides of the
ratio, and this reuses that machinery rather than inventing a new one.

Distinct from (and composable with) Phase 15's suite-level
`cost_ceiling_usd`, which bounds total spend across MANY (config, task)
pairs in one `run_suite` call — this one bounds a single task's own
agent-retry spend, at the `run_with_retries` layer.
"""

from __future__ import annotations

from pathlib import Path

from verdict.runner import run_with_retries
from verdict.schema import AttemptResult, VerdictStatus
from verdict.suite import BenchConfig, run_suite
from verdict.suite.loader import SuiteTask


class _AlwaysFailsButCostsAdapter:
    """A real agent attempt that never fixes the bug and reports a real,
    known cost every time — the adapter this ceiling is meant to bound.
    """

    name = "always-fails"

    def __init__(self, cost_per_attempt: float = 0.01) -> None:
        self.calls = 0
        self._cost_per_attempt = cost_per_attempt

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        self.calls += 1
        return AttemptResult(diff="", tokens_input=10, tokens_output=5, cost_usd=self._cost_per_attempt)


class _UnknownCostAdapter:
    """Never reports its own cost — stands in for OpenHands/Codex today.
    A dollar ceiling can never fire against this adapter, honestly.
    """

    name = "no-cost-reporting"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        self.calls += 1
        return AttemptResult(diff="", tokens_input=10, tokens_output=5, cost_usd=None)


# --- run_with_retries: the per-run ceiling ------------------------------


def test_cost_ceiling_stops_further_retries_once_reached(git_repo: Path) -> None:
    adapter = _AlwaysFailsButCostsAdapter(cost_per_attempt=0.01)
    task_run = run_with_retries(
        task="fix it", repo=git_repo, adapter=adapter, max_attempts=5, cost_ceiling_usd=0.02
    )

    # $0.01, $0.02 — ceiling met after the 2nd attempt, so a 3rd never runs
    # even though max_attempts=5 allowed up to 3 more.
    assert adapter.calls == 2
    assert task_run.attempt_count == 3  # 2 real attempts + 1 abort marker
    assert task_run.final.status is VerdictStatus.ERROR
    assert "cost ceiling" in (task_run.final.error or "")
    assert task_run.errored is True
    assert task_run.done is False


def test_cost_ceiling_preserves_every_real_attempt_already_made(git_repo: Path) -> None:
    adapter = _AlwaysFailsButCostsAdapter(cost_per_attempt=0.01)
    task_run = run_with_retries(
        task="fix it", repo=git_repo, adapter=adapter, max_attempts=10, cost_ceiling_usd=0.015
    )

    # Ceiling ($0.015) is crossed after 2 real attempts ($0.02 total).
    real_attempts = task_run.attempts[:-1]
    abort = task_run.attempts[-1]
    assert len(real_attempts) == 2
    assert all(v.status is VerdictStatus.NOT_DONE for v in real_attempts)
    assert all(v.attempt.cost_usd == 0.01 for v in real_attempts)
    assert abort.status is VerdictStatus.ERROR
    assert abort.attempt.cost_usd == 0.0  # the marker itself spends nothing new
    # Total cost across the whole TaskRun still reflects real spend, marker included.
    assert task_run.total_cost_usd == 0.02


def test_cost_ceiling_does_not_abort_a_run_that_finishes_within_budget(git_repo: Path) -> None:
    adapter = _AlwaysFailsButCostsAdapter(cost_per_attempt=0.01)
    task_run = run_with_retries(
        task="fix it", repo=git_repo, adapter=adapter, max_attempts=2, cost_ceiling_usd=100.0
    )

    assert adapter.calls == 2
    assert task_run.attempt_count == 2  # no abort marker — never even close to the ceiling
    assert task_run.final.status is VerdictStatus.NOT_DONE
    assert task_run.errored is False


def test_cost_ceiling_never_appends_a_marker_on_the_final_allowed_attempt(git_repo: Path) -> None:
    """If the ceiling is only crossed on the LAST attempt `max_attempts`
    would have allowed anyway, there's no further attempt being skipped —
    no synthetic marker should be added; the run just ends normally.
    """
    adapter = _AlwaysFailsButCostsAdapter(cost_per_attempt=0.01)
    task_run = run_with_retries(
        task="fix it", repo=git_repo, adapter=adapter, max_attempts=2, cost_ceiling_usd=0.015
    )

    assert adapter.calls == 2
    assert task_run.attempt_count == 2  # both attempts real — no abort marker needed
    assert task_run.final.status is VerdictStatus.NOT_DONE


def test_cost_ceiling_never_fires_for_an_adapter_with_unknown_cost(git_repo: Path) -> None:
    adapter = _UnknownCostAdapter()
    task_run = run_with_retries(
        task="fix it", repo=git_repo, adapter=adapter, max_attempts=3, cost_ceiling_usd=0.0001
    )

    # Cost is always unknown, so `_known_total_cost` is always None —
    # the ceiling can never trigger; the run uses its full max_attempts.
    assert adapter.calls == 3
    assert task_run.attempt_count == 3
    assert task_run.total_cost_usd is None


def test_cost_ceiling_disabled_by_default(git_repo: Path) -> None:
    adapter = _AlwaysFailsButCostsAdapter(cost_per_attempt=1000.0)
    task_run = run_with_retries(task="fix it", repo=git_repo, adapter=adapter, max_attempts=3)

    assert adapter.calls == 3  # no ceiling given — real spend is real spend, never capped silently
    assert task_run.attempt_count == 3


def test_cost_ceiling_abort_stops_the_agent_from_being_called_again(git_repo: Path) -> None:
    """The abort is a hard stop, not just a marker after the fact — the
    adapter genuinely never gets called a 3rd time once the ceiling fires.
    """
    adapter = _AlwaysFailsButCostsAdapter(cost_per_attempt=0.02)
    run_with_retries(task="fix it", repo=git_repo, adapter=adapter, max_attempts=10, cost_ceiling_usd=0.02)
    assert adapter.calls == 1  # a single $0.02 attempt already meets a $0.02 ceiling


# --- run_suite / leaderboard: excluded from pass-rate math ------------


def test_run_suite_cost_ceiling_abort_is_excluded_from_the_leaderboard(git_repo: Path) -> None:
    """The brief's own acceptance case, end to end through `run_suite`:
    a per-run cost ceiling abort must not count as a failure on the
    leaderboard — `ConfigResult.pass_rate`'s denominator must exclude it,
    exactly as it already excludes any other ERROR-status task
    (Phase 10/11). `run_cost_ceiling_usd` is `run_suite`'s own knob for
    this (distinct from Phase 15's suite-wide `cost_ceiling_usd`), threaded
    straight down to each `(config, task)` pair's own `run_with_retries`.
    """
    task = SuiteTask(name="fix-it", task="fix it", repo=git_repo)
    pricey = _AlwaysFailsButCostsAdapter(cost_per_attempt=0.05)
    cheap = _AlwaysFailsButCostsAdapter(cost_per_attempt=0.0)
    cheap.name = "cheap"
    configs = [BenchConfig(label="pricey", adapter=pricey), BenchConfig(label="cheap", adapter=cheap)]

    results = run_suite([task], configs, max_attempts=5, run_cost_ceiling_usd=0.06)

    pricey_result, cheap_result = results
    assert pricey_result.tasks_total == 1
    assert pricey_result.tasks_errored == 1
    assert pricey_result.tasks_done == 0
    assert pricey_result.pass_rate == 0.0  # honest zero-graded-tasks floor, not a fabricated number
    assert pricey.calls == 2  # $0.05, $0.10 — ceiling ($0.06) crossed after 2

    # The ceiling is per-(config, task) — "cheap" never approaches it and
    # runs its full max_attempts, completely unaffected by "pricey"'s abort.
    assert cheap_result.tasks_errored == 0
    assert cheap.calls == 5
