"""Accounting math for Phase 3: cost must be summed across every attempt
(including failed/abandoned ones), never just the winning one, and the
pass-rate-per-dollar formula must never silently guess at an unknown cost.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from verdict.config import TokenPricing, load_config
from verdict.economics import rank
from verdict.runner import _apply_pricing_fallback, run_with_retries
from verdict.schema import AttemptResult, ConfigResult, GateStatus, Provenance, Signal, TaskRun, Verdict


def _verdict(*, done: bool, cost_usd: float | None, tokens_in: int = 100, tokens_out: int = 50) -> Verdict:
    status = GateStatus.PASS if done else GateStatus.FAIL
    return Verdict(
        task="t", agent="mock", repo="/tmp/x",
        attempt=AttemptResult(diff="", tokens_input=tokens_in, tokens_output=tokens_out, cost_usd=cost_usd),
        signals=[Signal(name="test", provenance=Provenance.PROVEN, status=status, detail="")],
    )


# --- TaskRun accounting ------------------------------------------------

def test_task_run_sums_cost_across_all_attempts_not_just_the_winner() -> None:
    task_run = TaskRun(
        task="t", agent="mock", repo="/tmp/x",
        attempts=[
            _verdict(done=False, cost_usd=0.50),   # dead end
            _verdict(done=False, cost_usd=0.30),   # another dead end
            _verdict(done=True, cost_usd=0.20),    # the winner
        ],
    )
    assert task_run.attempt_count == 3
    assert task_run.failed_attempt_count == 2
    assert task_run.total_cost_usd == pytest.approx(1.00)
    assert task_run.total_tokens_input == 300
    assert task_run.total_tokens_output == 150
    assert task_run.done is True


def test_task_run_total_cost_is_none_if_any_attempt_cost_is_unknown() -> None:
    task_run = TaskRun(
        task="t", agent="mock", repo="/tmp/x",
        attempts=[_verdict(done=False, cost_usd=0.50), _verdict(done=True, cost_usd=None)],
    )
    # a partial sum presented as "total" would understate real spend
    assert task_run.total_cost_usd is None


def test_task_run_single_attempt_that_succeeds_has_zero_failed() -> None:
    task_run = TaskRun(task="t", agent="mock", repo="/tmp/x", attempts=[_verdict(done=True, cost_usd=0.10)])
    assert task_run.failed_attempt_count == 0
    assert task_run.attempt_count == 1


# --- ConfigResult / pass-rate-per-dollar --------------------------------

def _task_run(done: bool, cost: float) -> TaskRun:
    return TaskRun(task="t", agent="mock", repo="/tmp/x", attempts=[_verdict(done=done, cost_usd=cost)])


def test_pass_rate_per_dollar_formula() -> None:
    config = ConfigResult(
        label="claude-code / sonnet",
        task_runs=[_task_run(True, 1.0), _task_run(True, 1.0), _task_run(False, 1.0), _task_run(False, 1.0)],
    )
    assert config.tasks_done == 2
    assert config.tasks_total == 4
    assert config.pass_rate == 0.5
    assert config.total_cost_usd == pytest.approx(4.0)
    # 2 done tasks / $4 total = 0.5 verdict-pts/$
    assert config.pass_rate_per_dollar == pytest.approx(0.5)


def test_pass_rate_per_dollar_is_none_when_cost_unknown() -> None:
    task_run = TaskRun(
        task="t", agent="mock", repo="/tmp/x", attempts=[_verdict(done=True, cost_usd=None)]
    )
    config = ConfigResult(label="x", task_runs=[task_run])
    assert config.total_cost_usd is None
    assert config.pass_rate_per_dollar is None


def test_pass_rate_per_dollar_is_none_when_cost_is_zero() -> None:
    config = ConfigResult(label="x", task_runs=[_task_run(True, 0.0)])
    # zero spend with a done task is undefined, not infinitely good
    assert config.pass_rate_per_dollar is None


def test_pass_rate_per_dollar_none_with_no_tasks() -> None:
    config = ConfigResult(label="x", task_runs=[])
    assert config.pass_rate == 0.0
    assert config.pass_rate_per_dollar is None


# --- leaderboard ranking --------------------------------------------------

def test_rank_orders_by_pass_rate_per_dollar_descending() -> None:
    cheap_and_good = ConfigResult(label="cheap", task_runs=[_task_run(True, 1.0)])       # 1.0 pt/$
    expensive_and_good = ConfigResult(label="expensive", task_runs=[_task_run(True, 4.0)])  # 0.25 pt/$

    ranked = rank([expensive_and_good, cheap_and_good])
    assert [c.label for c in ranked] == ["cheap", "expensive"]


def test_rank_puts_unknown_cost_entries_last() -> None:
    known = ConfigResult(label="known", task_runs=[_task_run(True, 1.0)])
    unknown_run = TaskRun(
        task="t", agent="mock", repo="/tmp/x", attempts=[_verdict(done=True, cost_usd=None)]
    )
    unknown = ConfigResult(label="unknown", task_runs=[unknown_run])
    ranked = rank([unknown, known])
    assert [c.label for c in ranked] == ["known", "unknown"]


def test_rank_breaks_unknown_cost_ties_by_pass_rate() -> None:
    def unknown_cost_config(label: str, done: bool) -> ConfigResult:
        run = TaskRun(
            task="t", agent="mock", repo="/tmp/x", attempts=[_verdict(done=done, cost_usd=None)]
        )
        return ConfigResult(label=label, task_runs=[run])

    ranked = rank([unknown_cost_config("worse", False), unknown_cost_config("better", True)])
    assert [c.label for c in ranked] == ["better", "worse"]


# --- token pricing config -------------------------------------------------

def test_token_pricing_computes_cost() -> None:
    pricing = TokenPricing(input_per_1k=0.003, output_per_1k=0.015)
    cost = pricing.cost_usd(tokens_input=2000, tokens_output=1000)
    assert cost == pytest.approx(2 * 0.003 + 1 * 0.015)


def test_load_config_parses_cost_section(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text(
        "cost:\n  price_per_1k_tokens:\n    input: 0.003\n    output: 0.015\n"
    )
    config = load_config(tmp_path)
    assert config.token_pricing is not None
    assert config.token_pricing.input_per_1k == 0.003
    assert config.token_pricing.output_per_1k == 0.015


def test_load_config_tolerates_malformed_cost_section(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text("cost:\n  price_per_1k_tokens: not-a-mapping\n")
    config = load_config(tmp_path)
    assert config.token_pricing is None


# --- pricing fallback in runner.py -----------------------------------------

def test_pricing_fallback_used_only_when_adapter_reports_no_cost() -> None:
    from verdict.config import VerdictConfig

    priced_config = VerdictConfig(gate_overrides={}, token_pricing=TokenPricing(0.003, 0.015))

    unpriced_attempt = AttemptResult(diff="", tokens_input=2000, tokens_output=1000, cost_usd=None)
    result = _apply_pricing_fallback(unpriced_attempt, priced_config)
    assert result.cost_usd == pytest.approx(2 * 0.003 + 1 * 0.015)

    already_priced_attempt = AttemptResult(diff="", tokens_input=2000, tokens_output=1000, cost_usd=9.99)
    result2 = _apply_pricing_fallback(already_priced_attempt, priced_config)
    assert result2.cost_usd == 9.99  # adapter's own figure is never overridden


def test_pricing_fallback_is_noop_without_config(tmp_path: Path) -> None:
    from verdict.config import VerdictConfig

    unpriced = AttemptResult(diff="", tokens_input=2000, tokens_output=1000, cost_usd=None)
    result = _apply_pricing_fallback(unpriced, VerdictConfig(gate_overrides={}))
    assert result.cost_usd is None


# --- run_with_retries end-to-end -------------------------------------------

class _FlakyThenFixAdapter:
    """Fails the first two attempts, fixes it on the third — used to
    prove run_with_retries keeps every attempt's cost, not just the last.
    """

    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, task: str, worktree: Path) -> AttemptResult:
        self.calls += 1
        if self.calls >= 3:
            (worktree / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
        return AttemptResult(diff="", tokens_input=100, tokens_output=50, cost_usd=0.10)


def test_run_with_retries_accumulates_cost_across_dead_ends(git_repo: Path) -> None:
    adapter = _FlakyThenFixAdapter()
    task_run = run_with_retries(task="fix it", repo=git_repo, adapter=adapter, max_attempts=5)

    assert adapter.calls == 3  # stopped as soon as it succeeded
    assert task_run.attempt_count == 3
    assert task_run.failed_attempt_count == 2
    assert task_run.done is True
    assert task_run.total_cost_usd == pytest.approx(0.30)
    assert task_run.total_tokens_input == 300


def test_run_with_retries_stops_at_max_attempts_if_never_done(git_repo: Path) -> None:
    class AlwaysFailsAdapter:
        name = "always-fails"

        def run(self, task: str, worktree: Path) -> AttemptResult:
            return AttemptResult(diff="", tokens_input=10, tokens_output=5, cost_usd=0.01)

    task_run = run_with_retries(task="t", repo=git_repo, adapter=AlwaysFailsAdapter(), max_attempts=3)
    assert task_run.attempt_count == 3
    assert task_run.done is False
    assert task_run.total_cost_usd == pytest.approx(0.03)


def test_run_with_retries_default_is_a_single_attempt(git_repo: Path) -> None:
    class CountingAdapter:
        name = "counting"

        def __init__(self) -> None:
            self.calls = 0

        def run(self, task: str, worktree: Path) -> AttemptResult:
            self.calls += 1
            return AttemptResult(diff="", cost_usd=0.0)

    adapter = CountingAdapter()
    run_with_retries(task="t", repo=git_repo, adapter=adapter)
    assert adapter.calls == 1
