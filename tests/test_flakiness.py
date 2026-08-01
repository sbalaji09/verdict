"""Phase 7: flakiness detection. Wilson intervals and the two-proportion
z-test are pure math, tested directly against known values; `run_flakiness`
itself is tested end to end against a real git repo, the same standard
`test_runner.py`/`test_economics.py` already hold themselves to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from verdict.flakiness import (
    ComparisonVerdict,
    FlakinessResult,
    compare_flakiness,
    run_flakiness,
    wilson_interval,
)
from verdict.schema import AttemptResult

# --- wilson_interval ---------------------------------------------------


def test_wilson_interval_is_symmetric_around_fifty_percent() -> None:
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(1 - high, abs=1e-9)
    assert low < 0.5 < high


def test_wilson_interval_narrows_as_sample_size_grows() -> None:
    small_low, small_high = wilson_interval(8, 10)
    large_low, large_high = wilson_interval(80, 100)
    assert (large_high - large_low) < (small_high - small_low)


def test_wilson_interval_never_claims_zero_width_certainty_at_p_one() -> None:
    # Unlike the naive normal-approximation interval, Wilson doesn't collapse
    # to [1.0, 1.0] just because every trial in a small sample passed.
    low, high = wilson_interval(3, 3)
    assert high == pytest.approx(1.0)
    assert low > 0.0
    assert low < 1.0


def test_wilson_interval_rejects_unsupported_confidence_level() -> None:
    with pytest.raises(ValueError, match="unsupported confidence_level"):
        wilson_interval(5, 10, confidence_level=0.5)


def test_wilson_interval_rejects_zero_trials() -> None:
    with pytest.raises(ValueError, match="n must be"):
        wilson_interval(0, 0)


# --- compare_flakiness ---------------------------------------------------


def _result(*, trials: int, passes: int) -> FlakinessResult:
    return FlakinessResult(task="t", agent="a", repo="/tmp/x", trials=trials, passes=passes)


def test_identical_pass_rates_are_never_a_regression() -> None:
    comparison = compare_flakiness(_result(trials=20, passes=15), _result(trials=20, passes=15))
    assert comparison.verdict is ComparisonVerdict.NOISE
    assert comparison.p_value == pytest.approx(1.0)


def test_small_sample_gap_is_noise_not_a_regression() -> None:
    # 8/10 vs 6/10 is a 20-point raw drop, but with only 10 trials each
    # side it's well within sampling noise at alpha=0.05 — this is exactly
    # the case a raw-percentage comparison would misreport as "worse."
    comparison = compare_flakiness(_result(trials=10, passes=8), _result(trials=10, passes=6))
    assert comparison.verdict is ComparisonVerdict.NOISE


def test_large_significant_drop_is_a_regression() -> None:
    comparison = compare_flakiness(_result(trials=200, passes=190), _result(trials=200, passes=140))
    assert comparison.verdict is ComparisonVerdict.REGRESSION
    assert comparison.p_value is not None
    assert comparison.p_value < 0.05


def test_large_significant_rise_is_an_improvement() -> None:
    comparison = compare_flakiness(_result(trials=200, passes=140), _result(trials=200, passes=190))
    assert comparison.verdict is ComparisonVerdict.IMPROVEMENT


def test_all_pass_both_sides_is_noise_not_a_crash() -> None:
    # p_pool == 1.0 here, which drives the pooled standard error to zero —
    # must be handled as a degenerate "samples agree completely" case
    # rather than dividing by zero.
    comparison = compare_flakiness(_result(trials=10, passes=10), _result(trials=10, passes=10))
    assert comparison.verdict is ComparisonVerdict.NOISE
    assert comparison.z_statistic == 0.0


# --- run_flakiness end-to-end -------------------------------------------


class _AlternatingAdapter:
    """Fixes the bug on odd-numbered calls, leaves it broken on even ones —
    a deterministic stand-in for a real flaky agent, used to prove
    run_flakiness aggregates independent outcomes across real, separate
    `run()` calls rather than caching or reusing a single worktree's state.
    """

    name = "alternating"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, task: str, worktree: Path) -> AttemptResult:
        self.calls += 1
        if self.calls % 2 == 1:
            (worktree / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
        return AttemptResult(diff="", tokens_input=10, tokens_output=5, cost_usd=0.01)


def test_run_flakiness_aggregates_independent_trials(git_repo: Path) -> None:
    adapter = _AlternatingAdapter()
    result = run_flakiness(task="fix it", repo=git_repo, adapter=adapter, trials=6)

    assert adapter.calls == 6
    assert result.trials == 6
    assert result.passes == 3
    assert result.pass_rate == pytest.approx(0.5)
    assert result.total_cost_usd == pytest.approx(0.06)
    low, high = result.confidence_interval
    assert low < result.pass_rate < high


def test_run_flakiness_does_not_mutate_source_repo(git_repo: Path) -> None:
    original = (git_repo / "calculator.py").read_text()
    run_flakiness(task="fix it", repo=git_repo, adapter=_AlternatingAdapter(), trials=3)
    assert (git_repo / "calculator.py").read_text() == original


def test_run_flakiness_rejects_zero_trials(git_repo: Path) -> None:
    with pytest.raises(ValueError, match="trials must be"):
        run_flakiness(task="t", repo=git_repo, adapter=_AlternatingAdapter(), trials=0)
