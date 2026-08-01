"""Flakiness detection: run the same task multiple times, report a pass
rate with a real confidence interval, and separate a genuine regression
from run-to-run noise.

Every phase through Phase 6 asks "did this one attempt pass" — a perfectly
reasonable question for a merge gate (one PR, one verdict), but the wrong
question for "is agent/config A actually better than B" or "did this change
make the task harder." A single run can't answer either: an agent that
passes 8/10 trials and one that passes 9/10 look identical on any single
attempt, and a raw pass-rate delta between two small samples ("70% vs 60%")
looks like a regression right up until you compute how much of that gap is
just sampling noise. This module runs the same `(task, repo, adapter)`
several times, reports the pass rate *with* a Wilson score interval around
it, and — when comparing two such results — runs a real two-proportion
z-test rather than eyeballing the raw percentages.

No new trust bucket, no schema migration to `Verdict`/`TaskRun`: this
module is a statistical layer *over* `runner.run()`, the same way
`economics.py` is a ranking layer over `ConfigResult` without needing
`Verdict` to know anything about dollars. Every individual trial is a real,
independent `run()` call — its own fresh worktree, its own gates, its own
verdict — never a cached or simulated result.
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, computed_field
from rich.console import Console
from rich.table import Table

from verdict.adapters import Adapter
from verdict.runner import run

DEFAULT_TRIALS = 10
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_ALPHA = 0.05
"""Significance threshold for `compare_flakiness` — a p-value below this is
what earns the label REGRESSION or IMPROVEMENT rather than NOISE."""

# Two-sided critical z-values for the confidence levels this module is
# willing to vouch for. Computing an arbitrary confidence level's z-value
# needs the normal quantile function (the inverse of `_standard_normal_cdf`
# below), which isn't in the standard library — rather than hand-roll an
# approximation of *that* (a real source of subtle error), only these three
# well-known values are supported; anything else raises rather than
# guessing at a number close enough to be wrong in a way nobody would
# notice.
_Z_FOR_CONFIDENCE: dict[float, float] = {
    0.90: 1.6448536269514722,
    0.95: 1.9599639845400545,
    0.99: 2.5758293035489004,
}


def _z_for_confidence(confidence_level: float) -> float:
    try:
        return _Z_FOR_CONFIDENCE[confidence_level]
    except KeyError:
        supported = ", ".join(str(c) for c in sorted(_Z_FOR_CONFIDENCE))
        raise ValueError(
            f"unsupported confidence_level {confidence_level!r} (supported: {supported})"
        ) from None


def wilson_interval(
    successes: int, n: int, confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
) -> tuple[float, float]:
    """The Wilson score interval for a binomial proportion — chosen over
    the naive `p ± z*sqrt(p(1-p)/n)` normal-approximation interval because
    the naive one breaks exactly where flakiness detection lives: small `n`
    (a handful of trials, not thousands) and `p` near 0 or 1 (a mostly-
    reliable agent, which is the common case). At `p=1.0`, the naive
    interval collapses to a zero-width `[1.0, 1.0]` — "100% confident,
    forever, after 3 trials" — which is a false claim of certainty Wilson
    doesn't make; its interval still narrows toward 1.0 as `n` grows but
    never claims zero width from a small sample.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    z = _z_for_confidence(confidence_level)
    phat = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class FlakinessResult(BaseModel):
    """`trials` independent runs of the same `(task, repo, agent)`, boiled
    down to a pass count and a Wilson interval around the pass rate.
    """

    task: str
    agent: str
    repo: str
    trials: int
    passes: int
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    total_cost_usd: float | None = None
    """None if any trial's cost was unknown — same "never sum a partial
    figure and call it total" discipline as `TaskRun.total_cost_usd`."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> float:
        return self.passes / self.trials

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence_interval(self) -> tuple[float, float]:
        return wilson_interval(self.passes, self.trials, self.confidence_level)


def run_flakiness(
    task: str,
    repo: Path,
    adapter: Adapter,
    trials: int = DEFAULT_TRIALS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> FlakinessResult:
    """Run `adapter` on `task` against `repo` `trials` independent times —
    each its own fresh `runner.run()` call, its own worktree, its own
    verdict — and summarize the outcomes. Deliberately calls `run()`, not
    `run_with_retries()`: retries exist to *get past* a flaky failure on a
    single task attempt; this function exists to *measure* how often that
    failure happens in the first place, so averaging over a "keep retrying
    until it passes" loop would hide the exact variance this is measuring.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")

    passes = 0
    total_cost = 0.0
    cost_known = True
    for _ in range(trials):
        verdict = run(task=task, repo=repo, adapter=adapter)
        if verdict.done:
            passes += 1
        if verdict.attempt.cost_usd is None:
            cost_known = False
        elif cost_known:
            total_cost += verdict.attempt.cost_usd

    return FlakinessResult(
        task=task,
        agent=adapter.name,
        repo=str(repo),
        trials=trials,
        passes=passes,
        confidence_level=confidence_level,
        total_cost_usd=total_cost if cost_known else None,
    )


class ComparisonVerdict(str, Enum):
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    NOISE = "noise"
    """The difference between the two pass rates doesn't clear `alpha` —
    real variance can produce a gap this size even with nothing else
    changed. Distinct from REGRESSION/IMPROVEMENT the same way Phase 2's
    `INCONCLUSIVE` is distinct from `REGRESSION`: an honest "can't tell,"
    never a guess in either direction."""


class FlakinessComparison(BaseModel):
    """A two-proportion z-test between two `FlakinessResult`s — the honest
    answer to "did the pass rate actually change, or is this just noise
    from a small sample." `verdict` is only ever REGRESSION or IMPROVEMENT
    when the difference clears `alpha`; anything else is NOISE, on
    purpose — the same "say 'can't tell' rather than guess" discipline
    Phase 2's bisector uses for `skip`.
    """

    baseline_label: str
    candidate_label: str
    baseline: FlakinessResult
    candidate: FlakinessResult
    alpha: float = DEFAULT_ALPHA
    z_statistic: float | None = None
    p_value: float | None = None
    verdict: ComparisonVerdict = Field(default=ComparisonVerdict.NOISE)


def compare_flakiness(
    baseline: FlakinessResult,
    candidate: FlakinessResult,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    alpha: float = DEFAULT_ALPHA,
) -> FlakinessComparison:
    """Pooled two-proportion z-test: is `candidate`'s pass rate really
    different from `baseline`'s, or is the raw percentage gap between two
    small samples explainable by chance alone? A p-value under `alpha`
    means "no" — the difference is a REGRESSION (candidate worse) or an
    IMPROVEMENT (candidate better). A p-value at or above `alpha`, or a
    degenerate case where both proportions collapse to the same pooled
    rate, is NOISE: the data doesn't support calling this a real change,
    even if the raw numbers differ.
    """
    n1, x1 = baseline.trials, baseline.passes
    n2, x2 = candidate.trials, candidate.passes
    p1, p2 = x1 / n1, x2 / n2

    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if se == 0:
        # p_pool is exactly 0 or 1, which only happens when every trial in
        # both samples had the same outcome — i.e. p1 == p2 already.
        # Nothing to test; the samples agree completely.
        return FlakinessComparison(
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            baseline=baseline,
            candidate=candidate,
            alpha=alpha,
            z_statistic=0.0,
            p_value=1.0,
            verdict=ComparisonVerdict.NOISE,
        )

    z = (p2 - p1) / se
    p_value = 2 * (1 - _standard_normal_cdf(abs(z)))

    if p_value >= alpha:
        verdict = ComparisonVerdict.NOISE
    elif p2 < p1:
        verdict = ComparisonVerdict.REGRESSION
    else:
        verdict = ComparisonVerdict.IMPROVEMENT

    return FlakinessComparison(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline=baseline,
        candidate=candidate,
        alpha=alpha,
        z_statistic=z,
        p_value=p_value,
        verdict=verdict,
    )


def render_flakiness(
    result: FlakinessResult, label: str | None = None, console: Console | None = None
) -> None:
    console = console or Console()
    ci_low, ci_high = result.confidence_interval
    table = Table(title=f"Flakiness — {label or result.agent} × {result.task[:60]!r}")
    table.add_column("trials", justify="right")
    table.add_column("passes", justify="right")
    table.add_column("pass rate", justify="right")
    table.add_column(f"{result.confidence_level:.0%} CI", justify="right")
    table.add_row(
        str(result.trials),
        str(result.passes),
        f"{result.pass_rate:.0%}",
        f"[{ci_low:.0%}, {ci_high:.0%}]",
    )
    console.print(table)


def render_comparison(comparison: FlakinessComparison, console: Console | None = None) -> None:
    console = console or Console()
    render_flakiness(comparison.baseline, label=comparison.baseline_label, console=console)
    render_flakiness(comparison.candidate, label=comparison.candidate_label, console=console)

    color = {
        ComparisonVerdict.REGRESSION: "red",
        ComparisonVerdict.IMPROVEMENT: "green",
        ComparisonVerdict.NOISE: "yellow",
    }[comparison.verdict]
    p = f"{comparison.p_value:.4f}" if comparison.p_value is not None else "n/a"
    console.print(
        f"[bold {color}]{comparison.verdict.upper()}[/bold {color}] "
        f"(p={p}, alpha={comparison.alpha}) — "
        f"{comparison.baseline_label}: {comparison.baseline.pass_rate:.0%} vs. "
        f"{comparison.candidate_label}: {comparison.candidate.pass_rate:.0%}"
    )
