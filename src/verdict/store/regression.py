"""Regression detection over persisted history — reuses Phase 7's pooled
two-proportion z-test (`flakiness.compare_flakiness`) completely
unchanged. This module's only job is turning a `Store`'s historical
`(task, agent, repo, config)` outcomes into the exact `(trials, passes)`
shape `FlakinessResult` already represents, so the identical statistical
machinery that already tells "an agent got flakier across repeated trials
of one run" apart from noise does the same job across repeated runs
*over time* — no new statistics, no new significance test, just a
different source for the two samples being compared.

Why this reuse is legitimate, not a coincidence being forced: a two-
proportion z-test only needs two binomial samples — `(n, k)` pairs. Phase
7's `n` was "how many independent trials of the same task did we run just
now"; here `n` is "how many independent historical runs recorded an
outcome for this task." The test doesn't know or care which; the honest
"NOISE unless the gap clears `alpha`" discipline that keeps flakiness
detection from crying wolf over small-sample noise is exactly as valuable
here, guarding against "this task happened to fail once and the whole
dashboard lit up red."
"""

from __future__ import annotations

from dataclasses import dataclass

from verdict.flakiness import (
    DEFAULT_ALPHA,
    ComparisonVerdict,
    FlakinessComparison,
    FlakinessResult,
    compare_flakiness,
)
from verdict.schema import ConfigResult
from verdict.store.base import Store, TaskOutcome

DEFAULT_BASELINE_WINDOW = 20
"""How many of the most recent PRIOR historical outcomes count as the
baseline sample. Not "all of history" — an agent/repo pair with hundreds
of recorded runs shouldn't have a regression from the last five diluted
into invisibility by five hundred old ones; a bounded recent window is
what "compare against a historical baseline" actually means for a moving
target like an agent's pass rate."""


@dataclass
class TaskRegression:
    """One `(config, task)` pair whose current outcome, compared against
    its own recent history, cleared the z-test's significance bar in the
    worse direction. Carries the full `FlakinessComparison` (p-value,
    z-statistic, both samples) rather than just a label — the same "point
    at the evidence" instinct behind `Attribution.explanation` and the
    flakiness comparison's own render function.
    """

    config_label: str
    task: str
    agent: str
    repo: str
    comparison: FlakinessComparison


def _pooled(task: str, agent: str, repo: str, outcomes: list[TaskOutcome]) -> FlakinessResult | None:
    if not outcomes:
        return None
    passes = sum(1 for o in outcomes if o.done)
    return FlakinessResult(task=task, agent=agent, repo=repo, trials=len(outcomes), passes=passes)


def detect_regressions(
    store: Store,
    config_results: list[ConfigResult],
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
    alpha: float = DEFAULT_ALPHA,
    exclude_run_id: str | None = None,
) -> list[TaskRegression]:
    """For every `(config, task)` pair in `config_results` (the run being
    evaluated — NOT read back from the store, so this works whether or
    not that run has already been recorded), pool the `baseline_window`
    most recent PRIOR outcomes from `store.history(...)` into a baseline
    `FlakinessResult`, treat this run's own single outcome as a one-trial
    candidate `FlakinessResult`, and run `compare_flakiness` between them.

    `exclude_run_id` — pass the `run_id` `Store.record_run` just returned,
    if this run has already been persisted, so its own outcome can't leak
    into its own baseline (`store.history`'s `exclude_run_id` handles the
    actual filtering; this function never assumes a particular call
    order relative to `record_run`).

    A `(config, task)` pair with no prior history at all is skipped, not
    reported as anything — "no baseline exists yet" and "compared against
    a baseline and found no regression" are different claims, and this
    function only ever reports the second kind, the same "an unknown stays
    unknown" discipline the rest of this codebase's statistics already
    follow (`FlakinessComparison`'s own NOISE-not-a-guess verdict,
    `TaskRun.total_cost_usd`'s None-on-unknown, ...). Only
    `REGRESSION`-verdict comparisons are returned — an honest empty list
    when everything is NOISE or IMPROVEMENT, never noise dressed up as a
    finding.
    """
    regressions: list[TaskRegression] = []
    for config_result in config_results:
        for task_run in config_result.task_runs:
            baseline_outcomes = store.history(
                task_run.task,
                task_run.agent,
                task_run.repo,
                config_label=config_result.label,
                exclude_run_id=exclude_run_id,
                limit=baseline_window,
            )
            baseline = _pooled(task_run.task, task_run.agent, task_run.repo, baseline_outcomes)
            if baseline is None:
                continue

            candidate = FlakinessResult(
                task=task_run.task,
                agent=task_run.agent,
                repo=task_run.repo,
                trials=1,
                passes=1 if task_run.done else 0,
            )
            comparison = compare_flakiness(
                baseline,
                candidate,
                baseline_label=f"history ({baseline.trials} prior run(s))",
                candidate_label="this run",
                alpha=alpha,
            )
            if comparison.verdict is ComparisonVerdict.REGRESSION:
                regressions.append(
                    TaskRegression(
                        config_label=config_result.label,
                        task=task_run.task,
                        agent=task_run.agent,
                        repo=task_run.repo,
                        comparison=comparison,
                    )
                )
    return regressions
