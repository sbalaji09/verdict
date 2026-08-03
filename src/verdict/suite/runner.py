"""Runs every (config, task) pair in a suite and assembles one
`ConfigResult` per config — the scorecard `economics.py`'s `rank`/`render`
already know how to leaderboard. No new scoring logic lives here: each
(config, task) pair is exactly one `run_with_retries` call, reusing
Phase 0-3's full pipeline (isolation, gates, attribution, cost accounting)
completely unchanged. A suite run is "do this many times, then aggregate,"
not a different kind of grading.

Phase 10 taught `run_suite` to not lose an entire suite over one repo's
Postgres never coming up. Phase 11 moved the actual catch-and-retry logic
down into `runner.py::run_with_retries` itself (see its docstring, and
`_run_attempt`'s bounded infra-retry loop) — `run_with_retries` now never
lets `_EVALUATION_ERRORS` propagate at all, single `verdict run` included,
so this module no longer needs its own catch site. `_run_task` below is
now a thin, single-line wrapper again; kept as a named function (rather
than calling `run_with_retries` inline) purely for readability, and
because `Executor` (Phase 15, see `executor.py`) needs a plain,
module-level, picklable callable to hand to a process pool — a closure or
inline lambda couldn't cross a process boundary.

## Phase 15: parallel execution, same aggregation

`run_suite` always ran every (config, task) pair independently — one
config's cost or failure never affected another's, one task's result
never gated whether the next task ran (see the module docstring for
`run_suite` below). That independence is exactly what makes the pairs
safe to run concurrently: nothing about how one pair is graded depends on
another pair having happened yet, or on what order they happen in. Phase
15 exploits that by routing every pair through an `Executor`
(`executor.py`) instead of a bare Python loop — `SerialExecutor` (the
default, unchanged behavior) or `LocalProcessPoolExecutor` (a bounded
pool of OS processes, one work item per sandbox, same as before — see
`runner.py::run`'s own isolation, untouched by this phase).

The one thing that has to hold regardless of which `Executor` ran the
work: **the returned `list[ConfigResult]` is byte-identical to what the
serial loop would have produced**, order included. That's why aggregation
below never relies on completion order — work items are built in the
same fixed `(config, task)` nesting the original serial loop used, handed
to `Executor.run` (which is contractually order-preserving — see its own
docstring), and the flat result list is sliced back into per-config
chunks by *position*, not by whichever one happened to finish first.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Manager
from multiprocessing.managers import SyncManager

from verdict.adapters import Adapter
from verdict.runner import DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_ERROR_RETRIES, run_with_retries
from verdict.sandbox import SandboxConfig
from verdict.schema import AttemptResult, ConfigResult, TaskRun, Verdict
from verdict.suite.executor import Executor, SerialExecutor
from verdict.suite.loader import SuiteTask


@dataclass
class BenchConfig:
    """One row of the eventual leaderboard: a label plus the adapter that
    produces it. `label` is free-form (mirrors `ConfigResult.label` from
    Phase 3) so "claude-code / sonnet" vs. "claude-code / opus" can both
    run through the same adapter class with a different model configured
    however that adapter exposes it (env var, constructor arg, ...) —
    Verdict doesn't need to understand what varies.
    """

    label: str
    adapter: Adapter


def _run_task(
    task: SuiteTask,
    config: BenchConfig,
    max_attempts: int,
    sandbox_config: SandboxConfig | None,
    max_error_retries: int,
    run_cost_ceiling_usd: float | None = None,
) -> TaskRun:
    return run_with_retries(
        task=task.task,
        repo=task.repo,
        adapter=config.adapter,
        max_attempts=max_attempts,
        sandbox_config=sandbox_config,
        max_error_retries=max_error_retries,
        # Phase 12: the task's OWN `allow_test_changes` — sourced from
        # `task.yml` by `suite/loader.py`, the trusted side of the
        # boundary `integrity.TestChangeAllowance` documents. Never a
        # suite-wide override here; each task's own declaration is what
        # governs it.
        allow_test_changes=task.allow_test_changes,
        # Phase 13: the task's own held-out FAIL_TO_PASS/PASS_TO_PASS
        # tests, if it declared any — see `acceptance.py`.
        acceptance=task.acceptance,
        # Phase 18: a hard per-(config, task) spend cap — distinct from
        # this module's own `cost_ceiling_usd` (a suite-WIDE cap, checked
        # by `_run_task_within_ceiling` below before a task even starts).
        # This one bounds one task's own agent-retry loop, threaded down
        # to `run_with_retries` unchanged; see that function's docstring
        # for exactly how it aborts and why the abort isn't an agent
        # failure.
        cost_ceiling_usd=run_cost_ceiling_usd,
    )


class _CostCeiling:
    """A running-total spend tracker shared across every worker — a plain
    float only ever sees its own process's updates, which defeats a
    *global* ceiling the moment more than one worker process exists.
    Backed by `multiprocessing.Manager` proxies specifically because those
    are designed to be shared with (and pickled to) arbitrary worker
    processes, not just ones forked from this one — the same primitive
    works unchanged whether `run_suite` ends up using `SerialExecutor` or
    `LocalProcessPoolExecutor`.

    Enforcement is cooperative, not preemptive: a work item already
    running when the ceiling is crossed is never killed mid-attempt (that
    would mean tearing down a live sandbox from outside its own `with`
    block, a much bigger change for a rare edge). Only *new* work items
    check `exceeded()`, at the moment they'd otherwise start — exact under
    `SerialExecutor` (one item finishes, updates the total, *then* the
    next item's check runs), best-effort under a parallel pool (several
    items can already be in flight, each unaware the others are about to
    push the total over). This is the same "report the true, sometimes
    coarser, thing rather than a precise-looking guess" discipline as
    `SandboxConfig.attempt_budget_seconds`'s own per-attempt ceiling.
    """

    def __init__(self, limit: float, manager: SyncManager) -> None:
        self.limit = limit
        self._spent = manager.Value("d", 0.0)
        self._lock = manager.Lock()

    def exceeded(self) -> bool:
        with self._lock:
            return self._spent.value >= self.limit

    def add(self, cost: float) -> None:
        with self._lock:
            self._spent.value += cost


def _skipped_task_run(task: SuiteTask, config: BenchConfig, ceiling: _CostCeiling) -> TaskRun:
    error = (
        f"skipped: suite cost ceiling (${ceiling.limit:.2f}) was already reached before this "
        "task could start"
    )
    verdict = Verdict(
        task=task.task,
        agent=config.adapter.name,
        repo=str(task.repo),
        attempt=AttemptResult(diff=""),
        signals=[],
        error=error,
    )
    return TaskRun(task=task.task, agent=config.adapter.name, repo=str(task.repo), attempts=[verdict])


def _run_task_within_ceiling(
    task: SuiteTask,
    config: BenchConfig,
    max_attempts: int,
    sandbox_config: SandboxConfig | None,
    max_error_retries: int,
    ceiling: _CostCeiling | None,
    run_cost_ceiling_usd: float | None = None,
) -> TaskRun:
    """The picklable unit of work handed to `Executor.run` — `_run_task`
    itself plus the SUITE-wide cost-ceiling check, kept as one function so
    a single work item is one self-contained callable rather than the
    executor needing to know about two separate steps per item.
    """
    if ceiling is not None and ceiling.exceeded():
        return _skipped_task_run(task, config, ceiling)
    task_run = _run_task(task, config, max_attempts, sandbox_config, max_error_retries, run_cost_ceiling_usd)
    if ceiling is not None and task_run.total_cost_usd is not None:
        ceiling.add(task_run.total_cost_usd)
    return task_run


def run_suite(
    tasks: list[SuiteTask],
    configs: list[BenchConfig],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sandbox_config: SandboxConfig | None = None,
    max_error_retries: int = DEFAULT_MAX_ERROR_RETRIES,
    executor: Executor | None = None,
    cost_ceiling_usd: float | None = None,
    run_cost_ceiling_usd: float | None = None,
) -> list[ConfigResult]:
    """Every config runs against every task, independently — one config's
    cost or failure has no bearing on another's, and one task's result
    doesn't gate whether the next task runs. Returns one `ConfigResult` per
    config, in the same order `configs` was given; ranking them is
    `economics.rank`'s job, not this function's.

    `executor` defaults to `SerialExecutor()` — a library caller (or a
    test) that doesn't ask for parallelism gets exactly the old
    behavior, one `(config, task)` pair at a time, in this process. Pass
    `LocalProcessPoolExecutor(max_workers=N)` (`executor.py`) to run up
    to `N` pairs concurrently, each still in its own isolated worktree
    and sandbox (`runner.py::run` — unchanged by this phase).

    `cost_ceiling_usd`, if set, is a global spend cap across every pair in
    this call — see `_CostCeiling`'s docstring for exactly what "global"
    means once more than one worker is involved. `None` (the default)
    disables it, matching every other None-disables-the-cap knob in this
    codebase (`SandboxConfig.attempt_budget_seconds`, etc.).

    `run_cost_ceiling_usd` (Phase 18) is a DIFFERENT, finer-grained cap:
    passed straight through to every `run_with_retries` call as its own
    `cost_ceiling_usd`, bounding one `(config, task)` pair's own agent-
    retry spend rather than the whole suite's. The two compose freely —
    a suite can bound both "never spend more than $X total" and "never
    spend more than $Y retrying any single task" at once, since they're
    checked at different layers (`_CostCeiling` here vs. `run_with_
    retries`'s own loop) with no interaction between them.

    Work items are built in the same `(config, task)` nesting the
    original serial loop used and results are sliced back out by that
    same fixed position — never by completion order — so the returned
    list is identical whichever `Executor` produced it. See the module
    docstring's "Phase 15" section for why that invariant is safe to rely
    on: every pair is independent by construction, so nothing about *how*
    or *in what order* pairs actually ran can change what any individual
    pair's own result is.
    """
    executor = executor or SerialExecutor()

    manager: SyncManager | None = None
    ceiling: _CostCeiling | None = None
    if cost_ceiling_usd is not None:
        manager = Manager()
        ceiling = _CostCeiling(cost_ceiling_usd, manager)

    try:
        items = [
            (task, config, max_attempts, sandbox_config, max_error_retries, ceiling, run_cost_ceiling_usd)
            for config in configs
            for task in tasks
        ]
        flat_task_runs = executor.run(_run_task_within_ceiling, items)
    finally:
        if manager is not None:
            manager.shutdown()

    results: list[ConfigResult] = []
    for config_index, config in enumerate(configs):
        start = config_index * len(tasks)
        task_runs = flat_task_runs[start : start + len(tasks)]
        results.append(ConfigResult(label=config.label, task_runs=task_runs))
    return results
