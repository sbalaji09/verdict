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
than calling `run_with_retries` inline in the list comprehension below)
purely for readability at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass

from verdict.adapters import Adapter
from verdict.runner import DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_ERROR_RETRIES, run_with_retries
from verdict.sandbox import SandboxConfig
from verdict.schema import ConfigResult, TaskRun
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
) -> TaskRun:
    return run_with_retries(
        task=task.task,
        repo=task.repo,
        adapter=config.adapter,
        max_attempts=max_attempts,
        sandbox_config=sandbox_config,
        max_error_retries=max_error_retries,
    )


def run_suite(
    tasks: list[SuiteTask],
    configs: list[BenchConfig],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sandbox_config: SandboxConfig | None = None,
    max_error_retries: int = DEFAULT_MAX_ERROR_RETRIES,
) -> list[ConfigResult]:
    """Every config runs against every task, independently — one config's
    cost or failure has no bearing on another's, and one task's result
    doesn't gate whether the next task runs. Returns one `ConfigResult` per
    config, in the same order `configs` was given; ranking them is
    `economics.rank`'s job, not this function's.
    """
    results: list[ConfigResult] = []
    for config in configs:
        task_runs = [
            _run_task(task, config, max_attempts, sandbox_config, max_error_retries) for task in tasks
        ]
        results.append(ConfigResult(label=config.label, task_runs=task_runs))
    return results
