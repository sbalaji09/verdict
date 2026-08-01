"""Runs every (config, task) pair in a suite and assembles one
`ConfigResult` per config — the scorecard `economics.py`'s `rank`/`render`
already know how to leaderboard. No new scoring logic lives here: each
(config, task) pair is exactly one `run_with_retries` call, reusing
Phase 0-3's full pipeline (isolation, gates, attribution, cost accounting)
completely unchanged. A suite run is "do this many times, then aggregate,"
not a different kind of grading.

Phase 10's one addition: a `SandboxError` (sandbox provisioning failed, a
declared service never became healthy, a setup step hung) from any one
`(config, task)` pair is caught HERE, per task, rather than being left to
crash the whole suite — see DESIGN.md's Phase 10 section on why this is
the one place the ERROR-routing decision actually changes runner
behavior: a single `verdict run`/`verdict gate` still lets the same
exception propagate uncaught to the CLI boundary, exit code 2, no report
at all. A suite grading dozens of (agent, task) combinations shouldn't
lose everything else it already computed because one repo's Postgres
never came up.
"""

from __future__ import annotations

from dataclasses import dataclass

from verdict.adapters import Adapter
from verdict.runner import DEFAULT_MAX_ATTEMPTS, run_with_retries
from verdict.sandbox import SandboxConfig
from verdict.sandbox.base import SandboxError
from verdict.schema import AttemptResult, ConfigResult, TaskRun, Verdict
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
    task: SuiteTask, config: BenchConfig, max_attempts: int, sandbox_config: SandboxConfig | None
) -> TaskRun:
    try:
        return run_with_retries(
            task=task.task,
            repo=task.repo,
            adapter=config.adapter,
            max_attempts=max_attempts,
            sandbox_config=sandbox_config,
        )
    except SandboxError as exc:
        # No real attempt happened — cost is genuinely 0, not unknown —
        # and `signals`/`attributions` stay empty; `error` is what makes
        # `Verdict.status` ERROR (see `schema.py`), never NOT_DONE.
        errored = Verdict(
            task=task.task,
            agent=config.adapter.name,
            repo=str(task.repo),
            attempt=AttemptResult(diff="", files_changed=[], cost_usd=0.0),
            signals=[],
            error=str(exc),
        )
        return TaskRun(task=task.task, agent=config.adapter.name, repo=str(task.repo), attempts=[errored])


def run_suite(
    tasks: list[SuiteTask],
    configs: list[BenchConfig],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sandbox_config: SandboxConfig | None = None,
) -> list[ConfigResult]:
    """Every config runs against every task, independently — one config's
    cost or failure has no bearing on another's, and one task's result
    doesn't gate whether the next task runs. Returns one `ConfigResult` per
    config, in the same order `configs` was given; ranking them is
    `economics.rank`'s job, not this function's.
    """
    results: list[ConfigResult] = []
    for config in configs:
        task_runs = [_run_task(task, config, max_attempts, sandbox_config) for task in tasks]
        results.append(ConfigResult(label=config.label, task_runs=task_runs))
    return results
