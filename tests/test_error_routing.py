"""Phase 10's ERROR-routing decision, tested at each layer it touches:

- `schema.py`: `Verdict.error` set forces `status` to ERROR regardless of
  `signals`, and `ConfigResult.pass_rate` excludes errored tasks from its
  denominator (not just the numerator).
- `suite/runner.py`: a `SandboxError` from one `(config, task)` pair is
  caught per-task — the suite keeps going, everything else in the
  `ConfigResult` is unaffected.
- `runner.py`: unchanged from Phase 9 — a single `run()`/`run_with_retries`
  call still lets a `SandboxError` propagate uncaught (see
  `test_timeouts.py`'s `test_provisioning_timeout_raises_and_never_produces_a_signal`
  for that half of the contract; this file focuses on the suite path,
  which is what Phase 10 actually adds).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from verdict.adapters.mock import SuiteMockAdapter
from verdict.sandbox.base import SetupError
from verdict.schema import AttemptResult, GateStatus, Provenance, Signal, Verdict, VerdictStatus
from verdict.suite import BenchConfig, run_suite
from verdict.suite.loader import SuiteTask


def _init_git(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


# --- schema-level status/accounting rules -----------------------------------


def _verdict(signals: list[Signal], error: str | None = None) -> Verdict:
    return Verdict(
        task="t",
        agent="mock",
        repo="r",
        attempt=AttemptResult(diff="", files_changed=[]),
        signals=signals,
        error=error,
    )


def test_error_field_forces_error_status_even_with_no_signals() -> None:
    verdict = _verdict([], error="service 'db' never became healthy")
    assert verdict.status is VerdictStatus.ERROR
    assert not verdict.done


def test_error_field_wins_over_passing_signals() -> None:
    passing = Signal(name="test", provenance=Provenance.PROVEN, status=GateStatus.PASS, detail="ok")
    verdict = _verdict([passing], error="sandbox never came up")
    assert verdict.status is VerdictStatus.ERROR


def test_config_result_excludes_errored_tasks_from_pass_rate_denominator() -> None:
    from verdict.schema import ConfigResult, TaskRun

    done = TaskRun(task="a", agent="mock", repo="r", attempts=[_verdict([_passing()])])
    errored = TaskRun(task="b", agent="mock", repo="r", attempts=[_verdict([], error="db never healthy")])
    result = ConfigResult(label="cfg", task_runs=[done, errored])

    assert result.tasks_total == 2
    assert result.tasks_done == 1
    assert result.tasks_errored == 1
    # 1 done out of 1 GRADED task — the errored one isn't in the
    # denominator at all, so this isn't 1/2 = 50%.
    assert result.pass_rate == 1.0


def test_config_result_pass_rate_is_zero_not_none_when_everything_errored() -> None:
    from verdict.schema import ConfigResult, TaskRun

    errored = TaskRun(task="a", agent="mock", repo="r", attempts=[_verdict([], error="boom")])
    result = ConfigResult(label="cfg", task_runs=[errored])
    assert result.pass_rate == 0.0


def _passing() -> Signal:
    return Signal(name="test", provenance=Provenance.PROVEN, status=GateStatus.PASS, detail="ok")


# --- suite/runner.py: catches per-task, keeps going -------------------------


class _AlwaysErrorsAdapter:
    name = "always-errors"

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        raise SetupError("service 'db' (postgres:16) never became healthy within 30s")


def test_run_suite_records_one_erroring_task_and_continues_to_the_next(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calculator.py").write_text("def add(a, b):\n    return a - b  # bug\n")
    (repo / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "pytest.ini").write_text("[pytest]\n")
    _init_git(repo)

    good = SuiteMockAdapter({"fix it": {"calculator.py": "def add(a, b):\n    return a + b\n"}})
    tasks = [
        SuiteTask(name="task-a", task="fix it", repo=repo),
        SuiteTask(name="task-b", task="fix it", repo=repo),
    ]

    error_config = BenchConfig(label="errors", adapter=_AlwaysErrorsAdapter())
    good_config = BenchConfig(label="good", adapter=good)

    results = run_suite(tasks, [error_config, good_config])

    error_result, good_result = results
    assert error_result.tasks_total == 2
    assert error_result.tasks_errored == 2
    assert error_result.tasks_done == 0
    for task_run in error_result.task_runs:
        assert task_run.final.status is VerdictStatus.ERROR
        assert "never became healthy" in (task_run.final.error or "")

    # The OTHER config is completely unaffected by the first one erroring —
    # independent per (config, task) pair, exactly as `run_suite`'s own
    # docstring already promises for ordinary failures.
    assert good_result.tasks_done == 2
    assert good_result.tasks_errored == 0
