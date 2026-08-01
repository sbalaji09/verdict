"""Phase 11's retry policy: bounded auto-retry for ERROR (infra couldn't
evaluate the attempt), structurally separate from the opt-in, per-agent
`max_attempts` retry that only ever applies to a legitimate NOT_DONE.

- `runner.py::_run_attempt` is the ONE place `_EVALUATION_ERRORS` is
  caught and retried — up to `max_error_retries` extra times — before
  being reported as a real `VerdictStatus.ERROR` `Verdict`.
- `run_with_retries`'s own `max_attempts` loop never sees the exception at
  all (it's already been converted to a Verdict by the time that loop
  looks at it), and it stops immediately on ERROR rather than spending
  further agent attempts against infra that's already proven broken.
- A `NOT_DONE` `Verdict` — the agent's code was evaluated and found
  wanting — is never something `_run_attempt`'s retry loop can see or
  act on: it only ever triggers on a *raised* exception, and a NOT_DONE
  is always a *returned* Verdict. This file proves that boundary holds
  for both directions: an infra failure gets retried, an agent failure
  does not.
- `suite/runner.py::run_suite` end-to-end: an always-erroring config is
  retried, ends ERROR, and is excluded from the leaderboard's pass-rate
  denominator, while a normal config alongside it is unaffected.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from verdict.adapters.claude_code import ClaudeCodeAdapterError
from verdict.runner import run_with_retries
from verdict.sandbox.base import SetupError
from verdict.schema import AttemptResult, VerdictStatus
from verdict.suite import BenchConfig, run_suite
from verdict.suite.loader import SuiteTask


class _AlwaysErrorsAdapter:
    """Every call raises an infra-style error — stands in for a sandbox
    that never comes up, a service that's never healthy, etc.
    """

    name = "always-errors"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        self.calls += 1
        raise SetupError("service 'db' never became healthy within 30s")


class _ErrorsThenRecoversAdapter:
    """Raises an infra error the first N calls, then succeeds — stands in
    for a transient infra flake the bounded retry is meant to survive.
    """

    name = "flaky-infra"

    def __init__(self, errors_before_success: int) -> None:
        self.calls = 0
        self._errors_before_success = errors_before_success

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        self.calls += 1
        if self.calls <= self._errors_before_success:
            raise SetupError(f"transient infra flake (call {self.calls})")
        (worktree / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
        return AttemptResult(diff="", tokens_input=10, tokens_output=5, cost_usd=0.01)


class _AlwaysFailsAdapter:
    """A real agent attempt that just never fixes the bug — a legitimate
    NOT_DONE, never an exception.
    """

    name = "always-fails"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        self.calls += 1
        return AttemptResult(diff="", tokens_input=10, tokens_output=5, cost_usd=0.01)


# --- bounded auto-retry for ERROR -------------------------------------------


def test_persistent_infra_failure_is_retried_bounded_times_then_reported_as_error(
    git_repo: Path,
) -> None:
    adapter = _AlwaysErrorsAdapter()
    task_run = run_with_retries(task="fix it", repo=git_repo, adapter=adapter, max_error_retries=2)

    # 1 initial try + 2 retries, all infra errors, none of them ever
    # reaching the agent-retry (`max_attempts`) loop at all.
    assert adapter.calls == 3
    assert task_run.attempt_count == 3
    assert all(v.status is VerdictStatus.ERROR for v in task_run.attempts)
    assert task_run.final.status is VerdictStatus.ERROR
    assert task_run.errored is True
    assert "never became healthy" in (task_run.final.error or "")


def test_transient_infra_failure_recovers_within_the_retry_budget(git_repo: Path) -> None:
    adapter = _ErrorsThenRecoversAdapter(errors_before_success=2)
    task_run = run_with_retries(task="fix it", repo=git_repo, adapter=adapter, max_error_retries=3)

    assert adapter.calls == 3  # 2 errors, then a real (successful) attempt
    assert task_run.attempt_count == 3
    assert task_run.attempts[0].status is VerdictStatus.ERROR
    assert task_run.attempts[1].status is VerdictStatus.ERROR
    assert task_run.final.status is VerdictStatus.DONE
    assert task_run.done is True
    assert task_run.errored is False


def test_error_retry_is_bounded_not_unlimited(git_repo: Path) -> None:
    adapter = _AlwaysErrorsAdapter()
    run_with_retries(task="fix it", repo=git_repo, adapter=adapter, max_error_retries=0)
    assert adapter.calls == 1  # zero extra retries means exactly one try


def test_error_final_status_stops_the_agent_retry_loop_early(git_repo: Path) -> None:
    """Even with a generous `max_attempts`, an ERROR that survived its own
    bounded infra-retry budget must not burn further agent attempts —
    retrying the agent again can't fix a sandbox that's already proven
    broken `max_error_retries` times in a row.
    """
    adapter = _AlwaysErrorsAdapter()
    task_run = run_with_retries(
        task="fix it", repo=git_repo, adapter=adapter, max_attempts=5, max_error_retries=1
    )
    assert adapter.calls == 2  # 1 + 1 error retry, then the agent loop stops
    assert task_run.final.status is VerdictStatus.ERROR


# --- a legitimate NOT_DONE is never auto-retried ----------------------------


def test_agent_not_done_is_never_retried_by_the_error_retry_path(git_repo: Path) -> None:
    """`max_attempts` defaults to 1 (agent retries are opt-in); a
    NOT_DONE Verdict must never trigger the separate, always-on
    `max_error_retries` path — that path only ever sees exceptions.
    """
    adapter = _AlwaysFailsAdapter()
    task_run = run_with_retries(task="fix it", repo=git_repo, adapter=adapter, max_error_retries=3)

    assert adapter.calls == 1
    assert task_run.attempt_count == 1
    assert task_run.final.status is VerdictStatus.NOT_DONE
    assert task_run.errored is False


def test_agent_not_done_only_retries_up_to_max_attempts_never_more(git_repo: Path) -> None:
    adapter = _AlwaysFailsAdapter()
    task_run = run_with_retries(
        task="fix it", repo=git_repo, adapter=adapter, max_attempts=3, max_error_retries=5
    )
    # max_attempts, not max_error_retries, bounds this — proves the two
    # loops are genuinely independent, not one falling back to the other.
    assert adapter.calls == 3
    assert task_run.attempt_count == 3
    assert task_run.final.status is VerdictStatus.NOT_DONE


def test_adapter_crash_is_treated_as_infra_error_not_agent_failure(git_repo: Path) -> None:
    """`Adapter.run`'s own contract: raise only when the adapter itself
    couldn't run, never for the agent merely failing the task. Phase 11
    routes that raise through the exact same ERROR path as a sandbox
    failure — `ClaudeCodeAdapterError` (and every other per-adapter error
    class) is a subclass of `verdict.adapters.AdapterError`.
    """

    class _CrashingAdapter:
        name = "crashes"

        def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
            raise ClaudeCodeAdapterError("claude CLI exited 127: command not found")

    task_run = run_with_retries(task="fix it", repo=git_repo, adapter=_CrashingAdapter(), max_error_retries=1)
    assert task_run.final.status is VerdictStatus.ERROR
    assert task_run.attempt_count == 2
    assert "command not found" in (task_run.final.error or "")


# --- excluded from the leaderboard, end to end via run_suite ----------------


def _init_git(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def test_run_suite_retries_an_infra_failure_and_excludes_it_from_the_leaderboard(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calculator.py").write_text("def add(a, b):\n    return a - b  # bug\n")
    (repo / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "pytest.ini").write_text("[pytest]\n")
    _init_git(repo)

    tasks = [SuiteTask(name="task-a", task="fix it", repo=repo)]
    erroring_adapter = _AlwaysErrorsAdapter()
    error_config = BenchConfig(label="errors", adapter=erroring_adapter)

    results = run_suite(tasks, [error_config], max_error_retries=2)
    (error_result,) = results

    # Retried: 1 + 2 retries, not just a single attempt.
    assert erroring_adapter.calls == 3
    assert error_result.tasks_errored == 1
    assert error_result.tasks_done == 0
    # Excluded from the denominator, not counted as a failure: with the
    # only task errored, pass_rate is the "nothing was graded" 0.0 floor,
    # never a "1 failure out of 1" 0% that would look like a real defect.
    assert error_result.pass_rate == 0.0
    assert error_result.tasks_total == 1


def test_run_suite_pass_rate_denominator_excludes_retried_error_alongside_a_real_pass(
    tmp_path: Path,
) -> None:
    from verdict.adapters.mock import SuiteMockAdapter

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calculator.py").write_text("def add(a, b):\n    return a - b  # bug\n")
    (repo / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "pytest.ini").write_text("[pytest]\n")
    _init_git(repo)

    tasks = [
        SuiteTask(name="task-a", task="fix it", repo=repo),
        SuiteTask(name="task-b", task="fix it", repo=repo),
    ]
    good = SuiteMockAdapter({"fix it": {"calculator.py": "def add(a, b):\n    return a + b\n"}})
    mixed_adapter = _MixedAdapter(good)
    config = BenchConfig(label="mixed", adapter=mixed_adapter)

    results = run_suite(tasks, [config], max_error_retries=1)
    (result,) = results

    assert result.tasks_total == 2
    assert result.tasks_errored == 1
    assert result.tasks_done == 1
    # 1 done out of 1 GRADED task, not 1/2 — the errored task never
    # dilutes the rate in either direction.
    assert result.pass_rate == 1.0


class _MixedAdapter:
    """Errors on its very first call (task-a's only attempt, since
    `max_error_retries=1` still exhausts after 2 tries), then delegates to
    a real adapter for every call after — simulating one task hitting
    infra trouble while the rest of the suite is unaffected.
    """

    name = "mixed"

    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.calls = 0

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        self.calls += 1
        if self.calls <= 2:
            raise SetupError("service never became healthy")
        return self.delegate.run(task, worktree, sandbox=sandbox)
