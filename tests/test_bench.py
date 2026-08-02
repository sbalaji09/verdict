"""End-to-end for `run_suite`: real git repos, real pytest, real worktree
isolation per (config, task) pair — the same "no mocked pipeline" standard
`test_runner.py` and `test_attribution.py` hold themselves to, since this
is exactly the class of orchestration logic (does every config really run
against every task, independently) that's easy to get subtly wrong on
paper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from verdict.adapters.mock import SuiteMockAdapter
from verdict.economics import rank
from verdict.schema import AttemptResult
from verdict.suite import BenchConfig, LocalProcessPoolExecutor, load_suite, run_suite


def _init_git(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


@pytest.fixture
def suite_dir(tmp_path: Path) -> Path:
    suite = tmp_path / "suite"

    task_a = suite / "fix-add"
    repo_a = task_a / "repo"
    repo_a.mkdir(parents=True)
    (repo_a / "calculator.py").write_text("def add(a, b):\n    return a - b  # bug\n")
    (repo_a / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo_a / "pytest.ini").write_text("[pytest]\n")
    _init_git(repo_a)
    (task_a / "task.yml").write_text('task: "fix add()"\ncategory: bug-fix\n')

    task_b = suite / "fix-greet"
    repo_b = task_b / "repo"
    repo_b.mkdir(parents=True)
    (repo_b / "greet.py").write_text("def greet(name):\n    return 'bye ' + name  # bug\n")
    (repo_b / "test_greet.py").write_text(
        "from greet import greet\n\n\ndef test_greet():\n    assert greet('sam') == 'hello sam'\n"
    )
    (repo_b / "pytest.ini").write_text("[pytest]\n")
    _init_git(repo_b)
    (task_b / "task.yml").write_text('task: "fix greet()"\ncategory: bug-fix\n')

    return suite


def _good_and_bad_adapters(tasks) -> tuple[SuiteMockAdapter, SuiteMockAdapter]:
    by_name = {t.name: t.task for t in tasks}
    good = SuiteMockAdapter(
        {
            by_name["fix-add"]: {"calculator.py": "def add(a, b):\n    return a + b\n"},
            by_name["fix-greet"]: {"greet.py": "def greet(name):\n    return 'hello ' + name\n"},
        }
    )
    bad = SuiteMockAdapter(
        {
            by_name["fix-add"]: {"README.md": "noop\n"},
            by_name["fix-greet"]: {"README.md": "noop\n"},
        }
    )
    return good, bad


def test_run_suite_produces_one_config_result_per_config_in_order(suite_dir: Path) -> None:
    tasks = load_suite(suite_dir)
    good, bad = _good_and_bad_adapters(tasks)
    configs = [BenchConfig(label="good", adapter=good), BenchConfig(label="bad", adapter=bad)]

    results = run_suite(tasks, configs)

    assert [r.label for r in results] == ["good", "bad"]
    assert results[0].tasks_total == 2
    assert results[1].tasks_total == 2


def test_run_suite_runs_every_config_against_every_task_independently(suite_dir: Path) -> None:
    tasks = load_suite(suite_dir)
    good, bad = _good_and_bad_adapters(tasks)
    results = run_suite(tasks, [BenchConfig("good", good), BenchConfig("bad", bad)])

    good_result, bad_result = results
    assert good_result.tasks_done == 2
    assert bad_result.tasks_done == 0


def test_rank_places_the_fully_passing_config_first(suite_dir: Path) -> None:
    tasks = load_suite(suite_dir)
    good, bad = _good_and_bad_adapters(tasks)
    results = run_suite(tasks, [BenchConfig("bad", bad), BenchConfig("good", good)])

    ranked = rank(results)
    # both configs cost $0 (SuiteMockAdapter), so pass_rate_per_dollar is
    # undefined for both — rank falls back to raw pass rate as the tiebreak.
    assert ranked[0].label == "good"


class _FlakyThenFixAdapter:
    """Fails a task's first attempt, fixes it on the second — proves
    `run_suite` threads `max_attempts` through to `run_with_retries` per
    task, not just per config.
    """

    name = "flaky"

    def __init__(self, fix_patches: dict[str, dict[str, str]]) -> None:
        self._fix_patches = fix_patches
        self.calls_by_task: dict[str, int] = {}

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        self.calls_by_task[task] = self.calls_by_task.get(task, 0) + 1
        if self.calls_by_task[task] >= 2:
            for relative_path, contents in self._fix_patches[task].items():
                (worktree / relative_path).write_text(contents)
        return AttemptResult(diff="", tokens_input=10, tokens_output=5, cost_usd=0.01)


def test_run_suite_threads_max_attempts_through_to_every_task(suite_dir: Path) -> None:
    tasks = load_suite(suite_dir)
    by_name = {t.name: t.task for t in tasks}
    adapter = _FlakyThenFixAdapter(
        {
            by_name["fix-add"]: {"calculator.py": "def add(a, b):\n    return a + b\n"},
            by_name["fix-greet"]: {"greet.py": "def greet(name):\n    return 'hello ' + name\n"},
        }
    )

    results = run_suite(tasks, [BenchConfig("flaky", adapter)], max_attempts=2)

    assert results[0].tasks_done == 2
    for task_run in results[0].task_runs:
        assert task_run.attempt_count == 2


def _leaderboard_snapshot(results):
    """The parts of a `list[ConfigResult]` that a leaderboard actually
    reports on — everything `Verdict.created_at`-independent. Used to
    compare a parallel run against a serial one: the two will never share
    a `created_at` timestamp, but that's not part of "the leaderboard,"
    just bookkeeping metadata nothing downstream ranks or renders on.
    """
    return [
        (
            r.label,
            r.tasks_total,
            r.tasks_done,
            r.tasks_errored,
            r.pass_rate,
            r.pass_rate_per_dollar,
            r.total_cost_usd,
            [
                (t.task, t.agent, t.repo, t.done, t.attempt_count, t.total_cost_usd)
                for t in r.task_runs
            ],
        )
        for r in results
    ]


def test_run_suite_parallel_produces_the_same_leaderboard_as_serial(suite_dir: Path) -> None:
    """Phase 15's core correctness claim: running the exact same suite
    through a bounded process pool must rank and score identically to
    running it one pair at a time — aggregation can't depend on which
    order work actually finished in.
    """
    tasks = load_suite(suite_dir)
    good, bad = _good_and_bad_adapters(tasks)
    configs = [BenchConfig(label="good", adapter=good), BenchConfig(label="bad", adapter=bad)]

    serial_results = run_suite(tasks, configs)

    good2, bad2 = _good_and_bad_adapters(tasks)
    configs2 = [BenchConfig(label="good", adapter=good2), BenchConfig(label="bad", adapter=bad2)]
    parallel_results = run_suite(tasks, configs2, executor=LocalProcessPoolExecutor(max_workers=3))

    assert _leaderboard_snapshot(serial_results) == _leaderboard_snapshot(parallel_results)


class _FixedCostAdapter:
    """Applies a task's canned patch (like `SuiteMockAdapter`) but reports
    a fixed, nonzero `cost_usd` — `SuiteMockAdapter` always costs $0, which
    can never trip a cost ceiling. Exists only so
    `test_run_suite_cost_ceiling` has something real to spend against.
    """

    name = "mock"

    def __init__(self, patches_by_task: dict[str, dict[str, str]], cost_usd: float) -> None:
        self._patches_by_task = patches_by_task
        self._cost_usd = cost_usd

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        for relative_path, contents in self._patches_by_task[task].items():
            (worktree / relative_path).write_text(contents)
        return AttemptResult(diff="", tokens_input=10, tokens_output=5, cost_usd=self._cost_usd)


def test_run_suite_cost_ceiling_skips_work_once_reached(suite_dir: Path) -> None:
    """Enforcement is exact (not just best-effort) under the default
    `SerialExecutor`: each of the 2 tasks costs $0.01, the ceiling is
    $0.015, so the first config's 2 tasks ($0.02 total) push the running
    total past the ceiling and the second config's 2 tasks are skipped
    (recorded ERROR, excluded from pass_rate) before ever starting.
    """
    tasks = load_suite(suite_dir)
    by_name = {t.name: t.task for t in tasks}
    patches = {
        by_name["fix-add"]: {"calculator.py": "def add(a, b):\n    return a + b\n"},
        by_name["fix-greet"]: {"greet.py": "def greet(name):\n    return 'hello ' + name\n"},
    }
    configs = [
        BenchConfig("first", _FixedCostAdapter(patches, cost_usd=0.01)),
        BenchConfig("second", _FixedCostAdapter(patches, cost_usd=0.01)),
    ]

    results = run_suite(tasks, configs, cost_ceiling_usd=0.015)

    first, second = results
    assert first.tasks_done == 2
    assert first.tasks_errored == 0
    assert second.tasks_done == 0
    assert second.tasks_errored == 2
    for task_run in second.task_runs:
        assert "cost ceiling" in (task_run.final.error or "")
