"""Phase 17: `Store` round-trips (`SQLiteStore`, the default/only shipped
backend) and regression detection across recorded runs. `test_reporters.py`
already established the "build a `ConfigResult` tree directly, don't run a
full grading pipeline" pattern for testing report/aggregation code that
only cares about the shape, not how it was produced — this file follows
the same pattern.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from verdict.flakiness import ComparisonVerdict
from verdict.schema import AttemptResult, ConfigResult, GateStatus, Provenance, Signal, TaskRun, Verdict
from verdict.store import SQLiteStore, detect_regressions
from verdict.store.regression import DEFAULT_BASELINE_WINDOW


def _verdict(*, done: bool, task: str = "fix the bug", agent: str = "mock", repo: str = "/tmp/x") -> Verdict:
    status = GateStatus.PASS if done else GateStatus.FAIL
    return Verdict(
        task=task,
        agent=agent,
        repo=repo,
        attempt=AttemptResult(diff="", tokens_input=10, tokens_output=5, cost_usd=0.01),
        signals=[Signal(name="test", provenance=Provenance.PROVEN, status=status, detail="junit")],
    )


def _task_run(*, done: bool, task: str = "fix the bug", agent: str = "mock", repo: str = "/tmp/x") -> TaskRun:
    verdict = _verdict(done=done, task=task, agent=agent, repo=repo)
    return TaskRun(task=task, agent=agent, repo=repo, attempts=[verdict])


def _config_result(label: str, task_runs: list[TaskRun]) -> ConfigResult:
    return ConfigResult(label=label, task_runs=task_runs)


# --- round-trip ------------------------------------------------------


def test_record_and_get_config_results_round_trips_exactly(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "verdict.db")
    original = [_config_result("good", [_task_run(done=True), _task_run(done=False, task="other task")])]

    run_id = store.record_run(original, commit_sha="abc123", label="nightly")
    restored = store.get_config_results(run_id)

    assert len(restored) == 1
    assert restored[0].label == "good"
    assert restored[0].tasks_total == 2
    assert restored[0].tasks_done == 1
    # verdicts, signals, cost — the whole nested tree survives the round trip
    assert restored[0].task_runs[0].attempts[0].signals[0].status == GateStatus.PASS
    assert restored[0].task_runs[0].attempts[0].attempt.cost_usd == 0.01


def test_get_config_results_returns_empty_list_for_unknown_run_id(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "verdict.db")
    assert store.get_config_results("no-such-run") == []


def test_record_run_persists_artifact_path_references(tmp_path: Path) -> None:
    """Artifact references (screenshots/videos/logs) are just paths
    carried on `Signal.artifact_path` — proving they round-trip proves
    the whole nested `ConfigResult` JSON survives storage, no special
    artifact-table handling needed (see `store/base.py`'s docstring)."""
    verdict = Verdict(
        task="frontend check",
        agent="mock",
        repo="/tmp/x",
        attempt=AttemptResult(diff=""),
        signals=[
            Signal(
                name="frontend:glitch_scan:load",
                provenance=Provenance.PROVEN,
                status=GateStatus.FAIL,
                detail="flicker detected",
                artifact_path="/tmp/verdict-capture/load.webm",
            )
        ],
    )
    task_run = TaskRun(task=verdict.task, agent="mock", repo="/tmp/x", attempts=[verdict])
    store = SQLiteStore(tmp_path / "verdict.db")

    run_id = store.record_run([_config_result("cfg", [task_run])])
    restored = store.get_config_results(run_id)

    assert restored[0].task_runs[0].attempts[0].signals[0].artifact_path == "/tmp/verdict-capture/load.webm"


def test_runs_lists_recorded_runs_most_recent_first(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "verdict.db")
    first = store.record_run(
        [_config_result("cfg", [_task_run(done=True)])],
        commit_sha="sha1",
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    second = store.record_run(
        [_config_result("cfg", [_task_run(done=True)])],
        commit_sha="sha2",
        recorded_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    runs = store.runs()

    assert [r.run_id for r in runs] == [second, first]
    assert runs[0].commit_sha == "sha2"
    assert runs[0].config_labels == ("cfg",)


def test_history_is_most_recent_first_and_respects_limit(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "verdict.db")
    for day in range(1, 6):
        store.record_run(
            [_config_result("cfg", [_task_run(done=True)])],
            recorded_at=datetime(2026, 1, day, tzinfo=UTC),
        )

    outcomes = store.history("fix the bug", "mock", "/tmp/x", limit=3)

    assert len(outcomes) == 3
    assert [o.recorded_at.day for o in outcomes] == [5, 4, 3]


def test_history_filters_by_config_label(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "verdict.db")
    store.record_run(
        [_config_result("cfg-a", [_task_run(done=True)]), _config_result("cfg-b", [_task_run(done=False)])]
    )

    a_outcomes = store.history("fix the bug", "mock", "/tmp/x", config_label="cfg-a")
    b_outcomes = store.history("fix the bug", "mock", "/tmp/x", config_label="cfg-b")

    assert len(a_outcomes) == 1 and a_outcomes[0].done is True
    assert len(b_outcomes) == 1 and b_outcomes[0].done is False


def test_history_excludes_a_given_run_id(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "verdict.db")
    older = store.record_run(
        [_config_result("cfg", [_task_run(done=True)])], recorded_at=datetime(2026, 1, 1, tzinfo=UTC)
    )
    newer = store.record_run(
        [_config_result("cfg", [_task_run(done=False)])], recorded_at=datetime(2026, 1, 2, tzinfo=UTC)
    )

    without_newer = store.history("fix the bug", "mock", "/tmp/x", exclude_run_id=newer)
    without_older = store.history("fix the bug", "mock", "/tmp/x", exclude_run_id=older)

    assert [o.run_id for o in without_newer] == [older]
    assert [o.run_id for o in without_older] == [newer]


# --- regression detection ---------------------------------------------


def _seed_history(store: SQLiteStore, n: int, done: bool, task: str = "fix the bug") -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(n):
        store.record_run(
            [_config_result("cfg", [_task_run(done=done, task=task)])],
            recorded_at=base + timedelta(days=i),
        )


def test_detect_regressions_flags_a_clear_break_from_a_long_clean_history(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "verdict.db")
    _seed_history(store, n=20, done=True)  # 20/20 passing historically

    candidate = [_config_result("cfg", [_task_run(done=False)])]  # today's run fails
    regressions = detect_regressions(store, candidate)

    assert len(regressions) == 1
    assert regressions[0].task == "fix the bug"
    assert regressions[0].config_label == "cfg"
    assert regressions[0].comparison.verdict is ComparisonVerdict.REGRESSION
    assert regressions[0].comparison.p_value is not None
    assert regressions[0].comparison.p_value < 0.05


def test_detect_regressions_does_not_flag_noise_from_a_small_history(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "verdict.db")
    # 3 passes, 1 fail historically — already noisy; one more fail today
    # shouldn't clear statistical significance.
    _seed_history(store, n=3, done=True)
    _seed_history(store, n=1, done=False)

    candidate = [_config_result("cfg", [_task_run(done=False)])]
    regressions = detect_regressions(store, candidate)

    assert regressions == []


def test_detect_regressions_skips_tasks_with_no_prior_history(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "verdict.db")  # empty — nothing recorded yet

    candidate = [_config_result("cfg", [_task_run(done=False)])]
    regressions = detect_regressions(store, candidate)

    assert regressions == []


def test_detect_regressions_never_flags_an_improvement(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "verdict.db")
    _seed_history(store, n=20, done=False)  # always failing historically

    candidate = [_config_result("cfg", [_task_run(done=True)])]  # today it passes
    regressions = detect_regressions(store, candidate)

    assert regressions == []  # an IMPROVEMENT, never reported as a regression


def test_detect_regressions_excludes_the_current_run_from_its_own_baseline(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "verdict.db")
    _seed_history(store, n=20, done=True)

    candidate = [_config_result("cfg", [_task_run(done=False)])]
    run_id = store.record_run(candidate)  # persist the "current" run first

    regressions = detect_regressions(store, candidate, exclude_run_id=run_id)

    # Still flagged from the 20 prior clean runs — the just-recorded
    # failing run itself must not have been folded into its own baseline.
    assert len(regressions) == 1


def test_detect_regressions_respects_the_baseline_window(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "verdict.db")
    # Recent history is noisy (small effective baseline), but a much older,
    # larger clean run of passes exists too — a bounded window must not
    # reach back into it and manufacture significance from stale data.
    _seed_history(store, n=3, done=True)
    _seed_history(store, n=1, done=False)

    candidate = [_config_result("cfg", [_task_run(done=False)])]
    regressions = detect_regressions(store, candidate, baseline_window=DEFAULT_BASELINE_WINDOW)

    assert regressions == []
