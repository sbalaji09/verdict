from __future__ import annotations

from pathlib import Path

import pytest

from verdict.suite.loader import SuiteLoadError, load_suite


@pytest.fixture
def suite_dir(tmp_path: Path) -> Path:
    suite = tmp_path / "suite"
    task_a = suite / "b-task"
    (task_a / "repo").mkdir(parents=True)
    (task_a / "task.yml").write_text('task: "do the b thing"\ncategory: bug-fix\n')

    task_b = suite / "a-task"
    (task_b / "repo").mkdir(parents=True)
    (task_b / "task.yml").write_text('task: "do the a thing"\n')

    return suite


def test_load_suite_parses_every_task_dir_with_a_task_yml(suite_dir: Path) -> None:
    tasks = load_suite(suite_dir)
    assert {t.name for t in tasks} == {"a-task", "b-task"}


def test_load_suite_sorts_by_directory_name_for_determinism(suite_dir: Path) -> None:
    tasks = load_suite(suite_dir)
    assert [t.name for t in tasks] == ["a-task", "b-task"]


def test_load_suite_parses_task_text_and_optional_category(suite_dir: Path) -> None:
    tasks = {t.name: t for t in load_suite(suite_dir)}
    assert tasks["b-task"].task == "do the b thing"
    assert tasks["b-task"].category == "bug-fix"
    assert tasks["a-task"].category is None


def test_load_suite_resolves_repo_relative_to_task_dir(suite_dir: Path) -> None:
    tasks = {t.name: t for t in load_suite(suite_dir)}
    assert tasks["a-task"].repo == suite_dir / "a-task" / "repo"


def test_load_suite_supports_custom_repo_key(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    task_dir = suite / "custom"
    (task_dir / "checkout").mkdir(parents=True)
    (task_dir / "task.yml").write_text('task: "x"\nrepo: checkout\n')

    tasks = load_suite(suite)
    assert tasks[0].repo == task_dir / "checkout"


def test_load_suite_ignores_subdirectories_without_task_yml(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    real_task = suite / "real"
    (real_task / "repo").mkdir(parents=True)
    (real_task / "task.yml").write_text('task: "x"\n')
    (suite / "not_a_task").mkdir(parents=True)

    tasks = load_suite(suite)
    assert [t.name for t in tasks] == ["real"]


def test_load_suite_raises_when_directory_missing(tmp_path: Path) -> None:
    with pytest.raises(SuiteLoadError, match="does not exist"):
        load_suite(tmp_path / "nope")


def test_load_suite_raises_when_no_tasks_found(tmp_path: Path) -> None:
    empty = tmp_path / "empty_suite"
    empty.mkdir()
    with pytest.raises(SuiteLoadError, match="no task.yml"):
        load_suite(empty)


def test_load_suite_raises_when_task_key_missing(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    task_dir = suite / "broken"
    (task_dir / "repo").mkdir(parents=True)
    (task_dir / "task.yml").write_text("category: bug-fix\n")

    with pytest.raises(SuiteLoadError, match="missing required"):
        load_suite(suite)


def test_load_suite_raises_when_repo_directory_missing(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    task_dir = suite / "broken"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yml").write_text('task: "x"\n')

    with pytest.raises(SuiteLoadError, match="does not exist"):
        load_suite(suite)
