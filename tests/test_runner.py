from __future__ import annotations

from pathlib import Path

from verdict.adapters.mock import MockAdapter
from verdict.runner import run


def test_run_end_to_end_reports_done_after_fix(git_repo: Path) -> None:
    adapter = MockAdapter(patches={"calculator.py": "def add(a, b):\n    return a + b\n"})
    verdict = run(task="fix add()", repo=git_repo, adapter=adapter)

    assert verdict.done is True
    assert verdict.signals[0].name == "test"
    assert verdict.signals[0].passed is True
    assert verdict.attempt.files_changed == ["calculator.py"]


def test_run_end_to_end_reports_not_done_when_bug_untouched(git_repo: Path) -> None:
    adapter = MockAdapter(patches={"README.md": "unrelated\n"})
    verdict = run(task="do nothing useful", repo=git_repo, adapter=adapter)

    assert verdict.done is False
    assert verdict.signals[0].passed is False


def test_run_does_not_mutate_source_repo(git_repo: Path) -> None:
    original = (git_repo / "calculator.py").read_text()
    adapter = MockAdapter(patches={"calculator.py": "def add(a, b):\n    return a + b\n"})
    run(task="fix add()", repo=git_repo, adapter=adapter)

    assert (git_repo / "calculator.py").read_text() == original
