from __future__ import annotations

from pathlib import Path

from verdict.adapters.mock import MockAdapter
from verdict.runner import run
from verdict.schema import GateStatus, VerdictStatus


def _signal(verdict, name: str):
    return next(s for s in verdict.signals if s.name == name)


def test_run_end_to_end_reports_done_after_fix(git_repo: Path) -> None:
    adapter = MockAdapter(patches={"calculator.py": "def add(a, b):\n    return a + b\n"})
    verdict = run(task="fix add()", repo=git_repo, adapter=adapter)

    assert verdict.done is True
    assert verdict.status is VerdictStatus.DONE
    assert _signal(verdict, "test").status is GateStatus.PASS
    assert verdict.attempt.files_changed == ["calculator.py"]


def test_run_end_to_end_reports_not_done_when_bug_untouched(git_repo: Path) -> None:
    adapter = MockAdapter(patches={"README.md": "unrelated\n"})
    verdict = run(task="do nothing useful", repo=git_repo, adapter=adapter)

    assert verdict.done is False
    assert verdict.status is VerdictStatus.NOT_DONE
    assert _signal(verdict, "test").status is GateStatus.FAIL


def test_run_only_emits_applicable_gates(git_repo: Path) -> None:
    # git_repo fixture has no tsconfig.json, no package.json build script,
    # no eslint/ruff config — only pytest applies.
    adapter = MockAdapter(patches={"calculator.py": "def add(a, b):\n    return a + b\n"})
    verdict = run(task="fix add()", repo=git_repo, adapter=adapter)

    by_name = {s.name: s.status for s in verdict.signals}
    assert by_name["test"] is GateStatus.PASS
    assert by_name["typecheck"] is GateStatus.NA
    assert by_name["build"] is GateStatus.NA
    assert by_name["lint"] is GateStatus.NA


def test_run_does_not_mutate_source_repo(git_repo: Path) -> None:
    original = (git_repo / "calculator.py").read_text()
    adapter = MockAdapter(patches={"calculator.py": "def add(a, b):\n    return a + b\n"})
    run(task="fix add()", repo=git_repo, adapter=adapter)

    assert (git_repo / "calculator.py").read_text() == original
