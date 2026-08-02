"""Phase 12: defend the thesis. `integrity.py` compares an attempt's final
commit against the pre-agent base commit and flags test-tampering — this
file proves each detector independently (a deleted test file, a weakened
assertion, a newly added skip/xfail marker, a hardcoded expected output, a
dropped collected-test-count, a coverage drop), proves the whole thing
forces `NOT_DONE` end-to-end through `runner.run()`, and proves the
allowance trust boundary: a task-declared allowance lets a legitimate
test-editing task through, but nothing inside the graded repo's own
`verdict.yml` can grant one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from verdict.adapters import Adapter
from verdict.integrity import (
    DENY_ALL,
    TestChangeAllowance,
    check_test_integrity,
    coverage_regression_finding,
)
from verdict.runner import run
from verdict.sandbox import SandboxConfig
from verdict.schema import AttemptResult, GateStatus, Provenance, Signal, VerdictStatus


def _commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _check(repo: Path, base: str, final: str, allowance: TestChangeAllowance = DENY_ALL) -> Signal:
    return check_test_integrity(
        repo=repo,
        base_commit=base,
        final_commit=final,
        final_worktree_path=repo,
        final_signals=[],
        allowance=allowance,
        sandbox_config=SandboxConfig(backend="local"),
        measure_coverage=False,
    )


# --- individual detectors, each its own fixture -----------------------------


def test_deleted_test_file_is_flagged(git_repo: Path) -> None:
    base = _head(git_repo)
    (git_repo / "test_calculator.py").unlink()
    final = _commit(git_repo, "delete the test file")

    signal = _check(git_repo, base, final)
    assert signal.status is GateStatus.FAIL
    kinds = {f.code for f in signal.failures}
    assert "test_file_deleted" in kinds


def test_weakened_assertion_is_flagged(git_repo: Path) -> None:
    base = _head(git_repo)
    (git_repo / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    pass\n"
    )
    final = _commit(git_repo, "gut the assertion")

    signal = _check(git_repo, base, final)
    assert signal.status is GateStatus.FAIL
    kinds = {f.code for f in signal.failures}
    assert "assertions_weakened" in kinds


def test_added_xfail_marker_is_flagged(git_repo: Path) -> None:
    base = _head(git_repo)
    (git_repo / "test_calculator.py").write_text(
        "import pytest\nfrom calculator import add\n\n\n"
        "@pytest.mark.xfail\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    final = _commit(git_repo, "mark the test xfail")

    signal = _check(git_repo, base, final)
    assert signal.status is GateStatus.FAIL
    kinds = {f.code for f in signal.failures}
    assert "skip_marker_added" in kinds


def test_added_skip_marker_is_flagged(git_repo: Path) -> None:
    base = _head(git_repo)
    (git_repo / "test_calculator.py").write_text(
        "import pytest\nfrom calculator import add\n\n\n"
        "@pytest.mark.skip(reason='flaky')\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    final = _commit(git_repo, "skip the test")

    signal = _check(git_repo, base, final)
    assert signal.status is GateStatus.FAIL
    kinds = {f.code for f in signal.failures}
    assert "skip_marker_added" in kinds


def test_hardcoded_expected_output_is_flagged(git_repo: Path) -> None:
    base = _head(git_repo)
    (git_repo / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == -1\n"
    )
    final = _commit(git_repo, "match whatever the buggy code returns")

    signal = _check(git_repo, base, final)
    assert signal.status is GateStatus.FAIL
    kinds = {f.code for f in signal.failures}
    assert "hardcoded_expected_output" in kinds
    finding = next(f for f in signal.failures if f.code == "hardcoded_expected_output")
    assert "5" in finding.message and "-1" in finding.message


def test_collected_test_count_drop_is_flagged(git_repo: Path) -> None:
    """Doesn't touch a `test_*.py` file's content at all — deletes
    `pytest.ini` so autodetection stops finding the suite, exactly the
    "disable discovery without touching a test file" vector the
    execution-based count check (not the diff-based file check) exists
    to catch.
    """
    base = _head(git_repo)
    (git_repo / "pytest.ini").unlink()
    final = _commit(git_repo, "remove pytest.ini")

    final_signal = Signal(
        name="test", provenance=Provenance.PROVEN, status=GateStatus.PASS, detail="", tests_collected=0
    )
    signal = check_test_integrity(
        repo=git_repo,
        base_commit=base,
        final_commit=final,
        final_worktree_path=git_repo,
        final_signals=[final_signal],
        allowance=DENY_ALL,
        sandbox_config=SandboxConfig(backend="local"),
        measure_coverage=False,
    )
    assert signal.status is GateStatus.FAIL
    kinds = {f.code for f in signal.failures}
    assert "collected_test_count_dropped" in kinds


def test_coverage_drop_finding_pure_function() -> None:
    assert coverage_regression_finding(90.0, 60.0) is not None
    assert coverage_regression_finding(90.0, 89.0) is None  # within tolerance
    assert coverage_regression_finding(None, 60.0) is None  # nothing to compare
    assert coverage_regression_finding(90.0, None) is None


def test_coverage_drop_forces_a_fail_when_wired_through_check(git_repo: Path, monkeypatch) -> None:
    """Proves the wiring, not just the pure comparison: with
    `measure_pytest_coverage` stubbed to canned before/after numbers, a
    real drop shows up in the `integrity` Signal's findings.
    """
    import verdict.integrity as integrity_module

    base = _head(git_repo)
    (git_repo / "README.md").write_text("unrelated change\n")
    final = _commit(git_repo, "unrelated change")

    calls = iter([40.0, 80.0])  # first call: final worktree (low); second: base commit (high)
    monkeypatch.setattr(integrity_module, "measure_pytest_coverage", lambda *a, **k: next(calls))

    signal = check_test_integrity(
        repo=git_repo,
        base_commit=base,
        final_commit=final,
        final_worktree_path=git_repo,
        final_signals=[],
        allowance=DENY_ALL,
        sandbox_config=SandboxConfig(backend="local"),
        measure_coverage=True,
    )
    assert signal.status is GateStatus.FAIL
    kinds = {f.code for f in signal.failures}
    assert "coverage_dropped" in kinds


# --- allowance / trust boundary ---------------------------------------------


def test_allowance_lets_a_declared_test_edit_through(git_repo: Path) -> None:
    base = _head(git_repo)
    (git_repo / "test_calculator.py").unlink()
    final = _commit(git_repo, "delete the test file")

    signal = _check(git_repo, base, final, allowance=TestChangeAllowance(allowed=True))
    assert signal.status is GateStatus.PASS
    assert signal.failures == []


def test_allowance_scoped_to_specific_paths_does_not_excuse_other_files(git_repo: Path) -> None:
    base = _head(git_repo)
    (git_repo / "test_calculator.py").unlink()
    final = _commit(git_repo, "delete the test file")

    scoped = TestChangeAllowance(allowed=True, allowed_paths=("tests/test_new_feature.py",))
    signal = _check(git_repo, base, final, allowance=scoped)
    assert signal.status is GateStatus.FAIL


def test_newly_added_test_file_is_never_flagged(git_repo: Path) -> None:
    """Adding tests is never suspicious — only editing/deleting ones that
    already existed at the base commit is."""
    base = _head(git_repo)
    (git_repo / "test_extra.py").write_text("def test_extra():\n    assert True\n")
    final = _commit(git_repo, "add another test")

    # Realistic final count (base had 1, this adds a 2nd) — otherwise the
    # collected-count-drop check has nothing but `_check`'s empty default
    # to compare against and would spuriously see "0 tests now".
    final_signal = Signal(
        name="test", provenance=Provenance.PROVEN, status=GateStatus.PASS, detail="", tests_collected=2
    )
    signal = check_test_integrity(
        repo=git_repo,
        base_commit=base,
        final_commit=final,
        final_worktree_path=git_repo,
        final_signals=[final_signal],
        allowance=DENY_ALL,
        sandbox_config=SandboxConfig(backend="local"),
        measure_coverage=False,
    )
    assert signal.status is GateStatus.PASS


def test_default_allowance_is_deny_all() -> None:
    assert DENY_ALL.allowed is False
    assert DENY_ALL.permits("tests/test_x.py") is False


def test_permits_respects_glob_scoping() -> None:
    allowance = TestChangeAllowance(allowed=True, allowed_paths=("tests/test_new_*.py",))
    assert allowance.permits("tests/test_new_feature.py") is True
    assert allowance.permits("tests/test_other.py") is False


# --- end to end: forces NOT_DONE through runner.run() -----------------------


class _GutsTheTestsAdapter:
    """A "cheating" agent: makes the test pass by deleting the assertion
    entirely, changing nothing about the actual bug.
    """

    name = "cheater"

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        (worktree / "test_calculator.py").write_text(
            "from calculator import add\n\n\ndef test_add():\n    pass\n"
        )
        return AttemptResult(diff="", tokens_input=10, tokens_output=5, cost_usd=0.01)


class _LegitimatelyEditsTestsAdapter:
    """Fixes the real bug AND updates the test — a legitimate "add
    tests"/"fix the bug and its test" task, not a cheat.
    """

    name = "legit"

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        (worktree / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
        (worktree / "test_calculator.py").write_text(
            "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
            "\n\ndef test_add_negative():\n    assert add(-1, -1) == -2\n"
        )
        return AttemptResult(diff="", tokens_input=10, tokens_output=5, cost_usd=0.01)


def test_gutting_tests_forces_not_done_even_though_the_test_gate_passes(git_repo: Path) -> None:
    adapter: Adapter = _GutsTheTestsAdapter()  # type: ignore[assignment]
    verdict = run(
        task="fix it", repo=git_repo, adapter=adapter, sandbox_config=SandboxConfig(backend="local")
    )

    test_signal = next(s for s in verdict.signals if s.name == "test")
    assert test_signal.status is GateStatus.PASS  # the gutted test suite "passes"

    integrity_signal = next(s for s in verdict.signals if s.name == "integrity")
    assert integrity_signal.status is GateStatus.FAIL

    # The whole point: a task "passed" by disabling its tests is NOT done.
    assert verdict.status is VerdictStatus.NOT_DONE
    assert verdict.done is False


def test_allowance_lets_a_legitimate_test_edit_reach_done(git_repo: Path) -> None:
    adapter: Adapter = _LegitimatelyEditsTestsAdapter()  # type: ignore[assignment]
    verdict = run(
        task="fix add() and strengthen its tests",
        repo=git_repo,
        adapter=adapter,
        sandbox_config=SandboxConfig(backend="local"),
        allow_test_changes=TestChangeAllowance(allowed=True),
    )

    integrity_signal = next(s for s in verdict.signals if s.name == "integrity")
    assert integrity_signal.status is GateStatus.PASS
    assert verdict.status is VerdictStatus.DONE


def test_without_allowance_the_same_legitimate_edit_is_flagged_for_review(git_repo: Path) -> None:
    """Strict-by-default: even a GOOD-faith test edit gets flagged absent
    an explicit allowance — the false-positive cost (a human reviews an
    expected FAIL) is intentional, see `TestChangeAllowance`'s docstring.
    """
    adapter: Adapter = _LegitimatelyEditsTestsAdapter()  # type: ignore[assignment]
    verdict = run(task="fix add() and strengthen its tests", repo=git_repo, adapter=adapter,
                  sandbox_config=SandboxConfig(backend="local"))

    integrity_signal = next(s for s in verdict.signals if s.name == "integrity")
    assert integrity_signal.status is GateStatus.FAIL
    assert verdict.status is VerdictStatus.NOT_DONE


# --- suite/loader.py: task.yml is the trusted source ------------------------


def test_suite_task_parses_allow_test_changes_bool(tmp_path: Path) -> None:
    from verdict.suite.loader import load_suite

    suite = tmp_path / "suite"
    task_dir = suite / "add-tests"
    repo = task_dir / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "README.md").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    (task_dir / "task.yml").write_text('task: "add tests"\nallow_test_changes: true\n')

    (task,) = load_suite(suite)
    assert task.allow_test_changes.allowed is True
    assert task.allow_test_changes.allowed_paths is None


def test_suite_task_parses_allow_test_changes_path_list(tmp_path: Path) -> None:
    from verdict.suite.loader import load_suite

    suite = tmp_path / "suite"
    task_dir = suite / "add-tests"
    repo = task_dir / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "README.md").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    (task_dir / "task.yml").write_text(
        'task: "add tests"\nallow_test_changes:\n  - "tests/test_new_feature.py"\n'
    )

    (task,) = load_suite(suite)
    assert task.allow_test_changes.allowed is True
    assert task.allow_test_changes.allowed_paths == ("tests/test_new_feature.py",)


def test_suite_task_defaults_to_deny_all_without_the_key(tmp_path: Path) -> None:
    from verdict.suite.loader import load_suite

    suite = tmp_path / "suite"
    task_dir = suite / "bug-fix"
    repo = task_dir / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "README.md").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    (task_dir / "task.yml").write_text('task: "fix the bug"\n')

    (task,) = load_suite(suite)
    assert task.allow_test_changes.allowed is False


def test_suite_task_rejects_a_malformed_allow_test_changes_value(tmp_path: Path) -> None:
    from verdict.suite import SuiteLoadError
    from verdict.suite.loader import load_suite

    suite = tmp_path / "suite"
    task_dir = suite / "bad"
    repo = task_dir / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "README.md").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    (task_dir / "task.yml").write_text('task: "bad"\nallow_test_changes: "yes please"\n')

    with pytest.raises(SuiteLoadError):
        load_suite(suite)


def test_verdict_config_has_no_allow_test_changes_field() -> None:
    """The trust boundary, asserted structurally: `VerdictConfig` (parsed
    from the graded repo's OWN `verdict.yml`) must never grow an
    `allow_test_changes` field — see `TestChangeAllowance`'s docstring
    for why. If this test starts failing because someone added the
    field, that's the bug, not this test.
    """
    from verdict.config import VerdictConfig

    instance = VerdictConfig(gate_overrides={})
    assert not hasattr(instance, "allow_test_changes")
