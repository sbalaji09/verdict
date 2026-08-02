"""Phase 13: held-out FAIL_TO_PASS/PASS_TO_PASS acceptance tests, applied
after the agent finishes and never visible to it. The one scenario this
phase exists for: a task whose VISIBLE suite passes trivially (it never
exercised the actual feature) must still come back NOT_DONE if the
held-out test proving the feature was really built still fails.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from verdict.acceptance import (
    NONE_ACCEPTANCE,
    AcceptanceSpec,
    check_acceptance,
)
from verdict.adapters import Adapter
from verdict.runner import run
from verdict.sandbox import SandboxConfig
from verdict.schema import AttemptResult, GateStatus, VerdictStatus


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


def _make_patch(base_repo: Path, tmp_path: Path, files: dict[str, str]) -> str:
    """A real, `git apply`-able unified diff adding/modifying `files` —
    built via a throwaway clone rather than hand-written, so the patch
    text is exactly what `git` itself produces (correct headers, correct
    "new file mode", no hand-typed index-hash guessing).
    """
    clone = tmp_path / f"patch_clone_{len(list(tmp_path.glob('patch_clone_*')))}"
    subprocess.run(["git", "clone", "-q", str(base_repo), str(clone)], check=True)
    for rel, content in files.items():
        p = clone / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True)
    result = subprocess.run(
        ["git", "diff", "--cached", "--no-color"], cwd=clone, capture_output=True, text=True, check=True
    )
    return result.stdout


@pytest.fixture
def feature_repo(tmp_path: Path) -> Path:
    """A repo whose VISIBLE test suite is trivial and passes no matter
    what the agent does — the exact scenario this phase exists for. The
    real feature (`add`) is buggy; nothing visible ever checks it.
    """
    repo = tmp_path / "feature_repo"
    repo.mkdir()
    (repo / "feature.py").write_text("def add(a, b):\n    return a - b  # bug\n")
    (repo / "test_feature.py").write_text("def test_trivial():\n    assert True\n")
    (repo / "pytest.ini").write_text("[pytest]\n")

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    return repo


_HIDDEN_TEST = (
    "from feature import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
)


# --- check_acceptance, directly ---------------------------------------------


def test_none_declared_returns_no_signal(git_repo: Path) -> None:
    signal = check_acceptance(
        git_repo, _head(git_repo), NONE_ACCEPTANCE, SandboxConfig(backend="local")
    )
    assert signal is None


def test_fail_to_pass_still_failing_is_a_proven_fail(feature_repo: Path, tmp_path: Path) -> None:
    patch = _make_patch(feature_repo, tmp_path, {"test_hidden.py": _HIDDEN_TEST})
    final = _head(feature_repo)  # agent did nothing — bug still present

    spec = AcceptanceSpec(fail_to_pass=("test_hidden.py::test_add",), patch=patch)
    signal = check_acceptance(feature_repo, final, spec, SandboxConfig(backend="local"))

    assert signal is not None
    assert signal.status is GateStatus.FAIL
    kinds = {f.code for f in signal.failures}
    assert "fail_to_pass" in kinds


def test_fail_to_pass_now_passing_is_a_proven_pass(feature_repo: Path, tmp_path: Path) -> None:
    patch = _make_patch(feature_repo, tmp_path, {"test_hidden.py": _HIDDEN_TEST})
    (feature_repo / "feature.py").write_text("def add(a, b):\n    return a + b\n")
    final = _commit(feature_repo, "fix the bug for real")

    spec = AcceptanceSpec(fail_to_pass=("test_hidden.py::test_add",), patch=patch)
    signal = check_acceptance(feature_repo, final, spec, SandboxConfig(backend="local"))

    assert signal is not None
    assert signal.status is GateStatus.PASS
    assert signal.failures == []


def test_pass_to_pass_regression_is_flagged(feature_repo: Path, tmp_path: Path) -> None:
    patch = _make_patch(feature_repo, tmp_path, {"test_hidden.py": _HIDDEN_TEST})
    # Agent "fixes" add() but breaks the trivial pre-existing test along the way.
    (feature_repo / "feature.py").write_text("def add(a, b):\n    return a + b\n")
    (feature_repo / "test_feature.py").write_text("def test_trivial():\n    assert False\n")
    final = _commit(feature_repo, "fix add but break the existing test")

    spec = AcceptanceSpec(
        fail_to_pass=("test_hidden.py::test_add",),
        pass_to_pass=("test_feature.py::test_trivial",),
        patch=patch,
    )
    signal = check_acceptance(feature_repo, final, spec, SandboxConfig(backend="local"))

    assert signal is not None
    assert signal.status is GateStatus.FAIL
    kinds = {f.code for f in signal.failures}
    assert "pass_to_pass" in kinds
    assert "fail_to_pass" not in kinds  # the FAIL_TO_PASS side genuinely passed


def test_a_node_id_that_never_collects_is_flagged(feature_repo: Path, tmp_path: Path) -> None:
    patch = _make_patch(feature_repo, tmp_path, {"test_hidden.py": _HIDDEN_TEST})
    final = _head(feature_repo)

    spec = AcceptanceSpec(fail_to_pass=("test_hidden.py::test_does_not_exist",), patch=patch)
    signal = check_acceptance(feature_repo, final, spec, SandboxConfig(backend="local"))

    assert signal is not None
    assert signal.status is GateStatus.FAIL
    finding = signal.failures[0]
    assert "never collected" in finding.message


def test_a_patch_that_fails_to_apply_is_a_proven_fail_not_an_error(feature_repo: Path) -> None:
    bogus_patch = (
        "diff --git a/nonexistent_dir/file.py b/nonexistent_dir/file.py\n"
        "--- a/nonexistent_dir/file.py\n"
        "+++ b/nonexistent_dir/file.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-this does not exist in the repo\n"
        "+neither does this\n"
    )
    spec = AcceptanceSpec(fail_to_pass=("whatever::test_x",), patch=bogus_patch)
    signal = check_acceptance(feature_repo, _head(feature_repo), spec, SandboxConfig(backend="local"))

    assert signal is not None
    assert signal.status is GateStatus.FAIL
    kinds = {f.code for f in signal.failures}
    assert "tests_patch_apply_failed" in kinds


# --- end to end: the exact scenario this phase exists for -------------------


class _NoOpAdapter:
    """Does nothing at all — the visible suite still passes (it's
    trivial), so a pre-Phase-13 grade would have reported DONE.
    """

    name = "noop"

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        return AttemptResult(diff="", tokens_input=1, tokens_output=1, cost_usd=0.001)


class _RealFixAdapter:
    name = "real-fix"

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        (worktree / "feature.py").write_text("def add(a, b):\n    return a + b\n")
        return AttemptResult(diff="", tokens_input=1, tokens_output=1, cost_usd=0.001)


def test_visible_suite_passes_but_held_out_fail_to_pass_fails_is_not_done(
    feature_repo: Path, tmp_path: Path
) -> None:
    patch = _make_patch(feature_repo, tmp_path, {"test_hidden.py": _HIDDEN_TEST})
    spec = AcceptanceSpec(fail_to_pass=("test_hidden.py::test_add",), patch=patch)

    adapter: Adapter = _NoOpAdapter()  # type: ignore[assignment]
    verdict = run(
        task="add real add()",
        repo=feature_repo,
        adapter=adapter,
        sandbox_config=SandboxConfig(backend="local"),
        acceptance=spec,
    )

    test_signal = next(s for s in verdict.signals if s.name == "test")
    assert test_signal.status is GateStatus.PASS  # the trivial visible suite "passes"

    acceptance_signal = next(s for s in verdict.signals if s.name == "acceptance")
    assert acceptance_signal.status is GateStatus.FAIL

    # The whole point of this phase: visible-green is not enough.
    assert verdict.status is VerdictStatus.NOT_DONE
    assert verdict.done is False


def test_a_real_fix_that_satisfies_the_held_out_test_reaches_done(
    feature_repo: Path, tmp_path: Path
) -> None:
    patch = _make_patch(feature_repo, tmp_path, {"test_hidden.py": _HIDDEN_TEST})
    spec = AcceptanceSpec(
        fail_to_pass=("test_hidden.py::test_add",),
        pass_to_pass=("test_feature.py::test_trivial",),
        patch=patch,
    )

    adapter: Adapter = _RealFixAdapter()  # type: ignore[assignment]
    verdict = run(
        task="add real add()",
        repo=feature_repo,
        adapter=adapter,
        sandbox_config=SandboxConfig(backend="local"),
        acceptance=spec,
    )

    acceptance_signal = next(s for s in verdict.signals if s.name == "acceptance")
    assert acceptance_signal.status is GateStatus.PASS
    assert verdict.status is VerdictStatus.DONE


def test_no_acceptance_spec_is_unchanged_pre_phase_13_behavior(feature_repo: Path) -> None:
    """A task with no held-out tests declared gets no `acceptance` signal
    at all — the trivial visible suite is enough to reach DONE, exactly
    as it would have before this phase existed.
    """
    adapter: Adapter = _NoOpAdapter()  # type: ignore[assignment]
    verdict = run(
        task="noop", repo=feature_repo, adapter=adapter, sandbox_config=SandboxConfig(backend="local")
    )
    assert not any(s.name == "acceptance" for s in verdict.signals)
    assert verdict.status is VerdictStatus.DONE


# --- suite/loader.py: task.yml + tests.patch parsing ------------------------


def _init_git(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "README.md").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


def test_suite_task_parses_fail_to_pass_and_pass_to_pass(tmp_path: Path) -> None:
    from verdict.suite.loader import load_suite

    suite = tmp_path / "suite"
    task_dir = suite / "add-feature"
    repo = task_dir / "repo"
    repo.mkdir(parents=True)
    _init_git(repo)
    (task_dir / "tests.patch").write_text("diff --git a/x b/x\n--- a/x\n+++ b/x\n")
    (task_dir / "task.yml").write_text(
        'task: "add the feature"\n'
        "fail_to_pass:\n  - \"tests/test_x.py::test_new\"\n"
        "pass_to_pass:\n  - \"tests/test_x.py::test_old\"\n"
    )

    (task,) = load_suite(suite)
    assert task.acceptance.fail_to_pass == ("tests/test_x.py::test_new",)
    assert task.acceptance.pass_to_pass == ("tests/test_x.py::test_old",)
    assert task.acceptance.declared is True


def test_suite_task_requires_tests_patch_when_lists_are_declared(tmp_path: Path) -> None:
    from verdict.suite import SuiteLoadError
    from verdict.suite.loader import load_suite

    suite = tmp_path / "suite"
    task_dir = suite / "add-feature"
    repo = task_dir / "repo"
    repo.mkdir(parents=True)
    _init_git(repo)
    (task_dir / "task.yml").write_text(
        'task: "add the feature"\nfail_to_pass:\n  - "tests/test_x.py::test_new"\n'
    )

    with pytest.raises(SuiteLoadError, match="tests.patch"):
        load_suite(suite)


def test_suite_task_defaults_to_no_acceptance_without_the_keys(tmp_path: Path) -> None:
    from verdict.suite.loader import load_suite

    suite = tmp_path / "suite"
    task_dir = suite / "bug-fix"
    repo = task_dir / "repo"
    repo.mkdir(parents=True)
    _init_git(repo)
    (task_dir / "task.yml").write_text('task: "fix the bug"\n')

    (task,) = load_suite(suite)
    assert task.acceptance.declared is False
    assert task.acceptance is NONE_ACCEPTANCE
