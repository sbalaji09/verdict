from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from verdict.worktree import WorktreeError, diff_against_base, isolated_worktree


def _branches(repo: Path) -> str:
    return subprocess.run(
        ["git", "branch", "--list"], cwd=repo, capture_output=True, text=True
    ).stdout


def test_worktree_is_isolated_from_source_repo(git_repo: Path) -> None:
    original = (git_repo / "calculator.py").read_text()

    with isolated_worktree(git_repo) as wt:
        assert wt.path.exists()
        assert (wt.path / "calculator.py").read_text() == original
        (wt.path / "calculator.py").write_text("mutated")
        # the source repo's working tree must be untouched
        assert (git_repo / "calculator.py").read_text() == original

    # and cleaned up afterward
    assert not wt.path.exists()
    assert wt.branch not in _branches(git_repo)


def test_worktree_cleans_up_even_if_body_raises(git_repo: Path) -> None:
    captured_path = None
    with pytest.raises(RuntimeError), isolated_worktree(git_repo) as wt:
        captured_path = wt.path
        raise RuntimeError("boom")

    assert captured_path is not None
    assert not captured_path.exists()


def test_isolated_worktree_rejects_non_git_dir(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    with pytest.raises(WorktreeError), isolated_worktree(not_a_repo):
        pass


def test_diff_against_base_reports_changed_files(git_repo: Path) -> None:
    with isolated_worktree(git_repo) as wt:
        (wt.path / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
        diff, files = diff_against_base(wt.path, wt.base_commit)

    assert files == ["calculator.py"]
    assert "return a + b" in diff


def test_diff_against_base_ignores_head_moving_forward(git_repo: Path) -> None:
    # if the agent commits mid-run, HEAD moves — the diff must still be
    # computed against the original base commit, not current HEAD, or it
    # would silently miss everything the agent already committed.
    with isolated_worktree(git_repo) as wt:
        (wt.path / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
        subprocess.run(["git", "add", "-A"], cwd=wt.path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "agent commit"], cwd=wt.path, check=True)

        diff, files = diff_against_base(wt.path, wt.base_commit)

    assert files == ["calculator.py"]
    assert "return a + b" in diff


def test_worktree_records_base_commit(git_repo: Path) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=git_repo, capture_output=True, text=True
    ).stdout.strip()
    with isolated_worktree(git_repo) as wt:
        assert wt.base_commit == head


def test_scratch_worktree_is_detached_and_cleans_up(git_repo: Path) -> None:
    from verdict.worktree import scratch_worktree

    with scratch_worktree(git_repo, "HEAD") as path:
        assert path.exists()
        branch_name = subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=path, capture_output=True, text=True
        )
        assert branch_name.returncode != 0  # detached: no symbolic ref

    assert not path.exists()
