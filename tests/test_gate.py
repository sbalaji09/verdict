"""Phase 6's merge-gate mode: grading a repo exactly as it's already
checked out, no adapter involved — the entry point the GitHub Action
drives via `verdict gate`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from verdict.cli import app
from verdict.runner import grade_existing_diff
from verdict.schema import AttributionKind, VerdictStatus


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)


def _commit_all(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def base_repo(tmp_path: Path) -> tuple[Path, str]:
    """A repo with one commit — a bug — standing in for a PR's base branch."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "calculator.py").write_text("def add(a, b):\n    return a - b  # bug\n")
    (repo / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "pytest.ini").write_text("[pytest]\n")
    base = _commit_all(repo, "base")
    return repo, base


# --- grade_existing_diff, unit-level -----------------------------------


def test_grade_existing_diff_passes_when_the_pr_fixes_the_bug(base_repo: tuple[Path, str]) -> None:
    repo, base = base_repo
    (repo / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    _commit_all(repo, "fix")

    verdict = grade_existing_diff(repo=repo, base_ref=base)

    assert verdict.status is VerdictStatus.DONE
    assert verdict.attempt.files_changed == ["calculator.py"]
    assert verdict.attempt.cost_usd is None  # no adapter ran; cost is unknown, not zero


def test_grade_existing_diff_fails_and_attributes_a_pre_existing_bug(base_repo: tuple[Path, str]) -> None:
    repo, base = base_repo
    (repo / "NOTES.md").write_text("unrelated change\n")
    _commit_all(repo, "noop")

    verdict = grade_existing_diff(repo=repo, base_ref=base)

    assert verdict.status is VerdictStatus.NOT_DONE
    assert len(verdict.attributions) == 1
    assert verdict.attributions[0].kind is AttributionKind.PRE_EXISTING


def test_grade_existing_diff_bisects_a_real_regression_across_two_commits(tmp_path: Path) -> None:
    # Base is already-working code (unlike base_repo above, whose base
    # commit is itself buggy) — the "PR" then adds two real commits, one
    # unrelated, one that actually breaks the test, so attribution has to
    # bisect rather than trivially blame "the only changed commit."
    repo = tmp_path / "regression_repo"
    _init_repo(repo)
    (repo / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "pytest.ini").write_text("[pytest]\n")
    base = _commit_all(repo, "base — working")

    (repo / "NOTES.md").write_text("unrelated documentation change\n")
    _commit_all(repo, "docs: add notes")

    (repo / "calculator.py").write_text("def add(a, b):\n    return a - b  # introduced bug\n")
    _commit_all(repo, "oops, broke it")

    verdict = grade_existing_diff(repo=repo, base_ref=base)

    assert verdict.status is VerdictStatus.NOT_DONE
    assert len(verdict.attributions) == 1
    assert verdict.attributions[0].kind is AttributionKind.REGRESSION
    assert verdict.attributions[0].culprit_file == "calculator.py"


def test_grade_existing_diff_never_mutates_the_real_repo(base_repo: tuple[Path, str]) -> None:
    repo, base = base_repo
    (repo / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    final = _commit_all(repo, "fix")

    grade_existing_diff(repo=repo, base_ref=base)

    assert _head(repo) == final
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert branch  # still on a real branch, not left detached by bisection


# --- end-to-end via the CLI, against the real examples/sample_repo -----


def test_gate_cli_end_to_end_against_the_sample_repo_example(tmp_path: Path) -> None:
    """The most direct feasible stand-in for "run the action against a real
    PR" available in this sandbox (no real GitHub infrastructure to drive):
    bootstrap the actual `examples/sample_repo` (the same `setup.sh` a CI
    workflow would run), copy it so the checked-in example is never
    mutated, commit a real fix as a second commit standing in for the PR's
    own commits, and drive the exact `verdict gate` command the GitHub
    Action's `run: verdict gate ...` step invokes.
    """
    source = Path(__file__).resolve().parents[1] / "examples" / "sample_repo"
    subprocess.run(["bash", str(source / "setup.sh")], check=True, capture_output=True, text=True)

    repo = tmp_path / "sample_repo_copy"
    shutil.copytree(source, repo)
    base = _head(repo)

    (repo / "sample_repo" / "calculator.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    _commit_all(repo, "fix the seeded bug")

    output_dir = tmp_path / "verdict-report"
    result = CliRunner().invoke(
        app,
        [
            "gate", "--repo", str(repo), "--base", base, "--sandbox-backend", "local",
            "--report", "json", "--report", "html", "--output-dir", str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads((output_dir / "verdict-report.json").read_text())
    task_run = report["configs"][0]["task_runs"][0]
    assert task_run["done"] is True
    assert task_run["attempts"][-1]["status"] == "done"
    assert (output_dir / "verdict-report.html").exists()


def test_gate_cli_end_to_end_fails_when_the_pr_does_not_fix_the_bug(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "examples" / "sample_repo"
    subprocess.run(["bash", str(source / "setup.sh")], check=True, capture_output=True, text=True)

    repo = tmp_path / "sample_repo_copy_unfixed"
    shutil.copytree(source, repo)
    base = _head(repo)

    (repo / "NOTES.md").write_text("an unrelated change that doesn't touch the bug\n")
    _commit_all(repo, "irrelevant commit")

    output_dir = tmp_path / "verdict-report"
    result = CliRunner().invoke(
        app,
        [
            "gate", "--repo", str(repo), "--base", base, "--sandbox-backend", "local",
            "--report", "json", "--output-dir", str(output_dir),
        ],
    )

    assert result.exit_code == 1
    report = json.loads((output_dir / "verdict-report.json").read_text())
    assert report["configs"][0]["task_runs"][0]["done"] is False
