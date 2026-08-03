"""Phase 20: does Verdict's own DONE/NOT_DONE/UNVERIFIED status agree with
a human? Split the same way `test_calibration.py` splits: pure-arithmetic
claims (confusion matrix, precision/recall/F1, the zero-division-stays-
None discipline) tested directly against known counts; loader defensive
parsing tested against every malformed manifest shape; `run_ground_truth`
proven end to end against real git repos through the real `runner.run()`
pipeline, not a mocked comparison.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from verdict.calibration import DatasetLoadError
from verdict.ground_truth import (
    ClassMetrics,
    GroundTruthResult,
    load_ground_truth_dataset,
    run_ground_truth,
)
from verdict.sandbox import SandboxConfig
from verdict.schema import VerdictStatus


def _init_git(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)


def _bug_repo(base: Path, name: str, extra_files: dict[str, str] | None = None) -> Path:
    repo = base / name
    repo.mkdir()
    (repo / "calculator.py").write_text("def add(a, b):\n    return a - b  # bug\n")
    (repo / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "pytest.ini").write_text("[pytest]\n")
    for path, contents in (extra_files or {}).items():
        (repo / path).write_text(contents)
    _init_git(repo)
    return repo


def _write_manifest(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    return manifest


# --- ClassMetrics / GroundTruthResult arithmetic ----------------------------


def test_class_metrics_precision_recall_f1() -> None:
    metrics = ClassMetrics(label=VerdictStatus.DONE, true_positives=3, false_positives=1, false_negatives=2)
    assert metrics.precision == pytest.approx(3 / 4)
    assert metrics.recall == pytest.approx(3 / 5)
    assert metrics.f1 == pytest.approx(2 * (3 / 4) * (3 / 5) / ((3 / 4) + (3 / 5)))


def test_class_metrics_zero_denominator_is_none_not_zero() -> None:
    metrics = ClassMetrics(
        label=VerdictStatus.UNVERIFIED, true_positives=0, false_positives=0, false_negatives=0
    )
    assert metrics.precision is None
    assert metrics.recall is None
    assert metrics.f1 is None


def _confusion(**cells: int) -> dict[str, dict[str, int]]:
    labels = ("done", "not_done", "unverified")
    matrix = {h: {v: 0 for v in labels} for h in labels}
    for key, count in cells.items():
        human, verdict = key.split("__")
        matrix[human][verdict] = count
    return matrix


def test_ground_truth_result_perfect_agreement() -> None:
    result = GroundTruthResult(
        total=3,
        confusion=_confusion(done__done=1, not_done__not_done=1, unverified__unverified=1),
    )
    assert result.correct == 3
    assert result.accuracy == 1.0
    assert result.meets_threshold is True
    assert result.macro_f1 == 1.0
    for metrics in result.per_class:
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0


def test_ground_truth_result_confusion_matrix_drives_per_class_counts() -> None:
    # 1 true done predicted done, 1 true done predicted not_done (a miss),
    # 1 true not_done predicted done (a false alarm for "done").
    result = GroundTruthResult(
        total=3,
        confusion=_confusion(done__done=1, done__not_done=1, not_done__done=1),
    )
    done_metrics = next(m for m in result.per_class if m.label is VerdictStatus.DONE)
    assert done_metrics.true_positives == 1
    assert done_metrics.false_positives == 1  # not_done->done
    assert done_metrics.false_negatives == 1  # done->not_done
    assert done_metrics.precision == 0.5
    assert done_metrics.recall == 0.5
    assert result.accuracy == pytest.approx(1 / 3)


def test_ground_truth_result_accuracy_is_none_with_zero_graded_examples() -> None:
    result = GroundTruthResult(total=2, confusion=_confusion(), errored=2, errored_examples=["a", "b"])
    assert result.graded == 0
    assert result.accuracy is None
    assert result.meets_threshold is False


def test_ground_truth_result_threshold_gate() -> None:
    result = GroundTruthResult(
        total=2, confusion=_confusion(done__done=1, not_done__done=1), threshold=0.9
    )
    assert result.accuracy == 0.5
    assert result.meets_threshold is False


# --- load_ground_truth_dataset defensive parsing ----------------------------


def test_load_dataset_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoadError):
        load_ground_truth_dataset(tmp_path / "does-not-exist.json")


def test_load_dataset_not_a_json_array(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [])  # placeholder, overwritten below
    manifest.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(DatasetLoadError, match="JSON array"):
        load_ground_truth_dataset(manifest)


def test_load_dataset_missing_required_key(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [{"name": "x", "repo": "repo", "task": "do it", "patch": {"a.py": "x"}}],  # no human_label
    )
    with pytest.raises(DatasetLoadError, match="human_label"):
        load_ground_truth_dataset(manifest)


def test_load_dataset_rejects_empty_patch(tmp_path: Path) -> None:
    repo = _bug_repo(tmp_path, "repo")
    manifest = _write_manifest(
        tmp_path,
        [{"name": "x", "repo": "repo", "task": "do it", "patch": {}, "human_label": "done"}],
    )
    with pytest.raises(DatasetLoadError, match="patch"):
        load_ground_truth_dataset(manifest)
    assert repo.exists()  # sanity: fixture actually ran


def test_load_dataset_rejects_unknown_label(tmp_path: Path) -> None:
    _bug_repo(tmp_path, "repo")
    manifest = _write_manifest(
        tmp_path,
        [{"name": "x", "repo": "repo", "task": "do it", "patch": {"a.py": "x"}, "human_label": "maybe"}],
    )
    with pytest.raises(DatasetLoadError, match="human_label"):
        load_ground_truth_dataset(manifest)


def test_load_dataset_rejects_error_as_a_human_label(tmp_path: Path) -> None:
    # "error" is a real VerdictStatus value but not something a human ever
    # judges — it means infra couldn't evaluate the attempt.
    _bug_repo(tmp_path, "repo")
    manifest = _write_manifest(
        tmp_path,
        [{"name": "x", "repo": "repo", "task": "do it", "patch": {"a.py": "x"}, "human_label": "error"}],
    )
    with pytest.raises(DatasetLoadError, match="human_label"):
        load_ground_truth_dataset(manifest)


def test_load_dataset_rejects_a_repo_that_is_not_a_git_repo(tmp_path: Path) -> None:
    (tmp_path / "repo").mkdir()
    manifest = _write_manifest(
        tmp_path,
        [{"name": "x", "repo": "repo", "task": "do it", "patch": {"a.py": "x"}, "human_label": "done"}],
    )
    with pytest.raises(DatasetLoadError, match="not a git repo"):
        load_ground_truth_dataset(manifest)


def test_load_dataset_rejects_zero_examples(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [])
    with pytest.raises(DatasetLoadError, match="zero examples"):
        load_ground_truth_dataset(manifest)


def test_load_dataset_reads_a_real_manifest(tmp_path: Path) -> None:
    _bug_repo(tmp_path, "repo")
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "name": "fix-add",
                "repo": "repo",
                "task": "fix add()",
                "patch": {"calculator.py": "def add(a, b):\n    return a + b\n"},
                "human_label": "done",
            }
        ],
    )
    examples = load_ground_truth_dataset(manifest)
    assert len(examples) == 1
    assert examples[0].human_label is VerdictStatus.DONE
    assert examples[0].repo_path == (tmp_path / "repo").resolve()


# --- run_ground_truth end to end --------------------------------------------


def test_run_ground_truth_agrees_when_the_fix_is_real_and_labeled_done(tmp_path: Path) -> None:
    repo = _bug_repo(tmp_path, "repo")
    example = load_ground_truth_dataset(
        _write_manifest(
            tmp_path,
            [
                {
                    "name": "clean-fix",
                    "repo": "repo",
                    "task": "fix add()",
                    "patch": {"calculator.py": "def add(a, b):\n    return a + b\n"},
                    "human_label": "done",
                }
            ],
        )
    )
    result = run_ground_truth(example)

    assert result.total == 1
    assert result.correct == 1
    assert result.accuracy == 1.0
    assert result.disagreements == []
    assert repo.exists()


def test_run_ground_truth_agrees_when_still_broken_and_labeled_not_done(tmp_path: Path) -> None:
    _bug_repo(tmp_path, "repo")
    examples = load_ground_truth_dataset(
        _write_manifest(
            tmp_path,
            [
                {
                    "name": "still-broken",
                    "repo": "repo",
                    "task": "fix add()",
                    "patch": {"NOTES.md": "wip\n"},
                    "human_label": "not_done",
                }
            ],
        )
    )
    result = run_ground_truth(examples)
    assert result.correct == 1
    assert result.confusion["not_done"]["not_done"] == 1


def test_run_ground_truth_agrees_on_unverified_when_nothing_had_a_chance_to_run(tmp_path: Path) -> None:
    # UNVERIFIED is reachable through `run()`'s normal path only via
    # `budget_exceeded` — see DESIGN.md's Phase 20 section on why "no
    # detectable stack" alone no longer produces it now that Phase 12's
    # integrity signal is always a real PROVEN PASS/FAIL, never NA. This
    # mirrors `test_timeouts.py`'s own
    # `test_runner_marks_budget_exceeded_and_never_reports_done_when_the_
    # budget_is_zero` — same mechanism, deterministic, no timing race.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# widget scheduler\n")
    (repo / "CHANGELOG.md").write_text("## Unreleased\n")
    _init_git(repo)
    examples = load_ground_truth_dataset(
        _write_manifest(
            tmp_path,
            [
                {
                    "name": "budget-exhausted",
                    "repo": "repo",
                    "task": "note the new retry option in the changelog",
                    "patch": {"CHANGELOG.md": "## Unreleased\n- Added a retry option.\n"},
                    "human_label": "unverified",
                }
            ],
        )
    )
    sandbox_config = SandboxConfig(backend="local", attempt_budget_seconds=0)
    result = run_ground_truth(examples, sandbox_config=sandbox_config)
    assert result.correct == 1
    assert result.confusion["unverified"]["unverified"] == 1


def test_run_ground_truth_surfaces_a_genuine_disagreement_with_evidence(tmp_path: Path) -> None:
    # The agent correctly fixes add(), but a pre-existing, unrelated test
    # keeps the whole test gate FAIL — Verdict says NOT_DONE (any failing
    # PROVEN gate sinks the run), while a task-scoped human reviewer calls
    # it done. A real, structural disagreement, not a bug in either side.
    _bug_repo(
        tmp_path,
        "repo",
        extra_files={
            "test_known_issue.py": (
                "def test_tracked_separately():\n"
                "    # pre-existing, unrelated to add() — tracked elsewhere\n"
                "    assert False\n"
            )
        },
    )
    examples = load_ground_truth_dataset(
        _write_manifest(
            tmp_path,
            [
                {
                    "name": "preexisting-failure",
                    "repo": "repo",
                    "task": "fix add()",
                    "patch": {"calculator.py": "def add(a, b):\n    return a + b\n"},
                    "human_label": "done",
                }
            ],
        )
    )
    result = run_ground_truth(examples)

    assert result.correct == 0
    assert result.accuracy == 0.0
    assert len(result.disagreements) == 1
    disagreement = result.disagreements[0]
    assert disagreement.human_label is VerdictStatus.DONE
    assert disagreement.verdict_status is VerdictStatus.NOT_DONE
    assert "test" in disagreement.detail  # points at the real failing signal, not a vague claim


def test_run_ground_truth_mixed_dataset_produces_the_expected_confusion_matrix(tmp_path: Path) -> None:
    _bug_repo(tmp_path, "clean-fix")
    _bug_repo(tmp_path, "still-broken")
    entries: list[dict[str, object]] = [
        {
            "name": "clean-fix",
            "repo": "clean-fix",
            "task": "fix add()",
            "patch": {"calculator.py": "def add(a, b):\n    return a + b\n"},
            "human_label": "done",
        },
        {
            "name": "still-broken",
            "repo": "still-broken",
            "task": "fix add()",
            "patch": {"NOTES.md": "wip\n"},
            "human_label": "not_done",
        },
    ]
    examples = load_ground_truth_dataset(_write_manifest(tmp_path, entries))
    result = run_ground_truth(examples)

    assert result.total == 2
    assert result.correct == 2
    assert result.accuracy == 1.0
    assert result.confusion["done"]["done"] == 1
    assert result.confusion["not_done"]["not_done"] == 1
    assert result.disagreements == []
