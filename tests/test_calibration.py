"""Phase 7: judge calibration. Concordance is plain arithmetic over a
labeled dataset, tested against fake judges with known, deliberately
imperfect behavior — no real vision-model integration exists to test
against (see DESIGN.md's Phase 4 section), so a fake judge that we know the
right answer for is what proves the accounting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verdict.calibration import (
    DatasetLoadError,
    load_labeled_dataset,
    run_calibration,
)
from verdict.frontend.vision_judge import VisionJudgment


def _write_dataset(tmp_path: Path, entries: list[dict], screenshot_name: str = "shot.png") -> Path:
    (tmp_path / screenshot_name).write_bytes(b"not-really-a-png-but-judges-dont-care-in-tests")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    return manifest


class _PerfectJudge:
    """Agrees with the human label every time — a control case."""

    name = "perfect"

    def judge(self, screenshot_png: bytes, intent: str) -> VisionJudgment:
        # Cheats by reading the intent text itself, which the fixtures
        # below encode the expected human_label into (see "should-pass"/
        # "should-fail" naming) — good enough for a deterministic fixture
        # judge with no image to actually see.
        return VisionJudgment(passed="should-pass" in intent, rationale="perfect judge")


class _AlwaysPassesJudge:
    """The real MockVisionJudge's behavior, isolated here so calibration's
    own math is tested independently of frontend/vision_judge.py."""

    name = "always-passes"

    def judge(self, screenshot_png: bytes, intent: str) -> VisionJudgment:
        return VisionJudgment(passed=True, rationale="always passes")


# --- load_labeled_dataset -------------------------------------------------


def test_load_labeled_dataset_reads_a_valid_manifest(tmp_path: Path) -> None:
    manifest = _write_dataset(
        tmp_path,
        [{"name": "a", "screenshot": "shot.png", "intent": "should-pass", "human_label": True}],
    )
    examples = load_labeled_dataset(manifest)
    assert len(examples) == 1
    assert examples[0].name == "a"
    assert examples[0].screenshot_path == (tmp_path / "shot.png").resolve()
    assert examples[0].human_label is True


def test_load_labeled_dataset_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoadError, match="couldn't read/parse"):
        load_labeled_dataset(tmp_path / "nope.json")


def test_load_labeled_dataset_rejects_non_array_json(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(DatasetLoadError, match="JSON array"):
        load_labeled_dataset(manifest)


def test_load_labeled_dataset_rejects_missing_key(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"name": "a", "screenshot": "shot.png", "intent": "x"}]))
    with pytest.raises(DatasetLoadError, match="missing a required key"):
        load_labeled_dataset(manifest)


def test_load_labeled_dataset_rejects_missing_screenshot_file(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"name": "a", "screenshot": "ghost.png", "intent": "x", "human_label": True}])
    )
    with pytest.raises(DatasetLoadError, match="no such file"):
        load_labeled_dataset(manifest)


def test_load_labeled_dataset_rejects_empty_list(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    with pytest.raises(DatasetLoadError, match="zero examples"):
        load_labeled_dataset(manifest)


def test_load_labeled_dataset_rejects_non_bool_human_label(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"name": "a", "screenshot": "shot.png", "intent": "x", "human_label": "yes"}])
    )
    (tmp_path / "shot.png").write_bytes(b"x")
    with pytest.raises(DatasetLoadError, match="must be true/false"):
        load_labeled_dataset(manifest)


def test_load_labeled_dataset_resolves_screenshot_relative_to_manifest_dir(tmp_path: Path) -> None:
    sub = tmp_path / "nested"
    sub.mkdir()
    manifest = _write_dataset(
        sub, [{"name": "a", "screenshot": "shot.png", "intent": "x", "human_label": True}]
    )
    examples = load_labeled_dataset(manifest)
    assert examples[0].screenshot_path == (sub / "shot.png").resolve()


# --- run_calibration -------------------------------------------------


def test_run_calibration_reports_full_concordance_for_a_perfect_judge(tmp_path: Path) -> None:
    manifest = _write_dataset(
        tmp_path,
        [
            {"name": "a", "screenshot": "shot.png", "intent": "should-pass", "human_label": True},
            {"name": "b", "screenshot": "shot.png", "intent": "should-fail", "human_label": False},
        ],
    )
    examples = load_labeled_dataset(manifest)
    result = run_calibration(_PerfectJudge(), examples)

    assert result.total == 2
    assert result.agreements == 2
    assert result.concordance == pytest.approx(1.0)
    assert result.meets_threshold is True
    assert result.disagreements == []


def test_run_calibration_flags_disagreements_and_low_concordance(tmp_path: Path) -> None:
    manifest = _write_dataset(
        tmp_path,
        [
            {"name": "a", "screenshot": "shot.png", "intent": "x", "human_label": True},
            {"name": "b", "screenshot": "shot.png", "intent": "y", "human_label": False},
            {"name": "c", "screenshot": "shot.png", "intent": "z", "human_label": False},
        ],
    )
    examples = load_labeled_dataset(manifest)
    result = run_calibration(_AlwaysPassesJudge(), examples, threshold=0.95)

    assert result.total == 3
    assert result.agreements == 1  # only the human_label=True case agrees
    assert result.concordance == pytest.approx(1 / 3)
    assert result.meets_threshold is False
    assert {d.name for d in result.disagreements} == {"b", "c"}


def test_calibration_result_concordance_is_none_for_zero_examples() -> None:
    from verdict.calibration import CalibrationResult

    result = CalibrationResult(judge_name="x", total=0, agreements=0)
    assert result.concordance is None
    assert result.meets_threshold is False
