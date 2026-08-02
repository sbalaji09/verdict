"""Phase 7: judge calibration. Concordance is plain arithmetic over a
labeled dataset, tested against fake judges with known, deliberately
imperfect behavior. Phase 16 adds a real `VisionJudge` (`RealVisionJudge`,
provider-agnostic, backed by a `VisionModelTransport`) — calibrated the
same way, against a scripted FAKE transport (never real network; see
`test_run_calibration_end_to_end_against_the_real_judge_with_a_fake_
transport` below), so this module's own accounting is now provably
exercised through the real judge implementation, not just hand-rolled
fakes of `VisionJudge` itself.
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
from verdict.frontend.vision_judge import (
    RealVisionJudge,
    TransportResult,
    VisionJudgment,
    VisionTransportError,
)


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


# --- Phase 16: end-to-end against RealVisionJudge, transport faked -------


class _ScriptedTransport:
    """A `VisionModelTransport` double scripted by intent text — the
    "mocked provider transport" the real judge is calibrated against here.
    Never touches the network; `RealVisionJudge` (the real, shipped
    integration) is otherwise exercised completely unmodified, so this
    proves calibration's accounting against the actual judge code path,
    not a hand-rolled `VisionJudge` fake.
    """

    name = "scripted"

    def __init__(self, scripted: dict[str, TransportResult | Exception]) -> None:
        self._scripted = scripted

    def complete(self, screenshot_png: bytes, system_prompt: str, user_text: str) -> TransportResult:
        for intent, outcome in self._scripted.items():
            if intent in user_text:
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        raise AssertionError(f"no scripted outcome for user_text={user_text!r}")


def test_run_calibration_end_to_end_against_the_real_judge_with_a_fake_transport(tmp_path: Path) -> None:
    """The full stack this phase adds: `load_labeled_dataset` ->
    `RealVisionJudge.judge` (real prompt-building, real injection-hardened
    system prompt, real degrade-on-failure contract) -> a scripted fake
    `VisionModelTransport` -> `run_calibration`'s concordance/unavailable
    accounting. Two agreements, one disagreement, one transport failure —
    proving all three buckets flow through correctly from a labeled subset.
    """
    manifest = _write_dataset(
        tmp_path,
        [
            {"name": "agree-pass", "screenshot": "shot.png", "intent": "cta visible", "human_label": True},
            {"name": "agree-fail", "screenshot": "shot.png", "intent": "modal closed", "human_label": False},
            {"name": "disagree", "screenshot": "shot.png", "intent": "footer link", "human_label": True},
            {"name": "flaky-call", "screenshot": "shot.png", "intent": "nav logo", "human_label": True},
        ],
    )
    examples = load_labeled_dataset(manifest)

    transport = _ScriptedTransport(
        {
            "cta visible": TransportResult(
                passed=True, rationale="cta is visible", tokens_input=10, tokens_output=5
            ),
            "modal closed": TransportResult(
                passed=False, rationale="modal is closed", tokens_input=10, tokens_output=5
            ),
            "footer link": TransportResult(
                passed=False, rationale="no footer link seen", tokens_input=10, tokens_output=5
            ),
            "nav logo": VisionTransportError("simulated API timeout"),
        }
    )
    judge = RealVisionJudge(transport)

    result = run_calibration(judge, examples, threshold=0.9)

    assert result.total == 4
    assert result.unavailable == 1
    assert result.unavailable_examples == ["flaky-call"]
    assert result.graded == 3
    assert result.agreements == 2
    assert result.concordance == pytest.approx(2 / 3)
    assert {d.name for d in result.disagreements} == {"disagree"}
    assert judge.name == "vision-judge:scripted"
