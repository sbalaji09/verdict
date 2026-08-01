"""Judge calibration: how often does a `VisionJudge`'s opinion actually
match a human reviewer's, on a labeled sample?

Every JUDGED signal in this codebase (Phase 4's vision-intent check) is
deliberately advisory — `Verdict.status` never lets one flip a failing
PROVEN check to DONE, or vice versa. That discipline answers "can a bad
opinion sink a good run," but it doesn't answer a different, equally real
question a team adopting a vision judge needs answered before they trust
its comments at all: *is this judge's opinion actually any good?* A judge
that agrees with a human reviewer 55% of the time is barely better than a
coin flip, and nothing in the PROVEN/JUDGED split itself would ever surface
that — an advisory signal that's usually wrong still renders in the PR
comment looking exactly as confident as one that's usually right. This
module measures that gap directly: run the judge over a labeled dataset,
report concordance (fraction of exact agreement with the human label)
against a target threshold, and name every individual disagreement rather
than only the aggregate number.

No new trust bucket, no schema change to `Signal`/`Verdict` — a
`CalibrationResult` is a report *about* a `VisionJudge`, generated once
against a labeled sample, never something a `Verdict` itself carries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, computed_field
from rich.console import Console
from rich.table import Table

from verdict.frontend.vision_judge import VisionJudge

DEFAULT_CONCORDANCE_THRESHOLD = 0.95


@dataclass
class LabeledExample:
    """One human-reviewed case: a real screenshot, the intent it was
    checked against, and the human's own verdict on whether the intent was
    met. `screenshot_path` is resolved and read lazily by `run_calibration`
    — kept as a path here (not loaded bytes) so a large dataset manifest
    stays cheap to parse and hold in memory before any judging happens.
    """

    name: str
    screenshot_path: Path
    intent: str
    human_label: bool


class DatasetLoadError(Exception):
    """The manifest at a given path couldn't be parsed into labeled
    examples — malformed JSON, a missing required key, or a screenshot path
    that doesn't resolve to a real file. Raised eagerly, at load time,
    rather than surfacing as a confusing failure mid-judging."""


def load_labeled_dataset(manifest_path: Path) -> list[LabeledExample]:
    """Reads a calibration dataset: a JSON file holding a list of
    `{"name", "screenshot", "intent", "human_label"}` objects. `screenshot`
    is a path relative to the manifest's own directory, the same
    "co-located, portable" convention `examples/starter_suite`'s `task.yml`
    `repo` field already uses. See `examples/calibration_dataset/
    manifest.json` for a worked, checked-in example.
    """
    try:
        raw = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetLoadError(f"couldn't read/parse {manifest_path}: {exc}") from exc

    if not isinstance(raw, list):
        raise DatasetLoadError(f"{manifest_path} must contain a JSON array of examples")

    examples = []
    base_dir = manifest_path.resolve().parent
    for i, entry in enumerate(raw):
        try:
            name = entry["name"]
            screenshot = entry["screenshot"]
            intent = entry["intent"]
            human_label = entry["human_label"]
        except (KeyError, TypeError) as exc:
            raise DatasetLoadError(
                f"{manifest_path} entry {i} is missing a required key "
                f"(name/screenshot/intent/human_label): {exc}"
            ) from exc
        if not isinstance(human_label, bool):
            raise DatasetLoadError(f"{manifest_path} entry {i} ({name!r}): human_label must be true/false")

        screenshot_path = (base_dir / screenshot).resolve()
        if not screenshot_path.is_file():
            raise DatasetLoadError(f"{manifest_path} entry {i} ({name!r}): no such file {screenshot_path}")

        examples.append(
            LabeledExample(name=name, screenshot_path=screenshot_path, intent=intent, human_label=human_label)
        )

    if not examples:
        raise DatasetLoadError(f"{manifest_path} contains zero examples")

    return examples


class Disagreement(BaseModel):
    """One case where the judge and the human reviewer landed on different
    verdicts — kept individually, not just tallied, so a calibration report
    points at exactly which screenshots to look at rather than only a
    percentage."""

    name: str
    intent: str
    human_label: bool
    judge_label: bool
    rationale: str


class CalibrationResult(BaseModel):
    """Concordance of `judge_name` against a labeled dataset, plus every
    individual disagreement. `concordance`/`meets_threshold` are computed,
    not stored — same discipline as `Verdict.status`: an unknown number
    (zero examples) is reported as unknown, never as a fabricated 0% or
    100%.
    """

    judge_name: str
    total: int
    agreements: int
    threshold: float = DEFAULT_CONCORDANCE_THRESHOLD
    disagreements: list[Disagreement] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def concordance(self) -> float | None:
        if self.total == 0:
            return None
        return self.agreements / self.total

    @computed_field  # type: ignore[prop-decorator]
    @property
    def meets_threshold(self) -> bool:
        concordance = self.concordance
        return concordance is not None and concordance >= self.threshold


def run_calibration(
    judge: VisionJudge,
    examples: list[LabeledExample],
    threshold: float = DEFAULT_CONCORDANCE_THRESHOLD,
) -> CalibrationResult:
    """Runs `judge` over every example and compares its verdict to the
    human label. Reads each screenshot's bytes from disk right before
    judging it (not preloaded in `LabeledExample`) so a large dataset
    doesn't hold every image in memory at once — the judge only ever needs
    one at a time.
    """
    agreements = 0
    disagreements: list[Disagreement] = []
    for example in examples:
        screenshot_bytes = example.screenshot_path.read_bytes()
        judgment = judge.judge(screenshot_bytes, example.intent)
        if judgment.passed == example.human_label:
            agreements += 1
        else:
            disagreements.append(
                Disagreement(
                    name=example.name,
                    intent=example.intent,
                    human_label=example.human_label,
                    judge_label=judgment.passed,
                    rationale=judgment.rationale,
                )
            )

    return CalibrationResult(
        judge_name=judge.name,
        total=len(examples),
        agreements=agreements,
        threshold=threshold,
        disagreements=disagreements,
    )


def render_calibration(result: CalibrationResult, console: Console | None = None) -> None:
    console = console or Console()

    concordance = result.concordance
    concordance_str = f"{concordance:.1%}" if concordance is not None else "unknown (0 examples)"
    color = "green" if result.meets_threshold else "red"

    table = Table(title=f"Judge calibration — {result.judge_name}")
    table.add_column("examples", justify="right")
    table.add_column("agreements", justify="right")
    table.add_column("concordance", justify="right")
    table.add_column("threshold", justify="right")
    table.add_row(
        str(result.total),
        str(result.agreements),
        f"[{color}]{concordance_str}[/{color}]",
        f"{result.threshold:.0%}",
    )
    console.print(table)

    if not result.meets_threshold:
        console.print(
            f"[bold red]WARNING[/bold red]: {result.judge_name}'s concordance "
            f"({concordance_str}) is below the {result.threshold:.0%} target — "
            "its JUDGED signals are advisory-only by design (see DESIGN.md's Phase 0 "
            "PROVEN/JUDGED split), but a judge disagreeing with humans this often is "
            "not worth reading as a meaningful opinion."
        )

    if result.disagreements:
        dis_table = Table(title="Disagreements")
        dis_table.add_column("name")
        dis_table.add_column("intent")
        dis_table.add_column("human")
        dis_table.add_column("judge")
        for d in result.disagreements:
            dis_table.add_row(d.name, d.intent, str(d.human_label), str(d.judge_label))
        console.print(dis_table)
