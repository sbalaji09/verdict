"""Ground-truth evaluation: how do we know Verdict's own DONE/NOT_DONE/
UNVERIFIED judgment is actually right?

Phase 7's `calibration.py` answered this question for the one JUDGED
signal in this codebase (a vision model's opinion) — run it over a
human-labeled sample, report concordance, name every disagreement. This
module asks the same question one level up, about `Verdict.status`
itself: the thing every other phase treats as ground truth (`ConfigResult.
pass_rate`, the merge gate's exit code, the whole PROVEN/JUDGED discipline)
has never itself been checked against an independent human judgment. A
tool that only ever grades itself has no way to notice it's been wrong in
the same direction the whole time.

The mechanism deliberately mirrors Phase 7's, reusing what generalizes
(`DatasetLoadError` — a malformed manifest is the same kind of problem
whether the examples are screenshots or task attempts; the "store raw
counts + every individual disagreement, compute derived stats as `None`
rather than a fabricated number when there's nothing to divide by"
discipline; a CLI command that warns rather than fails a build) and
introduces only what's genuinely different: three classes instead of two,
so "agrees or not" becomes a confusion matrix and per-class precision/
recall/F1 rather than a single concordance percentage.

**What counts as a "ground-truth example" here.** Not a screenshot — a
`(repo, task, patch, human_label)` tuple: a real git repo, a task
description, a fixed set of file edits standing in for "the agent's
attempt" (applied via `MockAdapter`, the same fixed-patch mechanism Phase
0 built so the pipeline could be exercised without spending real API
tokens), and a human's own DONE/NOT_DONE/UNVERIFIED judgment of that
attempt. `run_ground_truth` replays each one through the real `runner.run()`
— actual worktree isolation, actual gates, actual attribution — and
compares `Verdict.status` to the human label. This is deliberately the
same "real pipeline end to end, not mocked" standard Phase 2's bisection
fixtures hold themselves to: a synthetic in-memory comparison could
exercise the comparison arithmetic, but only running the genuine `run()`
call proves the *status this tool actually reports* is what's being
measured, not a stand-in for it.

See `examples/ground_truth_dataset/` for a small, honest dataset: three
cases where Verdict agrees with a human reviewer, and two genuine,
individually-explained disagreements — see that directory's own
`manifest.json` and DESIGN.md's Phase 20 section for why each disagreement
happens and what it reveals about what Verdict can and can't see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, computed_field
from rich.console import Console
from rich.table import Table

from verdict.adapters.mock import MockAdapter
from verdict.calibration import DatasetLoadError
from verdict.runner import run
from verdict.sandbox import SandboxConfig
from verdict.schema import GateStatus, Provenance, Verdict, VerdictStatus

DEFAULT_ACCURACY_THRESHOLD = 0.8

_LABELS: tuple[VerdictStatus, ...] = (VerdictStatus.DONE, VerdictStatus.NOT_DONE, VerdictStatus.UNVERIFIED)
"""The three statuses a human is ever asked to label. `VerdictStatus.ERROR`
is deliberately excluded — it means infra couldn't evaluate the attempt at
all, which isn't a judgment about the work a human could agree or disagree
with any more than they could label a screenshot the vision judge never
managed to load. A run that comes back ERROR is tallied separately (see
`GroundTruthResult.errored`), never forced into the confusion matrix.
"""


@dataclass
class GroundTruthExample:
    """One human-reviewed case: a real repo, a task, a fixed patch standing
    in for the agent's attempt, and the human's own verdict on it.
    `repo_path`/`patch` are read/held as-is (not pre-applied) so a large
    dataset manifest stays cheap to parse before any grading happens — the
    same "resolved lazily" shape `LabeledExample.screenshot_path` uses in
    `calibration.py`.
    """

    name: str
    repo_path: Path
    task: str
    patch: dict[str, str]
    human_label: VerdictStatus


def load_ground_truth_dataset(manifest_path: Path) -> list[GroundTruthExample]:
    """Reads a ground-truth dataset: a JSON file holding a list of
    `{"name", "repo", "task", "patch", "human_label"}` objects. `repo` is a
    path relative to the manifest's own directory, to a real git repo (see
    `examples/ground_truth_dataset/setup.sh` for how the checked-in demo
    bootstraps its own repos) — the same "co-located, portable" convention
    `calibration.py`'s `screenshot` field and `suite/loader.py`'s `repo`
    field both already use. `patch` is `{relative_path: new_contents}`,
    fed straight to `MockAdapter`. Raises `DatasetLoadError` (reused
    directly from `calibration.py` — a malformed manifest is the same
    class of problem regardless of what the examples contain) eagerly, at
    load time, rather than surfacing as a confusing failure mid-grading.
    """
    try:
        raw = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetLoadError(f"couldn't read/parse {manifest_path}: {exc}") from exc

    if not isinstance(raw, list):
        raise DatasetLoadError(f"{manifest_path} must contain a JSON array of examples")

    examples: list[GroundTruthExample] = []
    base_dir = manifest_path.resolve().parent
    for i, entry in enumerate(raw):
        try:
            name = entry["name"]
            repo = entry["repo"]
            task = entry["task"]
            patch = entry["patch"]
            human_label_raw = entry["human_label"]
        except (KeyError, TypeError) as exc:
            raise DatasetLoadError(
                f"{manifest_path} entry {i} is missing a required key "
                f"(name/repo/task/patch/human_label): {exc}"
            ) from exc

        if not isinstance(patch, dict) or not patch:
            raise DatasetLoadError(
                f"{manifest_path} entry {i} ({name!r}): patch must be a non-empty object of "
                "{relative_path: contents}"
            )
        try:
            human_label = VerdictStatus(human_label_raw)
        except ValueError as exc:
            raise DatasetLoadError(
                f"{manifest_path} entry {i} ({name!r}): human_label must be one of "
                f"{[s.value for s in _LABELS]}, got {human_label_raw!r}"
            ) from exc
        if human_label not in _LABELS:
            raise DatasetLoadError(
                f"{manifest_path} entry {i} ({name!r}): human_label {human_label.value!r} isn't "
                "something a human judges — a repo/task attempt is never labeled 'error' "
                "(that's an infra outcome, not a completion judgment)"
            )

        repo_path = (base_dir / repo).resolve()
        if not (repo_path / ".git").is_dir():
            raise DatasetLoadError(
                f"{manifest_path} entry {i} ({name!r}): {repo_path} is not a git repo "
                "(run its dataset's setup.sh first)"
            )

        examples.append(
            GroundTruthExample(
                name=name, repo_path=repo_path, task=str(task), patch=dict(patch), human_label=human_label
            )
        )

    if not examples:
        raise DatasetLoadError(f"{manifest_path} contains zero examples")

    return examples


class GroundTruthDisagreement(BaseModel):
    """One case where Verdict's own status and the human's label landed on
    different values — kept individually, not just tallied, so a report
    points at exactly which case to go inspect rather than only a number.
    Mirrors `calibration.Disagreement` field-for-field where the concepts
    line up (`rationale` there ~ `detail` here); `detail` is drawn straight
    from the real `Verdict` produced (its failing signals' own text, or its
    `error`) — never a new, separately-authored claim about why they
    disagree.
    """

    name: str
    task: str
    repo: str
    human_label: VerdictStatus
    verdict_status: VerdictStatus
    detail: str


class ClassMetrics(BaseModel):
    """Precision/recall/F1 for one label, derived from the confusion
    matrix's counts for that label — `None`, never a fabricated `0.0`,
    when the denominator itself is zero (e.g. a label neither the humans
    nor Verdict ever produced in this dataset), the same "unknown stays
    unknown" rule `CalibrationResult.concordance` already established.
    """

    label: VerdictStatus
    true_positives: int
    false_positives: int
    false_negatives: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision(self) -> float | None:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall(self) -> float | None:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or (precision + recall) == 0:
            return None
        return 2 * precision * recall / (precision + recall)


class GroundTruthResult(BaseModel):
    """The full report: a confusion matrix over real counts, per-class
    precision/recall/F1 derived from it, and every individual disagreement
    — same "store the raw counts, compute the derived stats, keep every
    disagreement" shape as `CalibrationResult`, extended from one binary
    concordance number to a 3x3 confusion matrix because there are three
    labels here, not two.
    """

    total: int
    confusion: dict[str, dict[str, int]]
    """`confusion[human_label.value][verdict_status.value]` — every one of
    the 3x3 cells is present (0 where no example landed there), never a
    sparse dict a reader has to guess the missing zeros for.
    """
    errored: int = 0
    """How many examples' real `run()` came back `VerdictStatus.ERROR` —
    infra couldn't evaluate the attempt, so there's no completion status to
    compare the human label against. Excluded from the confusion matrix and
    every metric below, same reasoning `CalibrationResult.unavailable`
    already uses for a judge call that produced no opinion at all: a run
    that never got graded shouldn't silently count as either an agreement
    or a disagreement in either direction.
    """
    errored_examples: list[str] = Field(default_factory=list)
    threshold: float = DEFAULT_ACCURACY_THRESHOLD
    disagreements: list[GroundTruthDisagreement] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def graded(self) -> int:
        return self.total - self.errored

    @computed_field  # type: ignore[prop-decorator]
    @property
    def correct(self) -> int:
        return sum(self.confusion[label.value][label.value] for label in _LABELS)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def accuracy(self) -> float | None:
        if self.graded == 0:
            return None
        return self.correct / self.graded

    @computed_field  # type: ignore[prop-decorator]
    @property
    def meets_threshold(self) -> bool:
        accuracy = self.accuracy
        return accuracy is not None and accuracy >= self.threshold

    @computed_field  # type: ignore[prop-decorator]
    @property
    def per_class(self) -> list[ClassMetrics]:
        metrics = []
        for label in _LABELS:
            tp = self.confusion[label.value][label.value]
            fp = sum(
                self.confusion[other.value][label.value] for other in _LABELS if other is not label
            )
            fn = sum(
                self.confusion[label.value][other.value] for other in _LABELS if other is not label
            )
            metrics.append(
                ClassMetrics(label=label, true_positives=tp, false_positives=fp, false_negatives=fn)
            )
        return metrics

    @computed_field  # type: ignore[prop-decorator]
    @property
    def macro_f1(self) -> float | None:
        scores = [m.f1 for m in self.per_class if m.f1 is not None]
        return sum(scores) / len(scores) if scores else None


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {human.value: {verdict.value: 0 for verdict in _LABELS} for human in _LABELS}


def run_ground_truth(
    examples: list[GroundTruthExample],
    sandbox_config: SandboxConfig | None = None,
    threshold: float = DEFAULT_ACCURACY_THRESHOLD,
) -> GroundTruthResult:
    """Runs the real `runner.run()` — actual worktree isolation, actual
    gates, actual attribution, nothing mocked below the adapter — over
    every example and compares `Verdict.status` to the human label.

    `MockAdapter(patches=example.patch)` stands in for "the agent already
    made this attempt," the same fixed-patch mechanism `git_repo`-based
    tests across this codebase already use — ground truth doesn't need a
    live agent call any more than any other gate/attribution test does; it
    needs a *known*, reproducible attempt to grade, which a live call
    can't guarantee twice.

    An example whose `Verdict.status` comes back `ERROR` (the sandbox
    itself never came up, not a judgment about the work) is tallied under
    `errored`/`errored_examples`, never forced into the confusion matrix —
    see `_LABELS`' own docstring for why.
    """
    confusion = _empty_confusion()
    errored_examples: list[str] = []
    disagreements: list[GroundTruthDisagreement] = []

    for example in examples:
        adapter = MockAdapter(patches=example.patch)
        verdict = run(
            task=example.task, repo=example.repo_path, adapter=adapter, sandbox_config=sandbox_config
        )

        if verdict.status is VerdictStatus.ERROR:
            errored_examples.append(example.name)
            continue

        confusion[example.human_label.value][verdict.status.value] += 1

        if verdict.status is not example.human_label:
            disagreements.append(
                GroundTruthDisagreement(
                    name=example.name,
                    task=example.task,
                    repo=str(example.repo_path),
                    human_label=example.human_label,
                    verdict_status=verdict.status,
                    detail=_disagreement_detail(verdict),
                )
            )

    return GroundTruthResult(
        total=len(examples),
        confusion=confusion,
        errored=len(errored_examples),
        errored_examples=errored_examples,
        threshold=threshold,
        disagreements=disagreements,
    )


def _disagreement_detail(verdict: Verdict) -> str:
    """A short, evidence-pointing summary of *why* Verdict landed where it
    did — never a new claim, just a rendering of the same `Signal`s the
    report/CLI already show, the same "point at the evidence" instinct
    behind `Attribution.explanation` and the calibration disagreement
    table.

    Three cases, in order: a failing PROVEN signal (the thing that decided
    NOT_DONE, if there is one) always wins — that's the concrete evidence
    a NOT_DONE-but-human-said-done disagreement turns on. Otherwise, for a
    DONE verdict the human disagreed with (e.g. a fix that games the exact
    test rather than fixing the general case — Verdict can only see the
    executed PASS, never that it doesn't generalize), the passing PROVEN
    signals ARE the evidence, so those are shown instead of nothing.
    `verdict.error`, then a last-resort "nothing executed" cover the
    remaining UNVERIFIED-shaped case.
    """
    proven = [
        s for s in verdict.signals if s.provenance is Provenance.PROVEN and s.status is not GateStatus.NA
    ]
    for signal in proven:
        if signal.status is GateStatus.FAIL:
            return f"{signal.name}: {signal.detail}"
    if proven:
        return "; ".join(f"{signal.name}: {signal.detail}" for signal in proven)
    if verdict.error:
        return verdict.error
    return "no PROVEN gate produced a signal (nothing executed to ground a status in)"


def render_ground_truth(result: GroundTruthResult, console: Console | None = None) -> None:
    console = console or Console()

    accuracy = result.accuracy
    accuracy_str = f"{accuracy:.1%}" if accuracy is not None else "unknown (0 graded examples)"
    color = "green" if result.meets_threshold else "red"

    summary = Table(title="Ground-truth accuracy")
    summary.add_column("examples", justify="right")
    summary.add_column("graded", justify="right")
    summary.add_column("correct", justify="right")
    summary.add_column("errored", justify="right")
    summary.add_column("accuracy", justify="right")
    summary.add_column("macro F1", justify="right")
    summary.add_column("threshold", justify="right")
    macro_f1_str = f"{result.macro_f1:.2f}" if result.macro_f1 is not None else "—"
    summary.add_row(
        str(result.total),
        str(result.graded),
        str(result.correct),
        str(result.errored),
        f"[{color}]{accuracy_str}[/{color}]",
        macro_f1_str,
        f"{result.threshold:.0%}",
    )
    console.print(summary)

    matrix = Table(title="Confusion matrix (rows: human label, columns: Verdict's status)")
    matrix.add_column("human \\ verdict")
    for label in _LABELS:
        matrix.add_column(label.value, justify="right")
    for human in _LABELS:
        matrix.add_row(human.value, *(str(result.confusion[human.value][v.value]) for v in _LABELS))
    console.print(matrix)

    per_class = Table(title="Per-class precision / recall / F1")
    per_class.add_column("label")
    per_class.add_column("precision", justify="right")
    per_class.add_column("recall", justify="right")
    per_class.add_column("f1", justify="right")
    for metrics in result.per_class:
        per_class.add_row(
            metrics.label.value,
            f"{metrics.precision:.2f}" if metrics.precision is not None else "—",
            f"{metrics.recall:.2f}" if metrics.recall is not None else "—",
            f"{metrics.f1:.2f}" if metrics.f1 is not None else "—",
        )
    console.print(per_class)

    if result.errored_examples:
        console.print(
            f"[yellow]{result.errored} example(s) errored[/yellow] (infra couldn't evaluate the "
            f"attempt, excluded from the confusion matrix): {', '.join(result.errored_examples)}"
        )

    if not result.meets_threshold:
        console.print(
            f"[bold red]WARNING[/bold red]: accuracy ({accuracy_str}) is below the "
            f"{result.threshold:.0%} target — see the disagreements below for exactly which "
            "cases Verdict and the human reviewer disagreed on."
        )

    if result.disagreements:
        dis_table = Table(title="Disagreements")
        dis_table.add_column("name")
        dis_table.add_column("human")
        dis_table.add_column("verdict")
        dis_table.add_column("detail")
        for d in result.disagreements:
            dis_table.add_row(d.name, d.human_label.value, d.verdict_status.value, d.detail)
        console.print(dis_table)
