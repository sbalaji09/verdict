"""The verdict schema: every signal is tagged PROVEN or JUDGED, never blended.

Phase 0 emitted a single PROVEN signal with a binary passed/failed outcome.
Phase 1 adds three of its own signals (typecheck/build/lint) and most repos
only use a subset of the four available stacks — a Python-only repo has no
tsconfig.json, a script-only repo has no build step — so a binary outcome
can't represent "this gate doesn't apply here" without either lying (calling
it a pass) or being unfairly strict (calling it a failure). GateStatus adds
the third state that was missing. The Judged enum member and the fields
that support it (e.g. Signal.detail carrying a model's rationale) exist now
so Phase 4's vision-intent judge slots in without a schema migration later.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class Provenance(str, Enum):
    """How a signal's pass/fail was decided."""

    PROVEN = "proven"
    """Executed. Deterministic. Reproducible. E.g. a test suite exit code."""

    JUDGED = "judged"
    """An LLM/vision model formed an opinion. Advisory, never load-bearing."""


class GateStatus(str, Enum):
    """The outcome of one signal."""

    PASS = "pass"
    FAIL = "fail"
    NA = "n/a"
    """The gate's stack wasn't detected in this repo (e.g. no tsconfig.json
    for typecheck). Not a failure — a repo property, not an agent defect —
    but also not a pass: it contributes nothing to done/not-done either way.
    """


class Confidence(str, Enum):
    """How much a DONE/NOT_DONE verdict is worth trusting."""

    HIGH = "high"
    LOW = "low"
    """Downgraded when the test gate itself is N/A. Typecheck/build/lint
    catch real defects, but only the test suite exercises *behavior* — a
    repo with no detectable tests can still be reported DONE (every gate
    that did run passed), but that DONE is worth less.
    """


class VerdictStatus(str, Enum):
    """The three-way outcome of a whole Verdict."""

    DONE = "done"
    NOT_DONE = "not_done"
    UNVERIFIED = "unverified"
    """Zero PROVEN gates actually ran (all were N/A, or no signals at all).
    Distinct from DONE: there's nothing executed to ground a claim of
    correctness in, so Verdict refuses to report success by default.
    """


class Signal(BaseModel):
    """One executed check (or, later, one judged opinion) and its outcome."""

    name: str
    provenance: Provenance
    status: GateStatus
    detail: str
    command: str | None = None
    exit_code: int | None = None


class AttemptResult(BaseModel):
    """What an agent did during one attempt: its diff and what it cost."""

    diff: str
    files_changed: list[str] = Field(default_factory=list)
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float | None = None
    raw_output: str | None = None


class Verdict(BaseModel):
    """The end-to-end result of grading one agent attempt at one task."""

    task: str
    agent: str
    repo: str
    attempt: AttemptResult
    signals: list[Signal]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def _proven_applicable(self) -> list[Signal]:
        return [
            s
            for s in self.signals
            if s.provenance is Provenance.PROVEN and s.status is not GateStatus.NA
        ]

    @property
    def status(self) -> VerdictStatus:
        """DONE only if every *applicable* PROVEN gate passed. JUDGED
        signals never affect this — an opinion can flag concern but can't
        override or stand in for executed fact. N/A gates are excluded
        from the check entirely, not treated as passes. If nothing
        executable actually ran, the verdict is UNVERIFIED rather than a
        default DONE — "nothing failed" is not the same claim as "this
        works", and Verdict never lets the former masquerade as the latter.
        """
        applicable = self._proven_applicable()
        if not applicable:
            return VerdictStatus.UNVERIFIED
        if any(s.status is GateStatus.FAIL for s in applicable):
            return VerdictStatus.NOT_DONE
        return VerdictStatus.DONE

    @property
    def done(self) -> bool:
        """Convenience boolean for callers that just need pass/fail (e.g.
        the CLI's exit code). UNVERIFIED counts as not-done: a verdict that
        couldn't verify anything has no basis to report success.
        """
        return self.status is VerdictStatus.DONE

    @property
    def confidence(self) -> Confidence:
        """LOW whenever the test gate didn't actually run (missing, or
        N/A). Typecheck/build/lint gates don't affect this — they catch
        real defects, but only tests exercise behavior, so a DONE verdict
        that never ran a test suite is real but weaker evidence.
        """
        test_signal = next(
            (
                s
                for s in self.signals
                if s.name == "test" and s.provenance is Provenance.PROVEN
            ),
            None,
        )
        if test_signal is None or test_signal.status is GateStatus.NA:
            return Confidence.LOW
        return Confidence.HIGH
