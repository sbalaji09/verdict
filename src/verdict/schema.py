"""The verdict schema: every signal is tagged PROVEN or JUDGED, never blended.

Phase 0 only emits PROVEN signals (one: the repo's test gate). The Judged
enum member and the fields that support it (e.g. Signal.detail carrying a
model's rationale) exist now so Phase 4's vision-intent judge slots in
without a schema migration later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Provenance(str, Enum):
    """How a signal's pass/fail was decided."""

    PROVEN = "proven"
    """Executed. Deterministic. Reproducible. E.g. a test suite exit code."""

    JUDGED = "judged"
    """An LLM/vision model formed an opinion. Advisory, never load-bearing."""


class Signal(BaseModel):
    """One executed check (or, later, one judged opinion) and its outcome."""

    name: str
    provenance: Provenance
    passed: bool
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def done(self) -> bool:
        """True only if every PROVEN signal passed.

        JUDGED signals never affect this: a judged opinion can flag concern
        but is never allowed to override or stand in for executed fact.
        A Verdict with zero proven signals is not done — there's nothing to
        ground the claim in.
        """
        proven = [s for s in self.signals if s.provenance is Provenance.PROVEN]
        if not proven:
            return False
        return all(s.passed for s in proven)
