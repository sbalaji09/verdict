"""The JUDGED bucket: a vision model's opinion on whether a screenshot
matches a task's intended UI outcome. Explicitly advisory — `Verdict.status`
(schema.py) only ever consults PROVEN signals, so a glowing vision judgment
can never flip a failing DOM/interaction/visual-diff check to DONE, and a
harsh one can never sink a run where every PROVEN check passed.

No real vision-model provider ships here. Wiring one up means picking a
specific vendor's API, handling its auth, and testing it against real image
inputs — a genuine integration project of its own, not something to fake
with an untested stub pretending to be real. `VisionJudge` is the extension
point (mirrors `Adapter` in `adapters/__init__.py`: one Protocol, pluggable
implementations); `MockVisionJudge` is the only implementation that ships,
playing the same role `MockAdapter` plays on the agent side — a
deterministic, zero-cost, zero-network stand-in so the rest of the pipeline
is runnable and testable without an API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from verdict.schema import GateStatus, Provenance, Signal


@dataclass
class VisionJudgment:
    passed: bool
    rationale: str


class VisionJudge(Protocol):
    """Scores a screenshot against a task's intended UI outcome. Every
    implementation feeds a JUDGED Signal, never a PROVEN one — see
    `to_signal` below, the only place a `VisionJudgment` turns into a
    `Signal`."""

    name: str

    def judge(self, screenshot_png: bytes, intent: str) -> VisionJudgment: ...


class MockVisionJudge:
    """Passes unconditionally — it has no way to actually inspect the
    image, and says so in its own rationale rather than pretending
    otherwise. Exists so frontend checks are runnable and testable without
    a real vision-model API key.
    """

    name = "mock-vision-judge"

    def judge(self, screenshot_png: bytes, intent: str) -> VisionJudgment:
        return VisionJudgment(
            passed=True,
            rationale=(
                "no real vision model configured — MockVisionJudge cannot actually "
                "inspect the screenshot, so this is a placeholder opinion, not a "
                "real assessment (see DESIGN.md's Phase 4 section)."
            ),
        )


def to_signal(check_name: str, judgment: VisionJudgment, judge_name: str) -> Signal:
    return Signal(
        name=f"frontend:vision_intent:{check_name}",
        provenance=Provenance.JUDGED,
        status=GateStatus.PASS if judgment.passed else GateStatus.FAIL,
        detail=f"[{judge_name}] {judgment.rationale}",
    )
