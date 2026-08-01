"""Deterministic glitch detection over a burst of real screenshots — not a
model's guess, a frame-by-frame comparison using the exact same perceptual
diff `visual_diff.py`'s before/after check already uses. Every input frame
is a screenshot Playwright actually took; every output finding is computed
from `perceptual_diff_ratio` calls, nothing else — so this stays in the
PROVEN bucket alongside the rest of Phase 4's executed checks.

Two distinct glitch shapes, because they need different comparisons to
catch:

- **Flicker.** A frame that differs sharply from *both* its neighbors,
  while those neighbors closely resemble *each other* — content that
  appeared and vanished again within roughly one capture interval (a flash
  of unstyled content, a modal that flashes open and immediately shut, a
  layout that jumps and jumps back). A plain before/after diff across the
  whole burst would miss this entirely: the first and last frame can look
  identical while something clearly glitched in between.
- **Never settled.** The last two captured frames still differ meaningfully
  from each other — the page was still visibly changing at the end of the
  capture window, when a well-behaved render or interaction is expected to
  have reached a stable final state by then (a spinner that never
  resolves, an animation stuck mid-loop, layout thrash that hasn't
  stopped).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verdict.frontend.visual_diff import perceptual_diff_ratio

FLICKER_NEIGHBOR_RATIO = 0.3
"""A candidate flicker frame's diff to each neighbor must both exceed
`diff_threshold`, while the neighbors' diff to *each other* (skipping the
candidate frame) must be no more than this fraction of the smaller of the
two — i.e. the neighbors mostly agree with each other, and the middle
frame is the outlier, not part of a real, sustained change."""


@dataclass
class GlitchFinding:
    kind: str
    """"flicker" or "unsettled"."""
    frame_index: int
    detail: str


@dataclass
class GlitchScanResult:
    frame_count: int
    findings: list[GlitchFinding] = field(default_factory=list)

    @property
    def has_glitch(self) -> bool:
        return bool(self.findings)


def scan_for_glitches(frames: list[bytes], diff_threshold: float) -> GlitchScanResult:
    """`diff_threshold` is the same kind of value as
    `frontend.screenshot_threshold` — a fraction of the normalized frame
    that must change (beyond `visual_diff.PIXEL_TOLERANCE`'s anti-aliasing
    tolerance) to count as a real difference, not render noise."""
    if len(frames) < 3:
        # Nothing to triangulate a flicker from, and "did it settle" needs
        # at least two frames to compare — an honest no-finding, not a
        # false negative dressed up as a clean scan.
        return GlitchScanResult(frame_count=len(frames))

    findings: list[GlitchFinding] = []
    for i in range(1, len(frames) - 1):
        left = perceptual_diff_ratio(frames[i - 1], frames[i])
        right = perceptual_diff_ratio(frames[i], frames[i + 1])
        skip = perceptual_diff_ratio(frames[i - 1], frames[i + 1])
        is_flicker = (
            left > diff_threshold
            and right > diff_threshold
            and skip <= min(left, right) * FLICKER_NEIGHBOR_RATIO
        )
        if is_flicker:
            findings.append(
                GlitchFinding(
                    kind="flicker",
                    frame_index=i,
                    detail=(
                        f"frame {i} differs from its neighbors ({left:.2%}, {right:.2%}) "
                        f"but its neighbors barely differ from each other ({skip:.2%}) — "
                        "something appeared and reverted within one capture interval"
                    ),
                )
            )

    last_diff = perceptual_diff_ratio(frames[-2], frames[-1])
    if last_diff > diff_threshold:
        findings.append(
            GlitchFinding(
                kind="unsettled",
                frame_index=len(frames) - 1,
                detail=(
                    f"the last two captured frames still differ by {last_diff:.2%} "
                    f"(threshold {diff_threshold:.2%}) — the page had not visibly "
                    "stabilized by the end of the capture window"
                ),
            )
        )

    return GlitchScanResult(frame_count=len(frames), findings=findings)
