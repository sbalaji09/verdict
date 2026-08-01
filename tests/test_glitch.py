from __future__ import annotations

from io import BytesIO

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from verdict.frontend.glitch import scan_for_glitches  # noqa: E402

DIFF_THRESHOLD = 0.05


def _png(gray: int) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (100, 100), (gray, gray, gray)).save(buf, format="PNG")
    return buf.getvalue()


def test_stable_burst_has_no_findings() -> None:
    frame = _png(100)
    result = scan_for_glitches([frame, frame, frame, frame], DIFF_THRESHOLD)
    assert result.frame_count == 4
    assert not result.has_glitch


def test_transient_spike_is_flagged_as_flicker() -> None:
    # Frame 1 spikes sharply away from frames 0 and 2, which are identical
    # to each other — content that appeared and reverted within one
    # capture interval.
    frames = [_png(100), _png(250), _png(100), _png(100)]
    result = scan_for_glitches(frames, DIFF_THRESHOLD)
    assert result.has_glitch
    kinds = {f.kind for f in result.findings}
    assert "flicker" in kinds
    flicker = next(f for f in result.findings if f.kind == "flicker")
    assert flicker.frame_index == 1


def test_a_page_that_never_settles_is_flagged() -> None:
    # The final two frames still differ sharply — the page was still
    # changing at the end of the capture window.
    frames = [_png(100), _png(100), _png(200)]
    result = scan_for_glitches(frames, DIFF_THRESHOLD)
    assert result.has_glitch
    kinds = {f.kind for f in result.findings}
    assert "unsettled" in kinds


def test_a_real_sustained_change_is_not_flagged_as_flicker() -> None:
    # A gradual, sustained change across the whole burst (not a spike-and-
    # revert) should not be mistaken for a flicker — the "skip" comparison
    # (frame 0 vs frame 2) also differs meaningfully here, unlike a real
    # flicker where the endpoints resemble each other.
    frames = [_png(100), _png(150), _png(200)]
    result = scan_for_glitches(frames, DIFF_THRESHOLD)
    kinds = {f.kind for f in result.findings}
    assert "flicker" not in kinds


def test_too_few_frames_reports_no_findings_honestly() -> None:
    result = scan_for_glitches([_png(100), _png(200)], DIFF_THRESHOLD)
    assert result.frame_count == 2
    assert not result.has_glitch
