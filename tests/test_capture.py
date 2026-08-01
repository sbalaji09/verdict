from __future__ import annotations

import itertools

from verdict.frontend.capture import capture_action_burst, capture_settle_burst


class FakePage:
    """A `Screenshotter` double — no browser needed to test the capture
    loop's control flow (first-frame-immediately, action timing, minimum
    frame count)."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = itertools.cycle(frames)
        self.calls = 0

    def screenshot(self, *, full_page: bool = False) -> bytes:
        self.calls += 1
        return next(self._frames)


def test_settle_burst_captures_first_frame_immediately_with_no_wait() -> None:
    page = FakePage([b"a", b"b", b"c"])
    frames = capture_settle_burst(page, duration_seconds=0.05, interval_seconds=0.02)
    assert frames[0] == b"a"
    assert len(frames) >= 2


def test_action_burst_runs_action_immediately_after_the_first_frame() -> None:
    page = FakePage([b"1", b"2", b"3"])
    order: list[str] = []

    def action() -> None:
        order.append("action")

    frames = capture_action_burst(page, action, duration_seconds=0.05, interval_seconds=0.02)
    assert frames[0] == b"1"
    assert order == ["action"]
    assert len(frames) >= 2
