"""Frame-burst capture: a short sequence of real screenshots taken close
together in time, used as raw material for `glitch.py`'s frame-by-frame
comparison. Deliberately not a decoded video file — Playwright's own
`record_video_dir` already gives us a real `.webm` for a human to watch
(wired up in `runner.py`); decoding *that* back into frames would mean
pulling in an image/video codec dependency (opencv, ffmpeg) for something
a loop of `page.screenshot()` calls already gives us directly, at the
resolution we actually need (a handful of samples, not 30fps).

**Timing is best-effort, not a real-time guarantee.** Each `page.screenshot()`
call itself takes real, variable time (tens of milliseconds), so the actual
gap between frames is `interval_seconds` *plus* however long the previous
screenshot took — never less than requested, sometimes more. This is fine
for what glitch detection needs (catching a state that appeared and
reverted within roughly one interval), and it's the same "polling, not a
promised deadline" honesty `frontend/server.py` already applies to
readiness checks.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol


class Screenshotter(Protocol):
    """The one Playwright `Page` method this module actually needs — kept
    narrow so tests can pass a fake without spinning up a real browser."""

    def screenshot(self, *, full_page: bool = False) -> bytes: ...


def capture_settle_burst(
    page: Screenshotter, duration_seconds: float, interval_seconds: float
) -> list[bytes]:
    """Capture frames for `duration_seconds`, spaced `interval_seconds`
    apart, starting immediately. Used right after a navigation to catch
    load-time flicker (flash of unstyled content, a skeleton that never
    resolves, content that appears then disappears)."""
    frames = [page.screenshot(full_page=False)]
    deadline = time.monotonic() + duration_seconds
    while time.monotonic() < deadline:
        time.sleep(interval_seconds)
        frames.append(page.screenshot(full_page=False))
    return frames


def capture_action_burst(
    page: Screenshotter,
    action: Callable[[], None],
    duration_seconds: float,
    interval_seconds: float,
) -> list[bytes]:
    """Capture one frame, perform `action` (e.g. a click), then keep
    capturing for `duration_seconds` afterward. Used around an interaction
    to catch a transient glitch the interaction itself causes (a modal that
    flashes open and shut, a layout that jumps and jumps back)."""
    frames = [page.screenshot(full_page=False)]
    action()
    deadline = time.monotonic() + duration_seconds
    while time.monotonic() < deadline:
        frames.append(page.screenshot(full_page=False))
        time.sleep(interval_seconds)
    return frames
