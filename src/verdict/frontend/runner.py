"""Orchestrates Phase 4's frontend truth checks around one worktree: starts
the repo's own dev server twice — once at the worktree's pre-agent
`base_commit`, once at the agent's final state — and runs the checks in
DESIGN.md's Phase 4 order of trust: DOM assertion, interaction drive,
perceptual screenshot diff, and glitch scan (all PROVEN), then the
vision-intent judge (JUDGED, always last, never load-bearing).

Capturing "before" by actually rendering `base_commit` (rather than
diffing the agent's textual diff, or skipping straight to "after") is what
makes the visual-diff check meaningful — it's a real render of the real
pre-agent page, in the same browser, at the same viewports, not a guess.

The glitch scan captures a short burst of real screenshots around page
load and around each interaction, with Playwright's own video recording
running alongside — so a detected glitch (flicker, a page that never
settles) comes with an actual `.webm` recording a human can watch, not just
a frame count. See `glitch.py` for the detection logic and DESIGN.md's
Phase 4 section for why both exist.

Any failure in getting a browser up (missing `playwright` package) or a
dev server up (bad `frontend.start` command, a broken app) is reported as
a `frontend:setup` PROVEN FAIL signal rather than raised — a single
frontend hiccup must never crash the whole `run()` and lose the gate/
attribution results already computed.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from verdict.config import FrontendCheckSpec, FrontendConfig, InteractionSpec, VerdictConfig
from verdict.frontend.capture import capture_action_burst, capture_settle_burst
from verdict.frontend.checks import dom_check, interaction_check
from verdict.frontend.glitch import GlitchScanResult, scan_for_glitches
from verdict.frontend.server import FrontendServerError, dev_server
from verdict.frontend.vision_judge import MockVisionJudge, VisionJudge, to_signal
from verdict.frontend.visual_diff import perceptual_diff_ratio
from verdict.schema import GateStatus, Provenance, Signal
from verdict.worktree import Worktree, copy_vendored_dependencies, scratch_worktree

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page

try:
    from playwright.sync_api import sync_playwright

    _PLAYWRIGHT_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    sync_playwright = None  # type: ignore[assignment]
    _PLAYWRIGHT_IMPORT_ERROR = exc

_NETWORK_IDLE_GRACE_MS = 2_000


def _setup_fail(detail: str) -> Signal:
    return Signal(name="frontend:setup", provenance=Provenance.PROVEN, status=GateStatus.FAIL, detail=detail)


def run_frontend_checks(
    repo: Path,
    worktree: Worktree,
    config: VerdictConfig,
    task: str,
    vision_judge: VisionJudge | None = None,
) -> list[Signal]:
    """Empty list if `verdict.yml` has no `frontend:` section — frontend
    checks are entirely opt-in, same as the gate overrides they sit
    alongside."""
    frontend = config.frontend
    if frontend is None:
        return []

    if sync_playwright is None:
        return [
            _setup_fail(
                "frontend checks are configured in verdict.yml but the `playwright` "
                f"package isn't installed ({_PLAYWRIGHT_IMPORT_ERROR}). Install the "
                "`frontend` extra (`pip install verdict-eval[frontend]` or "
                "`uv sync --extra frontend`) and run `playwright install chromium`."
            )
        ]

    judge = vision_judge or MockVisionJudge()

    # Recording is real disk cost (one .webm per page opened) — only pay it
    # when glitch_scan is on, and only *keep* the directory afterward if
    # something actually failed. A clean run deletes its recordings; a
    # failing one keeps the whole capture directory (every page opened
    # during this run, not just the one that failed) as supporting context,
    # rather than the added complexity of cherry-picking a single file.
    video_dir = Path(tempfile.mkdtemp(prefix="verdict-frontend-videos-")) if frontend.glitch_scan else None
    keep_video_dir = False
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                before_shots, before_error = _capture_before(repo, worktree, frontend, browser)
                after_shots, check_signals = _capture_after_and_run_checks(
                    worktree.path, frontend, judge, browser, video_dir
                )
            finally:
                browser.close()

        signals: list[Signal] = []
        if before_error is not None:
            signals.append(
                _setup_fail(
                    "could not render the pre-agent baseline to diff against — visual-diff "
                    f"check skipped: {before_error}"
                )
            )
        else:
            signals.extend(_visual_diff_signals(frontend, before_shots, after_shots))
        signals.extend(check_signals)

        keep_video_dir = any(
            s.status is GateStatus.FAIL for s in signals if s.provenance is Provenance.PROVEN
        )
        return signals
    finally:
        if video_dir is not None and not keep_video_dir:
            shutil.rmtree(video_dir, ignore_errors=True)


def _goto(page: Page, url: str) -> None:
    page.goto(url, wait_until="load")
    try:
        page.wait_for_load_state("networkidle", timeout=_NETWORK_IDLE_GRACE_MS)
    except Exception:  # noqa: BLE001 - best-effort settle time, not a correctness requirement
        pass


def _open_page(browser: Browser, width: int, height: int, video_dir: Path | None = None) -> Page:
    if video_dir is None:
        return browser.new_page(viewport={"width": width, "height": height})
    return browser.new_page(
        viewport={"width": width, "height": height},
        record_video_dir=str(video_dir),
        record_video_size={"width": width, "height": height},
    )


def _finalize_video(page: Page) -> str | None:
    """Must be called after `page.close()` — Playwright only guarantees the
    recording is flushed to disk once the page (and its context) is closed.
    Returns None for a page opened without `record_video_dir`."""
    video = getattr(page, "video", None)
    if video is None:
        return None
    try:
        return str(video.path())
    except Exception:  # noqa: BLE001 - a missing/unflushed recording is not fatal to the run
        return None


def _glitch_signal(check_name: str, result: GlitchScanResult, video_path: str | None) -> Signal:
    name = f"frontend:glitch_scan:{check_name}"
    if result.has_glitch:
        findings = "; ".join(f"{f.kind} (frame {f.frame_index}): {f.detail}" for f in result.findings)
        detail = f"{len(result.findings)} glitch(es) across {result.frame_count} captured frames: {findings}"
        return Signal(
            name=name,
            provenance=Provenance.PROVEN,
            status=GateStatus.FAIL,
            detail=detail,
            artifact_path=video_path,
        )
    # No artifact_path on a clean scan: the recording only survives past
    # this run if *something* in the run failed (see run_frontend_checks),
    # so a Signal pointing at a since-deleted file would be misleading.
    return Signal(
        name=name,
        provenance=Provenance.PROVEN,
        status=GateStatus.PASS,
        detail=f"no glitches detected across {result.frame_count} captured frames",
    )


def _screenshot(browser: Browser, url: str, viewport_width: int, viewport_height: int) -> bytes:
    page = _open_page(browser, viewport_width, viewport_height)
    try:
        _goto(page, url)
        return page.screenshot(full_page=False)
    finally:
        page.close()


def _load_glitch_scan(
    browser: Browser, frontend: FrontendConfig, video_dir: Path
) -> tuple[bytes, Signal]:
    """Capture a burst of frames right after navigation, at the primary
    viewport, to catch load-time flicker — a flash of unstyled content, a
    skeleton that never resolves, content that appears then vanishes."""
    width, height = frontend.viewports[0], frontend.viewport_height
    page = _open_page(browser, width, height, video_dir)
    try:
        page.goto(frontend.url, wait_until="load")
        frames = capture_settle_burst(
            page, frontend.glitch_capture_seconds, frontend.glitch_frame_interval_seconds
        )
        try:
            page.wait_for_load_state("networkidle", timeout=_NETWORK_IDLE_GRACE_MS)
        except Exception:  # noqa: BLE001
            pass
    finally:
        page.close()
    video_path = _finalize_video(page)
    result = scan_for_glitches(frames, frontend.glitch_diff_threshold)
    return frames[-1], _glitch_signal("load", result, video_path)


def _capture_before(
    repo: Path, worktree: Worktree, frontend: FrontendConfig, browser: Browser
) -> tuple[dict[int, bytes], str | None]:
    """Render `worktree.base_commit` — the repo exactly as it was before the
    agent touched anything — in a disposable scratch worktree, so the
    visual diff compares two real renders rather than a render against a
    guess. No video/glitch scanning here — that's about the agent's shipped
    result, not the pre-agent baseline."""
    try:
        with scratch_worktree(repo, worktree.base_commit) as base_path:
            copy_vendored_dependencies(repo, base_path)
            with dev_server(frontend.start, base_path, frontend.url, frontend.ready_timeout_seconds):
                shots = {
                    vw: _screenshot(browser, frontend.url, vw, frontend.viewport_height)
                    for vw in frontend.viewports
                }
        return shots, None
    except FrontendServerError as exc:
        return {}, str(exc)


def _capture_after_and_run_checks(
    worktree_path: Path,
    frontend: FrontendConfig,
    judge: VisionJudge,
    browser: Browser,
    video_dir: Path | None,
) -> tuple[dict[int, bytes], list[Signal]]:
    try:
        with dev_server(frontend.start, worktree_path, frontend.url, frontend.ready_timeout_seconds):
            after_shots: dict[int, bytes] = {}
            glitch_signals: list[Signal] = []
            primary_shot: bytes | None = None

            for i, viewport in enumerate(frontend.viewports):
                if i == 0 and frontend.glitch_scan and video_dir is not None:
                    shot, load_signal = _load_glitch_scan(browser, frontend, video_dir)
                    glitch_signals.append(load_signal)
                else:
                    shot = _screenshot(browser, frontend.url, viewport, frontend.viewport_height)
                after_shots[viewport] = shot
                if i == 0:
                    primary_shot = shot

            check_signals: list[Signal] = []
            for spec in frontend.checks:
                check_signals.extend(_run_one_check(browser, frontend, spec, judge, primary_shot, video_dir))

            return after_shots, glitch_signals + check_signals
    except FrontendServerError as exc:
        detail = f"could not start the frontend dev server to run checks against: {exc}"
        return {}, [_setup_fail(detail)]


def _run_interaction_with_glitch_scan(
    browser: Browser,
    frontend: FrontendConfig,
    interaction: InteractionSpec,
    check_name: str,
    video_dir: Path,
) -> list[Signal]:
    width, height = frontend.viewports[0], frontend.viewport_height
    page = _open_page(browser, width, height, video_dir)
    result_holder: list[Signal] = []
    try:
        _goto(page, frontend.url)

        def _do_interaction() -> None:
            result_holder.append(interaction_check(page, interaction, check_name))

        frames = capture_action_burst(
            page, _do_interaction, frontend.glitch_capture_seconds, frontend.glitch_frame_interval_seconds
        )
    finally:
        page.close()

    video_path = _finalize_video(page)
    glitch_result = scan_for_glitches(frames, frontend.glitch_diff_threshold)
    return [result_holder[0], _glitch_signal(check_name, glitch_result, video_path)]


def _run_interaction_plain(
    browser: Browser, frontend: FrontendConfig, interaction: InteractionSpec, check_name: str
) -> Signal:
    page = _open_page(browser, frontend.viewports[0], frontend.viewport_height)
    try:
        _goto(page, frontend.url)
        return interaction_check(page, interaction, check_name)
    finally:
        page.close()


def _run_one_check(
    browser: Browser,
    frontend: FrontendConfig,
    spec: FrontendCheckSpec,
    judge: VisionJudge,
    primary_after_shot: bytes | None,
    video_dir: Path | None,
) -> list[Signal]:
    signals: list[Signal] = []

    if spec.dom is not None:
        page = _open_page(browser, frontend.viewports[0], frontend.viewport_height)
        try:
            _goto(page, frontend.url)
            signals.append(dom_check(page, spec.dom, spec.name))
        finally:
            page.close()

    if spec.interaction is not None:
        if frontend.glitch_scan and video_dir is not None:
            signals.extend(
                _run_interaction_with_glitch_scan(browser, frontend, spec.interaction, spec.name, video_dir)
            )
        else:
            signals.append(_run_interaction_plain(browser, frontend, spec.interaction, spec.name))

    if spec.vision_intent is not None:
        if primary_after_shot is None:
            signals.append(
                Signal(
                    name=f"frontend:vision_intent:{spec.name}",
                    provenance=Provenance.JUDGED,
                    status=GateStatus.FAIL,
                    detail="no screenshot was available to judge (dev server never came up)",
                )
            )
        else:
            judgment = judge.judge(primary_after_shot, spec.vision_intent)
            signals.append(to_signal(spec.name, judgment, judge.name))

    return signals


def _visual_diff_signals(
    frontend: FrontendConfig, before_shots: dict[int, bytes], after_shots: dict[int, bytes]
) -> list[Signal]:
    signals: list[Signal] = []
    for vw in frontend.viewports:
        name = f"frontend:visual_diff:{vw}px"
        before = before_shots.get(vw)
        after = after_shots.get(vw)
        if before is None or after is None:
            signals.append(
                Signal(
                    name=name,
                    provenance=Provenance.PROVEN,
                    status=GateStatus.FAIL,
                    detail=f"missing a before and/or after screenshot at {vw}px to diff",
                )
            )
            continue

        ratio = perceptual_diff_ratio(before, after)
        status = GateStatus.PASS if ratio <= frontend.screenshot_threshold else GateStatus.FAIL
        signals.append(
            Signal(
                name=name,
                provenance=Provenance.PROVEN,
                status=status,
                detail=(
                    f"{ratio:.2%} of the {vw}px render changed beyond tolerance "
                    f"(threshold {frontend.screenshot_threshold:.2%})"
                ),
            )
        )
    return signals
