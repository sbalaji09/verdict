"""Orchestrates Phase 4's frontend truth checks around one worktree: starts
the repo's own dev server twice — once at the worktree's pre-agent
`base_commit`, once at the agent's final state — and runs the checks in
DESIGN.md's Phase 4 order of trust: DOM assertion, interaction drive, and
perceptual screenshot diff (all PROVEN), then the vision-intent judge
(JUDGED, always last, never load-bearing).

Capturing "before" by actually rendering `base_commit` (rather than
diffing the agent's textual diff, or skipping straight to "after") is what
makes the visual-diff check meaningful — it's a real render of the real
pre-agent page, in the same browser, at the same viewports, not a guess.

Any failure in getting a browser up (missing `playwright` package) or a
dev server up (bad `frontend.start` command, a broken app) is reported as
a `frontend:setup` PROVEN FAIL signal rather than raised — a single
frontend hiccup must never crash the whole `run()` and lose the gate/
attribution results already computed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from verdict.config import FrontendCheckSpec, FrontendConfig, VerdictConfig
from verdict.frontend.checks import dom_check, interaction_check
from verdict.frontend.server import FrontendServerError, dev_server
from verdict.frontend.vision_judge import MockVisionJudge, VisionJudge, to_signal
from verdict.frontend.visual_diff import perceptual_diff_ratio
from verdict.schema import GateStatus, Provenance, Signal
from verdict.worktree import Worktree, copy_vendored_dependencies, scratch_worktree

if TYPE_CHECKING:
    from playwright.sync_api import Browser

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

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            before_shots, before_error = _capture_before(repo, worktree, frontend, browser)
            after_shots, check_signals = _capture_after_and_run_checks(
                worktree.path, frontend, task, judge, browser
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
    return signals


def _goto(page: object, url: str) -> None:
    page.goto(url, wait_until="load")  # type: ignore[attr-defined]
    try:
        page.wait_for_load_state("networkidle", timeout=_NETWORK_IDLE_GRACE_MS)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - best-effort settle time, not a correctness requirement
        pass


def _screenshot(browser: Browser, url: str, viewport_width: int, viewport_height: int) -> bytes:
    page = browser.new_page(viewport={"width": viewport_width, "height": viewport_height})
    try:
        _goto(page, url)
        return page.screenshot(full_page=False)
    finally:
        page.close()


def _capture_before(
    repo: Path, worktree: Worktree, frontend: FrontendConfig, browser: Browser
) -> tuple[dict[int, bytes], str | None]:
    """Render `worktree.base_commit` — the repo exactly as it was before the
    agent touched anything — in a disposable scratch worktree, so the
    visual diff compares two real renders rather than a render against a
    guess."""
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
    task: str,
    judge: VisionJudge,
    browser: Browser,
) -> tuple[dict[int, bytes], list[Signal]]:
    try:
        with dev_server(frontend.start, worktree_path, frontend.url, frontend.ready_timeout_seconds):
            after_shots = {
                vw: _screenshot(browser, frontend.url, vw, frontend.viewport_height)
                for vw in frontend.viewports
            }
            primary_shot = after_shots.get(frontend.viewports[0])
            check_signals: list[Signal] = []
            for spec in frontend.checks:
                check_signals.extend(_run_one_check(browser, frontend, spec, judge, primary_shot))
        return after_shots, check_signals
    except FrontendServerError as exc:
        detail = f"could not start the frontend dev server to run checks against: {exc}"
        return {}, [_setup_fail(detail)]


def _run_one_check(
    browser: Browser,
    frontend: FrontendConfig,
    spec: FrontendCheckSpec,
    judge: VisionJudge,
    primary_after_shot: bytes | None,
) -> list[Signal]:
    signals: list[Signal] = []

    if spec.dom is not None or spec.interaction is not None:
        page = browser.new_page(
            viewport={"width": frontend.viewports[0], "height": frontend.viewport_height}
        )
        try:
            _goto(page, frontend.url)
            if spec.dom is not None:
                signals.append(dom_check(page, spec.dom, spec.name))
            if spec.interaction is not None:
                signals.append(interaction_check(page, spec.interaction, spec.name))
        finally:
            page.close()

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
