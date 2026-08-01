"""Phase 4 end-to-end: a real (tiny, dependency-free) static site served by
Python's own `http.server`, driven by a real headless Chromium via
Playwright — not mocked at any layer. Slower than a pure unit test (each
case starts a real dev server twice and opens a real browser), but this is
exactly the class of behavior (a real render, a real click, a real
navigation) that a mock can't actually prove — the same trade-off Phase 2's
attribution tests made for real `git bisect`.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import pytest

from verdict.adapters.mock import MockAdapter
from verdict.runner import run
from verdict.schema import GateStatus, Provenance, VerdictStatus

pytest.importorskip("playwright")

BUGGY_INDEX_HTML = (
    "<!doctype html><html><head>"
    '<link rel="stylesheet" href="/style.css">'
    "</head><body>"
    '<a id="cta" class="cta hidden" href="/other.html">Go</a>'
    "</body></html>"
)
FIXED_INDEX_HTML = (
    "<!doctype html><html><head>"
    '<link rel="stylesheet" href="/style.css">'
    "</head><body>"
    '<a id="cta" class="cta" href="/other.html">Go</a>'
    "</body></html>"
)
OTHER_HTML = "<!doctype html><html><body><h1>Other page</h1></body></html>"
STYLE_CSS = ".hidden { display: none; }\n"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)


@pytest.fixture
def frontend_repo(tmp_path: Path) -> Path:
    """A minimal real static site with one seeded bug: the CTA link carries
    a leftover `hidden` class, so it never renders even though clicking it
    is supposed to navigate to `/other.html`.
    """
    repo = tmp_path / "frontend_repo"
    repo.mkdir()
    (repo / "index.html").write_text(BUGGY_INDEX_HTML)
    (repo / "other.html").write_text(OTHER_HTML)
    (repo / "style.css").write_text(STYLE_CSS)

    port = _free_port()
    (repo / "verdict.yml").write_text(
        f"""
frontend:
  start: "{sys.executable} -m http.server {port} --bind 127.0.0.1"
  url: "http://127.0.0.1:{port}/index.html"
  viewports: [800]
  ready_timeout_seconds: 10
  screenshot_threshold: 0.02
  glitch_capture_seconds: 0.3
  glitch_frame_interval_seconds: 0.1
  checks:
    - name: cta
      dom:
        selector: "#cta"
        visible: true
        class_contains: "cta"
      interaction:
        click: "#cta"
        expect_url_contains: "/other.html"
      vision_intent: "A visible CTA link is present on the page."
"""
    )
    _init_git_repo(repo)
    return repo


def _signal(verdict, name: str):
    return next(s for s in verdict.signals if s.name == name)


def test_frontend_checks_pass_after_fix(frontend_repo: Path) -> None:
    adapter = MockAdapter(patches={"index.html": FIXED_INDEX_HTML})
    verdict = run(task="make the CTA visible and link to other.html", repo=frontend_repo, adapter=adapter)

    assert _signal(verdict, "frontend:dom:cta").status is GateStatus.PASS
    assert _signal(verdict, "frontend:interaction:cta").status is GateStatus.PASS
    assert _signal(verdict, "frontend:visual_diff:800px").status is GateStatus.PASS
    assert _signal(verdict, "frontend:glitch_scan:load").status is GateStatus.PASS
    assert _signal(verdict, "frontend:glitch_scan:cta").status is GateStatus.PASS
    vision = _signal(verdict, "frontend:vision_intent:cta")
    assert vision.provenance is Provenance.JUDGED
    assert vision.status is GateStatus.PASS
    assert verdict.status is VerdictStatus.DONE


def test_frontend_checks_fail_when_bug_untouched(frontend_repo: Path) -> None:
    adapter = MockAdapter(patches={"README.md": "unrelated\n"})
    verdict = run(task="do nothing useful", repo=frontend_repo, adapter=adapter)

    assert _signal(verdict, "frontend:dom:cta").status is GateStatus.FAIL
    assert _signal(verdict, "frontend:interaction:cta").status is GateStatus.FAIL
    assert verdict.status is VerdictStatus.NOT_DONE


def test_judged_vision_signal_never_flips_a_proven_failure(frontend_repo: Path) -> None:
    """MockVisionJudge passes unconditionally — it can't actually see the
    image — so this asserts the one property Phase 4 is built around: a
    glowing JUDGED opinion can't rescue a run with a failing PROVEN check.
    """
    adapter = MockAdapter(patches={"README.md": "unrelated\n"})
    verdict = run(task="do nothing useful", repo=frontend_repo, adapter=adapter)

    vision = _signal(verdict, "frontend:vision_intent:cta")
    assert vision.status is GateStatus.PASS
    assert verdict.status is VerdictStatus.NOT_DONE


def test_no_frontend_signals_without_frontend_config(git_repo: Path) -> None:
    """`git_repo` (tests/conftest.py) has no `verdict.yml` at all — frontend
    checks are entirely opt-in and must contribute nothing when unconfigured."""
    adapter = MockAdapter(patches={"calculator.py": "def add(a, b):\n    return a + b\n"})
    verdict = run(task="fix add()", repo=git_repo, adapter=adapter)

    assert not any(s.name.startswith("frontend:") for s in verdict.signals)


def test_frontend_setup_failure_is_reported_not_raised(tmp_path: Path) -> None:
    """A `frontend.start` command that can never come up (here: `false`,
    which exits immediately) must surface as an honest FAIL signal, not
    crash the whole run and lose the gate results already computed.
    """
    repo = tmp_path / "broken_frontend_repo"
    repo.mkdir()
    (repo / "verdict.yml").write_text(
        f"""
frontend:
  start: "false"
  url: "http://127.0.0.1:{_free_port()}/"
  ready_timeout_seconds: 2
"""
    )
    _init_git_repo(repo)

    adapter = MockAdapter(patches={"README.md": "noop\n"})
    verdict = run(task="noop", repo=repo, adapter=adapter)

    setup_signals = [s for s in verdict.signals if s.name == "frontend:setup"]
    assert setup_signals
    assert setup_signals[0].status is GateStatus.FAIL
    assert setup_signals[0].provenance is Provenance.PROVEN


GLITCHY_INDEX_HTML = """<!doctype html><html><body style="margin:0">
<div id="banner" style="display:none;width:100%;height:400px;background:red"></div>
<script>
  let visible = false;
  setInterval(function () {
    visible = !visible;
    document.getElementById('banner').style.display = visible ? 'block' : 'none';
  }, 120);
</script>
</body></html>"""


@pytest.fixture
def glitchy_frontend_repo(tmp_path: Path) -> Path:
    """A page that keeps toggling a banner's visibility every 120ms — a
    real, continuous flicker, not a one-shot flash — so the glitch scan has
    a high chance of sampling it regardless of exact frame timing.
    """
    repo = tmp_path / "glitchy_frontend_repo"
    repo.mkdir()
    (repo / "index.html").write_text(GLITCHY_INDEX_HTML)

    port = _free_port()
    (repo / "verdict.yml").write_text(
        f"""
frontend:
  start: "{sys.executable} -m http.server {port} --bind 127.0.0.1"
  url: "http://127.0.0.1:{port}/index.html"
  viewports: [800]
  ready_timeout_seconds: 10
  glitch_capture_seconds: 1.2
  glitch_frame_interval_seconds: 0.1
"""
    )
    _init_git_repo(repo)
    return repo


def test_glitch_scan_catches_a_real_flickering_page(glitchy_frontend_repo: Path) -> None:
    adapter = MockAdapter(patches={"README.md": "noop\n"})
    verdict = run(task="noop", repo=glitchy_frontend_repo, adapter=adapter)

    load_scan = _signal(verdict, "frontend:glitch_scan:load")
    assert load_scan.status is GateStatus.FAIL
    assert load_scan.provenance is Provenance.PROVEN
    assert verdict.status is VerdictStatus.NOT_DONE


def test_glitch_scan_failure_keeps_its_video_recording(glitchy_frontend_repo: Path) -> None:
    adapter = MockAdapter(patches={"README.md": "noop\n"})
    verdict = run(task="noop", repo=glitchy_frontend_repo, adapter=adapter)

    load_scan = _signal(verdict, "frontend:glitch_scan:load")
    assert load_scan.artifact_path is not None
    assert Path(load_scan.artifact_path).exists()


def test_stable_page_glitch_scan_passes_and_discards_its_recording(frontend_repo: Path) -> None:
    adapter = MockAdapter(patches={"index.html": FIXED_INDEX_HTML})
    verdict = run(task="fix cta", repo=frontend_repo, adapter=adapter)

    load_scan = _signal(verdict, "frontend:glitch_scan:load")
    assert load_scan.status is GateStatus.PASS
    assert load_scan.artifact_path is None
