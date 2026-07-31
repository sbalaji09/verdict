"""Starts and tears down the repo's own frontend dev server (`frontend.start`
in `verdict.yml`) around a block of Playwright checks.

Two choices worth calling out, both mirroring decisions already made
elsewhere in this codebase for the same reasons:

- **`shell=True`, no argv parsing.** `frontend.start` is a user-supplied
  string, exactly like the `verdict.yml` gate overrides in `gates/base.py`'s
  `raw_signal` — we don't control the invocation, so there's no structured
  command to build, and this is the one place `shell=True` is a considered
  exception rather than an oversight.
- **`start_new_session=True`.** A dev server started via `npm run dev`
  usually forks a real child process (node) that outlives the shell if
  killed directly — a plain `proc.terminate()` would leave that child
  running and its port bound. Starting a new process group lets teardown
  kill the whole group at once.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_POLL_INTERVAL_SECONDS = 0.25
_TERMINATE_GRACE_SECONDS = 5


class FrontendServerError(RuntimeError):
    """The dev server never became ready, or exited before it did."""


def _tail(text: str, lines: int = 15) -> str:
    text = text.strip()
    if not text:
        return "(no output)"
    return "\n".join(text.splitlines()[-lines:])


def _url_is_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return bool(resp.status < 500)
    except urllib.error.HTTPError as exc:
        # Any HTTP response — even a 404 — means something is listening and
        # answering. Good enough for "ready"; a genuinely wrong page is the
        # frontend checks' job to catch, not this readiness probe's.
        return exc.code < 500
    except (OSError, ValueError):
        return False


def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except ProcessLookupError:
        pass


@contextmanager
def dev_server(
    command: str, cwd: Path, url: str, ready_timeout_seconds: int
) -> Iterator[None]:
    """Run `command` in `cwd`, poll `url` until it answers (or the process
    exits, or the timeout elapses), yield once ready, and always tear the
    process group down afterward — regardless of what the caller does with
    it. Output is captured to a temp file (not a pipe) so a chatty dev
    server can never deadlock this process by filling an unread pipe buffer.
    """
    log_file = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + ready_timeout_seconds
        while True:
            if proc.poll() is not None:
                log_file.seek(0)
                raise FrontendServerError(
                    f"frontend dev server (`{command}`) exited early "
                    f"(code {proc.returncode}) before answering at {url}:\n"
                    f"{_tail(log_file.read())}"
                )
            if _url_is_ready(url):
                break
            if time.monotonic() >= deadline:
                log_file.seek(0)
                raise FrontendServerError(
                    f"frontend dev server (`{command}`) did not answer at {url} "
                    f"within {ready_timeout_seconds}s:\n{_tail(log_file.read())}"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)
        yield
    finally:
        _terminate(proc)
        log_file.close()
