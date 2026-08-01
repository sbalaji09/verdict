"""Starts and tears down the repo's own frontend dev server (`frontend.start`
in `verdict.yml`) around a block of Playwright checks.

`frontend.start` is a raw shell string sourced from the repo being graded —
exactly like the `verdict.yml` gate overrides in `gates/base.py`'s
`raw_signal`. Before Phase 8 this ran via host `shell=True`; now it's
wrapped as `["sh", "-c", command]` and handed to a `Sandbox`, so the shell
interpretation happens inside the sandbox boundary, and process-tree
teardown (the old `os.killpg`/process-group dance) is the `Sandbox`
backend's job (`BackgroundHandle.terminate()`), not this module's.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from verdict.sandbox import Sandbox

_POLL_INTERVAL_SECONDS = 0.25


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


@contextmanager
def dev_server(
    command: str,
    cwd: Path,
    url: str,
    ready_timeout_seconds: int,
    sandbox: Sandbox,
) -> Iterator[None]:
    """Run `command` in `cwd` inside `sandbox`, poll `url` until it answers
    (or the process exits, or the timeout elapses), yield once ready, and
    always tear the background process down afterward — regardless of what
    the caller does with it.

    `network=True`: a dev server that can't reach the loopback address (or,
    for many frameworks, the wider network for HMR/asset fetching) can't be
    reasonably graded as "broken" by Phase 8's sandbox — this is the same
    conservative call `runner.py`'s install-step boundary makes explicit
    elsewhere. It's a real, intentional exception to gates' network-off
    default, not an oversight; see DESIGN.md's Phase 8 section.
    """
    handle = sandbox.exec_background(["sh", "-c", command], cwd=cwd, network=True)
    try:
        deadline = time.monotonic() + ready_timeout_seconds
        while True:
            if not handle.is_alive():
                raise FrontendServerError(
                    f"frontend dev server (`{command}`) exited early before answering at "
                    f"{url}:\n{_tail(handle.read_output())}"
                )
            if _url_is_ready(url):
                break
            if time.monotonic() >= deadline:
                raise FrontendServerError(
                    f"frontend dev server (`{command}`) did not answer at {url} "
                    f"within {ready_timeout_seconds}s:\n{_tail(handle.read_output())}"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)
        yield
    finally:
        handle.terminate()
