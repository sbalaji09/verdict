"""Boots the repo's own backend service (`backend.start` in `verdict.yml`)
inside the sandbox and waits for it to answer its health endpoint —
structurally the same primitive `frontend/server.py::dev_server` already
is for a frontend dev server (background process + HTTP poll loop), kept
as its own small, backend-named module rather than merged into one shared
one: the two call sites' failure semantics read differently enough
("dev server" vs. "service") that duplicating ~40 lines was the smaller
cost than a cross-domain shared abstraction neither side individually
needs more of.
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


class BackendServiceError(RuntimeError):
    """The service never became ready, or exited before it did."""


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
        # Any HTTP response — even a 4xx — means something is listening
        # and answering; a wrong status is what the health check itself
        # (or a smoke request) judges, not this readiness probe.
        return exc.code < 500
    except (OSError, ValueError):
        return False


@contextmanager
def boot_service(
    command: str,
    cwd: Path,
    health_url: str,
    ready_timeout_seconds: int,
    sandbox: Sandbox,
    env: dict[str, str] | None = None,
) -> Iterator[None]:
    """Run `command` in `cwd` inside `sandbox`, poll `health_url` until it
    answers (or the process exits, or the timeout elapses), yield once
    ready, and always tear the background process down afterward.

    `network=True`: a service that can't reach its own health endpoint or
    the loopback address it's meant to be listening on can't be
    reasonably graded as "broken" by Phase 8's sandbox — the identical,
    intentional exception to gates' network-off default that `frontend/
    server.py::dev_server` already makes for the same reason. See
    DESIGN.md's Phase 8 section.
    """
    handle = sandbox.exec_background(["sh", "-c", command], cwd=cwd, env=env, network=True)
    try:
        deadline = time.monotonic() + ready_timeout_seconds
        while True:
            if not handle.is_alive():
                raise BackendServiceError(
                    f"backend service (`{command}`) exited early before answering at "
                    f"{health_url}:\n{_tail(handle.read_output())}"
                )
            if _url_is_ready(health_url):
                break
            if time.monotonic() >= deadline:
                raise BackendServiceError(
                    f"backend service (`{command}`) did not answer at {health_url} "
                    f"within {ready_timeout_seconds}s:\n{_tail(handle.read_output())}"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)
        yield
    finally:
        handle.terminate()
