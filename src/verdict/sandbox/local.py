"""`LocalSandbox` — today's actual behavior (direct subprocess execution,
no isolation), wrapped behind the `Sandbox` interface so it's a drop-in
swap with `DockerSandbox`. Opt-in only: never the default for a real
`verdict run`/`verdict gate` invocation (see `sandbox/config.py`).

This is also the ONE place in the codebase a `Sandbox` implementation is
allowed to call `subprocess` directly with agent-influenced argv — every
other module (gates, adapters, the dev server, bisect) goes through a
`Sandbox` instance instead of `subprocess` itself. Whether that instance is
this one or `DockerSandbox` is a config choice, not something those callers
decide.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import warnings
from pathlib import Path

from rich.console import Console

from verdict.sandbox.base import (
    DEFAULT_TIMEOUT_SECONDS,
    BackgroundHandle,
    ExecResult,
    ResourceLimits,
)

_UNSAFE_WARNING = (
    "UNSAFE — no isolation. Agent-generated code will execute with this "
    "process's full privileges (filesystem, network, credentials). Use "
    "only for local development on trusted repos. See DESIGN.md, Phase 8."
)

_TERMINATE_GRACE_SECONDS = 5.0

_console = Console(stderr=True)
_warned = False


def _warn_unsafe_once() -> None:
    global _warned
    if _warned:
        return
    _warned = True
    warnings.warn(_UNSAFE_WARNING, stacklevel=3)
    _console.print(f"[bold red]⚠️  {_UNSAFE_WARNING}[/bold red]")


def _baseline_env() -> dict[str, str]:
    """Deliberately NOT `dict(os.environ)` — a sandboxed process (local or
    Docker) only ever sees this fixed baseline plus whatever the caller
    explicitly declares via `env=`. `PATH` still has to come from the host
    here, since LocalSandbox really does run on the host and needs to find
    real binaries on it; that's the one baseline concession this backend
    makes that DockerSandbox doesn't need (its image supplies its own PATH).
    """
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }


def _kill_process_group(proc: subprocess.Popen[str], grace_seconds: float) -> None:
    """SIGTERM, then SIGKILL after `grace_seconds`, aimed at `proc`'s whole
    process *group* rather than just `proc` itself — every child/grandchild
    it spawned (a dev server's forked node, a test runner's own worker
    processes) dies with it. Requires `proc` to have been started with
    `start_new_session=True`, which makes `proc.pid` its own process-group
    leader; killing a plain child PID (no group) is exactly the bug that
    used to leave orphans holding ports — see DESIGN.md's Phase 9 section.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
    except ProcessLookupError:
        pass


class _LocalBackgroundHandle:
    def __init__(self, proc: subprocess.Popen[str], log_path: Path) -> None:
        self._proc = proc
        self._log_path = log_path
        self._lock = threading.Lock()

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def read_output(self) -> str:
        with self._lock:
            try:
                return self._log_path.read_text(errors="replace")
            except OSError:
                return ""

    def terminate(self, grace_seconds: float = _TERMINATE_GRACE_SECONDS) -> None:
        _kill_process_group(self._proc, grace_seconds)


class LocalSandbox:
    """No isolation. `network`/`limits` are accepted (Protocol conformance)
    but not enforced — there is no boundary here to enforce them against.
    """

    def __init__(self) -> None:
        _warn_unsafe_once()
        self._log_dir: Path | None = None

    def __enter__(self) -> "LocalSandbox":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def exec(
        self,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        limits: ResourceLimits | None = None,
        network: bool = False,
    ) -> ExecResult:
        full_env = {**_baseline_env(), **(env or {})}
        try:
            # `start_new_session=True` (not plain Popen) so a timeout can
            # kill the *whole* process group `cmd` spawned, not just `cmd`
            # itself — plain `subprocess.run(timeout=...)` only ever kills
            # the one PID it started, leaving any children `cmd` forked
            # (a hung test's own subprocess, a lint tool's worker pool)
            # running as orphans. See `_kill_process_group`.
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=full_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            return ExecResult(exit_code=127, stdout="", stderr=str(exc))

        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            return ExecResult(exit_code=proc.returncode, stdout=stdout, stderr=stderr)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc, _TERMINATE_GRACE_SECONDS)
            try:
                stdout, stderr = proc.communicate(timeout=_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            return ExecResult(
                exit_code=124,
                stdout=stdout or "",
                stderr=(stderr or "") + f"\ntimed out after {timeout_seconds}s",
                timed_out=True,
                killed_reason="timeout",
            )

    def exec_background(
        self,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        network: bool = False,
    ) -> BackgroundHandle:
        import tempfile

        full_env = {**_baseline_env(), **(env or {})}
        log_fd, log_path_str = tempfile.mkstemp(prefix="verdict-sandbox-", suffix=".log")
        log_path = Path(log_path_str)
        log_file = os.fdopen(log_fd, "w")
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=full_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        return _LocalBackgroundHandle(proc, log_path)
