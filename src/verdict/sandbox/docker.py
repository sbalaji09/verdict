"""`DockerSandbox` — the default `Sandbox` backend. One ephemeral container
per instance, bound to one worktree, torn down unconditionally on exit.

Defaults: no network egress (`--network none`), no host secrets/env
(only what a caller explicitly passes via `env=`), the worktree mounted
read-write at `/workspace` (the agent needs to edit it) and nothing else
writable, resource limits passed through to `docker run`.

Network is a session-level choice, not a per-`exec()` toggle: Docker fixes
a container's network mode at `docker run` time, it can't be flipped
per-`docker exec`. Callers that need both a networked phase (dependency
install) and a network-off phase (gates) construct two separate
`DockerSandbox` instances against the same worktree rather than expecting
one instance to do both — see `sandbox/install.py` and `runner.py`.

This module IS allowed to call `subprocess` directly — but only ever with
argv Verdict itself constructs to invoke the trusted `docker` CLI
(`docker run`, `docker exec`, `docker rm`). The untrusted payload (`cmd`,
`env`, `cwd`) is data passed as arguments to `docker`, never interpolated
into a shell string, so this doesn't reopen the injection surface the rest
of Phase 8 closes.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from pathlib import Path

from verdict.sandbox.base import (
    DEFAULT_TIMEOUT_SECONDS,
    BackgroundHandle,
    ExecResult,
    ResourceLimits,
    SandboxUnavailableError,
)

DEFAULT_IMAGE = "verdict-sandbox:0.1.0"
_WORKSPACE = "/workspace"
_TERMINATE_GRACE_SECONDS = 5.0


def _docker(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise SandboxUnavailableError("`docker` CLI not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise SandboxUnavailableError(f"docker {args[0]} timed out") from exc


def _check_daemon_reachable() -> None:
    result = _docker(["info"], timeout=10)
    if result.returncode != 0:
        raise SandboxUnavailableError(
            "Docker daemon unreachable — is Docker running? "
            f"(docker info exited {result.returncode}: {result.stderr.strip()})"
        )


class _DockerBackgroundHandle:
    def __init__(self, container: str, pidfile: str, log_path: str) -> None:
        self._container = container
        self._pidfile = pidfile
        self._log_path = log_path

    def is_alive(self) -> bool:
        result = _docker(["exec", self._container, "sh", "-c", f"test -f {self._pidfile}"])
        if result.returncode != 0:
            return False
        pid_result = _docker(["exec", self._container, "sh", "-c", f"cat {self._pidfile}"])
        pid = pid_result.stdout.strip()
        if not pid:
            return False
        check = _docker(["exec", self._container, "sh", "-c", f"kill -0 {pid}"])
        return check.returncode == 0

    def read_output(self) -> str:
        result = _docker(["exec", self._container, "sh", "-c", f"cat {self._log_path} 2>/dev/null || true"])
        return result.stdout

    def terminate(self, grace_seconds: float = _TERMINATE_GRACE_SECONDS) -> None:
        pid_result = _docker(
            ["exec", self._container, "sh", "-c", f"cat {self._pidfile} 2>/dev/null || true"]
        )
        pid = pid_result.stdout.strip()
        if not pid:
            return
        _docker(["exec", self._container, "sh", "-c", f"kill -TERM {pid} 2>/dev/null || true"])
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if not self.is_alive():
                return
            time.sleep(0.25)
        _docker(["exec", self._container, "sh", "-c", f"kill -KILL {pid} 2>/dev/null || true"])


class DockerSandbox:
    """One instance = one `docker run -d ... sleep infinity` container that
    `exec()`/`exec_background()` target via `docker exec`, plus a
    `docker rm -f` on `__exit__` — the standard "keep a container alive as
    an exec target" pattern, since `docker run` per command would lose the
    shared filesystem/network state between calls.
    """

    def __init__(
        self,
        worktree: Path,
        image: str = DEFAULT_IMAGE,
        limits: ResourceLimits | None = None,
        network: bool = False,
    ) -> None:
        if shutil.which("docker") is None:
            raise SandboxUnavailableError("`docker` CLI not found on PATH")
        self._worktree = worktree
        self._image = image
        self._limits = limits or ResourceLimits()
        self._network = network
        self._container: str | None = None

    def __enter__(self) -> "DockerSandbox":
        _check_daemon_reachable()
        self._container = f"verdict-sandbox-{uuid.uuid4().hex[:12]}"
        run_args = [
            "run",
            "-d",
            "--name",
            self._container,
            "-v",
            f"{self._worktree}:{_WORKSPACE}",
            # Everything outside the worktree mount is read-only image
            # content: an agent process can edit /workspace freely (the
            # bind mount keeps its own read-write setting regardless of
            # this flag) but can't write anywhere else in the container's
            # filesystem. /tmp is the one exception, as a small tmpfs —
            # some tools (and this module's own exec_background pidfile/
            # logfile handling) need a scratch write location outside the
            # worktree.
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=256m",
            "--network",
            "bridge" if self._network else "none",
            "--cpus",
            str(self._limits.cpu_cores),
            "--memory",
            f"{self._limits.memory_mb}m",
            "--pids-limit",
            str(self._limits.pids),
            self._image,
            "sleep",
            "infinity",
        ]
        result = _docker(run_args, timeout=60)
        if result.returncode != 0:
            self._container = None
            raise SandboxUnavailableError(f"docker run failed: {result.stderr.strip()}")
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._container is None:
            return
        _docker(["rm", "-f", self._container], timeout=30)
        self._container = None

    def _cwd_in_container(self, cwd: Path) -> str:
        try:
            rel = cwd.resolve().relative_to(self._worktree.resolve())
        except ValueError:
            # cwd isn't inside the mounted worktree — not something a
            # correctly-behaving caller should ever pass, but fail loud
            # rather than silently execute somewhere unexpected.
            raise SandboxUnavailableError(
                f"{cwd} is not inside the sandboxed worktree {self._worktree}"
            ) from None
        return f"{_WORKSPACE}/{rel}" if str(rel) != "." else _WORKSPACE

    def exec(
        self,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        limits: ResourceLimits | None = None,
        network: bool = False,
    ) -> ExecResult:
        if self._container is None:
            raise SandboxUnavailableError("exec() called outside a `with` block")
        exec_args = ["exec", "-w", self._cwd_in_container(cwd)]
        for key, value in (env or {}).items():
            exec_args += ["-e", f"{key}={value}"]
        exec_args += [self._container, *cmd]
        try:
            result = _docker(exec_args, timeout=timeout_seconds)
        except SandboxUnavailableError:
            return ExecResult(
                exit_code=124,
                stdout="",
                stderr=f"timed out after {timeout_seconds}s",
                timed_out=True,
                killed_reason="timeout",
            )
        return ExecResult(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

    def exec_background(
        self,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        network: bool = False,
    ) -> BackgroundHandle:
        if self._container is None:
            raise SandboxUnavailableError("exec_background() called outside a `with` block")
        token = uuid.uuid4().hex[:8]
        pidfile = f"/tmp/verdict-bg-{token}.pid"
        log_path = f"/tmp/verdict-bg-{token}.log"
        # Record the backgrounded process's own PID (not `docker exec`'s
        # PID, which exits immediately with -d) so terminate() has
        # something real to signal later.
        wrapped = f"echo $$ > {pidfile}; exec {' '.join(_shell_quote(c) for c in cmd)} > {log_path} 2>&1"
        exec_args = ["exec", "-d", "-w", self._cwd_in_container(cwd)]
        for key, value in (env or {}).items():
            exec_args += ["-e", f"{key}={value}"]
        exec_args += [self._container, "sh", "-c", wrapped]
        result = _docker(exec_args, timeout=30)
        if result.returncode != 0:
            raise SandboxUnavailableError(f"docker exec -d failed: {result.stderr.strip()}")
        return _DockerBackgroundHandle(self._container, pidfile, log_path)

    def publish_port(self, container_port: int) -> None:
        """Not implemented in Phase 8: `docker run -p` publishes ports at
        container-create time, and today's `__enter__` doesn't accept one.
        Frontend checks that need the host to reach a port inside this
        container (Playwright connecting to a browser server, e.g.) are a
        known gap — see DESIGN.md's Phase 8 section, "deferred" list.
        """
        raise NotImplementedError(
            "DockerSandbox port publishing is not implemented yet — see DESIGN.md Phase 8"
        )


def _shell_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)
