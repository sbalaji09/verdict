"""`DockerSandbox` — the default `Sandbox` backend. One ephemeral container
per instance, bound to one worktree, torn down unconditionally on exit.

Defaults: no network egress (`--network none`), no host secrets/env
(only what a caller explicitly passes via `env=`), the worktree mounted
read-write at `/workspace` (the agent needs to edit it) and nothing else
writable, resource limits passed through to `docker run` (Phase 9 also
attempts a best-effort disk quota — see `_run_args`).

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

Phase 9's kill-tree requirement (`exec()` must be able to kill an entire
process tree on timeout, not just the one process it started) means both
`exec()` and `exec_background()` share one mechanism: launch the command
detached, under `setsid` (so it becomes its own session/process-group
leader — `$$` inside it *is* the group id), and track it via a pidfile a
later call can read back and signal. Killing `-PID` (negative — a process
*group*) reaches every child the command forked; killing plain `PID` (what
Phase 8 originally did for `exec_background`) does not, and left orphans
holding ports exactly as flagged going into this phase.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from verdict.sandbox.base import (
    DEFAULT_TIMEOUT_SECONDS,
    BackgroundHandle,
    ExecResult,
    ProvisioningTimeoutError,
    ResourceLimits,
    SandboxUnavailableError,
)

DEFAULT_IMAGE = "verdict-sandbox:0.1.0"
DEFAULT_PROVISION_TIMEOUT_SECONDS = 120
_WORKSPACE = "/workspace"
_TERMINATE_GRACE_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.2


def _shell_quote(s: str) -> str:
    return shlex.quote(s)


def _docker(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    """Runs the trusted `docker` CLI itself. `timeout` here bounds the
    `docker` client call (e.g. a hung daemon connection) — it is NOT how
    workload timeouts are enforced; see `_run_tracked`/`exec()` for that.
    """
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


def _read_file(container: str, path: str) -> str:
    result = _docker(["exec", container, "sh", "-c", f"cat {path} 2>/dev/null || true"])
    return result.stdout


def _tracked_pid(container: str, pidfile: str) -> str:
    return _docker(["exec", container, "sh", "-c", f"cat {pidfile} 2>/dev/null || true"]).stdout.strip()


def _tracked_alive(container: str, pidfile: str) -> bool:
    pid = _tracked_pid(container, pidfile)
    if not pid:
        return False
    # Signal 0 the process-group leader specifically (positive pid) — this
    # answers "is the group's leader still around," which is what
    # `is_alive()` promises; killing/reaping is `_kill_tracked`'s job.
    return _docker(["exec", container, "sh", "-c", f"kill -0 {pid}"]).returncode == 0


def _kill_tracked(container: str, pidfile: str, grace_seconds: float) -> None:
    """Kill the WHOLE process group `pidfile`'s pid leads — see module
    docstring for why this has to be a group kill (`-PID`), not a plain
    `kill PID`.
    """
    pid = _tracked_pid(container, pidfile)
    if not pid:
        return
    _docker(["exec", container, "sh", "-c", f"kill -TERM -{pid} 2>/dev/null || true"])
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _tracked_alive(container, pidfile):
            return
        time.sleep(0.25)
    _docker(["exec", container, "sh", "-c", f"kill -KILL -{pid} 2>/dev/null || true"])


class _DockerBackgroundHandle:
    def __init__(self, container: str, pidfile: str, log_path: str) -> None:
        self._container = container
        self._pidfile = pidfile
        self._log_path = log_path

    def is_alive(self) -> bool:
        return _tracked_alive(self._container, self._pidfile)

    def read_output(self) -> str:
        return _read_file(self._container, self._log_path)

    def terminate(self, grace_seconds: float = _TERMINATE_GRACE_SECONDS) -> None:
        _kill_tracked(self._container, self._pidfile, grace_seconds)


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
        provision_timeout_seconds: int = DEFAULT_PROVISION_TIMEOUT_SECONDS,
    ) -> None:
        if shutil.which("docker") is None:
            raise SandboxUnavailableError("`docker` CLI not found on PATH")
        self._worktree = worktree
        self._image = image
        self._limits = limits or ResourceLimits()
        self._network = network
        self._provision_timeout_seconds = provision_timeout_seconds
        self._container: str | None = None

    def _run_args(self, with_storage_opt: bool) -> list[str]:
        assert self._container is not None
        args = [
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
            # some tools (and this module's own tracked-exec pidfile/
            # log-file handling) need a scratch write location outside the
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
        ]
        if with_storage_opt:
            # Best-effort: `--storage-opt size=` only works with specific
            # storage-driver/backing-filesystem combinations (overlay2 on
            # xfs with pquota, notably — NOT the common ext4 default).
            # Unsupported is the common case, not the exception, so this
            # is retried without the flag on failure rather than treated
            # as a hard requirement. See DESIGN.md's Phase 9 section.
            args += ["--storage-opt", f"size={self._limits.disk_mb}m"]
        args += [self._image, "sleep", "infinity"]
        return args

    def __enter__(self) -> "DockerSandbox":
        _check_daemon_reachable()
        self._container = f"verdict-sandbox-{uuid.uuid4().hex[:12]}"
        try:
            result = _docker(self._run_args(with_storage_opt=True), timeout=self._provision_timeout_seconds)
        except SandboxUnavailableError as exc:
            self._container = None
            raise ProvisioningTimeoutError(
                f"sandbox container did not start within {self._provision_timeout_seconds}s"
            ) from exc

        if result.returncode != 0:
            # Retry once without the disk-quota flag — a storage-driver
            # mismatch is the overwhelmingly likely reason `docker run`
            # itself (not a timeout) failed immediately; anything else
            # fails again below with its own real error.
            retry = _docker(self._run_args(with_storage_opt=False), timeout=self._provision_timeout_seconds)
            if retry.returncode != 0:
                self._container = None
                raise SandboxUnavailableError(f"docker run failed: {retry.stderr.strip()}")
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

    def _start_tracked(
        self,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str] | None,
        stdout_path: str,
        stderr_path: str | None,
        pidfile: str,
        exitfile: str | None,
    ) -> None:
        """Launch `cmd` detached, under `setsid`, recording its (group)
        pid to `pidfile` before running it. `stderr_path=None` combines
        stderr into `stdout_path` (used by `exec_background` — a dev
        server's log doesn't need the split); `exitfile` set records the
        exit code for `exec()` to poll for.
        """
        assert self._container is not None
        cmd_str = " ".join(_shell_quote(c) for c in cmd)
        redirect = f"> {stdout_path} 2> {stderr_path}" if stderr_path else f"> {stdout_path} 2>&1"
        body = f"echo $$ > {pidfile}; {cmd_str} {redirect}"
        if exitfile:
            body += f"; echo $? > {exitfile}"
        wrapped = f"setsid sh -c {_shell_quote(body)}"
        exec_args = ["exec", "-d", "-w", self._cwd_in_container(cwd)]
        for key, value in (env or {}).items():
            exec_args += ["-e", f"{key}={value}"]
        exec_args += [self._container, "sh", "-c", wrapped]
        result = _docker(exec_args, timeout=30)
        if result.returncode != 0:
            raise SandboxUnavailableError(f"docker exec -d failed: {result.stderr.strip()}")

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

        token = uuid.uuid4().hex[:8]
        pidfile = f"/tmp/verdict-{token}.pid"
        exitfile = f"/tmp/verdict-{token}.exit"
        stdout_path = f"/tmp/verdict-{token}.stdout"
        stderr_path = f"/tmp/verdict-{token}.stderr"
        self._start_tracked(cmd, cwd, env, stdout_path, stderr_path, pidfile, exitfile)

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if _docker(
                ["exec", self._container, "sh", "-c", f"test -f {exitfile}"], timeout=10
            ).returncode == 0:
                raw_exit = _read_file(self._container, exitfile).strip()
                exit_code = int(raw_exit) if raw_exit.isdigit() else 1
                stdout = _read_file(self._container, stdout_path)
                stderr = _read_file(self._container, stderr_path)
                self._cleanup(pidfile, exitfile, stdout_path, stderr_path)
                return ExecResult(exit_code=exit_code, stdout=stdout, stderr=stderr)
            time.sleep(_POLL_INTERVAL_SECONDS)

        # Timed out: the exit-code file never appeared. Kill the WHOLE
        # process group before returning — a timed-out gate must never
        # leave its process tree running in a container that's about to
        # be reused for the next gate in the same sandbox session.
        stdout = _read_file(self._container, stdout_path)
        stderr = _read_file(self._container, stderr_path)
        _kill_tracked(self._container, pidfile, _TERMINATE_GRACE_SECONDS)
        self._cleanup(pidfile, exitfile, stdout_path, stderr_path)
        return ExecResult(
            exit_code=124,
            stdout=stdout,
            stderr=stderr + f"\ntimed out after {timeout_seconds}s",
            timed_out=True,
            killed_reason="timeout",
        )

    def _cleanup(self, *paths: str) -> None:
        if self._container is None:
            return
        joined = " ".join(paths)
        _docker(["exec", self._container, "sh", "-c", f"rm -f {joined}"], timeout=10)

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
        self._start_tracked(cmd, cwd, env, log_path, None, pidfile, None)
        return _DockerBackgroundHandle(self._container, pidfile, log_path)

    def publish_port(self, container_port: int) -> None:
        """Not implemented: `docker run -p` publishes ports at
        container-create time, and today's `__enter__` doesn't accept one.
        Frontend checks that need the host to reach a port inside this
        container (Playwright connecting to a browser server, e.g.) are a
        known gap — see DESIGN.md's Phase 8 section, "deferred" list.
        """
        raise NotImplementedError(
            "DockerSandbox port publishing is not implemented yet — see DESIGN.md Phase 8"
        )
