"""The `Sandbox` abstraction: the one interface every execution of
agent-influenced code (gate tools, coding-agent CLIs, dependency installs,
the frontend dev server, bisect re-runs) goes through, instead of a direct
`subprocess` call with this process's own privileges.

Two implementations live alongside this module: `LocalSandbox` (today's
behavior — no isolation, opt-in only) and `DockerSandbox` (the default —
an ephemeral, network-off-by-default container per worktree). Callers code
against this Protocol, never against a concrete backend, so swapping which
one runs is a config change, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_TIMEOUT_SECONDS = 600


class SandboxError(RuntimeError):
    """Base class for sandbox failures."""


class SandboxUnavailableError(SandboxError):
    """A sandbox backend couldn't be constructed or used — e.g. the Docker
    daemon isn't reachable. Never caught to silently fall back to a
    different (less isolated) backend; that would defeat the point of
    choosing DockerSandbox in the first place. Callers that want a fallback
    must do so explicitly and visibly (see LocalSandbox's own warning).
    """


@dataclass(frozen=True)
class ResourceLimits:
    """CPU/memory/pids/disk ceilings for one sandboxed process tree.

    Phase 8 threads these through to `docker run` (`--cpus`, `--memory`,
    `--pids-limit`) but doesn't yet do anything with `disk_mb` or detect
    *why* a container died (OOM vs. pids vs. a plain crash) — that
    enforcement/detection work is Phase 9. The field exists now so call
    sites don't need to change signatures again when it lands.
    """

    cpu_cores: float = 2.0
    memory_mb: int = 2048
    pids: int = 256
    disk_mb: int = 4096


@dataclass
class ExecResult:
    """What actually happened, independent of backend."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    killed_reason: str | None = None
    """"timeout" today; "oom" / "pids" once Phase 9 wires real detection."""


class BackgroundHandle(Protocol):
    """A still-running process started via `Sandbox.exec_background` (the
    dev server, primarily) — long-lived, so it needs its own lifecycle
    separate from the blocking `exec()` call.
    """

    def is_alive(self) -> bool: ...

    def read_output(self) -> str:
        """Everything captured so far (stdout+stderr, interleaved)."""
        ...

    def terminate(self, grace_seconds: float = 5.0) -> None:
        """SIGTERM the whole process tree, escalate to SIGKILL after
        `grace_seconds` if it hasn't exited. Idempotent — safe to call on
        an already-dead process.
        """
        ...


class Sandbox(Protocol):
    """One sandbox instance = one isolated workspace bound to one worktree,
    live for the lifetime of a `with` block. Multiple `exec()`/
    `exec_background()` calls against the same instance share that one
    workspace (and, for DockerSandbox, the same container/network
    namespace) — this is what lets a dev server started by one call be
    reachable by a browser driven through another.
    """

    def __enter__(self) -> "Sandbox": ...

    def __exit__(self, *exc_info: object) -> None:
        """Tear down unconditionally, even on an exception — never leaves
        a container or process running past this call.
        """
        ...

    def exec(
        self,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        limits: ResourceLimits | None = None,
        network: bool = False,
    ) -> ExecResult:
        """Run `cmd` (always an argv list — never a shell string; a caller
        holding a shell command from an untrusted source, e.g. a
        `verdict.yml` gate override, must wrap it itself as
        `["sh", "-c", command]` so the shell interpretation happens inside
        the sandbox boundary, not via host `shell=True`).

        `env` is the *complete* set of extra variables the process sees,
        merged with a small fixed baseline — never the caller's full
        `os.environ`. Secrets an adapter genuinely needs (e.g. an API key)
        must be passed here explicitly; nothing is inherited implicitly.

        `network` defaults to off. Sandboxes are not required to support
        `network=True` for every call — DockerSandbox implements it as a
        session-level policy (see docker.py) rather than a per-call toggle,
        since Docker's network mode is fixed at container-create time.
        """
        ...

    def exec_background(
        self,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        network: bool = False,
    ) -> BackgroundHandle:
        """Like `exec`, but returns immediately with a handle instead of
        blocking — for long-lived processes (the frontend dev server).
        """
        ...
