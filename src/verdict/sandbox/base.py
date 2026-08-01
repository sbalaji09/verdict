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


class ProvisioningTimeoutError(SandboxUnavailableError):
    """A sandbox never became usable within its provisioning budget — the
    container never started, or the dependency-install step never
    finished. Phase 9's explicit distinction: this is an INFRASTRUCTURE
    failure, not a gate result. It must never become a PROVEN FAIL Signal
    (that would misattribute an infra problem to the agent) — it aborts
    the whole attempt instead, the same way an adapter CLI itself hanging
    already did before this phase (see `cli.py`'s `_RUN_ERRORS`).

    A subclass of `SandboxUnavailableError` on purpose: both mean "this
    attempt couldn't be evaluated," they just differ in *why* — one
    timed out, the other found no usable backend at all. Phase 11's ERROR
    outcome is expected to fold this whole family into one Verdict-level
    status rather than an attempt-aborting exception; keeping them in one
    hierarchy now means that later change has one family to catch, not
    several scattered handlers.
    """


class SetupError(SandboxUnavailableError):
    """Phase 10's setup stage (dependency install, language-version-pin
    resolution, service-dependency startup) failed in a way that isn't
    the agent's fault: an unrecognized `services:` type/version in
    `verdict.yml` (a real, satisfiable-looking request Verdict can't
    fulfill — see `sandbox/services.py`'s allowlist), a declared service
    that never became healthy, an unresolvable `.nvmrc`/`.python-version`
    pin. A sibling of `ProvisioningTimeoutError` under the same
    `SandboxUnavailableError` ancestor for the same reason that one is a
    subclass of it: both mean "this attempt couldn't be evaluated," Phase
    11's ERROR outcome is expected to fold the whole family together, and
    Phase 10 already pulls forward a minimal `VerdictStatus.ERROR` for
    exactly this family (see `schema.py`) rather than waiting.
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
