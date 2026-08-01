"""Verdict's own sandbox settings — deliberately NOT part of `VerdictConfig`
(`config.py`), which is loaded from `verdict.yml` *inside the worktree
being graded*. Sandbox policy must come only from Verdict's own invocation
(CLI flags today), never from the repo under test — otherwise an untrusted
PR could ship a `verdict.yml` that turns off its own sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from verdict.sandbox.base import ResourceLimits, Sandbox
from verdict.sandbox.docker import DEFAULT_IMAGE, DockerSandbox
from verdict.sandbox.local import LocalSandbox

Backend = Literal["docker", "local"]


@dataclass
class SandboxConfig:
    backend: Backend = "local"
    """Library-level default is "local" so embedding code / tests don't
    silently require Docker. The `verdict` CLI itself wires `--sandbox-
    backend` with a default of "docker" (see cli.py) — the *product's*
    default is Docker, per Phase 8; this dataclass's own default is the
    safer no-surprise choice for callers that construct it directly.
    """

    image: str = DEFAULT_IMAGE
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    network: bool = False
    env_passthrough_allowlist: tuple[str, ...] = ()
    """Names of host env vars a caller is *permitted* to forward if it
    explicitly asks (e.g. an adapter's own API key) — still never automatic;
    this only bounds what a caller is allowed to request, it doesn't do any
    forwarding itself.
    """

    # --- Phase 9: timeouts and the global attempt budget ------------------
    gate_timeout_seconds: int = 600
    """Per-gate wall-clock ceiling (test/typecheck/build/lint each get their
    own timer). A gate that times out is reported as a real PROVEN FAIL —
    see DESIGN.md's Phase 9 section for why a hang is treated as the
    agent's fault, not infrastructure's."""

    provision_timeout_seconds: int = 120
    """How long DockerSandbox will wait for `docker run` to bring up a
    usable container before giving up. Provisioning, not a gate — exceeding
    this raises `ProvisioningTimeoutError` and aborts the whole attempt
    rather than becoming any kind of Signal."""

    install_timeout_seconds: int = 300
    """Ceiling for the dependency-install step (`sandbox/install.py`).
    Also provisioning, not a gate, for the same reason: the agent didn't
    write the install command, so a hung `npm install` isn't the agent's
    fault to be graded on."""

    attempt_budget_seconds: int | None = 1800
    """Global wall-clock ceiling across one whole attempt — adapter run,
    install, every gate, attribution, and frontend checks combined. `None`
    disables it. Exceeding it does not fail anything that already ran; it
    stops the attempt from starting anything further and marks the
    resulting `Verdict.budget_exceeded = True` (see `schema.py`)."""

    # --- Phase 10: service-dependency networking ---------------------------
    network_name: str | None = None
    """Set only by `runner.py`, per-attempt, when `verdict.yml` declares
    `services:` — joins the gate/adapter container to that specific
    `--internal` Docker network (`sandbox/services.py`) instead of the
    plain `network` bool's none-vs-bridge choice, so gates can reach
    declared services by name without gaining general internet egress.
    Never set by a caller constructing `SandboxConfig` directly; this is
    attempt-scoped state threaded through, not a standing setting."""

    health_timeout_seconds: int = 30
    """How long a declared service gets to pass its health check before
    `sandbox/services.py::start_services` gives up and raises
    `SetupError` — infra, not the agent's fault, same bucket as every
    other Phase 9/10 setup failure."""


def create_sandbox(worktree: Path, config: SandboxConfig | None = None) -> Sandbox:
    """Construct (but do not yet `__enter__`) a `Sandbox` for `worktree`
    per `config`. Callers are expected to use this via `with
    create_sandbox(...) as sandbox:`.
    """
    config = config or SandboxConfig()
    if config.backend == "docker":
        return DockerSandbox(
            worktree,
            image=config.image,
            limits=config.limits,
            network=config.network,
            provision_timeout_seconds=config.provision_timeout_seconds,
            network_name=config.network_name,
        )
    if config.backend == "local":
        return LocalSandbox()
    raise ValueError(f"unknown sandbox backend: {config.backend!r}")


_fallback: Sandbox | None = None


def fallback_sandbox() -> Sandbox:
    """A lazily-constructed, process-wide `LocalSandbox` used only by call
    sites that haven't been threaded through with an explicit `sandbox=`
    (chiefly unit tests exercising parsing/detection logic in isolation).
    Real entry points (`runner.py`) always build and pass an explicit
    `Sandbox` via `create_sandbox`, so production runs never touch this.
    """
    global _fallback
    if _fallback is None:
        _fallback = create_sandbox(Path("/"), SandboxConfig(backend="local"))
        _fallback.__enter__()
    return _fallback
