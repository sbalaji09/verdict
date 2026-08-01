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


def create_sandbox(worktree: Path, config: SandboxConfig | None = None) -> Sandbox:
    """Construct (but do not yet `__enter__`) a `Sandbox` for `worktree`
    per `config`. Callers are expected to use this via `with
    create_sandbox(...) as sandbox:`.
    """
    config = config or SandboxConfig()
    if config.backend == "docker":
        return DockerSandbox(worktree, image=config.image, limits=config.limits, network=config.network)
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
