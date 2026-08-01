"""A minimal, explicitly scoped dependency-install step: detect an install
command, run it with network ON, in its own sandbox session separate from
the one gates run in (which stays network OFF for the entire run — see
`runner.py`).

Deliberately minimal per Phase 8's scope (see DESIGN.md): detection only
covers the common "no lockfile-free install directory present yet" case,
just enough that `examples/sample_node_repo` can demo without a
pre-vendored `node_modules`. Explicitly OUT of scope here, deferred to
Phase 10: broader/more accurate autodetection, dependency caching across
runs, resolving a repo's pinned language version (the fat image's
`pyenv`/`nvm` exist for this, unused so far), and service dependencies
(databases, etc.).

Phase 9 draws one explicit line through this module's error handling: a
sandbox that can't be provisioned AT ALL (no Docker daemon, image missing)
degrades silently, same as before — gates downstream will honestly surface
whatever missing-dependency consequence that has, and there's nothing more
specific to say. But a install COMMAND that hangs (`npm install` stuck on
a broken registry, say) is provisioning infrastructure timing out, not the
agent's fault to grade — see DESIGN.md's Phase 9 section — so that specific
case raises `ProvisioningTimeoutError` and aborts the attempt instead of
being swallowed into "well, gates will notice eventually."
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from verdict.sandbox.base import ProvisioningTimeoutError, SandboxError
from verdict.sandbox.config import SandboxConfig, create_sandbox


def _detect_install_command(worktree: Path) -> list[str] | None:
    """Only fires when the dependency directory looks genuinely absent —
    if `copy_vendored_dependencies` (worktree.py) already populated one,
    installing again would be redundant work this phase doesn't need to
    do, and (for npm) can be slower/flakier than just using what's there.
    """
    if (worktree / "package.json").exists() and not (worktree / "node_modules").exists():
        return ["npm", "install"]
    if (worktree / "requirements.txt").exists() and not (worktree / ".venv").exists():
        return ["pip", "install", "--user", "-r", "requirements.txt"]
    if (worktree / "go.mod").exists() and not (worktree / "vendor").exists():
        return ["go", "mod", "download"]
    return None


def run_install_step(worktree: Path, config: SandboxConfig) -> None:
    """A missing tool, unreachable daemon, or unreachable network — the
    sandbox simply couldn't be provisioned at all — degrades silently to
    "dependencies weren't installed," which gates downstream will then
    honestly report as their own real failure (a missing binary, an import
    error). A provisioning TIMEOUT is different and is not swallowed here
    — see module docstring: it raises `ProvisioningTimeoutError`, aborting
    the whole attempt the same way an adapter CLI hanging already does.
    """
    command = _detect_install_command(worktree)
    if command is None:
        return

    install_config = dataclasses.replace(config, network=True)
    try:
        with create_sandbox(worktree, install_config) as sandbox:
            result = sandbox.exec(
                command, cwd=worktree, network=True, timeout_seconds=config.install_timeout_seconds
            )
    except ProvisioningTimeoutError:
        raise
    except SandboxError:
        return

    if result.timed_out:
        raise ProvisioningTimeoutError(
            f"dependency install ({' '.join(command)}) timed out after "
            f"{config.install_timeout_seconds}s"
        )
