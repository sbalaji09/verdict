"""Phase 10's setup stage: resolve language-version pins, then run a
lockfile-aware dependency install — both inside ONE network-on sandbox
session, separate from the one gates run in (which stays network OFF for
the entire run — see `runner.py`).

Detection is still not exhaustive (see DESIGN.md's Phase 10 section for
what's explicitly still out of scope), but covers the common lockfile
shapes for Node (npm/yarn/pnpm), Python (pip/poetry/uv), and Go, and
prefers a lockfile-respecting install (`npm ci`, `--frozen-lockfile`,
`--frozen`) over a loose one wherever a lockfile is present — a repo that
committed a lockfile is asking for exactly what it locked, not whatever
the latest compatible versions happen to resolve to today.

Phase 9 drew one line through this module's error handling, unchanged by
Phase 10: a sandbox that can't be provisioned AT ALL (no Docker daemon,
image missing) degrades silently — gates downstream will honestly surface
whatever missing-dependency consequence that has. A setup step that
actively FAILS — an install command timing out, a version pin pyenv/nvm
can't resolve — does not degrade silently; it raises
(`ProvisioningTimeoutError`/`SetupError`, both under `SandboxUnavailableError`)
and aborts the attempt, the same "infra, not the agent" treatment every
other Phase 9/10 setup failure gets.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from verdict.sandbox.base import ProvisioningTimeoutError, SandboxError, SetupError
from verdict.sandbox.config import SandboxConfig, create_sandbox
from verdict.sandbox.versions import resolve_version_pins


def _detect_install_command(worktree: Path) -> list[str] | None:
    """Only fires when the dependency directory looks genuinely absent —
    if `copy_vendored_dependencies` (worktree.py) already populated one,
    installing again would be redundant work this phase doesn't need to
    do, and (for npm) can be slower/flakier than just using what's there.

    Lockfile presence picks the install flavor once a language's own
    marker file is found — `npm ci`/`--frozen-lockfile`/`--frozen` are all
    preferred over their loose equivalents specifically because they fail
    loudly on a lockfile/manifest mismatch instead of silently resolving
    something adjacent to what's committed.
    """
    if (worktree / "package.json").exists() and not (worktree / "node_modules").exists():
        if (worktree / "package-lock.json").exists():
            return ["npm", "ci"]
        if (worktree / "pnpm-lock.yaml").exists():
            return ["pnpm", "install", "--frozen-lockfile"]
        if (worktree / "yarn.lock").exists():
            return ["yarn", "install", "--frozen-lockfile"]
        return ["npm", "install"]

    has_venv = (worktree / ".venv").exists()
    if (worktree / "poetry.lock").exists() and not has_venv:
        return ["poetry", "install", "--no-interaction"]
    if (worktree / "uv.lock").exists() and not has_venv:
        return ["uv", "sync", "--frozen"]
    if (worktree / "requirements.txt").exists() and not has_venv:
        return ["pip", "install", "--user", "-r", "requirements.txt"]

    if (worktree / "go.mod").exists() and not (worktree / "vendor").exists():
        return ["go", "mod", "download"]

    return None


def run_setup_step(worktree: Path, config: SandboxConfig) -> dict[str, str]:
    """Resolves `.python-version`/`.nvmrc` pins, then runs the detected
    install command with that resolution already applied, both inside one
    network-on sandbox session. Returns the resulting env-var overlay
    (`PATH`/`PYENV_VERSION`) — empty if nothing was pinned — that the
    caller (`runner.py`) must merge into the adapter's and every gate's
    `exec()` calls for the rest of the attempt, so the pinned version is
    what actually runs everywhere, not just during this one step.
    """
    setup_config = dataclasses.replace(config, network=True)
    overlay: dict[str, str] = {}
    try:
        with create_sandbox(worktree, setup_config) as sandbox:
            overlay = resolve_version_pins(worktree, sandbox)
            command = _detect_install_command(worktree)
            if command is not None:
                result = sandbox.exec(
                    command,
                    cwd=worktree,
                    env=overlay,
                    network=True,
                    timeout_seconds=config.install_timeout_seconds,
                )
                if result.timed_out:
                    raise ProvisioningTimeoutError(
                        f"dependency install ({' '.join(command)}) timed out after "
                        f"{config.install_timeout_seconds}s"
                    )
    except (ProvisioningTimeoutError, SetupError):
        raise
    except SandboxError:
        return {}
    return overlay
