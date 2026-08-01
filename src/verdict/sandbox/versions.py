"""Language-version-pin resolution (Phase 10): honor a repo's own
`.python-version`/`.nvmrc` instead of silently running every gate against
whatever the sandbox image's baked-in default happens to be — the "repo
pins Node 18, image defaults to Node 20" case Phase 8's Dockerfile
comments flagged and explicitly left unhandled.

Both version managers are already baked into the image (root `Dockerfile`)
for exactly this purpose; this module is what actually invokes them. Not
implemented here, deliberately: resolving anything beyond a literal
`.python-version`/`.nvmrc` file (e.g. `pyproject.toml`'s
`requires-python`, `package.json`'s `engines.node`, a range like `>=18`
rather than a single pinned version) — see DESIGN.md's Phase 10 section.
"""

from __future__ import annotations

from pathlib import Path

from verdict.sandbox.base import Sandbox, SetupError

_PYENV_ROOT = "/opt/pyenv"
_NVM_DIR = "/opt/nvm"

# Resolving/installing a not-yet-present version needs the internet (pyenv/
# nvm both download a source or prebuilt release) — this is a provisioning
# step, same network exception the dependency-install step already gets.
_RESOLVE_TIMEOUT_SECONDS = 300


def _read_pin(worktree: Path, filename: str) -> str | None:
    path = worktree / filename
    if not path.exists():
        return None
    pin = path.read_text().strip()
    return pin or None


def resolve_version_pins(worktree: Path, sandbox: Sandbox) -> dict[str, str]:
    """Returns an env-var overlay (`PATH` prefix, `PYENV_VERSION`) that
    must be merged into every subsequent `exec()` call for the rest of
    the attempt — install, adapter, every gate — so all of them see the
    pinned version, not the image's default. Empty dict if neither pin
    file is present, which is a complete no-op relative to pre-Phase-10
    behavior (the common case: most repos don't pin, and the image's
    default versions keep being what runs).

    Raises `SetupError` if a pin is present but can't be resolved (pyenv/
    nvm themselves fail, a version string that doesn't parse) — this is
    infrastructure being unable to honor a real, well-formed request, not
    an agent defect, same bucket as every other Phase 10 setup failure.
    """
    overlay: dict[str, str] = {}

    python_pin = _read_pin(worktree, ".python-version")
    if python_pin:
        overlay.update(_resolve_python(python_pin, worktree, sandbox))

    node_pin = _read_pin(worktree, ".nvmrc")
    if node_pin:
        overlay.update(_resolve_node(node_pin, worktree, sandbox))

    return overlay


def _resolve_python(version: str, worktree: Path, sandbox: Sandbox) -> dict[str, str]:
    # pyenv's shims are already on PATH (see Dockerfile) — PYENV_VERSION
    # alone makes every `python`/`pip` invocation resolve to the pinned
    # version, no PATH surgery needed, as long as that version is
    # actually installed first.
    env = {"PYENV_VERSION": version}
    check = sandbox.exec(
        ["python", "--version"], cwd=worktree, env=env, timeout_seconds=15, network=False
    )
    if check.exit_code == 0:
        return env

    install = sandbox.exec(
        ["pyenv", "install", "--skip-existing", version],
        cwd=worktree,
        timeout_seconds=_RESOLVE_TIMEOUT_SECONDS,
        network=True,
    )
    if install.exit_code != 0:
        raise SetupError(
            f".python-version pins {version!r}, which pyenv couldn't install: "
            f"{(install.stderr or install.stdout).strip()}"
        )
    return env


def _resolve_node(version: str, worktree: Path, sandbox: Sandbox) -> dict[str, str]:
    # Unlike pyenv, nvm has no persistent shim directory on PATH in this
    # image (see Dockerfile — PATH is hardcoded to one installed version),
    # so resolving a pin means explicitly sourcing nvm.sh, installing the
    # version if needed, then resolving the concrete `node` binary's
    # directory to prepend onto PATH for every later call — nvm itself
    # isn't invoked again after this.
    source = f". {_NVM_DIR}/nvm.sh"
    install = sandbox.exec(
        ["sh", "-c", f"{source} && nvm install {version}"],
        cwd=worktree,
        timeout_seconds=_RESOLVE_TIMEOUT_SECONDS,
        network=True,
    )
    if install.exit_code != 0:
        raise SetupError(
            f".nvmrc pins {version!r}, which nvm couldn't install: "
            f"{(install.stderr or install.stdout).strip()}"
        )

    which = sandbox.exec(
        ["sh", "-c", f"{source} && nvm which {version}"], cwd=worktree, timeout_seconds=15, network=False
    )
    node_path = which.stdout.strip().splitlines()[-1] if which.stdout.strip() else ""
    if which.exit_code != 0 or not node_path:
        raise SetupError(f".nvmrc pins {version!r}, but nvm couldn't resolve its install path")

    bin_dir = str(Path(node_path).parent)
    return {"PATH": f"{bin_dir}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
