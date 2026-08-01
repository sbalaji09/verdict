"""Adversarial containment tests for `DockerSandbox` (Phase 8's actual
security boundary — `LocalSandbox` makes no isolation claim, so these
don't apply to it). Auto-skipped when no Docker daemon is reachable — see
`conftest.py::_docker_available`.

Each test simulates a "malicious repo" whose own code — the exact
situation an untrusted coding agent or PR represents — tries to (a) read a
secret env var, (b) reach the network, (c) write outside the mounted
worktree, and asserts the attempt is blocked or contained. Effects are
checked independently of the malicious process's own exit code/output
where practical (see the filesystem-escape test), since a compromised
process could lie about what it did.

Requires `verdict-sandbox:0.1.0` (this repo's `Dockerfile`) to already be
built and available to the local Docker daemon — these tests don't build
it themselves (that's a real, slow operation with its own separate
concerns), so a missing image surfaces as a `SandboxUnavailableError`-driven
failure rather than a silent skip. Build it once with:
    docker build -t verdict-sandbox:0.1.0 .
"""

from __future__ import annotations

from pathlib import Path

import pytest

from verdict.sandbox.base import SandboxUnavailableError
from verdict.sandbox.docker import DockerSandbox

pytestmark = pytest.mark.docker


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    wt = tmp_path / "worktree"
    wt.mkdir()
    return wt


def _sandbox(worktree: Path, network: bool = False) -> DockerSandbox:
    try:
        return DockerSandbox(worktree, network=network)
    except SandboxUnavailableError as exc:
        pytest.skip(f"docker sandbox unavailable: {exc}")


def test_malicious_repo_cannot_read_a_host_secret_env_var(
    worktree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERDICT_SECRET_PROBE", "leaked-value")

    with _sandbox(worktree) as sandbox:
        # The secret lives in *this test process's* environment — Phase 8's
        # contract is that nothing is forwarded unless `env=` explicitly
        # names it, so the container must never see it regardless of what
        # the host process has set.
        result = sandbox.exec(
            ["sh", "-c", 'echo "${VERDICT_SECRET_PROBE:-UNSET}"'], cwd=worktree
        )
    assert result.stdout.strip() == "UNSET"
    assert "leaked-value" not in result.stdout


def test_malicious_repo_cannot_reach_the_network(worktree: Path) -> None:
    with _sandbox(worktree, network=False) as sandbox:
        result = sandbox.exec(
            ["sh", "-c", "curl -m 3 -s -o /dev/null -w '%{http_code}' http://example.com"],
            cwd=worktree,
            timeout_seconds=10,
        )
    # --network none means no network interface at all — curl fails fast
    # (couldn't resolve/connect), never gets a real HTTP response.
    assert result.exit_code != 0


def test_malicious_repo_cannot_write_outside_the_worktree(worktree: Path) -> None:
    marker = "/etc/verdict-pwned-marker"
    with _sandbox(worktree) as sandbox:
        result = sandbox.exec(["sh", "-c", f"echo pwned > {marker}"], cwd=worktree)
        # The write itself must fail (read-only rootfs) ...
        assert result.exit_code != 0
        check = sandbox.exec(["sh", "-c", f"test -f {marker}"], cwd=worktree)
        assert check.exit_code != 0

    # ... and, independent of anything the container reported, the actual
    # host filesystem must be untouched — never trust a possibly-
    # compromised process's own exit code as the only evidence.
    assert not Path(marker).exists()


def test_malicious_repo_can_still_write_inside_the_worktree(worktree: Path) -> None:
    """The one legitimate write path must keep working — containment that
    also broke the agent's ability to edit its own checkout wouldn't be a
    usable sandbox."""
    with _sandbox(worktree) as sandbox:
        result = sandbox.exec(["sh", "-c", "echo hello > output.txt"], cwd=worktree)
        assert result.exit_code == 0
    assert (worktree / "output.txt").read_text().strip() == "hello"


def test_sandbox_is_torn_down_on_exit(worktree: Path) -> None:
    sandbox = _sandbox(worktree)
    with sandbox:
        container = sandbox._container  # noqa: SLF001 - white-box teardown check
        assert container is not None
    assert sandbox._container is None  # noqa: SLF001
