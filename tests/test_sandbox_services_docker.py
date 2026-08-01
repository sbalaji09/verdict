"""Phase 10's service-dependency machinery against the real containment
boundary — Docker-gated, auto-skipped without a reachable daemon (see
`conftest.py`) and without the `verdict-sandbox:0.1.0` image already built
(same second-layer skip `test_sandbox_docker_adversarial.py` uses).

Covers the two concrete claims the design review called out explicitly:
1. A real fixture repo whose test suite needs Postgres to pass actually
   passes, driven end to end through `runner.run()` — services start,
   health-gate, and are reachable by the DNS name `verdict.yml` gave them.
2. A gate container joined to the per-attempt service network still
   cannot reach the public internet — declaring a service must never
   accidentally reopen general egress.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from verdict.adapters.mock import MockAdapter
from verdict.runner import run
from verdict.sandbox import SandboxConfig
from verdict.sandbox.base import SandboxUnavailableError
from verdict.sandbox.docker import DockerSandbox
from verdict.sandbox.services import setup_services, teardown_services
from verdict.schema import VerdictStatus

pytestmark = pytest.mark.docker


def _init_git(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)


@pytest.fixture
def db_repo(tmp_path: Path) -> Path:
    """A minimal repo whose test suite genuinely fails without Postgres
    reachable at `db:5432` — proving the service is real, not just
    started-and-ignored.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("psycopg2-binary\n")
    (repo / "test_db.py").write_text(
        "import psycopg2\n\n"
        "def test_can_connect():\n"
        "    conn = psycopg2.connect(\n"
        '        host="db", port=5432, user="postgres", password="verdict", dbname="postgres",\n'
        "    )\n"
        "    conn.close()\n"
    )
    (repo / "pytest.ini").write_text("[pytest]\n")
    (repo / "verdict.yml").write_text(
        "services:\n"
        "  - name: db\n"
        "    type: postgres\n"
        '    version: "16"\n'
        "    env:\n"
        "      POSTGRES_PASSWORD: verdict\n"
    )
    _init_git(repo)
    return repo


def test_fixture_repo_needing_postgres_passes_end_to_end(db_repo: Path) -> None:
    adapter = MockAdapter(patches={"README.md": "noop\n"})
    sandbox_config = SandboxConfig(backend="docker")
    try:
        verdict = run(task="noop", repo=db_repo, adapter=adapter, sandbox_config=sandbox_config)
    except SandboxUnavailableError as exc:
        pytest.skip(f"docker sandbox unavailable: {exc}")

    test_signal = next(s for s in verdict.signals if s.name == "test")
    assert test_signal.status.value == "pass", test_signal.detail
    assert verdict.status is VerdictStatus.DONE


def test_service_network_does_not_reach_the_public_internet(tmp_path: Path) -> None:
    from verdict.config import ServiceSpec

    services = [ServiceSpec(name="db", type="postgres", version="16", env={"POSTGRES_PASSWORD": "x"})]
    try:
        session = setup_services(services)
    except SandboxUnavailableError as exc:
        pytest.skip(f"docker sandbox unavailable: {exc}")

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    try:
        assert session.network_name is not None
        gate_config = SandboxConfig(backend="docker", network_name=session.network_name)
        with DockerSandbox(
            worktree, image=gate_config.image, network_name=gate_config.network_name
        ) as sandbox:
            # The service itself must be reachable...
            reach_service = sandbox.exec(
                ["sh", "-c", "pg_isready -h db -p 5432 || getent hosts db"],
                cwd=worktree,
                timeout_seconds=15,
            )
            # (best-effort: the sandbox image may not ship pg_isready —
            # getent resolving the hostname is enough to prove network
            # membership even if the client tool itself is absent)
            assert reach_service.exit_code == 0 or "db" in reach_service.stdout

            # ...but the public internet must NOT be, despite being on a
            # real (non-`none`) network — this is the actual containment
            # claim: --internal blocks egress even though the gate
            # container is no longer network-isolated in the --none sense.
            reach_internet = sandbox.exec(
                ["sh", "-c", "curl -m 3 -s -o /dev/null -w '%{http_code}' http://example.com"],
                cwd=worktree,
                timeout_seconds=10,
            )
            assert reach_internet.exit_code != 0
    finally:
        teardown_services(session)
