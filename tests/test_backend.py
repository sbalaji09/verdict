"""Phase 14: mirrors `test_frontend.py`'s approach — a real, tiny,
dependency-free HTTP service (Python's own `http.server`, no Flask/etc.
needed) actually booted inside the sandbox, not mocked. The scenario this
phase exists for: a service that compiles and passes its (unrelated,
trivial) unit tests but crashes on startup must be graded NOT_DONE, not
DONE — something none of `test`/`typecheck`/`build`/`lint` can see on
their own, since none of them ever run the service as a process.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

from verdict.adapters.mock import MockAdapter
from verdict.runner import run
from verdict.sandbox import SandboxConfig
from verdict.schema import AttemptResult, GateStatus, VerdictStatus


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)


_BUGGY_APP_PY = (
    "import sys\n"
    "print('boom: app crashes on startup', file=sys.stderr)\n"
    "sys.exit(1)\n"
)


def _fixed_app_py(port: int) -> str:
    return (
        "import http.server\nimport socketserver\n\n"
        "class Handler(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        if self.path == '/api/widgets':\n"
        "            self.send_response(200)\n"
        "            self.end_headers()\n"
        "            self.wfile.write(b'{\"widgets\": []}')\n"
        "        else:\n"
        "            self.send_response(200)\n"
        "            self.end_headers()\n"
        "            self.wfile.write(b'ok')\n"
        "    def log_message(self, *a):\n"
        "        pass\n\n"
        f'with socketserver.TCPServer(("127.0.0.1", {port}), Handler) as httpd:\n'
        "    httpd.serve_forever()\n"
    )


def _make_backend_repo(tmp_path: Path, port: int, verdict_yml: str) -> Path:
    """A repo whose unit test suite is genuinely unrelated to the backend
    service — it stays green no matter what happens to `app.py`, exactly
    the "compiles and passes unit tests" half of this phase's scenario.
    """
    repo = tmp_path / "backend_repo"
    repo.mkdir()
    (repo / "app.py").write_text(_BUGGY_APP_PY)
    (repo / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "pytest.ini").write_text("[pytest]\n")
    (repo / "verdict.yml").write_text(verdict_yml)
    _init_git_repo(repo)
    return repo


class _NoOpAdapter:
    """Does nothing at all — `app.py` stays broken."""

    name = "noop"

    def run(self, task: str, worktree: Path, sandbox=None) -> AttemptResult:
        return AttemptResult(diff="", tokens_input=1, tokens_output=1, cost_usd=0.001)


def _signal(verdict, name: str):
    return next(s for s in verdict.signals if s.name == name)


def test_compiles_and_unit_passes_but_fails_to_boot_is_not_done(tmp_path: Path) -> None:
    port = _free_port()
    repo = _make_backend_repo(
        tmp_path,
        port,
        f"""
backend:
  start: "{sys.executable} app.py"
  health_url: "http://127.0.0.1:{port}/healthz"
  ready_timeout_seconds: 5
""",
    )

    verdict = run(
        task="do nothing useful",
        repo=repo,
        adapter=_NoOpAdapter(),  # type: ignore[arg-type]
        sandbox_config=SandboxConfig(backend="local"),
    )

    test_signal = _signal(verdict, "test")
    assert test_signal.status is GateStatus.PASS  # unit tests are genuinely unrelated, still green

    boot_signal = _signal(verdict, "backend:boot")
    assert boot_signal.status is GateStatus.FAIL
    assert "exited early" in boot_signal.detail

    # The whole point of this phase: a green test suite is not enough.
    assert verdict.status is VerdictStatus.NOT_DONE
    assert verdict.done is False


def test_a_real_fix_that_boots_and_passes_smoke_reaches_done(tmp_path: Path) -> None:
    port = _free_port()
    repo = _make_backend_repo(
        tmp_path,
        port,
        f"""
backend:
  start: "{sys.executable} app.py"
  health_url: "http://127.0.0.1:{port}/healthz"
  ready_timeout_seconds: 10
  smoke:
    - path: "/api/widgets"
      expect_status: 200
      expect_body_contains: "widgets"
""",
    )

    adapter = MockAdapter(patches={"app.py": _fixed_app_py(port)})
    verdict = run(
        task="fix the service so it boots",
        repo=repo,
        adapter=adapter,
        sandbox_config=SandboxConfig(backend="local"),
    )

    boot_signal = _signal(verdict, "backend:boot")
    assert boot_signal.status is GateStatus.PASS

    smoke_signal = _signal(verdict, "backend:smoke:GET /api/widgets")
    assert smoke_signal.status is GateStatus.PASS

    assert verdict.status is VerdictStatus.DONE


def test_a_smoke_check_that_answers_wrong_is_not_done_even_though_boot_succeeds(
    tmp_path: Path,
) -> None:
    port = _free_port()
    repo = _make_backend_repo(
        tmp_path,
        port,
        f"""
backend:
  start: "{sys.executable} app.py"
  health_url: "http://127.0.0.1:{port}/healthz"
  ready_timeout_seconds: 10
  smoke:
    - path: "/api/widgets"
      expect_status: 200
      expect_body_contains: "this substring will never appear"
""",
    )

    adapter = MockAdapter(patches={"app.py": _fixed_app_py(port)})
    verdict = run(
        task="fix the service", repo=repo, adapter=adapter, sandbox_config=SandboxConfig(backend="local")
    )

    boot_signal = _signal(verdict, "backend:boot")
    assert boot_signal.status is GateStatus.PASS  # it DID boot correctly

    smoke_signal = _signal(verdict, "backend:smoke:GET /api/widgets")
    assert smoke_signal.status is GateStatus.FAIL

    assert verdict.status is VerdictStatus.NOT_DONE


def test_no_backend_signals_without_backend_config(git_repo: Path) -> None:
    verdict = run(
        task="fix it",
        repo=git_repo,
        adapter=MockAdapter(patches={"calculator.py": "def add(a, b):\n    return a + b\n"}),
        sandbox_config=SandboxConfig(backend="local"),
    )
    assert not any(s.name.startswith("backend") for s in verdict.signals)


def test_a_failing_migration_skips_boot_and_smoke_entirely(tmp_path: Path) -> None:
    port = _free_port()
    repo = _make_backend_repo(
        tmp_path,
        port,
        f"""
backend:
  start: "{sys.executable} app.py"
  health_url: "http://127.0.0.1:{port}/healthz"
  ready_timeout_seconds: 5
  migrate: "false"
""",
    )

    verdict = run(
        task="do nothing",
        repo=repo,
        adapter=_NoOpAdapter(),  # type: ignore[arg-type]
        sandbox_config=SandboxConfig(backend="local"),
    )

    migrate_signal = _signal(verdict, "backend:migrate")
    assert migrate_signal.status is GateStatus.FAIL
    assert not any(s.name == "backend:boot" for s in verdict.signals)
    assert verdict.status is VerdictStatus.NOT_DONE


def test_a_successful_migration_runs_before_boot(tmp_path: Path) -> None:
    port = _free_port()
    repo = _make_backend_repo(
        tmp_path,
        port,
        f"""
backend:
  start: "{sys.executable} app.py"
  health_url: "http://127.0.0.1:{port}/healthz"
  ready_timeout_seconds: 10
  migrate: "true"
""",
    )

    adapter = MockAdapter(patches={"app.py": _fixed_app_py(port)})
    verdict = run(
        task="fix it", repo=repo, adapter=adapter, sandbox_config=SandboxConfig(backend="local")
    )

    migrate_signal = _signal(verdict, "backend:migrate")
    assert migrate_signal.status is GateStatus.PASS
    boot_signal = _signal(verdict, "backend:boot")
    assert boot_signal.status is GateStatus.PASS
    assert verdict.status is VerdictStatus.DONE
