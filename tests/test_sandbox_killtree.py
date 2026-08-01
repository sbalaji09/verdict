"""Phase 9's core containment claim: a timed-out `Sandbox.exec()` call
kills the WHOLE process tree it started, not just the one process it
launched directly. Before this phase, a timeout only killed the immediate
child — anything that child forked (a hung test's own subprocess, a dev
server's node process) kept running as an orphan, potentially still
holding a port.

These tests exercise `LocalSandbox` directly (fast, no Docker required —
`DockerSandbox`'s equivalent containment is exercised in
`test_sandbox_docker_adversarial.py`, Docker-gated). They don't just
trust the killed process's own reported exit code/status: they
independently verify the *effect* — the port is free again, the specific
child PID is gone — the same "check the effect, not just what the process
claims" discipline `test_sandbox_docker_adversarial.py` established for
Phase 8.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

import pytest

from verdict.sandbox.local import LocalSandbox

_CHILD_SCRIPT = """
import os, socket, sys, time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 0))
port = s.getsockname()[1]
with open(sys.argv[1], "w") as f:
    f.write(f"{port} {os.getpid()}")
s.listen(1)
time.sleep(60)
"""

_PARENT_SCRIPT = """
import subprocess, sys, time

# Spawn a grandchild (relative to the sandboxed process) that binds a
# port and sleeps — simulating a dev server / worker a hung test forked —
# then hang forever itself, simulating an agent-introduced infinite loop.
subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
time.sleep(60)
"""


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_marker(marker: Path, deadline_seconds: float = 5.0) -> tuple[int, int]:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        if marker.exists() and marker.stat().st_size > 0:
            port_str, pid_str = marker.read_text().split()
            return int(port_str), int(pid_str)
        time.sleep(0.05)
    raise TimeoutError("grandchild never wrote its marker file — test setup itself is broken")


def test_timeout_kills_the_whole_process_tree_not_just_the_direct_child(tmp_path: Path) -> None:
    child_script = tmp_path / "child.py"
    parent_script = tmp_path / "parent.py"
    marker = tmp_path / "marker.txt"
    child_script.write_text(_CHILD_SCRIPT)
    parent_script.write_text(_PARENT_SCRIPT)

    sandbox = LocalSandbox()
    result = sandbox.exec(
        [sys.executable, str(parent_script), str(child_script), str(marker)],
        cwd=tmp_path,
        timeout_seconds=2,
    )

    assert result.timed_out
    assert result.killed_reason == "timeout"
    assert result.exit_code == 124

    port, grandchild_pid = _wait_for_marker(marker, deadline_seconds=3.0)

    # The grandchild must be dead too, even though `exec()` only directly
    # launched the *parent* script — this is the actual kill-TREE claim,
    # not just "the one process I started is gone."
    deadline = time.monotonic() + 5.0
    while _pid_is_alive(grandchild_pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _pid_is_alive(grandchild_pid), "orphaned grandchild process still running after timeout"

    # And the port it bound must be free again — the concrete "orphan
    # holding a port" failure mode this phase exists to close.
    deadline = time.monotonic() + 5.0
    while not _port_is_free(port) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _port_is_free(port), f"port {port} still held after the sandboxed process tree was killed"


def test_normal_completion_is_unaffected_by_the_kill_tree_machinery(tmp_path: Path) -> None:
    """The process-group rewrite (Phase 9) must not change behavior on the
    ordinary, non-timeout path — same stdout/stderr/exit-code contract as
    before."""
    sandbox = LocalSandbox()
    result = sandbox.exec(["sh", "-c", "echo hello; echo oops >&2; exit 7"], cwd=tmp_path)
    assert result.exit_code == 7
    assert result.stdout.strip() == "hello"
    assert result.stderr.strip() == "oops"
    assert not result.timed_out
    assert result.killed_reason is None


def test_background_dev_server_terminate_kills_its_children_too(tmp_path: Path) -> None:
    """`exec_background` (the frontend dev server path) already used
    process groups before Phase 9 — this locks that contract in against
    regression now that `exec()` shares the same kill helper."""
    child_script = tmp_path / "child.py"
    parent_script = tmp_path / "parent.py"
    marker = tmp_path / "marker.txt"
    child_script.write_text(_CHILD_SCRIPT)
    parent_script.write_text(_PARENT_SCRIPT)

    sandbox = LocalSandbox()
    handle = sandbox.exec_background(
        [sys.executable, str(parent_script), str(child_script), str(marker)], cwd=tmp_path
    )
    try:
        port, grandchild_pid = _wait_for_marker(marker, deadline_seconds=5.0)
        assert not _port_is_free(port)
    finally:
        handle.terminate(grace_seconds=2)

    assert not handle.is_alive()
    deadline = time.monotonic() + 5.0
    while _pid_is_alive(grandchild_pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _pid_is_alive(grandchild_pid)
    deadline = time.monotonic() + 5.0
    while not _port_is_free(port) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _port_is_free(port)


@pytest.mark.docker
def test_docker_sandbox_kill_tree_smoke(tmp_path: Path) -> None:
    """Same claim as the LocalSandbox tests above, against the real
    containment boundary — Docker-gated, auto-skipped without a daemon.
    A lighter smoke test (not the full port/PID introspection, which is
    awkward to do from outside the container) — it just proves a timed-out
    `exec()` call reports itself correctly and doesn't hang the poll loop.
    """
    from verdict.sandbox.docker import DockerSandbox

    with DockerSandbox(tmp_path) as sandbox:
        result = sandbox.exec(["sh", "-c", "sleep 30"], cwd=tmp_path, timeout_seconds=2)
        assert result.timed_out
        assert result.exit_code == 124
        # The container must still be usable for a subsequent command —
        # proof the timeout killed only that one exec's tree, not the
        # whole session (gates 2-4 still need this container afterward).
        follow_up = sandbox.exec(["echo", "still alive"], cwd=tmp_path, timeout_seconds=10)
        assert follow_up.exit_code == 0
        assert follow_up.stdout.strip() == "still alive"
