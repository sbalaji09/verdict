from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _docker_available():
        return
    skip_docker = pytest.mark.skip(reason="no reachable Docker daemon")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip_docker)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A minimal real git repo: a tiny pytest suite with one failing test,
    fixable by writing a correct calculator.py.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "calculator.py").write_text(
        "def add(a, b):\n    return a - b  # bug\n"
    )
    (repo / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "pytest.ini").write_text("[pytest]\n")

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)

    return repo
