"""Phase 0's one PROVEN gate: run the repo's own test command and report its
exit code as executed fact. Later phases add typecheck/build/lint gates
alongside this one — this module stays scoped to "test" on purpose.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from verdict.schema import Provenance, Signal

DEFAULT_TIMEOUT_SECONDS = 600


def detect_test_command(repo: Path) -> str | None:
    """Guess how this repo runs its tests, preferring explicit config over
    guesswork. Returns None if nothing recognizable was found.
    """
    if (repo / "pytest.ini").exists() or (repo / "setup.cfg").exists():
        return "pytest -q"

    pyproject = repo / "pyproject.toml"
    if pyproject.exists() and "[tool.pytest" in pyproject.read_text(errors="ignore"):
        return "pytest -q"

    package_json = repo / "package.json"
    if package_json.exists():
        try:
            data = json.loads(package_json.read_text())
        except json.JSONDecodeError:
            data = {}
        if "test" in data.get("scripts", {}):
            return "npm test --silent"

    tests_dir = repo / "tests"
    if tests_dir.is_dir() and any(tests_dir.glob("test_*.py")):
        return "pytest -q"

    return None


def run_test_gate(
    worktree: Path,
    command: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Signal:
    """Execute the test command inside `worktree` and return a PROVEN
    Signal. If no command could be resolved at all, the gate itself fails
    (this is executed fact too: "no test command found" is deterministic).
    """
    resolved = command or detect_test_command(worktree)
    if resolved is None:
        return Signal(
            name="test",
            provenance=Provenance.PROVEN,
            passed=False,
            detail="no test command configured or autodetected",
            command=None,
        )

    try:
        result = subprocess.run(
            resolved,
            shell=True,
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return Signal(
            name="test",
            provenance=Provenance.PROVEN,
            passed=False,
            detail=f"timed out after {timeout_seconds}s",
            command=resolved,
        )

    output = (result.stdout + result.stderr).strip()
    tail = "\n".join(output.splitlines()[-15:])
    return Signal(
        name="test",
        provenance=Provenance.PROVEN,
        passed=result.returncode == 0,
        detail=tail or "(no output)",
        command=resolved,
        exit_code=result.returncode,
    )
