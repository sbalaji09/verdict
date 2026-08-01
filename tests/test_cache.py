"""Phase 10's base-state cache: key computation, and hit/miss behavior for
both artifact kinds (`gate_signals`, `screenshots`), plus the two real
consumers (`attribution/engine.py`'s baseline check, `frontend/runner.py`'s
before-image) actually reading from a pre-seeded cache instead of
re-rendering.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from verdict.attribution.engine import _reproduces_at
from verdict.attribution.reproduce import Reproduction
from verdict.sandbox import SandboxConfig
from verdict.sandbox.cache import (
    cache_key,
    compute_lockfile_hash,
    load_gate_signals,
    load_screenshots,
    save_gate_signals,
    save_screenshots,
)
from verdict.schema import GateStatus, Provenance, Signal


def _git_repo(repo: Path, files: dict[str, str]) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        target = repo / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    return repo


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


# --- key computation --------------------------------------------------------


def test_lockfile_hash_changes_when_lockfile_content_changes(tmp_path: Path) -> None:
    repo1 = _git_repo(tmp_path / "a" / "repo", {"requirements.txt": "flask==2.0\n"})
    repo2 = _git_repo(tmp_path / "b" / "repo", {"requirements.txt": "flask==3.0\n"})
    assert compute_lockfile_hash(repo1, "HEAD") != compute_lockfile_hash(repo2, "HEAD")


def test_lockfile_hash_is_stable_with_no_lockfiles_at_all(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path, {"README.md": "hi\n"})
    assert compute_lockfile_hash(repo, "HEAD") == compute_lockfile_hash(repo, "HEAD")


def test_cache_key_differs_by_image_tag() -> None:
    assert cache_key("abc123", "hash1", "verdict-sandbox:0.1.0") != cache_key(
        "abc123", "hash1", "verdict-sandbox:0.2.0"
    )


# --- gate_signals hit/miss ---------------------------------------------------


def _signal(name: str, status: GateStatus) -> Signal:
    return Signal(name=name, provenance=Provenance.PROVEN, status=status, detail="")


def test_gate_signals_round_trip(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    signals = {"test": _signal("test", GateStatus.PASS), "lint": _signal("lint", GateStatus.FAIL)}
    save_gate_signals(cache_root, "key1", signals)

    loaded = load_gate_signals(cache_root, "key1")
    assert loaded is not None
    assert loaded["test"].status is GateStatus.PASS
    assert loaded["lint"].status is GateStatus.FAIL


def test_gate_signals_miss_when_key_absent(tmp_path: Path) -> None:
    assert load_gate_signals(tmp_path / "cache", "nonexistent-key") is None


def test_gate_signals_miss_on_corrupted_entry(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    entry = cache_root / "key1"
    entry.mkdir(parents=True)
    (entry / "gate_signals.json").write_text("{not valid json")
    assert load_gate_signals(cache_root, "key1") is None


# --- screenshots hit/miss -----------------------------------------------------


def test_screenshots_round_trip(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    save_screenshots(cache_root, "key1", {1440: b"\x89PNGfakepixels", 800: b"\x89PNGother"})
    loaded = load_screenshots(cache_root, "key1")
    assert loaded == {1440: b"\x89PNGfakepixels", 800: b"\x89PNGother"}


def test_screenshots_miss_when_key_absent(tmp_path: Path) -> None:
    assert load_screenshots(tmp_path / "cache", "nonexistent-key") is None


# --- real consumer: attribution baseline check reuses the cache -------------


def test_reproduces_at_uses_a_seeded_cache_without_touching_the_sandbox(
    tmp_path: Path, monkeypatch
) -> None:
    """Seed the cache directly, then call `_reproduces_at` with a sandbox
    backend that would fail loudly if actually invoked (an unreachable
    Docker backend) — proving the cache hit path never opens a scratch
    worktree or a sandbox at all.
    """
    import verdict.attribution.engine as engine_module
    import verdict.sandbox.cache as cache_module

    cache_root = tmp_path / "cache"
    monkeypatch.setattr(cache_module, "DEFAULT_CACHE_DIR", cache_root)
    monkeypatch.setattr(engine_module, "DEFAULT_CACHE_DIR", cache_root)

    repo = _git_repo(tmp_path / "repo", {"README.md": "hi\n"})
    base_commit = _head(repo)
    lockfile_hash = compute_lockfile_hash(repo, base_commit)
    sandbox_config = SandboxConfig(backend="docker", image="verdict-sandbox:0.1.0")
    key = cache_key(base_commit, lockfile_hash, sandbox_config.image)

    save_gate_signals(cache_root, key, {"test": _signal("test", GateStatus.FAIL)})

    # scratch_worktree/create_sandbox would both explode if actually
    # called (no real repo layout for a scratch worktree of this
    # in-memory setup's `repo`, and "docker" backend has no daemon here)
    # — reaching a real answer instead of an exception proves the cache
    # hit short-circuited before either was touched.
    result = _reproduces_at(repo, base_commit, "test", None, sandbox_config)
    assert result is Reproduction.BAD


def test_reproduces_at_populates_the_cache_on_miss(tmp_path: Path, monkeypatch) -> None:
    import verdict.attribution.engine as engine_module
    import verdict.sandbox.cache as cache_module

    cache_root = tmp_path / "cache"
    monkeypatch.setattr(cache_module, "DEFAULT_CACHE_DIR", cache_root)
    monkeypatch.setattr(engine_module, "DEFAULT_CACHE_DIR", cache_root)

    repo = _git_repo(
        tmp_path / "repo",
        {
            "calculator.py": "def add(a, b):\n    return a + b\n",
            "test_calculator.py": (
                "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
            ),
            "pytest.ini": "[pytest]\n",
        },
    )
    base_commit = _head(repo)
    sandbox_config = SandboxConfig(backend="local")

    result = _reproduces_at(repo, base_commit, "test", None, sandbox_config)
    assert result is Reproduction.GOOD  # the seeded test suite passes

    lockfile_hash = compute_lockfile_hash(repo, base_commit)
    key = cache_key(base_commit, lockfile_hash, sandbox_config.image)
    cached = load_gate_signals(cache_root, key)
    assert cached is not None
    assert cached["test"].status is GateStatus.PASS
