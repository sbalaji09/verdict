"""Base-state cache (Phase 10): the repo's pre-agent state at
`worktree.base_commit` gets rendered, installed, and graded identically
every time it's needed. Before this phase, two independent call sites
each redid this from scratch on every use:

- `attribution/engine.py`'s baseline "was this already failing before the
  agent touched anything" check (Phase 2) — called once per failing
  signal being attributed, so up to `MAX_ATTRIBUTIONS_PER_GATE` times
  *per gate* in a single attempt, every one of them rendering the exact
  same `base_commit` from scratch.
- `frontend/runner.py`'s before-image capture (Phase 4) — rendered fresh
  on every `run_frontend_checks` call, even across `run_with_retries`
  attempts that share the same `base_commit`.

This module is deliberately just the cache primitives — key computation
plus host-filesystem read/write for two independent artifact kinds (gate
signals, screenshots). It does NOT orchestrate rendering itself: each
consumer still owns *how* to render its own artifact (running gates vs.
driving a real browser), it just asks this module "do I already have
this" first and "here's what I computed" after a miss. Kept this way on
purpose to avoid coupling this module to Playwright/browser types, which
belong to the (optional) `frontend` extra, not to the sandbox package.

A HOST filesystem cache, not anything living inside a container — it has
to survive across separate `docker run` sessions (each of the redundant
calls above used to open its own) and across separate `verdict` CLI
invocations entirely (a suite re-grading the same repo HEAD many times
over should reuse this too). Keyed on `(base_commit, lockfile_hash,
sandbox image tag)` — the image tag is part of the key because a
different image can carry different tool versions and so produce
different real results; base_commit + lockfile_hash alone would let a
stale cache entry silently survive a `verdict-sandbox` image upgrade.
No eviction/TTL in this phase — stale entries simply accumulate under
`--cache-dir`; see DESIGN.md's Phase 10 section for what's deferred.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from verdict.schema import Signal

DEFAULT_CACHE_DIR = Path(os.environ.get("VERDICT_CACHE_DIR", str(Path.home() / ".cache" / "verdict"))) / (
    "base-state"
)

# Every lockfile/manifest name this module knows to hash. Order doesn't
# matter for the resulting hash (each entry is folded in independently,
# name-prefixed) but does need to stay in sync with what
# `sandbox/install.py::_detect_install_command` actually looks for, or a
# lockfile change that changes install behavior could go unnoticed by the
# cache key.
_LOCKFILE_NAMES = (
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "uv.lock",
    "requirements.txt",
    "go.sum",
)


def _git_show(repo: Path, ref: str, path: str) -> bytes | None:
    """`None` if `path` doesn't exist at `ref` — distinct from an empty
    file, which still contributes its (empty) content to the hash.
    """
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=repo, capture_output=True, timeout=30
    )
    return result.stdout if result.returncode == 0 else None


def compute_lockfile_hash(repo: Path, ref: str) -> str:
    """Hashes whichever known lockfiles exist AT `ref`, via `git show` —
    no worktree checkout needed just to compute this. A repo with no
    lockfiles at all still gets a stable (constant) hash rather than an
    error, so gate autodetection alone (no lockfile, e.g. a pure-stdlib
    Python repo) still participates in caching.
    """
    hasher = hashlib.sha256()
    for name in _LOCKFILE_NAMES:
        content = _git_show(repo, ref, name)
        if content is not None:
            hasher.update(name.encode())
            hasher.update(content)
    return hasher.hexdigest()[:16]


def cache_key(base_commit: str, lockfile_hash: str, image: str) -> str:
    image_tag = image.replace("/", "_").replace(":", "_")
    return f"{base_commit}-{lockfile_hash}-{image_tag}"


def _entry_dir(cache_root: Path, key: str) -> Path:
    return cache_root / key


def load_gate_signals(cache_root: Path, key: str) -> dict[str, Signal] | None:
    path = _entry_dir(cache_root, key) / "gate_signals.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        return {name: Signal.model_validate(data) for name, data in raw.items()}
    except (json.JSONDecodeError, KeyError, OSError, ValueError):
        # A corrupted/partially-written cache entry is treated as a plain
        # miss — recompute and overwrite, never crash a grading run over
        # a cache-layer problem.
        return None


def save_gate_signals(cache_root: Path, key: str, signals: dict[str, Signal]) -> None:
    entry = _entry_dir(cache_root, key)
    entry.mkdir(parents=True, exist_ok=True)
    raw = {name: json.loads(signal.model_dump_json()) for name, signal in signals.items()}
    (entry / "gate_signals.json").write_text(json.dumps(raw))


def load_screenshots(cache_root: Path, key: str) -> dict[int, bytes] | None:
    shots_dir = _entry_dir(cache_root, key) / "screenshots"
    if not shots_dir.is_dir():
        return None
    try:
        return {int(p.stem): p.read_bytes() for p in shots_dir.glob("*.png")}
    except (OSError, ValueError):
        return None


def save_screenshots(cache_root: Path, key: str, screenshots: dict[int, bytes]) -> None:
    shots_dir = _entry_dir(cache_root, key) / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    for viewport, png_bytes in screenshots.items():
        (shots_dir / f"{viewport}.png").write_bytes(png_bytes)
