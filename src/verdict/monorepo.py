"""Monorepo package selection — "ask, don't guess."

Every gate autodetector through Phase 18 assumed the repo root itself is
the one project to grade. Phase 19 lifts that assumption for repos that
aren't shaped that way (multiple independent packages under one git repo)
without changing behavior for the single-project repos every earlier phase
and fixture already relies on: `resolve_package` returns `None` — "run
against the worktree root, exactly as before" — unless something in the
repo or the caller actually asks for a specific package.

The one new failure mode this module introduces is `PackageSelectionError`,
raised only when the correct package genuinely can't be inferred (multiple
sibling candidates, or an explicitly requested package that doesn't exist).
`runner.py` treats it as an evaluation error (`_EVALUATION_ERRORS`) — a
config problem, not an agent failure — same bucket as a sandbox that never
came up: never silently misgraded, never blamed on the agent.
"""

from __future__ import annotations

from pathlib import Path

from verdict.config import VerdictConfig

PROJECT_MARKERS = (
    "pyproject.toml",
    "setup.cfg",
    "pytest.ini",
    "package.json",
    "go.mod",
    "Cargo.toml",
    "Makefile",
    "makefile",
    "build.gradle",
    "build.gradle.kts",
    "WORKSPACE",
    "WORKSPACE.bazel",
    "BUILD.bazel",
)
"""Files whose presence in a directory marks it as looking like its own,
independent project root — the same signal a human skimming a repo tree
would use. Deliberately coarse (existence, not content) — this is only
used to decide whether picking a package is *ambiguous*, never to decide
which tool actually runs; that's still each gate's own `applicable()`."""

IGNORED_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".next",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "target",
        "vendor",
    }
)


class PackageSelectionError(Exception):
    """Raised instead of silently grading the wrong (or no) package — every
    message names exactly what's ambiguous and how to resolve it (pass
    `--package`, or add a `packages:` block to verdict.yml). The same "say
    the true thing instead of guessing" discipline `GateStatus.NA` already
    applies to a single missing stack, one level up: applied here to "which
    directory is even the project."
    """


def _has_markers(path: Path) -> bool:
    return any((path / marker).exists() for marker in PROJECT_MARKERS)


MAX_CANDIDATE_DEPTH = 2
"""How many directory levels below the worktree root this scan will
descend looking for a project marker. 2, not 1, because the two most
common real monorepo shapes — a package directly under the root
(`api/pyproject.toml`) and a package one level under a grouping folder
(`services/api/pyproject.toml`, `apps/web/package.json` — the Turborepo/
Nx convention) — both need to be found by the same scan. A directory with
its own markers is a leaf for this purpose: descending further into it
would just rediscover a nested test fixture or vendored dependency as a
spurious extra "candidate," not a second real package.
"""


def detect_sibling_candidates(worktree: Path) -> list[str]:
    """Directories (up to `MAX_CANDIDATE_DEPTH` deep) that look like their
    own independent project root. Only meaningful — and only ever
    consulted by `resolve_package` — when the worktree root itself has
    none of `PROJECT_MARKERS`; a repo with, say, a root `pyproject.toml`
    AND a `docs/` folder that happens to contain a stray `Makefile` is a
    normal single-project repo, not an ambiguous monorepo, and never
    reaches this scan.
    """
    if not worktree.is_dir():
        return []

    candidates: list[str] = []

    def _scan(directory: Path, relative: str, depth: int) -> None:
        for child in sorted(directory.iterdir()):
            if not child.is_dir() or child.name in IGNORED_DIRS or child.name.startswith("."):
                continue
            child_relative = f"{relative}/{child.name}" if relative else child.name
            if _has_markers(child):
                candidates.append(child_relative)
            elif depth < MAX_CANDIDATE_DEPTH:
                _scan(child, child_relative, depth + 1)

    _scan(worktree, "", 1)
    return candidates


def resolve_package(worktree: Path, config: VerdictConfig, requested: str | None) -> str | None:
    """Returns the relative path (from `worktree`) gates should be resolved
    and run against, or `None` for "the worktree root" — today's behavior,
    unchanged for every repo this doesn't apply to.

    Resolution order, each step preferring an explicit answer over a guess:

    1. `requested` (`--package`, or whatever a caller passes) always wins
       when given — it must name a real directory, or this raises rather
       than silently falling through to autodetection.
    2. A `packages:` block in verdict.yml naming exactly one package is
       taken as that package's own explicit, unambiguous declaration —
       naming more than one means the caller has to say which, since
       there's no "current package" concept a task implicitly targets.
    3. No `packages:` block and the worktree root itself looks like a
       project (has one of `PROJECT_MARKERS`) → `None`, exactly the
       pre-Phase-19 behavior for every existing single-project fixture.
    4. No `packages:` block, no markers at the root, and two or more
       sibling directories each look like independent projects → genuinely
       ambiguous, raises rather than picking one arbitrarily.
    5. No `packages:` block, no root markers, at most one sibling
       candidate → `None`. A single nested project isn't a choice between
       alternatives, so there's nothing to disambiguate; the gates simply
       autodetect (and correctly report NA) against the worktree root as
       they always have. Pointing at that one nested directory is still
       available via `--package` if that's what's wanted.
    """
    declared = sorted(config.packages)

    if requested is not None:
        if not (worktree / requested).is_dir():
            raise PackageSelectionError(
                f"--package {requested!r} is not a directory in this repo."
            )
        return requested

    if declared:
        if len(declared) == 1:
            return declared[0]
        raise PackageSelectionError(
            "verdict.yml declares multiple packages ("
            + ", ".join(declared)
            + ") — pass --package to pick which one this run targets."
        )

    if _has_markers(worktree):
        return None

    candidates = detect_sibling_candidates(worktree)
    if len(candidates) >= 2:
        raise PackageSelectionError(
            "this repo has no project files at its root, but "
            + ", ".join(candidates)
            + " each look like their own project (a monorepo shape) — pass --package, or add a "
            "`packages:` block to verdict.yml, to say which one this run should grade."
        )
    return None
