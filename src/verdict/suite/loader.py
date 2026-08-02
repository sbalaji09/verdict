"""Loads a benchmark suite: a directory of task subdirectories, each one a
real, executable acceptance test for an agent — never a prose rubric a
model has to interpret.

```text
my_suite/
  bug-fix-calculator/
    task.yml          # { task: "...", repo: "repo", category: "bug-fix" }
    repo/              # a real git repo with its own verdict.yml/tests
  refactor-tax-calc/
    task.yml
    repo/
```

`task.yml`'s only required key is `task` — the natural-language instruction
handed to the agent, exactly as `--task` would be for a single `verdict
run`. The *acceptance criteria* are never written down as prose in
`task.yml` at all: they're whatever `repo/`'s own gates (tests/typecheck/
build/lint) and `verdict.yml` (frontend checks, gate overrides) already
define, checked the same executable way a single `verdict run` checks
them. A suite task is a `verdict run` with its repo and task pre-wired —
nothing new to interpret, nothing new that could be judged instead of
proven.

An optional `allow_test_changes` key (Phase 12) declares that a task
legitimately requires editing tests (e.g. "add tests for the new
endpoint") — either `true` (any test file may be touched) or a list of
path globs (only those may be touched; every other test file is still
guarded). This is read HERE, from `task.yml`, before any agent worktree
exists — the trust boundary `integrity.TestChangeAllowance` documents:
this file is the benchmark author's, never the graded repo's own
`verdict.yml`, which an agent could edit itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from verdict.integrity import DENY_ALL, TestChangeAllowance


class SuiteLoadError(RuntimeError):
    """Raised when a suite directory or one of its tasks is malformed."""


@dataclass
class SuiteTask:
    name: str
    """The task subdirectory's name — used as its identity in reports."""
    task: str
    repo: Path
    category: str | None = None
    """Free-form grouping (e.g. "bug-fix"/"refactor"/"feature-add") used
    only for the failure-mode breakdown's reporting — never affects
    scoring."""
    allow_test_changes: TestChangeAllowance = DENY_ALL
    """From `task.yml`'s own `allow_test_changes` key — see this module's
    docstring and `TestChangeAllowance`'s own for the trust boundary this
    represents. Defaults to `DENY_ALL`, not something lenient, for every
    task that doesn't declare it explicitly."""


def _parse_allow_test_changes(data: dict[str, object], task_yml: Path) -> TestChangeAllowance:
    raw = data.get("allow_test_changes")
    if raw is None or raw is False:
        return DENY_ALL
    if raw is True:
        return TestChangeAllowance(allowed=True)
    if isinstance(raw, list) and all(isinstance(p, str) for p in raw):
        return TestChangeAllowance(allowed=True, allowed_paths=tuple(raw))
    raise SuiteLoadError(
        f"{task_yml}: allow_test_changes must be true/false or a list of path globs, got {raw!r}"
    )


def _load_one_task(task_dir: Path) -> SuiteTask:
    task_yml = task_dir / "task.yml"
    try:
        data = yaml.safe_load(task_yml.read_text()) or {}
    except yaml.YAMLError as exc:
        raise SuiteLoadError(f"{task_yml}: invalid YAML: {exc}") from exc

    if not isinstance(data, dict) or not data.get("task"):
        raise SuiteLoadError(f"{task_yml}: missing required `task` key")

    repo_name = str(data.get("repo", "repo"))
    repo = task_dir / repo_name
    if not repo.is_dir():
        raise SuiteLoadError(f"{task_yml}: repo directory {repo} does not exist")

    category = str(data["category"]) if data.get("category") else None
    allow_test_changes = _parse_allow_test_changes(data, task_yml)
    return SuiteTask(
        name=task_dir.name,
        task=str(data["task"]),
        repo=repo,
        category=category,
        allow_test_changes=allow_test_changes,
    )


def load_suite(suite_dir: Path) -> list[SuiteTask]:
    """Every immediate subdirectory of `suite_dir` containing a `task.yml`
    is one task. Sorted by directory name for a deterministic run order —
    a leaderboard shouldn't depend on filesystem iteration order.
    """
    if not suite_dir.is_dir():
        raise SuiteLoadError(f"suite directory does not exist: {suite_dir}")

    task_dirs = sorted(
        (p for p in suite_dir.iterdir() if p.is_dir() and (p / "task.yml").exists()),
        key=lambda p: p.name,
    )
    if not task_dirs:
        raise SuiteLoadError(f"no task.yml found in any subdirectory of {suite_dir}")

    return [_load_one_task(task_dir) for task_dir in task_dirs]
