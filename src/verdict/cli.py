from __future__ import annotations

import sys
from pathlib import Path

import typer

from verdict import economics
from verdict.adapters import Adapter
from verdict.adapters.aider import AiderAdapter, AiderAdapterError
from verdict.adapters.claude_code import ClaudeCodeAdapter, ClaudeCodeAdapterError
from verdict.adapters.codex import CodexAdapter, CodexAdapterError
from verdict.adapters.cursor import CursorAdapter, CursorAdapterError
from verdict.adapters.mock import MockAdapter, SuiteMockAdapter
from verdict.adapters.openhands import OpenHandsAdapter, OpenHandsAdapterError
from verdict.failure_modes import render_failure_modes
from verdict.report import render_task_run
from verdict.runner import run_with_retries
from verdict.suite import BenchConfig, SuiteLoadError, load_suite, run_suite
from verdict.worktree import WorktreeError

app = typer.Typer(add_completion=False, help="Grade AI coding agents on executable truth.")

# Every adapter this CLI can drive by name, beyond `mock` (which needs a
# per-repo or per-task canned patch, handled separately below). One entry
# per real Adapter implementation — adding a fifth, sixth, ... agent is
# exactly this: implement the class, add one line here, nothing else in
# this module changes.
_REAL_AGENTS: dict[str, type[Adapter]] = {
    "claude-code": ClaudeCodeAdapter,
    "cursor": CursorAdapter,
    "codex": CodexAdapter,
    "aider": AiderAdapter,
    "openhands": OpenHandsAdapter,
}

# Every AdapterError a real agent's subprocess can raise — caught the same
# way at both call sites below (the CLI's job is to report it and exit
# non-zero, not to recover from it).
_ADAPTER_ERRORS: tuple[type[Exception], ...] = (
    ClaudeCodeAdapterError,
    CursorAdapterError,
    CodexAdapterError,
    AiderAdapterError,
    OpenHandsAdapterError,
)

# Combined with WorktreeError once here so both `run` and `bench` catch the
# exact same set via a single, mypy-friendly tuple name.
_RUN_ERRORS: tuple[type[Exception], ...] = (WorktreeError, *_ADAPTER_ERRORS)


@app.callback()
def _callback() -> None:
    """Grade AI coding agents on executable truth, not opinion."""

# MockAdapter has no LLM to decide what to write, so it needs a literal
# patch handed to it. These happen to fix the seeded bug in each example
# repo, so `--agent mock` is demoable out of the box on either one.
_MOCK_PATCHES: dict[str, dict[str, str]] = {
    "sample_repo": {
        "sample_repo/calculator.py": (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        ),
    },
    "sample_node_repo": {
        "src/calculator.ts": (
            "export function add(a: number, b: number): number {\n"
            "  return a + b;\n"
            "}\n"
        ),
    },
    "sample_frontend_repo": {
        "public/index.html": (
            "<!doctype html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="utf-8">\n'
            "  <title>Acme Launch</title>\n"
            '  <link rel="stylesheet" href="/style.css">\n'
            "</head>\n"
            "<body>\n"
            "  <header><h1>Acme</h1></header>\n"
            "  <main>\n"
            "    <p>The fastest way to launch your next idea.</p>\n"
            '    <a id="cta" class="cta" href="/signup.html">Get Started</a>\n'
            "  </main>\n"
            "</body>\n"
            "</html>\n"
        ),
    },
}


def _build_adapter(agent: str, repo: Path) -> Adapter:
    if agent == "mock":
        patch = _MOCK_PATCHES.get(repo.name)
        if patch is None:
            raise typer.BadParameter(
                f"mock adapter has no canned patch for {repo.name!r}; "
                "use a real --agent, or point --repo at one of the examples/ repos"
            )
        return MockAdapter(patches=patch)
    factory = _REAL_AGENTS.get(agent)
    if factory is None:
        raise typer.BadParameter(f"unknown agent: {agent!r} (choices: mock, {', '.join(_REAL_AGENTS)})")
    return factory()


# `verdict bench` runs the same task text against every task in a suite —
# MockAdapter's single fixed-at-construction patch can't represent "a
# different canned fix per task," so SuiteMockAdapter looks its patch up by
# the task's own text instead (see adapters/mock.py). These happen to fix
# examples/starter_suite's three seeded tasks, so `--agent mock` is
# demoable against it out of the box, the same way `_MOCK_PATCHES` above
# is for single-repo `verdict run`.
_STARTER_SUITE_MOCK_PATCHES: dict[str, dict[str, str]] = {
    "Fix the bug in add() so it returns the correct sum instead of subtracting.": {
        "calculator.py": "def add(a: int, b: int) -> int:\n    return a + b\n",
    },
    "Add a multiply(a, b) function to calculator.py that returns the product of a and b, "
    "matching add()'s style.": {
        "calculator.py": (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "\n\n"
            "def multiply(a: int, b: int) -> int:\n"
            "    return a * b\n"
        ),
    },
    "The non-negative-amount validation in calculate_us_tax and calculate_eu_tax is duplicated. "
    "Extract it into a shared helper function, without changing either function's behavior.": {
        "tax.py": (
            "def _validate_non_negative(amount: float) -> None:\n"
            '    if amount < 0:\n'
            '        raise ValueError("amount must be non-negative")\n'
            "\n\n"
            "def calculate_us_tax(amount: float) -> float:\n"
            "    _validate_non_negative(amount)\n"
            "    return round(amount * 0.07, 2)\n"
            "\n\n"
            "def calculate_eu_tax(amount: float) -> float:\n"
            "    _validate_non_negative(amount)\n"
            "    return round(amount * 0.20, 2)\n"
        ),
    },
}


def _build_bench_adapter(agent: str) -> Adapter:
    if agent == "mock":
        return SuiteMockAdapter(_STARTER_SUITE_MOCK_PATCHES)
    factory = _REAL_AGENTS.get(agent)
    if factory is None:
        raise typer.BadParameter(f"unknown agent: {agent!r} (choices: mock, {', '.join(_REAL_AGENTS)})")
    return factory()


@app.command(name="run")
def run_cmd(
    task: str = typer.Option(..., "--task", help="Natural-language description of the work."),
    agent: str = typer.Option(
        ..., "--agent", help=f"Which adapter to drive: mock | {' | '.join(_REAL_AGENTS)}"
    ),
    repo: Path = typer.Option(..., "--repo", help="Path to a git repository to grade against."),
    max_attempts: int = typer.Option(
        1,
        "--max-attempts",
        help=(
            "Retry on failure up to this many times, stopping early on DONE. "
            "Cost is tracked across every attempt, dead ends included. "
            "Each retry re-runs the real agent — with a real --agent that's real spend."
        ),
    ),
) -> None:
    """Run an agent against one task (retrying on failure if --max-attempts > 1)
    and print its Verdict plus the cost across every attempt made.

    Gate commands are autodetected per-stack; override any of them — and
    configure token pricing — with a `verdict.yml` in the repo being graded
    (see README's Configuration section).
    """
    adapter = _build_adapter(agent, repo)
    try:
        task_run = run_with_retries(task=task, repo=repo, adapter=adapter, max_attempts=max_attempts)
    except _RUN_ERRORS as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    render_task_run(task_run)
    if not task_run.done:
        sys.exit(1)


@app.command(name="bench")
def bench_cmd(
    suite: Path = typer.Option(
        ..., "--suite", help="Path to a suite directory (see DESIGN.md's Phase 5 section for the format)."
    ),
    agent: list[str] = typer.Option(
        ...,
        "--agent",
        help=(
            f"Repeatable: one config to benchmark, e.g. --agent mock --agent claude-code. "
            f"Choices: mock | {' | '.join(_REAL_AGENTS)}"
        ),
    ),
    max_attempts: int = typer.Option(
        1, "--max-attempts", help="Retry each task up to this many times per config, stopping early on DONE."
    ),
) -> None:
    """Run every --agent against every task in --suite, then print a
    pass-rate-per-dollar leaderboard and a failure-mode breakdown.

    Unlike `verdict run`, this never exits non-zero for a bad score — it's
    a scorecard, not a merge gate; use `verdict run` in CI for that.
    """
    try:
        tasks = load_suite(suite)
    except SuiteLoadError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    configs = [BenchConfig(label=name, adapter=_build_bench_adapter(name)) for name in agent]

    try:
        results = run_suite(tasks, configs, max_attempts=max_attempts)
    except _RUN_ERRORS as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    economics.render(results)
    render_failure_modes(results)


if __name__ == "__main__":
    app()
