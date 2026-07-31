from __future__ import annotations

import sys
from pathlib import Path

import typer

from verdict.adapters import Adapter
from verdict.adapters.claude_code import ClaudeCodeAdapter, ClaudeCodeAdapterError
from verdict.adapters.mock import MockAdapter
from verdict.report import render_task_run
from verdict.runner import run_with_retries
from verdict.worktree import WorktreeError

app = typer.Typer(add_completion=False, help="Grade AI coding agents on executable truth.")


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
}


def _build_adapter(agent: str, repo: Path) -> Adapter:
    if agent == "mock":
        patch = _MOCK_PATCHES.get(repo.name)
        if patch is None:
            raise typer.BadParameter(
                f"mock adapter has no canned patch for {repo.name!r}; "
                "use --agent claude-code, or point --repo at one of the examples/ repos"
            )
        return MockAdapter(patches=patch)
    if agent == "claude-code":
        return ClaudeCodeAdapter()
    raise typer.BadParameter(f"unknown agent: {agent!r} (choices: mock, claude-code)")


@app.command(name="run")
def run_cmd(
    task: str = typer.Option(..., "--task", help="Natural-language description of the work."),
    agent: str = typer.Option(..., "--agent", help="Which adapter to drive: mock | claude-code"),
    repo: Path = typer.Option(..., "--repo", help="Path to a git repository to grade against."),
    max_attempts: int = typer.Option(
        1,
        "--max-attempts",
        help=(
            "Retry on failure up to this many times, stopping early on DONE. "
            "Cost is tracked across every attempt, dead ends included. "
            "Each retry re-runs the real agent — with --agent claude-code that's real spend."
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
    except (WorktreeError, ClaudeCodeAdapterError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    render_task_run(task_run)
    if not task_run.done:
        sys.exit(1)


if __name__ == "__main__":
    app()
