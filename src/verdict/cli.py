from __future__ import annotations

import sys
from pathlib import Path

import typer

from verdict.adapters import Adapter
from verdict.adapters.claude_code import ClaudeCodeAdapter, ClaudeCodeAdapterError
from verdict.adapters.mock import MockAdapter
from verdict.report import render
from verdict.runner import run
from verdict.worktree import WorktreeError

app = typer.Typer(add_completion=False, help="Grade AI coding agents on executable truth.")


@app.callback()
def _callback() -> None:
    """Grade AI coding agents on executable truth, not opinion."""

# Phase 0's canned fix: MockAdapter has no LLM to decide what to write, so it
# needs a literal patch handed to it. This one happens to fix the bug in
# examples/sample_repo, so `--agent mock` is demoable out of the box.
_MOCK_PATCH = {
    "sample_repo/calculator.py": (
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
    ),
}


def _build_adapter(agent: str) -> Adapter:
    if agent == "mock":
        return MockAdapter(patches=_MOCK_PATCH)
    if agent == "claude-code":
        return ClaudeCodeAdapter()
    raise typer.BadParameter(f"unknown agent: {agent!r} (choices: mock, claude-code)")


@app.command(name="run")
def run_cmd(
    task: str = typer.Option(..., "--task", help="Natural-language description of the work."),
    agent: str = typer.Option(..., "--agent", help="Which adapter to drive: mock | claude-code"),
    repo: Path = typer.Option(..., "--repo", help="Path to a git repository to grade against."),
    test_cmd: str | None = typer.Option(
        None, "--test-cmd", help="Override the autodetected test command."
    ),
) -> None:
    """Run one agent attempt against one task and print its Verdict."""
    adapter = _build_adapter(agent)
    try:
        verdict = run(task=task, repo=repo, adapter=adapter, test_command=test_cmd)
    except (WorktreeError, ClaudeCodeAdapterError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    render(verdict)
    if not verdict.done:
        sys.exit(1)


if __name__ == "__main__":
    app()
