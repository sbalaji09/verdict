"""Drives the real OpenHands CLI (`openhands`) headlessly against a
worktree.

This is the least standardized of the four CLI-subprocess adapters, and
that's reflected honestly in what this adapter reports rather than papered
over. OpenHands is typically configured via a workspace directory and an
LLM provider config rather than a single self-contained invocation the way
`claude -p`/`cursor-agent -p`/`codex exec` are — Verdict doesn't manage
that configuration (out of scope, the same way Phase 1 doesn't run `npm
install` for a repo, see DESIGN.md), so this adapter assumes the caller's
environment already has OpenHands configured (API keys, default model)
and only supplies `--task` plus the worktree as the working directory.

OpenHands also has no documented structured token/cost output the way
Claude Code's `--output-format json` does — so this adapter makes no
attempt to scrape one out of its logs. `tokens_input`/`tokens_output` are
always `0` and `cost_usd` is always `None`; `raw_output` carries the CLI's
full stdout so a human (or `verdict.yml`'s cost-pricing fallback, once
someone establishes a real per-token figure some other way) still has
something to look at. Guessing a number here would be strictly worse than
reporting "unknown," per this whole schema's own discipline about
`total_cost_usd`/`pass_rate_per_dollar` (see Phase 3's DESIGN.md section).
"""

from __future__ import annotations

from pathlib import Path

from verdict.adapters import AdapterError
from verdict.sandbox import Sandbox
from verdict.sandbox.config import fallback_sandbox
from verdict.schema import AttemptResult

DEFAULT_TIMEOUT_SECONDS = 1800
"""OpenHands' agentic loop tends to run longer per task than the other
three CLIs in practice, hence a longer default than
`ClaudeCodeAdapter`/`CursorAdapter`/`CodexAdapter`/`AiderAdapter`."""


class OpenHandsAdapterError(AdapterError):
    """Raised when the `openhands` CLI itself fails to run (not when the
    agent merely fails the task — that's reflected in AttemptResult/gates)."""


class OpenHandsAdapter:
    """Runs `openhands run --task <task> --no-auto-continue` inside the
    worktree, then diffs the worktree to see what it changed."""

    name = "openhands"

    def __init__(self, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds

    def run(self, task: str, worktree: Path, sandbox: Sandbox | None = None) -> AttemptResult:
        command = [
            "openhands",
            "run",
            "--task",
            task,
            "--no-auto-continue",  # stop after the task is addressed, don't keep prompting itself
        ]
        result = (sandbox or fallback_sandbox()).exec(
            command, cwd=worktree, timeout_seconds=self._timeout_seconds
        )
        if result.exit_code == 127:
            raise OpenHandsAdapterError(
                "`openhands` CLI not found on PATH. Install OpenHands, or use "
                "--agent mock to exercise the pipeline without it."
            )
        if result.timed_out:
            raise OpenHandsAdapterError(f"openhands did not finish within {self._timeout_seconds}s")

        if result.exit_code != 0:
            raise OpenHandsAdapterError(f"openhands exited {result.exit_code}:\n{result.stderr.strip()}")

        # diff/files_changed are filled in by runner.py, which knows the
        # worktree's base commit. No structured usage/cost — see module
        # docstring.
        return AttemptResult(
            diff="",
            files_changed=[],
            tokens_input=0,
            tokens_output=0,
            cost_usd=None,
            raw_output=result.stdout,
        )
