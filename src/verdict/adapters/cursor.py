"""Drives the real Cursor CLI (`cursor-agent`) headlessly against a
worktree — the same subprocess-boundary shape as `ClaudeCodeAdapter`, since
Cursor's CLI was deliberately modeled on Claude Code's: `-p`/`--print` for
non-interactive mode, `--output-format json` for structured output, a
force flag to auto-accept edits so a headless run never blocks on a
permission prompt.

This mirrors `cursor-agent`'s publicly documented CLI surface as of
writing; a CLI's exact flags and JSON shape can drift between versions, so
every field this adapter reads out of the JSON payload is read
defensively (`.get(...)`, type-checked, falls back to 0/None) — the same
discipline `ClaudeCodeAdapter` already applies, for the same reason: a
missing or renamed field should degrade the accounting, never crash the
adapter.
"""

from __future__ import annotations

import json
from pathlib import Path

from verdict.sandbox import Sandbox
from verdict.sandbox.config import fallback_sandbox
from verdict.schema import AttemptResult

DEFAULT_TIMEOUT_SECONDS = 900


class CursorAdapterError(RuntimeError):
    """Raised when the `cursor-agent` CLI itself fails to run (not when the
    agent merely fails the task — that's reflected in AttemptResult/gates)."""


class CursorAdapter:
    """Runs `cursor-agent -p <task>` inside the worktree in headless,
    force-accept mode, then diffs the worktree to see what it changed."""

    name = "cursor"

    def __init__(self, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds

    def run(self, task: str, worktree: Path, sandbox: Sandbox | None = None) -> AttemptResult:
        command = [
            "cursor-agent",
            "-p",
            task,
            "--output-format",
            "json",
            "-f",  # force: auto-accept edits, no interactive permission prompts
        ]
        result = (sandbox or fallback_sandbox()).exec(
            command, cwd=worktree, timeout_seconds=self._timeout_seconds
        )
        if result.exit_code == 127:
            raise CursorAdapterError(
                "`cursor-agent` CLI not found on PATH. Install Cursor's CLI, or use "
                "--agent mock to exercise the pipeline without it."
            )
        if result.timed_out:
            raise CursorAdapterError(f"cursor-agent did not finish within {self._timeout_seconds}s")

        payload: dict[str, object] = {}
        if result.stdout.strip():
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                payload = {}

        if result.exit_code != 0 and not payload:
            raise CursorAdapterError(
                f"cursor-agent exited {result.exit_code}:\n{result.stderr.strip()}"
            )

        usage_raw = payload.get("usage", {})
        usage: dict[str, object] = usage_raw if isinstance(usage_raw, dict) else {}

        cost_usd = payload.get("total_cost_usd")
        raw_result = payload.get("result")

        # diff/files_changed are filled in by runner.py, which knows the
        # worktree's base commit.
        return AttemptResult(
            diff="",
            files_changed=[],
            tokens_input=_as_int(usage.get("input_tokens")),
            tokens_output=_as_int(usage.get("output_tokens")),
            cost_usd=cost_usd if isinstance(cost_usd, (int, float)) else None,
            raw_output=raw_result if isinstance(raw_result, str) else result.stdout,
        )


def _as_int(value: object) -> int:
    return value if isinstance(value, int) else 0
