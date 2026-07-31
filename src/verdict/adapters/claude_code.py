"""Drives the real Claude Code CLI headlessly against a worktree.

We shell out to `claude -p` rather than importing an SDK: it's the same
binary a developer already has installed and authenticated, it's a stable
process boundary (a crash or hang in the agent can't corrupt our process),
and `--output-format json` hands back token/cost accounting for free —
exactly what AttemptResult needs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from verdict.schema import AttemptResult

# Generous: real coding tasks can involve many tool calls and file reads.
DEFAULT_TIMEOUT_SECONDS = 900


class ClaudeCodeAdapterError(RuntimeError):
    """Raised when the `claude` CLI itself fails to run (not when the agent
    merely fails the task — that's reflected in AttemptResult/gates)."""


class ClaudeCodeAdapter:
    """Runs `claude -p <task>` inside the worktree in headless, auto-accept
    mode, then diffs the worktree to see what it changed.
    """

    name = "claude-code"

    def __init__(self, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds

    def run(self, task: str, worktree: Path) -> AttemptResult:
        command = [
            "claude",
            "-p",
            task,
            "--output-format",
            "json",
            "--permission-mode",
            "acceptEdits",
        ]
        try:
            result = subprocess.run(
                command,
                cwd=worktree,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise ClaudeCodeAdapterError(
                "`claude` CLI not found on PATH. Install Claude Code, or use "
                "--agent mock to exercise the pipeline without it."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCodeAdapterError(
                f"claude did not finish within {self._timeout_seconds}s"
            ) from exc

        payload: dict[str, object] = {}
        if result.stdout.strip():
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                payload = {}

        if result.returncode != 0 and not payload:
            raise ClaudeCodeAdapterError(
                f"claude exited {result.returncode}:\n{result.stderr.strip()}"
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
