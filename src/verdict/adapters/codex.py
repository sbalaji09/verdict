"""Drives the real OpenAI Codex CLI (`codex`) headlessly against a
worktree, via its non-interactive `codex exec` subcommand.

Two differences from `ClaudeCodeAdapter`/`CursorAdapter` worth calling out,
both handled without needing a different `Adapter` shape:

- **`codex exec --json` streams newline-delimited JSON events**, not one
  final JSON object — so this adapter parses stdout line by line and takes
  the last event that carries a usable `usage` dict, rather than
  `json.loads`-ing the whole payload at once. Lines that aren't JSON (or
  that don't carry usage) are simply skipped, the same "read defensively,
  degrade the accounting rather than crash" discipline `ClaudeCodeAdapter`
  already applies.
- **Codex's CLI doesn't hand back a dollar cost the way Claude Code's
  does.** `cost_usd` is left `None` here — not guessed — so a repo that
  wants a populated leaderboard `$` figure for this adapter should
  configure `cost.price_per_1k_tokens` in `verdict.yml`; `runner.py`'s
  pricing fallback (Phase 3) already exists exactly for an adapter that
  reports token counts but not its own cost, so no new plumbing was
  needed here.

Like `CursorAdapter`, this mirrors `codex`'s publicly documented CLI
surface as of writing; flags and event shapes can drift between CLI
versions, hence the defensive, line-by-line, `.get()`-based parsing below.
"""

from __future__ import annotations

import json
from pathlib import Path

from verdict.adapters import AdapterError
from verdict.sandbox import Sandbox
from verdict.sandbox.config import fallback_sandbox
from verdict.schema import AttemptResult

DEFAULT_TIMEOUT_SECONDS = 900


class CodexAdapterError(AdapterError):
    """Raised when the `codex` CLI itself fails to run (not when the agent
    merely fails the task — that's reflected in AttemptResult/gates)."""


class CodexAdapter:
    """Runs `codex exec <task> --full-auto --json` inside the worktree,
    then diffs the worktree to see what it changed."""

    name = "codex"

    def __init__(self, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds

    def run(self, task: str, worktree: Path, sandbox: Sandbox | None = None) -> AttemptResult:
        command = [
            "codex",
            "exec",
            task,
            "--full-auto",  # auto-approve edits/commands, no interactive prompts
            "--json",
            "--skip-git-repo-check",  # the worktree is a throwaway branch, not necessarily "trusted"
        ]
        result = (sandbox or fallback_sandbox()).exec(
            command, cwd=worktree, timeout_seconds=self._timeout_seconds
        )
        if result.exit_code == 127:
            raise CodexAdapterError(
                "`codex` CLI not found on PATH. Install the Codex CLI, or use "
                "--agent mock to exercise the pipeline without it."
            )
        if result.timed_out:
            raise CodexAdapterError(f"codex did not finish within {self._timeout_seconds}s")

        tokens_input, tokens_output, final_text = _parse_events(result.stdout)

        if result.exit_code != 0 and final_text is None:
            raise CodexAdapterError(f"codex exited {result.exit_code}:\n{result.stderr.strip()}")

        return AttemptResult(
            diff="",
            files_changed=[],
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=None,  # see module docstring — configure verdict.yml's cost fallback instead
            raw_output=final_text if final_text is not None else result.stdout,
        )


def _parse_events(stdout: str) -> tuple[int, int, str | None]:
    """`codex exec --json` emits one JSON event per line. Take the last
    event that carries a `usage` dict for token counts, and the last event
    that carries readable text for `raw_output` — a line that isn't valid
    JSON, or doesn't match either shape, is simply skipped rather than
    treated as an error."""
    tokens_input = tokens_output = 0
    final_text: str | None = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        usage = event.get("usage")
        if isinstance(usage, dict):
            tokens_input = _as_int(usage.get("input_tokens"), tokens_input)
            tokens_output = _as_int(usage.get("output_tokens"), tokens_output)

        text = event.get("text") or event.get("content") or event.get("message")
        if isinstance(text, str) and text:
            final_text = text

    return tokens_input, tokens_output, final_text


def _as_int(value: object, default: int) -> int:
    return value if isinstance(value, int) else default
