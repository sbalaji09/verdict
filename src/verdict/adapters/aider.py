"""Drives the real Aider CLI (`aider`) headlessly against a worktree, via
its non-interactive single-message mode (`--message`, `--yes-always`).

Unlike Claude Code/Cursor/Codex, Aider has no `--output-format json` (or
equivalent) — it prints a human-readable summary line at the end of a run,
e.g. `Tokens: 2.3k sent, 456 received. Cost: $0.01 message, $0.03 session.`
This is exactly the situation `gates/typecheck.py`'s `tsc` runner already
handles for a different tool: no structured format exists, so the only
honest option is a regex over the tool's own default text — still a real,
executed number, just scraped rather than parsed from JSON. `_TOKENS_RE`/
`_COST_RE` below do that scraping, and both are `None`/`0` (not guessed)
when the summary line isn't found, e.g. because a run failed before
producing one.
"""

from __future__ import annotations

import re
from pathlib import Path

from verdict.sandbox import Sandbox
from verdict.sandbox.config import fallback_sandbox
from verdict.schema import AttemptResult

DEFAULT_TIMEOUT_SECONDS = 900

_TOKENS_RE = re.compile(r"Tokens:\s*([\d.]+k?)\s*sent,\s*([\d.]+k?)\s*received", re.IGNORECASE)
_COST_RE = re.compile(r"Cost:\s*\$([\d.]+)\s*message,\s*\$([\d.]+)\s*session", re.IGNORECASE)


class AiderAdapterError(RuntimeError):
    """Raised when the `aider` CLI itself fails to run (not when the agent
    merely fails the task — that's reflected in AttemptResult/gates)."""


class AiderAdapter:
    """Runs `aider --message <task> --yes-always` inside the worktree in
    one-shot, auto-confirm mode, then diffs the worktree to see what it
    changed."""

    name = "aider"

    def __init__(self, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds

    def run(self, task: str, worktree: Path, sandbox: Sandbox | None = None) -> AttemptResult:
        command = ["aider", "--message", task, "--yes-always"]
        result = (sandbox or fallback_sandbox()).exec(
            command, cwd=worktree, timeout_seconds=self._timeout_seconds
        )
        if result.exit_code == 127:
            raise AiderAdapterError(
                "`aider` CLI not found on PATH. Install Aider, or use "
                "--agent mock to exercise the pipeline without it."
            )
        if result.timed_out:
            raise AiderAdapterError(f"aider did not finish within {self._timeout_seconds}s")

        output = result.stdout + result.stderr
        tokens_input, tokens_output = _parse_tokens(output)
        cost_usd = _parse_session_cost(output)

        if result.exit_code != 0 and tokens_input == 0 and tokens_output == 0 and cost_usd is None:
            # No usable summary *and* a nonzero exit — genuinely couldn't run,
            # as opposed to running and merely failing the task (which still
            # produces a token/cost summary and should fall through to a
            # normal AttemptResult for the gates to grade).
            raise AiderAdapterError(f"aider exited {result.exit_code}:\n{result.stderr.strip()}")

        return AttemptResult(
            diff="",
            files_changed=[],
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            raw_output=result.stdout,
        )


def _parse_count(raw: str) -> int:
    raw = raw.strip().lower()
    try:
        if raw.endswith("k"):
            return int(float(raw[:-1]) * 1000)
        return int(float(raw))
    except ValueError:
        return 0


def _parse_tokens(output: str) -> tuple[int, int]:
    match = _TOKENS_RE.search(output)
    if not match:
        return 0, 0
    return _parse_count(match.group(1)), _parse_count(match.group(2))


def _parse_session_cost(output: str) -> float | None:
    match = _COST_RE.search(output)
    if not match:
        return None
    try:
        return float(match.group(2))  # session total, not the single-message cost
    except ValueError:
        return None
