"""A fake agent that applies a known set of file edits.

Exists so the rest of the pipeline (worktree isolation, gates, schema,
reporting) can be exercised in tests and demos without spending real API
tokens or depending on network access. It implements the same Adapter
protocol as ClaudeCodeAdapter, so swapping one for the other never touches
runner.py.
"""

from __future__ import annotations

from pathlib import Path

from verdict.schema import AttemptResult


class MockAdapter:
    """Applies a fixed map of {relative path: new file contents}."""

    name = "mock"

    def __init__(self, patches: dict[str, str]) -> None:
        if not patches:
            raise ValueError("MockAdapter needs at least one patch to apply")
        self._patches = patches

    def run(self, task: str, worktree: Path) -> AttemptResult:
        for relative_path, contents in self._patches.items():
            target = worktree / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)

        # diff/files_changed are filled in by runner.py, which knows the
        # worktree's base commit — an adapter shouldn't need to know about
        # git diffing to do its one job (edit files).
        return AttemptResult(
            diff="",
            files_changed=[],
            tokens_input=0,
            tokens_output=0,
            cost_usd=0.0,
            raw_output=f"[mock] applied fixed patch for task: {task!r}",
        )
