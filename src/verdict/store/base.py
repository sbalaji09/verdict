"""The `Store` abstraction: how a run's results outlive the process that
produced them. Mirrors this codebase's other pluggable-backend Protocols —
`Sandbox` (Phase 8), `Executor` (Phase 15) — one interface, callers never
code against a concrete backend. `SQLiteStore` (`sqlite_store.py`) is the
one implementation that ships; a future `PostgresStore`/`S3Store`/whatever
a team's own infra needs is a new class implementing this Protocol, never
a change to `regression.py` or any CLI call site.

## What actually gets persisted

A full `ConfigResult` (every `TaskRun`, every `Verdict`, every `Signal` —
`artifact_path` references (screenshots, videos, glitch-scan recordings)
included, since those are just path strings already carried on `Signal`,
not binary blobs this module needs to know anything special about) is
stored verbatim as JSON, keyed by `(run_id, config_label)` — `Store`
never reshapes what `ConfigResult`/`TaskRun`/`Verdict` already are, the
same "no new data, no new decisions" discipline `report_json.py` already
holds itself to.

Alongside that, `record_run` also writes one denormalized `TaskOutcome`
row per `TaskRun` — just `(task, agent, repo, config_label, done, status,
recorded_at, commit_sha)` — because "query history for this task" and
"pool N historical outcomes into a pass/fail count for a z-test" (see
`regression.py`) are the whole reason this module exists, and neither
needs to deserialize a full nested `ConfigResult` JSON blob to answer.
This is the identical reasoning Phase 2's `FailureLocation` gave for
keeping failures structured instead of only living inside `detail` text:
reuse the shape a real question needs, don't re-derive it by parsing
something coarser back apart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel

from verdict.schema import ConfigResult


class TaskOutcome(BaseModel):
    """One historical (config, task) observation — the slim, query-fast
    projection `history()`/`regression.py` actually work from. The full
    `TaskRun` (every attempt, every signal, every artifact reference) is
    still recoverable via `Store.get_config_results(run_id)` keyed off
    the same `run_id` this row carries.
    """

    run_id: str
    recorded_at: datetime
    commit_sha: str | None
    config_label: str
    task: str
    agent: str
    repo: str
    done: bool
    status: str
    """`VerdictStatus.value` of the deciding attempt — kept as a plain
    string here (not the enum) so this model has no import-time
    dependency beyond `schema.ConfigResult` itself; `done` is what every
    consumer in this module actually branches on."""


class RunRecord(BaseModel):
    """One `record_run` call — a whole `list[ConfigResult]`, as produced
    by one `verdict bench`/`gate` invocation, timestamped and optionally
    tagged with the commit it graded and a free-form label (mirrors
    `ConfigResult.label`'s own "caller supplies it, Store doesn't need to
    understand it" philosophy)."""

    run_id: str
    recorded_at: datetime
    commit_sha: str | None
    label: str | None
    config_labels: tuple[str, ...]


class Store(Protocol):
    """Callers code against this, never against `SQLiteStore` directly —
    see this module's own docstring for why."""

    def record_run(
        self,
        config_results: list[ConfigResult],
        commit_sha: str | None = None,
        label: str | None = None,
        recorded_at: datetime | None = None,
    ) -> str:
        """Persist a whole run and return its new `run_id`. `recorded_at`
        defaults to now — exposed as a parameter (not always
        `datetime.now()` internally) purely so tests can pin a
        deterministic timestamp without mocking the clock."""
        ...

    def history(
        self,
        task: str,
        agent: str,
        repo: str,
        config_label: str | None = None,
        exclude_run_id: str | None = None,
        limit: int | None = None,
    ) -> list[TaskOutcome]:
        """Every recorded observation of this exact `(task, agent, repo)`
        — optionally narrowed to one `config_label` and/or excluding one
        `run_id` (so a caller can query "history before this run" without
        that run's own just-recorded outcome polluting its own baseline —
        see `regression.py`). Ordered most-recent-first; `limit` caps how
        many are returned, never how many exist.
        """
        ...

    def runs(self, limit: int | None = None) -> list[RunRecord]:
        """Every recorded run, most-recent-first."""
        ...

    def get_config_results(self, run_id: str) -> list[ConfigResult]:
        """The full `list[ConfigResult]` (every `TaskRun`/`Verdict`/
        `Signal`, artifact references included) `record_run` was given
        for this `run_id` — `[]` if `run_id` doesn't exist. Never a
        partial/summarized reconstruction; this is exactly the JSON
        `record_run` stored, deserialized back."""
        ...
