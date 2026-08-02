"""`SQLiteStore`: the default `Store` implementation — one file, the
standard library's own `sqlite3`, no new dependency. SQLite specifically
(over, say, a flat JSON-lines file) because `regression.py`'s whole job is
"query this exact `(task, agent, repo, config_label)` combination's
history," and a real index (`idx_task_outcomes_lookup` below) makes that a
single indexed lookup instead of an O(n) scan over every run ever
recorded — the same reason `sandbox/cache.py`'s base-state cache uses a
keyed directory layout rather than a flat blob nobody could look anything
up in without reading everything.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from verdict.schema import ConfigResult
from verdict.store.base import RunRecord, TaskOutcome

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    recorded_at TEXT NOT NULL,
    commit_sha TEXT,
    label TEXT
);

CREATE TABLE IF NOT EXISTS config_results (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    config_label TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (run_id, config_label)
);

CREATE TABLE IF NOT EXISTS task_outcomes (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    config_label TEXT NOT NULL,
    task TEXT NOT NULL,
    agent TEXT NOT NULL,
    repo TEXT NOT NULL,
    done INTEGER NOT NULL,
    status TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    commit_sha TEXT
);

CREATE INDEX IF NOT EXISTS idx_task_outcomes_lookup
    ON task_outcomes(task, agent, repo, config_label, recorded_at);
"""


class SQLiteStore:
    """A new connection per operation, not one held open for the
    instance's lifetime — SQLite is a single file, and a short-lived
    connection per call is simpler to reason about than a shared handle
    across whatever concurrency a caller (a suite's worker pool included)
    might introduce, at a cost (a fresh connection per query) this
    module's actual call volume — a handful of writes/queries per `verdict
    bench`/`gate` invocation, never a hot inner loop — never notices.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record_run(
        self,
        config_results: list[ConfigResult],
        commit_sha: str | None = None,
        label: str | None = None,
        recorded_at: datetime | None = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        recorded_at = recorded_at or datetime.now(UTC)
        recorded_at_iso = recorded_at.isoformat()

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, recorded_at, commit_sha, label) VALUES (?, ?, ?, ?)",
                (run_id, recorded_at_iso, commit_sha, label),
            )
            for config_result in config_results:
                conn.execute(
                    "INSERT INTO config_results (run_id, config_label, data) VALUES (?, ?, ?)",
                    (run_id, config_result.label, config_result.model_dump_json()),
                )
                for task_run in config_result.task_runs:
                    conn.execute(
                        """INSERT INTO task_outcomes
                           (run_id, config_label, task, agent, repo, done, status, recorded_at, commit_sha)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            run_id,
                            config_result.label,
                            task_run.task,
                            task_run.agent,
                            task_run.repo,
                            int(task_run.done),
                            task_run.final.status.value,
                            recorded_at_iso,
                            commit_sha,
                        ),
                    )
        return run_id

    def history(
        self,
        task: str,
        agent: str,
        repo: str,
        config_label: str | None = None,
        exclude_run_id: str | None = None,
        limit: int | None = None,
    ) -> list[TaskOutcome]:
        query = (
            "SELECT run_id, recorded_at, commit_sha, config_label, task, agent, repo, done, status "
            "FROM task_outcomes WHERE task = ? AND agent = ? AND repo = ?"
        )
        params: list[object] = [task, agent, repo]
        if config_label is not None:
            query += " AND config_label = ?"
            params.append(config_label)
        if exclude_run_id is not None:
            query += " AND run_id != ?"
            params.append(exclude_run_id)
        query += " ORDER BY recorded_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            TaskOutcome(
                run_id=row[0],
                recorded_at=datetime.fromisoformat(row[1]),
                commit_sha=row[2],
                config_label=row[3],
                task=row[4],
                agent=row[5],
                repo=row[6],
                done=bool(row[7]),
                status=row[8],
            )
            for row in rows
        ]

    def runs(self, limit: int | None = None) -> list[RunRecord]:
        query = "SELECT run_id, recorded_at, commit_sha, label FROM runs ORDER BY recorded_at DESC"
        params: list[object] = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            run_rows = conn.execute(query, params).fetchall()
            records = []
            for run_id, recorded_at, commit_sha, label in run_rows:
                config_labels = tuple(
                    row[0]
                    for row in conn.execute(
                        "SELECT config_label FROM config_results WHERE run_id = ?", (run_id,)
                    ).fetchall()
                )
                records.append(
                    RunRecord(
                        run_id=run_id,
                        recorded_at=datetime.fromisoformat(recorded_at),
                        commit_sha=commit_sha,
                        label=label,
                        config_labels=config_labels,
                    )
                )
        return records

    def get_config_results(self, run_id: str) -> list[ConfigResult]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT data FROM config_results WHERE run_id = ?", (run_id,)
            ).fetchall()
        return [ConfigResult.model_validate_json(row[0]) for row in rows]
