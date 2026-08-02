"""The `Executor` abstraction: how `run_suite` turns a list of independent
work items into a list of results, without the rest of `runner.py` needing
to know *where* that work actually ran.

Two implementations ship in Phase 15:

- **`SerialExecutor`** — a plain `for` loop, in this process. What
  `run_suite` always did before this phase; still the default, so nothing
  that doesn't opt in to parallelism changes behavior or cost profile.
- **`LocalProcessPoolExecutor`** — a bounded pool of OS processes
  (`concurrent.futures.ProcessPoolExecutor`). Each work item already runs
  inside its own isolated sandbox (`runner.py::run` creates a fresh
  worktree + `Sandbox` per attempt, unchanged by this phase) — the process
  pool adds a second, coarser layer of isolation on top (one worker
  process's crash/hang can't corrupt another's, or this process's), and is
  what actually lets multiple `(config, task)` pairs execute concurrently
  instead of one at a time.

A future distributed executor (e.g. one that dispatches work items to a
queue consumed by other machines) is expected to be a third `Executor`
implementation, not a rewrite of `run_suite` — that's the entire reason
this Protocol exists rather than `run_suite` calling
`ProcessPoolExecutor` directly.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Protocol, TypeVar

R = TypeVar("R")


class Executor(Protocol):
    """Runs `fn(*item)` for every `item` in `items` and returns the
    results **in the same order as `items`**, regardless of which item
    actually finished first. Order-preserving output is what lets
    `run_suite` reassemble a deterministic `ConfigResult`/`TaskRun` tree
    no matter which backend ran the work or how work was interleaved —
    see `runner.py`'s own docstring on why that's load-bearing for
    "parallel and serial produce byte-identical leaderboards."

    `fn` must be a plain, module-level, picklable callable (never a
    closure or a bound method capturing unpicklable state) — a
    process-pool-backed `Executor` has to be able to ship it to another
    process. `items` themselves must be picklable for the same reason;
    `run_suite` only ever builds them from `SuiteTask`/`BenchConfig`/
    `SandboxConfig`, all plain dataclasses with no open handles or locks.
    """

    def run(self, fn: Callable[..., R], items: list[tuple[Any, ...]]) -> list[R]: ...


class SerialExecutor:
    """One item at a time, in this process — the library's own default.
    `run_suite`'s original behavior, preserved exactly (including: a
    later item never starts before an earlier one finishes), so any
    caller that doesn't pass `executor=` explicitly sees no behavior
    change from before this phase.
    """

    def run(self, fn: Callable[..., R], items: list[tuple[Any, ...]]) -> list[R]:
        return [fn(*item) for item in items]


class LocalProcessPoolExecutor:
    """A bounded pool of `max_workers` OS processes. Submits every item up
    front (`ProcessPoolExecutor.submit`, not `.map`) so all of them start
    racing concurrently (bounded by `max_workers`), then reads each
    future's `.result()` back in *submission* order — blocking on an
    earlier item that happens to finish later is fine, since every
    future is already running by the time any `.result()` call blocks;
    this just guarantees the returned list matches `items`' order
    regardless of which one actually finished first.

    `max_workers` is the actual mechanism that bounds concurrent resource
    use: at most `max_workers` sandboxes (each itself capped by its own
    `SandboxConfig.limits`) are ever running at once, so the aggregate
    ceiling across the whole pool is `max_workers * limits` by
    construction — no separate accounting needed on top of the pool size.
    """

    def __init__(self, max_workers: int) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self.max_workers = max_workers

    def run(self, fn: Callable[..., R], items: list[tuple[Any, ...]]) -> list[R]:
        results: list[R | None] = [None] * len(items)
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(fn, *item): index for index, item in enumerate(items)}
            for future in futures:
                results[futures[future]] = future.result()
        return results  # type: ignore[return-value]
