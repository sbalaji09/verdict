"""Unit tests for the `Executor` abstraction itself, independent of
`run_suite` — proves the two claims Phase 15 makes about it directly:
output order never depends on completion order, and `max_workers` is a
real concurrency bound, not just a config value nothing enforces.
"""

from __future__ import annotations

import multiprocessing
import time

from verdict.suite.executor import LocalProcessPoolExecutor, SerialExecutor


def _record_span(events, index: int, sleep_seconds: float) -> int:
    events.append(("start", index, time.monotonic()))
    time.sleep(sleep_seconds)
    events.append(("end", index, time.monotonic()))
    return index


def _max_concurrent(events) -> int:
    spans = sorted(events, key=lambda e: e[2])
    active = 0
    peak = 0
    for kind, _index, _ts in spans:
        active += 1 if kind == "start" else -1
        peak = max(peak, active)
    return peak


def test_serial_executor_preserves_item_order() -> None:
    results = SerialExecutor().run(lambda a, b: a + b, [(1, 2), (3, 4), (5, 6)])
    assert results == [3, 7, 11]


def test_local_process_pool_executor_preserves_item_order_regardless_of_finish_time() -> None:
    with multiprocessing.Manager() as manager:
        events = manager.list()
        # Earlier items sleep longer, so if output order tracked completion
        # order instead of input order, this would come back reversed.
        items = [(events, i, 0.3 - i * 0.05) for i in range(6)]

        results = LocalProcessPoolExecutor(max_workers=6).run(_record_span, items)

        assert results == list(range(6))


def test_local_process_pool_executor_honors_the_concurrency_bound() -> None:
    with multiprocessing.Manager() as manager:
        events = manager.list()
        items = [(events, i, 0.3) for i in range(8)]

        results = LocalProcessPoolExecutor(max_workers=2).run(_record_span, items)

        assert sorted(results) == list(range(8))
        assert _max_concurrent(events) == 2


def test_local_process_pool_executor_actually_runs_concurrently() -> None:
    """Not just bounded — genuinely parallel: with more workers than
    needed, 8 items that each take 0.3s should finish well under
    8 * 0.3s serial time.
    """
    with multiprocessing.Manager() as manager:
        events = manager.list()
        items = [(events, i, 0.3) for i in range(8)]

        start = time.monotonic()
        LocalProcessPoolExecutor(max_workers=4).run(_record_span, items)
        elapsed = time.monotonic() - start

        assert elapsed < 8 * 0.3 * 0.75
        assert _max_concurrent(events) >= 2
