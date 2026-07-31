"""Ranks agent/model/config combinations by pass-rate-per-dollar.

This is the ranking *engine* only: given a list of `ConfigResult`s (each
one caller-assembled from `TaskRun`s — e.g. by calling `run_with_retries`
once per task in a small suite), it sorts and renders them. There's no
suite file format or persistence here — nothing yet defines "run this
directory of tasks against these three agents automatically." That's
Phase 5's job; this module is what it will rank once it exists.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from verdict.schema import ConfigResult


def rank(entries: list[ConfigResult]) -> list[ConfigResult]:
    """Highest pass-rate-per-dollar first. Entries whose economics are
    undefined (unknown cost, or literally zero spend) sort after every
    entry with a real number — not first (that would reward not knowing
    your own cost) and not silently dropped (a leaderboard that hides
    entries it can't rank is worse than one that's honest about the gap).
    Ties within the undefined group fall back to raw pass rate, since
    that's still real signal even without a cost figure to divide by.
    """

    def sort_key(entry: ConfigResult) -> tuple[int, float]:
        rate = entry.pass_rate_per_dollar
        if rate is None:
            return (1, -entry.pass_rate)
        return (0, -rate)

    return sorted(entries, key=sort_key)


def render(entries: list[ConfigResult], console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="Leaderboard — ranked by pass-rate-per-dollar")
    table.add_column("rank", justify="right")
    table.add_column("config")
    table.add_column("pass rate", justify="right")
    table.add_column("total cost", justify="right")
    table.add_column("verdict-pts/$", justify="right")

    for i, entry in enumerate(rank(entries), start=1):
        cost = entry.total_cost_usd
        rate = entry.pass_rate_per_dollar
        table.add_row(
            str(i),
            entry.label,
            f"{entry.tasks_done}/{entry.tasks_total} ({entry.pass_rate:.0%})",
            f"${cost:.2f}" if cost is not None else "unknown",
            f"{rate:.2f}" if rate is not None else "—",
        )

    console.print(table)
