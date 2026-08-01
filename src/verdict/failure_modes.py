"""Failure-mode breakdown: for each config, which named check (a gate or a
frontend check) actually failed, and how often, across every suite task
that config didn't finish.

This is a tally, not a root-cause classifier. Phase 2's attribution already
answers "why did this specific failure happen" *per task* (bisected to a
culprit file); this answers a different, coarser question — "what kind of
thing does this config keep failing at, across a whole suite" — by
counting failing `Signal.name`s. It doesn't replace attribution and
doesn't try to: a suite-wide tally over check *names* is a different
altitude than a single task's causal chain.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from verdict.schema import ConfigResult, GateStatus, Provenance


@dataclass
class FailureModeBreakdown:
    label: str
    counts: Counter[str] = field(default_factory=Counter)

    @property
    def total_failures(self) -> int:
        return sum(self.counts.values())


def summarize_failure_modes(config_result: ConfigResult) -> FailureModeBreakdown:
    """Counts each PROVEN, FAILing signal name across every task run that
    didn't end DONE. A JUDGED signal is never counted here, for the same
    reason it never decides `Verdict.status`: it's an opinion, not a
    failure mode a config can be said to actually have.
    """
    counts: Counter[str] = Counter()
    for task_run in config_result.task_runs:
        if task_run.done:
            continue
        for signal in task_run.final.signals:
            if signal.provenance is Provenance.PROVEN and signal.status is GateStatus.FAIL:
                counts[signal.name] += 1
    return FailureModeBreakdown(label=config_result.label, counts=counts)


def render_failure_modes(config_results: list[ConfigResult], console: Console | None = None) -> None:
    console = console or Console()
    breakdowns = [summarize_failure_modes(cr) for cr in config_results]

    if not any(b.counts for b in breakdowns):
        console.print("[dim]No failures to break down — every config finished every task.[/dim]")
        return

    table = Table(title="Failure-mode breakdown — failing checks by config")
    table.add_column("config")
    table.add_column("check")
    table.add_column("failures", justify="right")

    for breakdown in breakdowns:
        for name, count in breakdown.counts.most_common():
            table.add_row(breakdown.label, name, str(count))

    console.print(table)
