"""Renders a Verdict (or a whole TaskRun, cost included) to the terminal,
keeping the PROVEN/JUDGED split visually explicit — that split is the
entire point of the tool.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from verdict.schema import (
    AttributionKind,
    Confidence,
    GateStatus,
    Provenance,
    TaskRun,
    Verdict,
    VerdictStatus,
)

console = Console()

_MARKS = {
    GateStatus.PASS: "[green]✓[/green]",
    GateStatus.FAIL: "[red]✗[/red]",
    GateStatus.NA: "[dim]·[/dim]",
}

_VERDICT_STYLE = {
    VerdictStatus.DONE: ("bold green", "DONE"),
    VerdictStatus.NOT_DONE: ("bold red", "NOT DONE"),
    VerdictStatus.UNVERIFIED: ("bold yellow", "UNVERIFIED"),
}

_ATTRIBUTION_MARK = {
    AttributionKind.REGRESSION: "[red]✗[/red]",
    AttributionKind.PRE_EXISTING: "[yellow]~[/yellow]",
    AttributionKind.INCONCLUSIVE: "[dim]?[/dim]",
}


def _render_header(verdict: Verdict) -> None:
    header = Text()
    header.append("Verdict", style="bold")
    header.append(f"   repo: {verdict.repo}   agent: {verdict.agent}\n", style="dim")
    header.append(f"task: {verdict.task}", style="dim")
    console.print(header)
    console.print()


def _render_signals(verdict: Verdict) -> None:
    proven = [s for s in verdict.signals if s.provenance is Provenance.PROVEN]
    judged = [s for s in verdict.signals if s.provenance is Provenance.JUDGED]

    if proven:
        console.print("[bold]PROVEN[/bold] [dim](executed)[/dim]")
        for s in proven:
            mark = _MARKS[s.status]
            cmd = f"  [dim]{s.command}[/dim]" if s.command else ""
            console.print(f"  {mark} {s.name:<12}{cmd}")
            if s.status is GateStatus.FAIL:
                console.print(Panel(s.detail, border_style="red", expand=False))
            elif s.status is GateStatus.NA:
                console.print(f"      [dim]{s.detail}[/dim]")
        console.print()

    if judged:
        console.print("[bold]JUDGED[/bold] [dim](model opinion)[/dim]")
        for s in judged:
            mark = "[green]~[/green]" if s.status is GateStatus.PASS else "[yellow]~[/yellow]"
            console.print(f"  {mark} {s.name:<12}  {s.detail}")
        console.print()

    if verdict.attributions:
        console.print("[bold]CAUSAL ANALYSIS[/bold] [dim](proven — bisection, not opinion)[/dim]")
        for a in verdict.attributions:
            mark = _ATTRIBUTION_MARK[a.kind]
            console.print(f"  {mark} {a.explanation}")
        console.print()


def _render_verdict_line(verdict: Verdict) -> None:
    style, label = _VERDICT_STYLE[verdict.status]
    line = f"\n[{style}]VERDICT: {label}[/{style}]"
    if verdict.status is not VerdictStatus.UNVERIFIED and verdict.confidence is Confidence.LOW:
        line += " [dim](low confidence: no test gate ran)[/dim]"
    console.print(line)


def render(verdict: Verdict) -> None:
    """Render a single attempt — no cost-across-attempts context, since
    there isn't any without a TaskRun. Used directly by callers that only
    ever make one attempt (e.g. tests, or scripts calling `runner.run`).
    """
    _render_header(verdict)
    _render_signals(verdict)
    if verdict.attempt.cost_usd is not None:
        console.print(
            f"[dim]tokens {verdict.attempt.tokens_input}+{verdict.attempt.tokens_output}"
            f" · cost ${verdict.attempt.cost_usd:.4f}[/dim]"
        )
    _render_verdict_line(verdict)


def render_task_run(task_run: TaskRun) -> None:
    """Render a TaskRun: the final attempt's signals/causal-analysis, plus
    the cost block accounting for *every* attempt — dead ends included,
    not just the one that ended up DONE.
    """
    _render_header(task_run.final)
    _render_signals(task_run.final)

    cost = task_run.total_cost_usd
    cost_str = f"${cost:.4f}" if cost is not None else "unknown"
    console.print(
        f"[dim]attempts {task_run.attempt_count} ({task_run.failed_attempt_count} failed)"
        f" · tokens {task_run.total_tokens_input}+{task_run.total_tokens_output}"
        f" · cost {cost_str}[/dim]"
    )

    _render_verdict_line(task_run.final)
