"""Renders a Verdict to the terminal, keeping the PROVEN/JUDGED split
visually explicit — that split is the entire point of the tool.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from verdict.schema import Provenance, Verdict

console = Console()


def render(verdict: Verdict) -> None:
    header = Text()
    header.append("Verdict", style="bold")
    header.append(f"   repo: {verdict.repo}   agent: {verdict.agent}\n", style="dim")
    header.append(f"task: {verdict.task}", style="dim")
    console.print(header)
    console.print()

    proven = [s for s in verdict.signals if s.provenance is Provenance.PROVEN]
    judged = [s for s in verdict.signals if s.provenance is Provenance.JUDGED]

    if proven:
        console.print("[bold]PROVEN[/bold] [dim](executed)[/dim]")
        for s in proven:
            mark = "[green]✓[/green]" if s.passed else "[red]✗[/red]"
            cmd = f"  [dim]{s.command}[/dim]" if s.command else ""
            console.print(f"  {mark} {s.name:<12}{cmd}")
            if not s.passed:
                console.print(Panel(s.detail, border_style="red", expand=False))
        console.print()

    if judged:
        console.print("[bold]JUDGED[/bold] [dim](model opinion)[/dim]")
        for s in judged:
            mark = "[green]~[/green]" if s.passed else "[yellow]~[/yellow]"
            console.print(f"  {mark} {s.name:<12}  {s.detail}")
        console.print()

    if verdict.attempt.cost_usd is not None:
        console.print(
            f"[dim]tokens {verdict.attempt.tokens_input}+{verdict.attempt.tokens_output}"
            f" · cost ${verdict.attempt.cost_usd:.4f}[/dim]"
        )

    if verdict.done:
        console.print("\n[bold green]VERDICT: DONE[/bold green]")
    else:
        console.print("\n[bold red]VERDICT: NOT DONE[/bold red]")
