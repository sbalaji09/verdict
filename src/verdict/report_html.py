"""The HTML reporter: a single, self-contained dashboard file (inline CSS,
inline JS, no external requests) suitable as a CI artifact — the same
`list[ConfigResult]` shape the JSON reporter and `bench`'s leaderboard
already work from, just rendered for a human instead of a script.

Three sections, same order of trust as everywhere else in this codebase:

1. **Leaderboard** — reuses `economics.rank`'s ordering, so the HTML
   report and the CLI's own leaderboard table can never disagree about
   who's ranked where.
2. **Failure-mode breakdown** — reuses `failure_modes.summarize_failure_modes`
   for the same reason.
3. **Per-task detail** — for every task run, every config: PROVEN signals,
   JUDGED signals (visually separated, never blended), causal analysis
   (attributions), and the attempt's cost. A JUDGED signal is rendered
   with an "opinion" badge and is never allowed to change a task's PASS/
   FAIL color — that color comes from `Verdict.status` alone, which
   already only consults PROVEN signals.

All interpolated text (task descriptions, signal detail, file paths) is
HTML-escaped — this file can contain a diff or a stack trace an agent
produced, and none of that is trusted markup.
"""

from __future__ import annotations

from html import escape

from verdict.economics import rank
from verdict.failure_modes import summarize_failure_modes
from verdict.schema import (
    Attribution,
    ConfigResult,
    GateStatus,
    Provenance,
    Signal,
    TaskRun,
    Verdict,
)

_STATUS_CLASS = {
    GateStatus.PASS: "pass",
    GateStatus.FAIL: "fail",
    GateStatus.NA: "na",
}


def _fmt_cost(cost: float | None) -> str:
    return f"${cost:.4f}" if cost is not None else "unknown"


def _fmt_rate(rate: float | None) -> str:
    return f"{rate:.2f}" if rate is not None else "—"


def _render_leaderboard_rows(config_results: list[ConfigResult]) -> str:
    rows = []
    for i, cr in enumerate(rank(config_results), start=1):
        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{escape(cr.label)}</td>"
            f"<td>{cr.tasks_done}/{cr.tasks_total} ({cr.pass_rate:.0%})</td>"
            f"<td>{_fmt_cost(cr.total_cost_usd)}</td>"
            f"<td>{_fmt_rate(cr.pass_rate_per_dollar)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _render_failure_mode_rows(config_results: list[ConfigResult]) -> str:
    rows = []
    for cr in config_results:
        breakdown = summarize_failure_modes(cr)
        for name, count in breakdown.counts.most_common():
            rows.append(
                f"<tr><td>{escape(cr.label)}</td><td>{escape(name)}</td><td>{count}</td></tr>"
            )
    if not rows:
        return '<tr><td colspan="3" class="dim">No failures — every config finished every task.</td></tr>'
    return "\n".join(rows)


def _render_signal_row(signal: Signal) -> str:
    status_class = _STATUS_CLASS[signal.status]
    kind = "opinion" if signal.provenance is Provenance.JUDGED else "proven"
    artifact = (
        f'<div class="artifact">recording: {escape(signal.artifact_path)}</div>'
        if signal.artifact_path
        else ""
    )
    return (
        f'<div class="signal {status_class} {kind}">'
        f'<span class="mark">{signal.status.value}</span> '
        f'<span class="name">{escape(signal.name)}</span> '
        f'<span class="badge">{"JUDGED" if kind == "opinion" else "PROVEN"}</span>'
        f'<div class="detail">{escape(signal.detail)}</div>'
        f"{artifact}"
        "</div>"
    )


def _render_attribution(attribution: Attribution) -> str:
    return f'<div class="attribution">{escape(attribution.explanation)}</div>'


def _render_verdict_detail(verdict: Verdict) -> str:
    proven = [s for s in verdict.signals if s.provenance is Provenance.PROVEN]
    judged = [s for s in verdict.signals if s.provenance is Provenance.JUDGED]

    parts = ['<div class="proven-signals">']
    parts.extend(_render_signal_row(s) for s in proven)
    parts.append("</div>")

    if judged:
        parts.append('<div class="judged-signals">')
        parts.extend(_render_signal_row(s) for s in judged)
        parts.append("</div>")

    if verdict.attributions:
        parts.append('<div class="attributions"><h4>Causal analysis</h4>')
        parts.extend(_render_attribution(a) for a in verdict.attributions)
        parts.append("</div>")

    cost = verdict.attempt.cost_usd
    parts.append(
        f'<div class="cost dim">tokens {verdict.attempt.tokens_input}+'
        f"{verdict.attempt.tokens_output} · cost {_fmt_cost(cost)}</div>"
    )
    return "\n".join(parts)


def _render_task_run(config_label: str, task_run: TaskRun) -> str:
    verdict = task_run.final
    status_class = "pass" if task_run.done else "fail"
    return (
        f'<details class="task {status_class}">'
        f'<summary><span class="status-dot"></span> '
        f"<strong>{escape(config_label)}</strong> — {escape(task_run.task)} "
        f'<span class="dim">({task_run.attempt_count} attempt'
        f'{"s" if task_run.attempt_count != 1 else ""})</span></summary>'
        f'<div class="task-body">{_render_verdict_detail(verdict)}</div>'
        "</details>"
    )


def _render_tasks(config_results: list[ConfigResult]) -> str:
    blocks = []
    for cr in config_results:
        for task_run in cr.task_runs:
            blocks.append(_render_task_run(cr.label, task_run))
    return "\n".join(blocks)


_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0; padding: 2rem; max-width: 960px; margin-inline: auto;
  background: #fafafa; color: #1a1a1a; line-height: 1.5;
}
@media (prefers-color-scheme: dark) {
  body { background: #16181c; color: #e6e6e6; }
  table { background: #1f2228 !important; }
  th { background: #262a32 !important; }
  td, th { border-color: #333 !important; }
  details.task { background: #1f2228 !important; border-color: #333 !important; }
  .signal { background: #262a32 !important; }
  .detail, .attribution, .artifact { background: #14161a !important; }
}
h1 { font-size: 1.5rem; }
h2 { font-size: 1.1rem; margin-top: 2.5rem; border-bottom: 1px solid #ddd; padding-bottom: 0.4rem; }
h4 { margin: 0.5rem 0; font-size: 0.95rem; }
table { width: 100%; border-collapse: collapse; margin-top: 0.75rem; overflow-x: auto; display: block; }
th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #e0e0e0; white-space: nowrap; }
th { background: #f0f0f0; font-weight: 600; }
.dim { color: #888; font-size: 0.9em; }
details.task {
  border: 1px solid #e0e0e0; border-radius: 8px; margin: 0.6rem 0; padding: 0.6rem 0.9rem;
  background: white;
}
details.task.fail { border-left: 4px solid #d33; }
details.task.pass { border-left: 4px solid #2a2; }
summary { cursor: pointer; list-style: none; }
summary::-webkit-details-marker { display: none; }
.status-dot { display: inline-block; width: 0.6em; height: 0.6em; border-radius: 50%; margin-right: 0.4em; }
.task.pass .status-dot { background: #2a2; }
.task.fail .status-dot { background: #d33; }
.task-body { margin-top: 0.75rem; }
.signal { margin: 0.4rem 0; padding: 0.5rem 0.75rem; border-radius: 6px; background: #f5f5f5; }
.signal.fail { border-left: 3px solid #d33; }
.signal.pass { border-left: 3px solid #2a2; }
.signal.na { border-left: 3px solid #999; opacity: 0.75; }
.signal.opinion { border-left-style: dashed; }
.mark { font-weight: 700; text-transform: uppercase; font-size: 0.8em; }
.badge {
  font-size: 0.7em; padding: 0.1em 0.5em; border-radius: 999px; border: 1px solid currentColor;
  margin-left: 0.4em; opacity: 0.7;
}
.name { font-family: ui-monospace, monospace; }
.detail, .attribution, .artifact {
  font-family: ui-monospace, monospace; font-size: 0.85em; white-space: pre-wrap;
  margin-top: 0.3rem; padding: 0.4rem 0.6rem; background: #fff; border-radius: 4px;
  max-height: 220px; overflow-y: auto;
}
.attributions { margin-top: 0.5rem; }
#filter-bar { margin: 1rem 0; }
.hidden-by-filter { display: none !important; }
"""

_JS = """
document.addEventListener('DOMContentLoaded', function () {
  var checkbox = document.getElementById('failures-only');
  if (!checkbox) return;
  checkbox.addEventListener('change', function () {
    document.querySelectorAll('details.task').forEach(function (el) {
      var isFail = el.classList.contains('fail');
      el.classList.toggle('hidden-by-filter', checkbox.checked && !isFail);
    });
  });
});
"""


def render_html(config_results: list[ConfigResult]) -> str:
    """A single, self-contained HTML file — no external stylesheet, script,
    font, or image request. Safe to open from disk or attach as a CI
    artifact with no build step."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verdict Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Verdict Report</h1>
<p class="dim">Proven signals are executed, deterministic checks — they decide pass/fail. Judged
signals are a model's opinion — advisory only, shown with a dashed border, never load-bearing.</p>

<h2>Leaderboard</h2>
<table>
<thead><tr>
<th>rank</th><th>config</th><th>pass rate</th><th>total cost</th><th>verdict-pts/$</th>
</tr></thead>
<tbody>
{_render_leaderboard_rows(config_results)}
</tbody>
</table>

<h2>Failure-mode breakdown</h2>
<table>
<thead><tr><th>config</th><th>check</th><th>failures</th></tr></thead>
<tbody>
{_render_failure_mode_rows(config_results)}
</tbody>
</table>

<h2>Tasks</h2>
<div id="filter-bar"><label><input type="checkbox" id="failures-only"> Show only failing tasks</label></div>
{_render_tasks(config_results)}

<script>{_JS}</script>
</body>
</html>
"""
