"""Builds the body of the GitHub Action's advisory PR comment: JUDGED
signals only, explicitly labeled as opinion, and explicitly never the
thing that decided the check's pass/fail — that's the action's own exit
code, driven entirely by PROVEN signals in `grade_existing_diff`'s Verdict.
See DESIGN.md's Phase 6 section for why these two are kept structurally
separate (a comment vs. the check itself) rather than merged into one
signal.

This module only builds the comment *text* — a pure function of the JSON
report's already-serialized data, directly testable with no network, no
`gh` CLI, no GitHub API. Posting it is the action's job (a couple of
`gh pr comment` calls in `action.yml`), kept out of this package entirely:
Verdict doesn't need a GitHub API client to do its actual job, and adding
one here would mean a network dependency for something a two-line shell
step already does better.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MARKER = "<!-- verdict-report:judged-signals -->"


def _judged_lines(config: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for task_run in config.get("task_runs", []):
        attempts = task_run.get("attempts") or []
        if not attempts:
            continue
        final = attempts[-1]
        judged = [s for s in final.get("signals", []) if s.get("provenance") == "judged"]
        if not judged:
            continue
        lines.append(f"**{task_run.get('task', '(task)')}**")
        for signal in judged:
            mark = "✅" if signal.get("status") == "pass" else "⚠️"
            name = signal.get("name", "?")
            detail = signal.get("detail", "")
            lines.append(f"- {mark} `{name}` — {detail}")
        lines.append("")
    return lines


def build_comment(report: dict[str, Any]) -> str:
    """`report` is the same dict `report_json.to_report_dict` produces —
    `{"schema_version": ..., "configs": [...]}`."""
    body = [
        MARKER,
        "### Verdict — judged signals (advisory only, not blocking)",
        "",
    ]

    any_judged = False
    for config in report.get("configs", []):
        lines = _judged_lines(config)
        if lines:
            any_judged = True
            body.append(f"#### `{config.get('label', '?')}`")
            body.extend(lines)

    if not any_judged:
        body.append("_No JUDGED signals were produced by this run._")

    body.append(
        "\n> Judged signals are a model's opinion and never affect the PASS/FAIL check above — "
        "see the uploaded `verdict-report` artifact for the full proven/judged breakdown and "
        "causal analysis."
    )
    return "\n".join(body)


def build_comment_from_file(report_path: Path) -> str:
    report = json.loads(report_path.read_text())
    return build_comment(report)
