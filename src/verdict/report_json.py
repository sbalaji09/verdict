"""The JSON reporter: a stable, machine-readable serialization of whatever
Verdict already computed — no new data, no new decisions. `Verdict`,
`TaskRun`, and `ConfigResult` are already Pydantic models (they have been
since Phase 0), so this module is deliberately thin: it doesn't reshape
anything, it wraps `ConfigResult.model_dump(mode="json")` in a small,
versioned envelope so a consumer (the PR-comment builder, a future
dashboard, a CI script) has one stable top-level shape to parse regardless
of which command produced the report.
"""

from __future__ import annotations

import json

from verdict.schema import ConfigResult

SCHEMA_VERSION = 1


def to_report_dict(config_results: list[ConfigResult]) -> dict[str, object]:
    """The envelope both `render_json` and the HTML reporter build from —
    exposed separately so callers that want the structured dict (rather
    than a JSON string) don't have to round-trip through `json.loads`.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "configs": [cr.model_dump(mode="json") for cr in config_results],
    }


def render_json(config_results: list[ConfigResult]) -> str:
    return json.dumps(to_report_dict(config_results), indent=2)
