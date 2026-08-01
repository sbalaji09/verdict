"""The json/html reporters both work from the same `list[ConfigResult]`
shape the CLI's `bench` command already produces — these tests build that
shape directly (following `test_economics.py`'s precedent) rather than
running a full grading pipeline, since the reporters' own job is
serialization/rendering, not grading.
"""

from __future__ import annotations

from html.parser import HTMLParser

from verdict.report_html import render_html
from verdict.report_json import render_json, to_report_dict
from verdict.schema import (
    AttemptResult,
    Attribution,
    AttributionKind,
    ConfigResult,
    GateStatus,
    Provenance,
    Signal,
    TaskRun,
    Verdict,
)


def _verdict(*, status: GateStatus, judged_status: GateStatus | None = None) -> Verdict:
    signals = [Signal(name="test", provenance=Provenance.PROVEN, status=status, detail="1 passed")]
    if judged_status is not None:
        signals.append(
            Signal(
                name="frontend:vision_intent:cta",
                provenance=Provenance.JUDGED,
                status=judged_status,
                detail="looks fine to me",
            )
        )
    attributions = []
    if status is GateStatus.FAIL:
        attributions.append(
            Attribution(
                kind=AttributionKind.REGRESSION,
                check_name="test",
                failure_id="test_x",
                culprit_file="calculator.py",
                method="single change",
                explanation="agent edited calculator.py, which caused test to fail.",
            )
        )
    return Verdict(
        task="fix <script>alert(1)</script> & things",
        agent="mock",
        repo="/tmp/x",
        attempt=AttemptResult(diff="", tokens_input=10, tokens_output=5, cost_usd=0.01),
        signals=signals,
        attributions=attributions,
    )


def _config_result(label: str, done: bool) -> ConfigResult:
    status = GateStatus.PASS if done else GateStatus.FAIL
    verdict = _verdict(status=status, judged_status=GateStatus.PASS)
    task_run = TaskRun(task=verdict.task, agent="mock", repo="/tmp/x", attempts=[verdict])
    return ConfigResult(label=label, task_runs=[task_run])


# --- JSON reporter -------------------------------------------------------


def test_json_report_includes_the_computed_status_fields() -> None:
    """Verdict.status/done/confidence are @property, not plain data — the
    report must not silently drop them, or a consumer would have to
    re-derive PROVEN-only pass/fail logic itself."""
    results = [_config_result("good", done=True)]
    report = to_report_dict(results)

    task_run = report["configs"][0]["task_runs"][0]
    assert task_run["done"] is True
    verdict = task_run["attempts"][-1]
    assert verdict["status"] == "done"
    assert verdict["done"] is True
    assert "confidence" in verdict


def test_json_report_includes_config_level_aggregates() -> None:
    results = [_config_result("good", done=True)]
    report = to_report_dict(results)
    config = report["configs"][0]
    assert config["tasks_done"] == 1
    assert config["tasks_total"] == 1
    assert config["pass_rate"] == 1.0


def test_render_json_is_valid_json_with_a_schema_version() -> None:
    import json

    output = render_json([_config_result("good", done=True)])
    parsed = json.loads(output)
    assert parsed["schema_version"] == 1
    assert len(parsed["configs"]) == 1


# --- HTML reporter --------------------------------------------------------


def test_html_report_parses_as_well_formed_html() -> None:
    html = render_html([_config_result("good", done=True), _config_result("bad", done=False)])
    HTMLParser().feed(html)  # raises on gross malformation
    assert "<!doctype html>" in html.lower()
    assert "<script>" in html
    assert "<style>" in html


def test_html_report_escapes_untrusted_task_text() -> None:
    """Task text (and signal detail) can contain anything an agent wrote —
    this must never be interpreted as markup."""
    html = render_html([_config_result("good", done=True)])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_html_report_shows_leaderboard_and_failure_mode_sections() -> None:
    html = render_html([_config_result("good", done=True), _config_result("bad", done=False)])
    assert "Leaderboard" in html
    assert "Failure-mode breakdown" in html
    assert "good" in html and "bad" in html


def test_html_report_includes_causal_analysis_for_failing_tasks() -> None:
    html = render_html([_config_result("bad", done=False)])
    assert "Causal analysis" in html
    assert "calculator.py" in html


def test_html_report_never_lets_a_judged_pass_look_like_the_verdict_passed() -> None:
    # _config_result("bad", ...) has a FAILing proven "test" signal but a
    # PASSing judged vision_intent signal — the task card must still be
    # marked failing.
    html = render_html([_config_result("bad", done=False)])
    assert 'class="task fail"' in html
