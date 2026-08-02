"""The json/html reporters both work from the same `list[ConfigResult]`
shape the CLI's `bench` command already produces — these tests build that
shape directly (following `test_economics.py`'s precedent) rather than
running a full grading pipeline, since the reporters' own job is
serialization/rendering, not grading.
"""

from __future__ import annotations

from datetime import UTC, datetime
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
from verdict.store.base import TaskOutcome


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


def test_html_report_links_an_artifact_path_instead_of_printing_it_as_text() -> None:
    verdict = Verdict(
        task="frontend glitch",
        agent="mock",
        repo="/tmp/x",
        attempt=AttemptResult(diff=""),
        signals=[
            Signal(
                name="frontend:glitch_scan:load",
                provenance=Provenance.PROVEN,
                status=GateStatus.FAIL,
                detail="flicker detected",
                artifact_path="/tmp/verdict-capture/run1/load.webm",
            )
        ],
    )
    task_run = TaskRun(task=verdict.task, agent="mock", repo="/tmp/x", attempts=[verdict])
    html = render_html([ConfigResult(label="cfg", task_runs=[task_run])])

    assert '<a class="artifact" href="/tmp/verdict-capture/run1/load.webm">' in html


# --- Phase 17: History / trend section -------------------------------


def test_html_report_omits_the_history_section_by_default() -> None:
    html = render_html([_config_result("good", done=True)])
    assert "<h2>History</h2>" not in html


def test_html_report_renders_a_history_section_when_history_is_supplied() -> None:
    outcomes = [
        TaskOutcome(
            run_id=f"run-{i}",
            recorded_at=datetime(2026, 1, i + 1, tzinfo=UTC),
            commit_sha=f"sha{i}",
            config_label="good",
            task="fix add()",
            agent="mock",
            repo="/tmp/x",
            done=(i != 2),  # one failure in the middle of an otherwise-clean run
            status="done" if i != 2 else "not_done",
        )
        for i in range(5)
    ]
    html = render_html(
        [_config_result("good", done=True)],
        history={("good", "fix add()", "mock", "/tmp/x"): outcomes},
    )

    assert "<h2>History</h2>" in html
    assert "<svg" in html
    assert "4/5" in html  # 4 of the 5 recorded outcomes passed
    HTMLParser().feed(html)  # still well-formed with the section present


def test_html_report_history_section_handles_a_task_with_no_recorded_history() -> None:
    html = render_html(
        [_config_result("good", done=True)],
        history={("good", "fix add()", "mock", "/tmp/x"): []},
    )
    assert "no history" in html
