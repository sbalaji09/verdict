from __future__ import annotations

from verdict.pr_comment import MARKER, build_comment


def _signal(name: str, provenance: str, status: str, detail: str) -> dict:
    return {"name": name, "provenance": provenance, "status": status, "detail": detail}


def _task_run(task: str, signals: list[dict]) -> dict:
    return {"task": task, "attempts": [{"signals": signals}]}


def _report(configs: list[dict]) -> dict:
    return {"schema_version": 1, "configs": configs}


def test_build_comment_lists_judged_signals_with_their_task() -> None:
    report = _report(
        [
            {
                "label": "claude-code",
                "task_runs": [
                    _task_run(
                        "make the CTA green",
                        [
                            _signal("test", "proven", "pass", "ok"),
                            _signal(
                                "frontend:vision_intent:cta", "judged", "pass", "looks green and prominent"
                            ),
                        ],
                    )
                ],
            }
        ]
    )

    comment = build_comment(report)

    assert MARKER in comment
    assert "make the CTA green" in comment
    assert "frontend:vision_intent:cta" in comment
    assert "looks green and prominent" in comment
    assert "`test`" not in comment  # the PROVEN signal must never be listed here


def test_build_comment_only_includes_judged_signals_not_proven_ones() -> None:
    report = _report(
        [
            {
                "label": "claude-code",
                "task_runs": [_task_run("fix the bug", [_signal("test", "proven", "fail", "1 failed")])],
            }
        ]
    )

    comment = build_comment(report)

    assert "No JUDGED signals were produced" in comment
    assert "1 failed" not in comment


def test_build_comment_handles_a_task_run_with_no_attempts() -> None:
    report = _report([{"label": "x", "task_runs": [{"task": "t", "attempts": []}]}])
    comment = build_comment(report)
    assert "No JUDGED signals were produced" in comment


def test_build_comment_states_it_is_advisory_and_non_blocking() -> None:
    comment = build_comment(_report([]))
    assert "never affect the PASS/FAIL check" in comment


def test_build_comment_covers_multiple_configs_independently() -> None:
    report = _report(
        [
            {
                "label": "config-a",
                "task_runs": [_task_run("t1", [_signal("vision", "judged", "pass", "good a")])],
            },
            {
                "label": "config-b",
                "task_runs": [_task_run("t2", [_signal("vision", "judged", "fail", "bad b")])],
            },
        ]
    )

    comment = build_comment(report)

    assert "config-a" in comment and "good a" in comment
    assert "config-b" in comment and "bad b" in comment
