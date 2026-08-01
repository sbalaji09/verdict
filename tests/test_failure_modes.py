from __future__ import annotations

from verdict.failure_modes import summarize_failure_modes
from verdict.schema import AttemptResult, ConfigResult, GateStatus, Provenance, Signal, TaskRun, Verdict


def _verdict(signals: list[Signal]) -> Verdict:
    return Verdict(
        task="t", agent="mock", repo="/tmp/x",
        attempt=AttemptResult(diff="", cost_usd=0.0),
        signals=signals,
    )


def _proven(name: str, status: GateStatus) -> Signal:
    return Signal(name=name, provenance=Provenance.PROVEN, status=status, detail="")


def _judged(name: str, status: GateStatus) -> Signal:
    return Signal(name=name, provenance=Provenance.JUDGED, status=status, detail="")


def test_counts_failing_proven_signals_across_not_done_task_runs() -> None:
    config = ConfigResult(
        label="agent-x",
        task_runs=[
            TaskRun(
                task="t1", agent="agent-x", repo="/tmp/x",
                attempts=[_verdict([_proven("test", GateStatus.FAIL)])],
            ),
            TaskRun(
                task="t2", agent="agent-x", repo="/tmp/x",
                attempts=[_verdict([_proven("test", GateStatus.FAIL), _proven("lint", GateStatus.FAIL)])],
            ),
        ],
    )

    breakdown = summarize_failure_modes(config)

    assert breakdown.label == "agent-x"
    assert breakdown.counts["test"] == 2
    assert breakdown.counts["lint"] == 1
    assert breakdown.total_failures == 3


def test_done_task_runs_do_not_contribute_to_the_breakdown() -> None:
    config = ConfigResult(
        label="agent-x",
        task_runs=[
            TaskRun(
                task="t1", agent="agent-x", repo="/tmp/x",
                attempts=[_verdict([_proven("test", GateStatus.PASS)])],
            ),
        ],
    )
    breakdown = summarize_failure_modes(config)
    assert breakdown.total_failures == 0


def test_na_signals_are_not_counted_as_failures() -> None:
    config = ConfigResult(
        label="agent-x",
        task_runs=[
            TaskRun(
                task="t1", agent="agent-x", repo="/tmp/x",
                attempts=[
                    _verdict(
                        [_proven("test", GateStatus.FAIL), _proven("typecheck", GateStatus.NA)]
                    )
                ],
            ),
        ],
    )
    breakdown = summarize_failure_modes(config)
    assert breakdown.counts["test"] == 1
    assert "typecheck" not in breakdown.counts


def test_judged_signals_never_count_as_a_failure_mode() -> None:
    """A JUDGED opinion is never a "failure mode" in this tally, for the
    same reason it never decides Verdict.status: it's advisory, not a
    proven defect."""
    config = ConfigResult(
        label="agent-x",
        task_runs=[
            TaskRun(
                task="t1", agent="agent-x", repo="/tmp/x",
                attempts=[
                    _verdict(
                        [
                            _proven("test", GateStatus.FAIL),
                            _judged("frontend:vision_intent:cta", GateStatus.FAIL),
                        ]
                    )
                ],
            ),
        ],
    )
    breakdown = summarize_failure_modes(config)
    assert breakdown.counts["test"] == 1
    assert "frontend:vision_intent:cta" not in breakdown.counts


def test_only_the_final_attempt_of_a_multi_attempt_task_run_is_counted() -> None:
    config = ConfigResult(
        label="agent-x",
        task_runs=[
            TaskRun(
                task="t1", agent="agent-x", repo="/tmp/x",
                attempts=[
                    _verdict([_proven("typecheck", GateStatus.FAIL)]),  # dead end
                    _verdict([_proven("lint", GateStatus.FAIL)]),  # final, still not done
                ],
            ),
        ],
    )
    breakdown = summarize_failure_modes(config)
    assert breakdown.counts["lint"] == 1
    assert "typecheck" not in breakdown.counts
