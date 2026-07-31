from __future__ import annotations

from verdict.schema import (
    AttemptResult,
    Confidence,
    GateStatus,
    Provenance,
    Signal,
    Verdict,
    VerdictStatus,
)


def _attempt() -> AttemptResult:
    return AttemptResult(diff="", files_changed=[])


def _signal(name: str, status: GateStatus, provenance: Provenance = Provenance.PROVEN) -> Signal:
    return Signal(name=name, provenance=provenance, status=status, detail="")


def test_done_true_when_all_proven_signals_pass() -> None:
    v = Verdict(
        task="t", agent="mock", repo="/tmp/x", attempt=_attempt(),
        signals=[_signal("test", GateStatus.PASS)],
    )
    assert v.status is VerdictStatus.DONE
    assert v.done is True


def test_done_false_when_any_proven_signal_fails() -> None:
    v = Verdict(
        task="t", agent="mock", repo="/tmp/x", attempt=_attempt(),
        signals=[_signal("test", GateStatus.PASS), _signal("typecheck", GateStatus.FAIL)],
    )
    assert v.status is VerdictStatus.NOT_DONE
    assert v.done is False


def test_judged_signal_never_makes_or_breaks_done() -> None:
    failing_proven = _signal("test", GateStatus.FAIL)
    glowing_judged = _signal("vibes", GateStatus.PASS, provenance=Provenance.JUDGED)
    v = Verdict(
        task="t", agent="mock", repo="/tmp/x", attempt=_attempt(),
        signals=[failing_proven, glowing_judged],
    )
    # a judged opinion can never rescue a failing proven signal
    assert v.status is VerdictStatus.NOT_DONE


def test_unverified_when_no_signals_at_all() -> None:
    v = Verdict(task="t", agent="mock", repo="/tmp/x", attempt=_attempt(), signals=[])
    assert v.status is VerdictStatus.UNVERIFIED
    assert v.done is False


def test_unverified_when_every_gate_is_na() -> None:
    v = Verdict(
        task="t", agent="mock", repo="/tmp/x", attempt=_attempt(),
        signals=[_signal("test", GateStatus.NA), _signal("lint", GateStatus.NA)],
    )
    assert v.status is VerdictStatus.UNVERIFIED


def test_na_gates_excluded_but_dont_block_done() -> None:
    v = Verdict(
        task="t", agent="mock", repo="/tmp/x", attempt=_attempt(),
        signals=[_signal("test", GateStatus.PASS), _signal("build", GateStatus.NA)],
    )
    assert v.status is VerdictStatus.DONE


def test_confidence_low_when_test_gate_missing() -> None:
    v = Verdict(
        task="t", agent="mock", repo="/tmp/x", attempt=_attempt(),
        signals=[_signal("lint", GateStatus.PASS)],
    )
    assert v.confidence is Confidence.LOW


def test_confidence_low_when_test_gate_na() -> None:
    v = Verdict(
        task="t", agent="mock", repo="/tmp/x", attempt=_attempt(),
        signals=[_signal("test", GateStatus.NA), _signal("build", GateStatus.PASS)],
    )
    assert v.confidence is Confidence.LOW


def test_confidence_high_when_test_gate_ran() -> None:
    v = Verdict(
        task="t", agent="mock", repo="/tmp/x", attempt=_attempt(),
        signals=[_signal("test", GateStatus.PASS)],
    )
    assert v.confidence is Confidence.HIGH
