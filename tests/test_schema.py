from __future__ import annotations

from verdict.schema import AttemptResult, Provenance, Signal, Verdict


def _attempt() -> AttemptResult:
    return AttemptResult(diff="", files_changed=[])


def test_done_true_when_all_proven_signals_pass() -> None:
    v = Verdict(
        task="t",
        agent="mock",
        repo="/tmp/x",
        attempt=_attempt(),
        signals=[Signal(name="test", provenance=Provenance.PROVEN, passed=True, detail="ok")],
    )
    assert v.done is True


def test_done_false_when_any_proven_signal_fails() -> None:
    v = Verdict(
        task="t",
        agent="mock",
        repo="/tmp/x",
        attempt=_attempt(),
        signals=[
            Signal(name="test", provenance=Provenance.PROVEN, passed=True, detail="ok"),
            Signal(name="typecheck", provenance=Provenance.PROVEN, passed=False, detail="bad"),
        ],
    )
    assert v.done is False


def test_judged_signal_never_makes_or_breaks_done() -> None:
    failing_proven = Signal(
        name="test", provenance=Provenance.PROVEN, passed=False, detail="fail"
    )
    glowing_judged = Signal(
        name="vibes", provenance=Provenance.JUDGED, passed=True, detail="looks great"
    )
    v = Verdict(
        task="t", agent="mock", repo="/tmp/x", attempt=_attempt(),
        signals=[failing_proven, glowing_judged],
    )
    # a judged opinion can never rescue a failing proven signal
    assert v.done is False


def test_done_false_with_no_signals_at_all() -> None:
    v = Verdict(task="t", agent="mock", repo="/tmp/x", attempt=_attempt(), signals=[])
    assert v.done is False
