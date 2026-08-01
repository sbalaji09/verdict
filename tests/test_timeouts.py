"""Phase 9's semantics, made explicit and tested separately rather than
lumped together (per the design decision in DESIGN.md's Phase 9 section):

- A gate that hangs is the AGENT's fault — a real PROVEN FAIL, bounded and
  reported cleanly, never a crash or a special status.
- A PROVISIONING timeout (sandbox startup, dependency install) is
  INFRASTRUCTURE's fault — it aborts the whole attempt via
  `ProvisioningTimeoutError`, the same way an adapter CLI hanging already
  did before this phase; it must never become a gate Signal.
- The global per-attempt budget is a third, separate thing: exceeding it
  doesn't fail anything that already ran, it just stops the attempt from
  starting anything further, and the resulting Verdict is never a clean
  DONE — see `schema.py`'s `Verdict.status`.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from verdict.adapters.mock import MockAdapter
from verdict.config import VerdictConfig
from verdict.gates.registry import run_all_gates
from verdict.runner import run
from verdict.sandbox import SandboxConfig
from verdict.sandbox.base import ProvisioningTimeoutError
from verdict.sandbox.local import LocalSandbox
from verdict.schema import AttemptResult, GateStatus, Provenance, Signal, Verdict, VerdictStatus


def _git_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative_path, content in files.items():
        target = repo / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    return repo


# --- 1. gate timeout == PROVEN FAIL, bounded and clean ---------------------

INFINITE_LOOP_TEST = (
    "def test_hangs():\n"
    "    while True:\n"
    "        pass\n"
)


@pytest.fixture
def infinite_loop_repo(tmp_path: Path) -> Path:
    """A repo whose only test never returns — an agent-introduced hang,
    not an infra problem."""
    return _git_repo(
        tmp_path,
        {
            "test_hangs.py": INFINITE_LOOP_TEST,
            "pytest.ini": "[pytest]\n",
        },
    )


def test_agent_introduced_infinite_loop_is_bounded_and_reported_as_a_proven_fail(
    infinite_loop_repo: Path,
) -> None:
    adapter = MockAdapter(patches={"README.md": "noop\n"})
    sandbox_config = SandboxConfig(backend="local", gate_timeout_seconds=2)

    started = time.monotonic()
    verdict = run(task="noop", repo=infinite_loop_repo, adapter=adapter, sandbox_config=sandbox_config)
    elapsed = time.monotonic() - started

    # Bounded: the gate timeout is what stopped this, not the test suite's
    # own patience running out.
    assert elapsed < 20

    test_signal = next(s for s in verdict.signals if s.name == "test")
    assert test_signal.provenance is Provenance.PROVEN
    assert test_signal.status is GateStatus.FAIL
    assert "timed out" in test_signal.detail

    # A real, observed FAIL — not budget-related, not an error — so this
    # is unambiguously NOT_DONE, same as any other failing gate.
    assert verdict.status is VerdictStatus.NOT_DONE
    assert not verdict.budget_exceeded


def test_run_all_gates_directly_reports_a_hang_as_fail_not_a_crash(infinite_loop_repo: Path) -> None:
    """Same claim, one layer down — exercises `gates/registry.py` directly
    rather than the full `runner.run()` pipeline."""
    sandbox = LocalSandbox()
    signals, budget_exceeded = run_all_gates(
        infinite_loop_repo, VerdictConfig(gate_overrides={}), sandbox=sandbox, timeout_seconds=2
    )
    assert not budget_exceeded
    test_signal = next(s for s in signals if s.name == "test")
    assert test_signal.status is GateStatus.FAIL
    assert test_signal.exit_code == 124


# --- 2. provisioning timeout aborts the attempt, never becomes a Signal ----


def test_provisioning_timeout_raises_and_never_produces_a_signal(tmp_path: Path, monkeypatch) -> None:
    """`run_setup_step` raising `ProvisioningTimeoutError` must propagate
    all the way out of `runner.run()` uncaught — this is deliberately NOT
    caught and turned into a gate Signal anywhere in the pipeline. As of
    Phase 11, `run()` itself is still where this contract is tested
    directly; callers built on top of it route it differently:
    `run_with_retries` (`verdict run`/`verdict bench`) now catches it and
    reports it as a `VerdictStatus.ERROR` `Verdict` (see
    `test_error_retry.py`), while `grade_existing_diff` (`verdict gate`)
    still lets it propagate to `cli.py`'s `_RUN_ERRORS`, unchanged.
    """
    import verdict.runner as runner_module

    def _raise_provisioning_timeout(worktree: Path, config: SandboxConfig) -> dict[str, str]:
        raise ProvisioningTimeoutError("dependency install (npm install) timed out after 1s")

    monkeypatch.setattr(runner_module, "run_setup_step", _raise_provisioning_timeout)

    repo = _git_repo(tmp_path, {"README.md": "hi\n"})
    adapter = MockAdapter(patches={"README.md": "bye\n"})

    with pytest.raises(ProvisioningTimeoutError):
        run(task="noop", repo=repo, adapter=adapter, sandbox_config=SandboxConfig(backend="local"))


def test_install_step_timeout_raises_provisioning_timeout_error(tmp_path: Path, monkeypatch) -> None:
    """The real (not monkeypatched) `run_setup_step`, with a stand-in
    install command that hangs — proves the timeout->exception wiring
    itself, not just that the exception type propagates correctly.
    """
    import verdict.sandbox.install as install_module

    monkeypatch.setattr(install_module, "_detect_install_command", lambda worktree: ["sleep", "5"])

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    config = SandboxConfig(backend="local", install_timeout_seconds=1)

    with pytest.raises(ProvisioningTimeoutError, match="timed out"):
        install_module.run_setup_step(worktree, config)


def test_install_step_no_command_detected_is_a_silent_noop(tmp_path: Path) -> None:
    """Confirms Phase 9's changes didn't touch the "nothing to install"
    path — no exception, nothing happens."""
    import verdict.sandbox.install as install_module

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    install_module.run_setup_step(worktree, SandboxConfig(backend="local"))  # must not raise


# --- 3. the global per-attempt budget --------------------------------------


def test_run_all_gates_skips_everything_once_the_deadline_has_already_passed(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path, {"README.md": "hi\n"})  # no detectable stack at all — irrelevant here
    sandbox = LocalSandbox()
    already_passed_deadline = time.monotonic() - 1

    signals, budget_exceeded = run_all_gates(
        repo, VerdictConfig(gate_overrides={}), sandbox=sandbox, deadline=already_passed_deadline
    )
    assert budget_exceeded
    assert signals == []


def test_runner_marks_budget_exceeded_and_never_reports_done_when_the_budget_is_zero(
    tmp_path: Path,
) -> None:
    repo = _git_repo(
        tmp_path,
        {
            "calculator.py": "def add(a, b):\n    return a + b\n",
            "test_calculator.py": (
                "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
            ),
            "pytest.ini": "[pytest]\n",
        },
    )
    adapter = MockAdapter(patches={"README.md": "noop\n"})
    sandbox_config = SandboxConfig(backend="local", attempt_budget_seconds=0)

    verdict = run(task="noop", repo=repo, adapter=adapter, sandbox_config=sandbox_config)

    assert verdict.budget_exceeded
    assert verdict.signals == []
    # Never DONE — "we ran out of time before proving it," not a win, even
    # though nothing failed (nothing ran).
    assert verdict.status is VerdictStatus.UNVERIFIED
    assert not verdict.done


def _verdict(signals: list[Signal], budget_exceeded: bool) -> Verdict:
    return Verdict(
        task="t",
        agent="mock",
        repo="r",
        attempt=AttemptResult(diff="", files_changed=[]),
        signals=signals,
        budget_exceeded=budget_exceeded,
    )


def _passing(name: str) -> Signal:
    return Signal(name=name, provenance=Provenance.PROVEN, status=GateStatus.PASS, detail="ok")


def _failing(name: str) -> Signal:
    return Signal(name=name, provenance=Provenance.PROVEN, status=GateStatus.FAIL, detail="bad")


def test_budget_exceeded_with_all_passing_signals_is_unverified_not_done() -> None:
    verdict = _verdict([_passing("test"), _passing("typecheck")], budget_exceeded=True)
    assert verdict.status is VerdictStatus.UNVERIFIED
    assert not verdict.done


def test_budget_exceeded_with_a_real_failure_is_still_not_done_the_failure_wins() -> None:
    """A real observed PROVEN FAIL counts regardless of budget — it doesn't
    get demoted to UNVERIFIED just because coverage elsewhere was cut
    short."""
    verdict = _verdict([_passing("test"), _failing("typecheck")], budget_exceeded=True)
    assert verdict.status is VerdictStatus.NOT_DONE


def test_budget_not_exceeded_with_all_passing_signals_is_done() -> None:
    """Control case: without `budget_exceeded`, the same all-pass signal
    set is a normal DONE — confirms the new field only changes behavior
    when it's actually set."""
    verdict = _verdict([_passing("test"), _passing("typecheck")], budget_exceeded=False)
    assert verdict.status is VerdictStatus.DONE
