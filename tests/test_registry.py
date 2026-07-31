from __future__ import annotations

from pathlib import Path

from verdict.config import VerdictConfig
from verdict.gates.registry import resolve_gate
from verdict.schema import GateStatus


def test_resolve_gate_not_applicable_when_no_stack_detected(tmp_path: Path) -> None:
    signal = resolve_gate("build", tmp_path, VerdictConfig(gate_overrides={}))
    assert signal.status is GateStatus.NA
    assert signal.name == "build"


def test_resolve_gate_honors_verdict_yml_override(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("hi")
    config = VerdictConfig(gate_overrides={"build": "true"})
    signal = resolve_gate("build", tmp_path, config)
    assert signal.status is GateStatus.PASS
    assert signal.command == "true"


def test_resolve_gate_override_failure_is_proven_fail(tmp_path: Path) -> None:
    config = VerdictConfig(gate_overrides={"lint": "false"})
    signal = resolve_gate("lint", tmp_path, config)
    assert signal.status is GateStatus.FAIL


def test_resolve_gate_picks_first_applicable_runner(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert True\n")
    signal = resolve_gate("test", tmp_path, VerdictConfig(gate_overrides={}))
    assert signal.command is not None
    assert "pytest" in signal.command
