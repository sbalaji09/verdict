"""A `Sandbox` fake for unit tests that need to exercise a call site's
command-construction/output-parsing logic without a real backend — the
same "hand-rolled Protocol fake" convention `adapters/mock.py` and
`frontend/vision_judge.py`'s `MockVisionJudge` already established.
"""

from __future__ import annotations

from pathlib import Path

from verdict.sandbox.base import BackgroundHandle, ExecResult, ResourceLimits


class FakeSandbox:
    """Records every `exec()` call and returns canned `ExecResult`s (one
    fixed result, or a queue popped in order for multi-call scenarios), or
    raises a canned exception instead — enough to drive any adapter/gate's
    error-handling branches without a real subprocess.
    """

    def __init__(
        self,
        result: ExecResult | None = None,
        results: list[ExecResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._result = result
        self._results = list(results) if results is not None else None
        self._error = error

    def __enter__(self) -> "FakeSandbox":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def exec(
        self,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout_seconds: int = 600,
        limits: ResourceLimits | None = None,
        network: bool = False,
    ) -> ExecResult:
        self.calls.append({"cmd": cmd, "cwd": cwd, "env": env, "network": network})
        if self._error is not None:
            raise self._error
        if self._results is not None:
            return self._results.pop(0)
        return self._result if self._result is not None else ExecResult(exit_code=0, stdout="", stderr="")

    def exec_background(
        self,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        network: bool = False,
    ) -> BackgroundHandle:
        raise NotImplementedError("FakeSandbox doesn't support exec_background")
