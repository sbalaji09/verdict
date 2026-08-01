"""The `build` gate: run the repo's own `npm run build`.

Unlike test/typecheck/lint, there's no cross-project convention for "the
build step" — the same command name (`next build`, `vite build`, `tsc -p .`,
a custom webpack invocation) means different things per project, and none
of them share a machine-readable pass/fail schema the way junit XML or
eslint's JSON reporter do. So this gate deliberately does not try to
detect *which* bundler is in use — it trusts the project's own `package.json`
"build" script, the same way the test gate prefers a repo's own pytest
config over guessing. The signal here is coarser than the other three
(exit code + log tail, no structured diagnostic list) as a direct
consequence, not an oversight — see DESIGN.md.

There's no equivalent autodetected build gate for Python: no comparable
"the build script" convention exists across Python projects. A verdict.yml
override is the only way to get a build gate on a non-npm repo.
"""

from __future__ import annotations

import json
from pathlib import Path

from verdict.gates.base import DEFAULT_TIMEOUT_SECONDS, ToolRunner, exec_command, tail
from verdict.sandbox import Sandbox
from verdict.schema import GateStatus, Provenance, Signal


class NpmBuildRunner:
    tool = "npm-build"
    gate = "build"

    def applicable(self, worktree: Path) -> bool:
        package_json = worktree / "package.json"
        if not package_json.exists():
            return False
        try:
            data = json.loads(package_json.read_text())
        except json.JSONDecodeError:
            return False
        return "build" in data.get("scripts", {})

    def run(
        self, worktree: Path, sandbox: Sandbox | None = None, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    ) -> Signal:
        command = ["npm", "run", "build", "--silent"]
        result = exec_command(command, cwd=worktree, sandbox=sandbox, timeout_seconds=timeout_seconds)
        detail = tail(result.stdout + result.stderr)
        status = GateStatus.PASS if result.returncode == 0 else GateStatus.FAIL
        return Signal(
            name=self.gate,
            provenance=Provenance.PROVEN,
            status=status,
            detail=detail,
            command=" ".join(command),
            exit_code=result.returncode,
        )


RUNNERS: list[ToolRunner] = [NpmBuildRunner()]
