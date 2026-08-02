"""Phase 14: mirrors Phase 4's frontend truth on the backend. A service can
compile and pass every unit test yet still fail to boot, fail a migration,
or answer its own API wrong — none of which `test`/`typecheck`/`build`/
`lint` can see, because all four grade the CODE, never the running
PROCESS. This module actually boots the repo's own backend (`backend.start`
in `verdict.yml`) inside the sandbox and checks it the way a human would:
does it come up at all, does its schema migration apply, does a real
request against it come back right.

Three PROVEN signals, run in dependency order — each one only runs if the
one before it didn't already fail, since there's no point booting a
service against an unmigrated database or smoke-testing one that never
came up:

1. **`backend:migrate`** — optional (`backend.migrate` in `verdict.yml`).
   Skipped (no signal at all) if not configured.
2. **`backend:boot`** — did `backend.start` come up and answer
   `backend.health_url` within `ready_timeout_seconds`. Skipped entirely
   if migrate ran and failed.
3. **`backend:smoke:<name>`** — one per configured `smoke:` request: a
   real HTTP call against the booted service, checked against an
   expected status code and (optionally) a body substring. Skipped
   entirely if boot failed — there is nothing to send a request to.

Every failure here is a boot/migration/response defect the AGENT's code
introduced, not infrastructure Verdict itself couldn't provide — by the
time this module runs, `runner.py` has already gotten the sandbox itself
created and any declared `services:` (Postgres, Redis, ...) health-checked
(Phase 10's `setup_services`, which raises `SetupError` — a `SandboxError`
— straight past this module entirely, caught only by `runner.py`'s
`_EVALUATION_ERRORS` machinery and reported `ERROR`, never NOT_DONE). So
by construction, everything this module can observe is genuinely the
agent's own service failing to boot/migrate/answer correctly — a real
PROVEN FAIL, same discipline as every other gate. A `SandboxUnavailableError`
raised while actually executing a command here (a `docker exec` itself
failing, not the command it ran) is deliberately NOT caught — it
propagates the same way it already does for every gate in `gates/`, up
through `runner.py` to the same ERROR-routing/retry machinery, never
silently reinterpreted as a boot failure.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from verdict.backend.server import BackendServiceError, boot_service
from verdict.config import BackendConfig, SmokeRequestSpec, VerdictConfig
from verdict.gates.base import exec_command, tail
from verdict.sandbox import Sandbox, SandboxConfig
from verdict.sandbox.config import fallback_sandbox
from verdict.schema import GateStatus, Provenance, Signal

_SMOKE_TIMEOUT_SECONDS = 10


def run_backend_checks(
    worktree_path: Path,
    config: VerdictConfig,
    sandbox: Sandbox | None = None,
    sandbox_config: SandboxConfig | None = None,
) -> list[Signal]:
    """Empty list if `verdict.yml` has no `backend:` section — entirely
    opt-in, the same "no section, no checks" contract `frontend:` already
    established.
    """
    backend = config.backend
    if backend is None:
        return []
    sandbox = sandbox or fallback_sandbox()
    sandbox_config = sandbox_config or SandboxConfig()

    signals: list[Signal] = []

    if backend.migrate is not None:
        migrate_signal = _run_migrate(backend, worktree_path, sandbox, sandbox_config)
        signals.append(migrate_signal)
        if migrate_signal.status is GateStatus.FAIL:
            return signals  # nothing downstream can be trusted against an unmigrated DB

    try:
        with boot_service(
            backend.start, worktree_path, backend.health_url, backend.ready_timeout_seconds, sandbox
        ):
            signals.append(
                Signal(
                    name="backend:boot",
                    provenance=Provenance.PROVEN,
                    status=GateStatus.PASS,
                    detail=f"answered {backend.health_url} within {backend.ready_timeout_seconds}s",
                    command=backend.start,
                )
            )
            for spec in backend.smoke:
                signals.append(_run_smoke_request(spec, backend.health_url))
    except BackendServiceError as exc:
        signals.append(
            Signal(
                name="backend:boot",
                provenance=Provenance.PROVEN,
                status=GateStatus.FAIL,
                detail=str(exc),
                command=backend.start,
            )
        )

    return signals


def _run_migrate(
    backend: BackendConfig, worktree_path: Path, sandbox: Sandbox, sandbox_config: SandboxConfig
) -> Signal:
    assert backend.migrate is not None
    result = exec_command(
        ["sh", "-c", backend.migrate],
        cwd=worktree_path,
        sandbox=sandbox,
        timeout_seconds=sandbox_config.gate_timeout_seconds,
    )
    status = GateStatus.PASS if result.returncode == 0 else GateStatus.FAIL
    detail = "migration applied cleanly" if status is GateStatus.PASS else tail(result.stdout + result.stderr)
    return Signal(
        name="backend:migrate",
        provenance=Provenance.PROVEN,
        status=status,
        detail=detail,
        command=backend.migrate,
        exit_code=result.returncode,
    )


def _run_smoke_request(spec: SmokeRequestSpec, health_url: str) -> Signal:
    name = f"backend:smoke:{spec.name or f'{spec.method} {spec.path}'}"
    url = urllib.parse.urljoin(health_url, spec.path)
    request = urllib.request.Request(
        url,
        data=spec.body.encode() if spec.body is not None else None,
        method=spec.method,
        headers=spec.headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=_SMOKE_TIMEOUT_SECONDS) as resp:
            status, body = resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode(errors="replace")
    except (OSError, ValueError) as exc:
        return Signal(
            name=name,
            provenance=Provenance.PROVEN,
            status=GateStatus.FAIL,
            detail=f"request to {url} failed: {exc}",
            command=f"{spec.method} {url}",
        )

    problems = []
    if status != spec.expect_status:
        problems.append(f"expected status {spec.expect_status}, got {status}")
    if spec.expect_body_contains is not None and spec.expect_body_contains not in body:
        problems.append(f"expected response body to contain {spec.expect_body_contains!r}")

    if problems:
        detail = "; ".join(problems) + f"\nresponse body:\n{tail(body)}"
        return Signal(
            name=name,
            provenance=Provenance.PROVEN,
            status=GateStatus.FAIL,
            detail=detail,
            command=f"{spec.method} {url}",
        )
    return Signal(
        name=name,
        provenance=Provenance.PROVEN,
        status=GateStatus.PASS,
        detail=f"{spec.method} {url} -> {status}",
        command=f"{spec.method} {url}",
    )
