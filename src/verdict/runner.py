"""Orchestrates one end-to-end grading run: isolate a worktree, let the
adapter do the work, run the gates, assemble a Verdict.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

from verdict.adapters import Adapter
from verdict.attribution.engine import attribute_failures
from verdict.config import VerdictConfig, load_config
from verdict.frontend.runner import run_frontend_checks
from verdict.gates.registry import run_all_gates
from verdict.sandbox import Sandbox, SandboxConfig, create_sandbox
from verdict.sandbox.install import run_setup_step
from verdict.sandbox.services import ServiceSession, setup_services, teardown_services
from verdict.schema import AttemptResult, Attribution, Signal, TaskRun, Verdict
from verdict.worktree import (
    Worktree,
    commit_all,
    copy_vendored_dependencies,
    diff_against_base,
    diff_between,
    isolated_worktree,
    rev_parse,
)

DEFAULT_MAX_ATTEMPTS = 1


def _budget_deadline(sandbox_config: SandboxConfig) -> float | None:
    """A `time.monotonic()` timestamp the global per-attempt budget
    (Phase 9) expires at, or None if unset. Computed once, at the start of
    an attempt — not re-derived per phase — so the budget is genuinely
    global (adapter + install + every gate + attribution + frontend
    combined), not an independent allowance per phase.
    """
    if sandbox_config.attempt_budget_seconds is None:
        return None
    return time.monotonic() + sandbox_config.attempt_budget_seconds


def _budget_exceeded(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _sandbox_config_for_services(sandbox_config: SandboxConfig, session: ServiceSession) -> SandboxConfig:
    """The gate/adapter sandbox joins the per-attempt service network
    instead of its plain none/bridge choice when any services were
    declared (`session.network_name` set) — see `sandbox/services.py`'s
    module docstring for why an `--internal` network is safe to add here
    without reopening general internet egress. No services declared
    (the common case) → `sandbox_config` passes through unchanged.
    """
    if session.network_name is None:
        return sandbox_config
    return dataclasses.replace(sandbox_config, network_name=session.network_name)


def run(
    task: str,
    repo: Path,
    adapter: Adapter,
    sandbox_config: SandboxConfig | None = None,
) -> Verdict:
    """Run `adapter` on `task` inside a throwaway worktree of `repo`, grade
    the result against every applicable gate (test/typecheck/build/lint),
    and return the Verdict. The worktree (and its branch) is always torn
    down before this returns, win or lose.

    `sandbox_config` governs how the adapter's CLI, the setup stage, gates,
    and the frontend dev server actually execute — see `sandbox/config.py`.
    Defaults to `SandboxConfig()` (backend "local") when not given; the
    `verdict` CLI itself passes an explicit config defaulting to "docker"
    (Phase 8's actual product default) via its `--sandbox-backend` flag.

    Phase 10's setup stage runs here, in order, all before gates: declared
    `services:` are started and health-gated (infra failures raise
    `SetupError`/`ProvisioningTimeoutError` and abort the whole attempt —
    never a gate Signal, never NOT_DONE — see DESIGN.md's Phase 10
    section), then language-version pins are resolved and dependencies
    installed (`run_setup_step`), producing an env overlay every
    subsequent gate call gets merged into.
    """
    sandbox_config = sandbox_config or SandboxConfig()
    deadline = _budget_deadline(sandbox_config)

    with isolated_worktree(repo) as worktree:
        copy_vendored_dependencies(repo, worktree.path)

        # Read once, early, purely to see what services (if any) are
        # declared — deliberately BEFORE the agent runs, so a service the
        # agent's own work depends on (e.g. it runs migrations against
        # `db`) is already up. Gate overrides/frontend config are still
        # read from the *agent's* post-edit verdict.yml further down,
        # unchanged from before this phase — only the services list is
        # read this early.
        early_config = load_config(worktree.path)
        service_session = setup_services(early_config.services, sandbox_config.health_timeout_seconds)

        try:
            setup_env = run_setup_step(worktree.path, sandbox_config)
            gate_sandbox_config = _sandbox_config_for_services(sandbox_config, service_session)

            with create_sandbox(worktree.path, gate_sandbox_config) as sandbox:
                attempt = adapter.run(task, worktree.path, sandbox=sandbox)
                diff, files_changed = diff_against_base(worktree.path, worktree.base_commit)
                attempt = attempt.model_copy(update={"diff": diff, "files_changed": files_changed})

                # Commit the agent's work now, before any gate runs — gates (pytest,
                # mypy, ...) can leave their own artifacts (__pycache__, etc.) in the
                # worktree as a side effect of executing, and attribution's bisection
                # must never mistake a gate's own byproduct for something the agent
                # changed. `attempt_commit` is what attribution treats as "final".
                attempt_commit = commit_all(worktree.path, "verdict: attempt final state")

                config = load_config(worktree.path)
                attempt = _apply_pricing_fallback(attempt, config)

                signals, attributions, budget_exceeded = _run_gates_and_attribution(
                    repo,
                    worktree.path,
                    worktree,
                    attempt_commit,
                    config,
                    sandbox,
                    sandbox_config,
                    deadline,
                    setup_env,
                )

                # Frontend checks run after gates/attribution, and their signals are
                # appended afterward rather than folded into `signals` beforehand —
                # attribution's bisector only knows the four gate names in
                # gates/registry.py's GATE_RUNNERS, and would crash trying to
                # `resolve_gate("frontend:...")`. A failing frontend check is real
                # PROVEN evidence for Verdict.status either way; it's just not
                # (yet) bisectable to a culprit file the way test/typecheck/build/
                # lint failures are.
                if budget_exceeded or _budget_exceeded(deadline):
                    budget_exceeded = True
                else:
                    signals = signals + run_frontend_checks(
                        repo, worktree, config, task, sandbox=sandbox, sandbox_config=sandbox_config
                    )
        finally:
            teardown_services(service_session)

    return Verdict(
        task=task,
        agent=adapter.name,
        repo=str(repo),
        attempt=attempt,
        signals=signals,
        attributions=attributions,
        budget_exceeded=budget_exceeded,
    )


def _run_gates_and_attribution(
    repo: Path,
    gate_worktree_path: Path,
    attribution_worktree: Worktree,
    final_commit: str,
    config: VerdictConfig,
    sandbox: Sandbox,
    sandbox_config: SandboxConfig,
    deadline: float | None,
    env: dict[str, str] | None = None,
) -> tuple[list[Signal], list[Attribution], bool]:
    """Shared by `run()` and `grade_existing_diff()`: run every gate (or
    stop early at `deadline`), then attribute failures — but only if the
    budget wasn't already exhausted, since bisection re-runs gates many
    times over and starting it with no time left would just manufacture
    more inconclusive/timed-out signals rather than a real answer.

    `gate_worktree_path` and `attribution_worktree` are separate params
    (rather than deriving one from the other) because `grade_existing_diff`
    grades `repo` in place — its `Worktree.path` and the `repo` gates run
    in are the same directory, but that's a coincidence of that mode, not
    something this shared helper should assume.

    `env` (Phase 10) is the version-pin overlay from `run_setup_step` —
    merged into every gate's own execution env, empty for the unpinned
    case. Attribution's own baseline/bisection re-renders resolve their
    own overlay independently (they render different worktree states, at
    different commits, than `gate_worktree_path`), so `env` is NOT passed
    to `attribute_failures` — only to the top-level gate run here.
    """
    signals, gates_budget_exceeded = run_all_gates(
        gate_worktree_path,
        config,
        sandbox=sandbox,
        timeout_seconds=sandbox_config.gate_timeout_seconds,
        deadline=deadline,
        env=env,
    )
    if gates_budget_exceeded or _budget_exceeded(deadline):
        return signals, [], True
    attributions = attribute_failures(repo, attribution_worktree, final_commit, signals, sandbox_config)
    return signals, attributions, False


def grade_existing_diff(
    repo: Path, base_ref: str, sandbox_config: SandboxConfig | None = None
) -> Verdict:
    """Grade `repo` exactly as it's already checked out against `base_ref`
    — no adapter, no worktree isolation. This is Phase 6's merge-gate entry
    point: a pull request's diff already exists as real commits sitting on
    disk (a human or an agent produced it, Verdict doesn't need to know or
    care which), so there's no task to hand an agent and nothing to
    isolate — `repo` itself is graded in place.

    Deliberately read-only with respect to git: `diff_between` never
    stages anything (unlike `diff_against_base`), and `attribute_failures`'
    bisection always runs inside its own disposable `scratch_worktree`s off
    `repo` — never `repo`'s own working directory or index. The only thing
    this function executes *in* `repo` is the gates/frontend checks
    themselves (pytest, tsc, a dev server, ...), exactly as a CI job
    already expects to happen in its own checkout.

    Unlike `run()`, there's no setup stage here beyond services — `repo`
    is graded exactly as already checked out, dependencies assumed
    already installed by whatever CI job invoked this (unchanged from
    before Phase 10). Declared `services:` still start and health-gate
    first, though — a CI checkout with a real test suite needing Postgres
    is a common, real case this command should support too.
    """
    repo = repo.resolve()
    base_commit = rev_parse(repo, base_ref)
    final_commit = rev_parse(repo, "HEAD")
    diff, files_changed = diff_between(repo, base_commit, final_commit)

    # cost_usd is None, not 0 — this diff wasn't produced by an adapter
    # Verdict drove, so "what it cost" is simply not a question this mode
    # answers, and reporting 0 would look like a real, known figure.
    attempt = AttemptResult(diff=diff, files_changed=files_changed, cost_usd=None)

    # A plain Worktree value pointing at the real checkout, not a fresh
    # `isolated_worktree()` — attribute_failures only ever *reads*
    # `worktree.path` (for the dependency graph scan) and reads
    # `worktree.base_commit` as bisection's starting point; it never writes
    # through this path itself.
    worktree = Worktree(path=repo, branch="HEAD", base_commit=base_commit)
    sandbox_config = sandbox_config or SandboxConfig()
    deadline = _budget_deadline(sandbox_config)

    config = load_config(repo)
    service_session = setup_services(config.services, sandbox_config.health_timeout_seconds)
    try:
        gate_sandbox_config = _sandbox_config_for_services(sandbox_config, service_session)
        with create_sandbox(repo, gate_sandbox_config) as sandbox:
            signals, attributions, budget_exceeded = _run_gates_and_attribution(
                repo, repo, worktree, final_commit, config, sandbox, sandbox_config, deadline
            )
            if budget_exceeded or _budget_exceeded(deadline):
                budget_exceeded = True
            else:
                signals = signals + run_frontend_checks(
                    repo, worktree, config, task="pull request diff", sandbox=sandbox,
                    sandbox_config=sandbox_config,
                )
    finally:
        teardown_services(service_session)

    return Verdict(
        task="grade existing diff",
        agent="gate",
        repo=str(repo),
        attempt=attempt,
        signals=signals,
        attributions=attributions,
        budget_exceeded=budget_exceeded,
    )


def _apply_pricing_fallback(attempt: AttemptResult, config: VerdictConfig) -> AttemptResult:
    """If the adapter didn't report its own cost (only ClaudeCodeAdapter
    does today) but verdict.yml configures token pricing, compute it —
    real spend is real spend whether or not the adapter happened to hand
    back a dollar figure directly.
    """
    if attempt.cost_usd is not None or config.token_pricing is None:
        return attempt
    computed = config.token_pricing.cost_usd(attempt.tokens_input, attempt.tokens_output)
    return attempt.model_copy(update={"cost_usd": computed})


def run_with_retries(
    task: str,
    repo: Path,
    adapter: Adapter,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sandbox_config: SandboxConfig | None = None,
) -> TaskRun:
    """Attempt `task` up to `max_attempts` times, stopping early on the
    first DONE. Every attempt is kept — including failed, abandoned ones —
    so `TaskRun`'s cost accounting reflects what the task actually cost,
    not just what the winning attempt cost.

    Defaults to a single attempt: retries are opt-in. Each retry re-runs
    the real adapter, which for `ClaudeCodeAdapter` means real spend —
    Verdict shouldn't silently multiply a bill the caller didn't ask for.
    """
    attempts: list[Verdict] = []
    for _ in range(max(max_attempts, 1)):
        verdict = run(task=task, repo=repo, adapter=adapter, sandbox_config=sandbox_config)
        attempts.append(verdict)
        if verdict.done:
            break
    return TaskRun(task=task, agent=adapter.name, repo=str(repo), attempts=attempts)
