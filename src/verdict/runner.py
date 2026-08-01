"""Orchestrates one end-to-end grading run: isolate a worktree, let the
adapter do the work, run the gates, assemble a Verdict.
"""

from __future__ import annotations

from pathlib import Path

from verdict.adapters import Adapter
from verdict.attribution.engine import attribute_failures
from verdict.config import VerdictConfig, load_config
from verdict.frontend.runner import run_frontend_checks
from verdict.gates.registry import run_all_gates
from verdict.sandbox import SandboxConfig, create_sandbox
from verdict.sandbox.install import run_install_step
from verdict.schema import AttemptResult, TaskRun, Verdict
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

    `sandbox_config` governs how the adapter's CLI, the install step, gates,
    and the frontend dev server actually execute — see `sandbox/config.py`.
    Defaults to `SandboxConfig()` (backend "local") when not given; the
    `verdict` CLI itself passes an explicit config defaulting to "docker"
    (Phase 8's actual product default) via its `--sandbox-backend` flag.
    """
    sandbox_config = sandbox_config or SandboxConfig()
    with isolated_worktree(repo) as worktree:
        copy_vendored_dependencies(repo, worktree.path)
        run_install_step(worktree.path, sandbox_config)

        with create_sandbox(worktree.path, sandbox_config) as sandbox:
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
            signals = run_all_gates(worktree.path, config, sandbox=sandbox)
            attributions = attribute_failures(repo, worktree, attempt_commit, signals, sandbox_config)

            # Frontend checks run after gates/attribution, and their signals are
            # appended afterward rather than folded into `signals` beforehand —
            # attribution's bisector only knows the four gate names in
            # gates/registry.py's GATE_RUNNERS, and would crash trying to
            # `resolve_gate("frontend:...")`. A failing frontend check is real
            # PROVEN evidence for Verdict.status either way; it's just not
            # (yet) bisectable to a culprit file the way test/typecheck/build/
            # lint failures are.
            frontend_signals = run_frontend_checks(repo, worktree, config, task, sandbox=sandbox)
            signals = signals + frontend_signals

    return Verdict(
        task=task,
        agent=adapter.name,
        repo=str(repo),
        attempt=attempt,
        signals=signals,
        attributions=attributions,
    )


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

    config = load_config(repo)
    with create_sandbox(repo, sandbox_config) as sandbox:
        signals = run_all_gates(repo, config, sandbox=sandbox)
        attributions = attribute_failures(repo, worktree, final_commit, signals, sandbox_config)
        signals = signals + run_frontend_checks(
            repo, worktree, config, task="pull request diff", sandbox=sandbox
        )

    return Verdict(
        task="grade existing diff",
        agent="gate",
        repo=str(repo),
        attempt=attempt,
        signals=signals,
        attributions=attributions,
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
