from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import typer
from rich.console import Console
from rich.table import Table

from verdict import economics
from verdict.adapters import Adapter
from verdict.adapters.aider import AiderAdapter, AiderAdapterError
from verdict.adapters.claude_code import ClaudeCodeAdapter, ClaudeCodeAdapterError
from verdict.adapters.codex import CodexAdapter, CodexAdapterError
from verdict.adapters.cursor import CursorAdapter, CursorAdapterError
from verdict.adapters.mock import MockAdapter, SuiteMockAdapter
from verdict.adapters.openhands import OpenHandsAdapter, OpenHandsAdapterError
from verdict.calibration import (
    DEFAULT_CONCORDANCE_THRESHOLD,
    DatasetLoadError,
    load_labeled_dataset,
    render_calibration,
    run_calibration,
)
from verdict.failure_modes import render_failure_modes
from verdict.flakiness import (
    DEFAULT_TRIALS,
    FlakinessResult,
    compare_flakiness,
    render_comparison,
    render_flakiness,
    run_flakiness,
)
from verdict.frontend.vision_judge import MockVisionJudge, RealVisionJudge, VisionJudge
from verdict.frontend.vision_transport import AnthropicVisionTransport
from verdict.ground_truth import (
    DEFAULT_ACCURACY_THRESHOLD,
    load_ground_truth_dataset,
    render_ground_truth,
    run_ground_truth,
)
from verdict.integrity import TestChangeAllowance
from verdict.monorepo import PackageSelectionError
from verdict.pr_comment import build_comment_from_file
from verdict.report import render_task_run
from verdict.report_html import HistoryKey, render_html
from verdict.report_json import render_json
from verdict.runner import DEFAULT_MAX_ERROR_RETRIES, grade_existing_diff, run_with_retries
from verdict.sandbox import ResourceLimits, SandboxConfig
from verdict.sandbox.base import SandboxUnavailableError
from verdict.schema import ConfigResult, TaskRun, VerdictStatus
from verdict.store import (
    DEFAULT_BASELINE_WINDOW,
    SQLiteStore,
    Store,
    TaskOutcome,
    TaskRegression,
    detect_regressions,
)
from verdict.suite import BenchConfig, LocalProcessPoolExecutor, SuiteLoadError, load_suite, run_suite
from verdict.worktree import WorktreeError, rev_parse

app = typer.Typer(add_completion=False, help="Grade AI coding agents on executable truth.")

_SANDBOX_BACKEND_HELP = (
    "How agent-influenced code actually executes: \"docker\" (default — an "
    "isolated, network-off-by-default container) or \"local\" (no "
    "isolation at all, prints an UNSAFE warning — trusted-repo local dev "
    "only). See DESIGN.md's Phase 8 section."
)


def _build_sandbox_config(
    sandbox_backend: str,
    sandbox_image: str,
    sandbox_cpus: float,
    sandbox_memory_mb: int,
    gate_timeout_seconds: int,
    provision_timeout_seconds: int,
    install_timeout_seconds: int,
    attempt_budget_seconds: int,
    health_timeout_seconds: int,
) -> SandboxConfig:
    if sandbox_backend not in ("docker", "local"):
        raise typer.BadParameter(f"unknown --sandbox-backend: {sandbox_backend!r} (choices: docker, local)")
    return SandboxConfig(
        backend=sandbox_backend,  # type: ignore[arg-type]
        image=sandbox_image,
        limits=ResourceLimits(cpu_cores=sandbox_cpus, memory_mb=sandbox_memory_mb),
        gate_timeout_seconds=gate_timeout_seconds,
        provision_timeout_seconds=provision_timeout_seconds,
        install_timeout_seconds=install_timeout_seconds,
        # 0 means "no budget" at the CLI layer — Typer options can't carry
        # a bare `None` from the command line, so 0 is the user-facing
        # spelling of SandboxConfig.attempt_budget_seconds=None.
        attempt_budget_seconds=attempt_budget_seconds if attempt_budget_seconds > 0 else None,
        health_timeout_seconds=health_timeout_seconds,
    )



# Every adapter this CLI can drive by name, beyond `mock` (which needs a
# per-repo or per-task canned patch, handled separately below). One entry
# per real Adapter implementation — adding a fifth, sixth, ... agent is
# exactly this: implement the class, add one line here, nothing else in
# this module changes.
_REAL_AGENTS: dict[str, type[Adapter]] = {
    "claude-code": ClaudeCodeAdapter,
    "cursor": CursorAdapter,
    "codex": CodexAdapter,
    "aider": AiderAdapter,
    "openhands": OpenHandsAdapter,
}

# Every AdapterError a real agent's subprocess can raise — caught the same
# way at both call sites below (the CLI's job is to report it and exit
# non-zero, not to recover from it).
_ADAPTER_ERRORS: tuple[type[Exception], ...] = (
    ClaudeCodeAdapterError,
    CursorAdapterError,
    CodexAdapterError,
    AiderAdapterError,
    OpenHandsAdapterError,
)

# Combined with WorktreeError once here so both `run` and `bench` catch the
# exact same set via a single, mypy-friendly tuple name. SandboxUnavailableError
# also covers its subclass ProvisioningTimeoutError (Phase 9) — a
# provisioning timeout is deliberately handled exactly like every other
# "this attempt couldn't be evaluated" cause already in this tuple: report
# and exit 2, never a Signal. See DESIGN.md's Phase 9 section.
_RUN_ERRORS: tuple[type[Exception], ...] = (WorktreeError, SandboxUnavailableError, *_ADAPTER_ERRORS)


@app.callback()
def _callback() -> None:
    """Grade AI coding agents on executable truth, not opinion."""

# MockAdapter has no LLM to decide what to write, so it needs a literal
# patch handed to it. These happen to fix the seeded bug in each example
# repo, so `--agent mock` is demoable out of the box on either one.
_MOCK_PATCHES: dict[str, dict[str, str]] = {
    "sample_repo": {
        "sample_repo/calculator.py": (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        ),
    },
    "sample_node_repo": {
        "src/calculator.ts": (
            "export function add(a: number, b: number): number {\n"
            "  return a + b;\n"
            "}\n"
        ),
    },
    "sample_frontend_repo": {
        "public/index.html": (
            "<!doctype html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="utf-8">\n'
            "  <title>Acme Launch</title>\n"
            '  <link rel="stylesheet" href="/style.css">\n'
            "</head>\n"
            "<body>\n"
            "  <header><h1>Acme</h1></header>\n"
            "  <main>\n"
            "    <p>The fastest way to launch your next idea.</p>\n"
            '    <a id="cta" class="cta" href="/signup.html">Get Started</a>\n'
            "  </main>\n"
            "</body>\n"
            "</html>\n"
        ),
    },
}


def _build_adapter(agent: str, repo: Path) -> Adapter:
    if agent == "mock":
        patch = _MOCK_PATCHES.get(repo.name)
        if patch is None:
            raise typer.BadParameter(
                f"mock adapter has no canned patch for {repo.name!r}; "
                "use a real --agent, or point --repo at one of the examples/ repos"
            )
        return MockAdapter(patches=patch)
    factory = _REAL_AGENTS.get(agent)
    if factory is None:
        raise typer.BadParameter(f"unknown agent: {agent!r} (choices: mock, {', '.join(_REAL_AGENTS)})")
    return factory()


# `verdict bench` runs the same task text against every task in a suite —
# MockAdapter's single fixed-at-construction patch can't represent "a
# different canned fix per task," so SuiteMockAdapter looks its patch up by
# the task's own text instead (see adapters/mock.py). These happen to fix
# examples/starter_suite's three seeded tasks, so `--agent mock` is
# demoable against it out of the box, the same way `_MOCK_PATCHES` above
# is for single-repo `verdict run`.
_STARTER_SUITE_MOCK_PATCHES: dict[str, dict[str, str]] = {
    "Fix the bug in add() so it returns the correct sum instead of subtracting.": {
        "calculator.py": "def add(a: int, b: int) -> int:\n    return a + b\n",
    },
    "Add a multiply(a, b) function to calculator.py that returns the product of a and b, "
    "matching add()'s style.": {
        "calculator.py": (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "\n\n"
            "def multiply(a: int, b: int) -> int:\n"
            "    return a * b\n"
        ),
    },
    "The non-negative-amount validation in calculate_us_tax and calculate_eu_tax is duplicated. "
    "Extract it into a shared helper function, without changing either function's behavior.": {
        "tax.py": (
            "def _validate_non_negative(amount: float) -> None:\n"
            '    if amount < 0:\n'
            '        raise ValueError("amount must be non-negative")\n'
            "\n\n"
            "def calculate_us_tax(amount: float) -> float:\n"
            "    _validate_non_negative(amount)\n"
            "    return round(amount * 0.07, 2)\n"
            "\n\n"
            "def calculate_eu_tax(amount: float) -> float:\n"
            "    _validate_non_negative(amount)\n"
            "    return round(amount * 0.20, 2)\n"
        ),
    },
}


def _build_bench_adapter(agent: str) -> Adapter:
    if agent == "mock":
        return SuiteMockAdapter(_STARTER_SUITE_MOCK_PATCHES)
    factory = _REAL_AGENTS.get(agent)
    if factory is None:
        raise typer.BadParameter(f"unknown agent: {agent!r} (choices: mock, {', '.join(_REAL_AGENTS)})")
    return factory()


_REPORT_FORMATS = ("cli", "json", "html")
_REPORT_HELP = f"Repeatable: which reporter(s) to produce. Choices: {' | '.join(_REPORT_FORMATS)}."


def _write_machine_reports(
    formats: list[str],
    output_dir: Path,
    config_results: list[ConfigResult],
    history: dict[HistoryKey, list[TaskOutcome]] | None = None,
) -> None:
    """Writes the `json`/`html` reporters if requested — `cli` is handled
    separately by each command's own rich-based renderer, since that one
    prints to the terminal rather than a file. Shared across `run`/`bench`/
    `gate` so all three reporters (and the merge-gate/scorecard commands
    that produce them) stay in exact lockstep: one `ConfigResult` shape,
    one place that serializes it.

    `history` (Phase 17) is optional and only ever threaded through to the
    HTML reporter's trend section — `render_json`/`to_report_dict` don't
    take it, since a machine consumer already has full query access to
    whatever `Store` produced it and doesn't need it re-embedded here.
    """
    unknown = set(formats) - set(_REPORT_FORMATS)
    if unknown:
        raise typer.BadParameter(
            f"unknown --report format(s): {', '.join(sorted(unknown))} "
            f"(choices: {', '.join(_REPORT_FORMATS)})"
        )

    if "json" not in formats and "html" not in formats:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    if "json" in formats:
        path = output_dir / "verdict-report.json"
        path.write_text(render_json(config_results))
        typer.echo(f"wrote {path}")
    if "html" in formats:
        path = output_dir / "verdict-report.html"
        path.write_text(render_html(config_results, history=history))
        typer.echo(f"wrote {path}")


def _build_store(store_path: Path | None) -> Store | None:
    """`--store` is opt-in — `None` (the default) means no persistence at
    all, the same "off unless explicitly requested" stance every other
    side-effecting flag in this CLI takes (`--max-attempts`,
    `--cost-ceiling-usd`, ...). `SQLiteStore` is the only backend wired to
    the CLI; a caller embedding Verdict as a library can pass any other
    `Store` implementation directly to `persist_run`/`detect_regressions`
    without touching this function at all.
    """
    return SQLiteStore(store_path) if store_path is not None else None


def _persist_run(
    store: Store | None,
    config_results: list[ConfigResult],
    commit_sha: str | None,
    run_label: str | None,
    baseline_window: int,
    regression_alpha: float,
) -> tuple[str | None, dict[HistoryKey, list[TaskOutcome]]]:
    """Persists `config_results` (if `--store` was given) and prints any
    regressions found against recent history. Returns the new `run_id`
    (`None` if no store) and a `history` dict ready for
    `render_html`/`_write_machine_reports`'s trend section.

    Regression detection here is informational only — it never changes a
    command's exit code. This mirrors Phase 7's own explicit stance on
    calibration/flakiness ("no automatic CI gate on a statistical
    signal" — see DESIGN.md): `verdict gate`'s exit code is governed
    purely by PROVEN checks against the diff actually under review: a
    historical trend is real signal worth a human's attention, but
    blocking a merge on it would be a different, weaker kind of promise
    than blocking on an executed check that failed on THIS change.
    """
    if store is None:
        return None, {}

    run_id = store.record_run(config_results, commit_sha=commit_sha, label=run_label)

    regressions = detect_regressions(
        store,
        config_results,
        baseline_window=baseline_window,
        alpha=regression_alpha,
        exclude_run_id=run_id,
    )
    _render_regressions(regressions)

    history: dict[HistoryKey, list[TaskOutcome]] = {}
    for config_result in config_results:
        for task_run in config_result.task_runs:
            key = (config_result.label, task_run.task, task_run.agent, task_run.repo)
            history[key] = list(
                store.history(task_run.task, task_run.agent, task_run.repo, config_label=config_result.label)
            )
    return run_id, history


def _render_regressions(regressions: list[TaskRegression]) -> None:
    if not regressions:
        return
    typer.secho(
        f"\n{len(regressions)} regression(s) flagged against recorded history:",
        fg=typer.colors.RED,
        bold=True,
    )
    for r in regressions:
        c = r.comparison
        p = f"{c.p_value:.4f}" if c.p_value is not None else "n/a"
        typer.secho(
            f"  [REGRESSION] {r.config_label} / {r.task!r}: "
            f"{c.baseline_label} {c.baseline.pass_rate:.0%} -> "
            f"{c.candidate_label} {c.candidate.pass_rate:.0%} (p={p}, alpha={c.alpha})",
            fg=typer.colors.RED,
        )


@app.command(name="run")
def run_cmd(
    task: str = typer.Option(..., "--task", help="Natural-language description of the work."),
    agent: str = typer.Option(
        ..., "--agent", help=f"Which adapter to drive: mock | {' | '.join(_REAL_AGENTS)}"
    ),
    repo: Path = typer.Option(..., "--repo", help="Path to a git repository to grade against."),
    package: str | None = typer.Option(
        None,
        "--package",
        help=(
            "Which package this task targets in a monorepo — a path relative to --repo, e.g. "
            "'services/api'. Required whenever the repo's shape is ambiguous (root verdict.yml "
            "declares more than one `packages:` entry, or the repo root has no project files but "
            "multiple subdirectories each look like their own project) — Verdict never guesses. "
            "Omit for a normal single-project repo."
        ),
    ),
    max_attempts: int = typer.Option(
        1,
        "--max-attempts",
        help=(
            "Retry on a legitimate agent failure up to this many times, stopping early on DONE. "
            "Cost is tracked across every attempt, dead ends included. "
            "Each retry re-runs the real agent — with a real --agent that's real spend."
        ),
    ),
    max_error_retries: int = typer.Option(
        DEFAULT_MAX_ERROR_RETRIES,
        "--max-error-retries",
        help=(
            "Separate from --max-attempts: how many times an infra failure (sandbox never came "
            "up, a service never healthy, the adapter's own CLI crashed) is auto-retried before "
            "being reported as ERROR. Never spent on a legitimate agent NOT_DONE."
        ),
    ),
    allow_test_changes: bool = typer.Option(
        False,
        "--allow-test-changes",
        help=(
            "This --task legitimately requires editing tests (e.g. \"add tests for X\") — skip the "
            "Phase 12 integrity gate's test-tampering checks. You're invoking this flag yourself, "
            "outside the repo being graded, which is exactly the trust boundary that makes it safe "
            "here (never settable from the repo's own verdict.yml). Off by default."
        ),
    ),
    report: list[str] = typer.Option(["cli"], "--report", help=_REPORT_HELP),
    output_dir: Path = typer.Option(
        Path("verdict-report"), "--output-dir", help="Where json/html reports are written."
    ),
    sandbox_backend: str = typer.Option("docker", "--sandbox-backend", help=_SANDBOX_BACKEND_HELP),
    sandbox_image: str = typer.Option(
        "verdict-sandbox:0.1.0", "--sandbox-image", help="Image DockerSandbox runs."
    ),
    sandbox_cpus: float = typer.Option(2.0, "--sandbox-cpus", help="CPU limit passed to DockerSandbox."),
    sandbox_memory_mb: int = typer.Option(
        2048, "--sandbox-memory-mb", help="Memory limit (MB) passed to DockerSandbox."
    ),
    gate_timeout_seconds: int = typer.Option(
        600, "--gate-timeout-seconds", help="Per-gate wall-clock timeout. A hang here is a real PROVEN FAIL."
    ),
    provision_timeout_seconds: int = typer.Option(
        120, "--provision-timeout-seconds", help="How long DockerSandbox waits for `docker run` to come up."
    ),
    install_timeout_seconds: int = typer.Option(
        300, "--install-timeout-seconds", help="Timeout for the dependency-install step."
    ),
    attempt_budget_seconds: int = typer.Option(
        1800,
        "--attempt-budget-seconds",
        help="Global wall-clock ceiling across one whole attempt. 0 disables it.",
    ),
    health_timeout_seconds: int = typer.Option(
        30, "--health-timeout-seconds", help="How long a declared service gets to pass its health check."
    ),
    cost_ceiling_usd: float = typer.Option(
        0.0,
        "--cost-ceiling-usd",
        help=(
            "Hard cap on spend across every --max-attempts retry of this one task. 0 disables it. "
            "Reached mid-retry-loop → the loop stops cleanly, every attempt already made is kept, "
            "and the abort is marked ERROR (excluded from pass-rate math), never a NOT_DONE agent "
            "failure — see DESIGN.md's Phase 18 section."
        ),
    ),
) -> None:
    """Run an agent against one task (retrying on failure if --max-attempts > 1)
    and print its Verdict plus the cost across every attempt made.

    Gate commands are autodetected per-stack; override any of them — and
    configure token pricing — with a `verdict.yml` in the repo being graded
    (see README's Configuration section).
    """
    adapter = _build_adapter(agent, repo)
    sandbox_config = _build_sandbox_config(
        sandbox_backend, sandbox_image, sandbox_cpus, sandbox_memory_mb,
        gate_timeout_seconds, provision_timeout_seconds, install_timeout_seconds, attempt_budget_seconds,
        health_timeout_seconds,
    )
    task_run = run_with_retries(
        task=task,
        repo=repo,
        adapter=adapter,
        max_attempts=max_attempts,
        sandbox_config=sandbox_config,
        max_error_retries=max_error_retries,
        allow_test_changes=TestChangeAllowance(allowed=allow_test_changes),
        # 0 means "no ceiling" at the CLI layer, same spelling every other
        # None-disables-the-cap knob in this CLI already uses.
        cost_ceiling_usd=cost_ceiling_usd if cost_ceiling_usd > 0 else None,
        package=package,
    )

    if "cli" in report:
        render_task_run(task_run)
    _write_machine_reports(report, output_dir, [ConfigResult(label=agent, task_runs=[task_run])])

    # ERROR (infra never evaluated the attempt) and NOT_DONE (evaluated,
    # found wanting) are deliberately different exit codes — the same
    # distinction `_RUN_ERRORS` used to draw by crashing with code 2
    # before Phase 11 taught `run_with_retries` to retry-then-report ERROR
    # instead. A CI script keying off exit code can still tell "the agent
    # failed" (1) apart from "we couldn't tell" (2).
    if task_run.final.status is VerdictStatus.ERROR:
        sys.exit(2)
    if not task_run.done:
        sys.exit(1)


@app.command(name="bench")
def bench_cmd(
    suite: Path = typer.Option(
        ..., "--suite", help="Path to a suite directory (see DESIGN.md's Phase 5 section for the format)."
    ),
    agent: list[str] = typer.Option(
        ...,
        "--agent",
        help=(
            f"Repeatable: one config to benchmark, e.g. --agent mock --agent claude-code. "
            f"Choices: mock | {' | '.join(_REAL_AGENTS)}"
        ),
    ),
    max_attempts: int = typer.Option(
        1, "--max-attempts", help="Retry each task up to this many times per config, stopping early on DONE."
    ),
    max_error_retries: int = typer.Option(
        DEFAULT_MAX_ERROR_RETRIES,
        "--max-error-retries",
        help=(
            "Separate from --max-attempts: how many times an infra failure is auto-retried per "
            "(config, task) before that task is recorded as ERROR and excluded from the leaderboard's "
            "pass-rate denominator."
        ),
    ),
    report: list[str] = typer.Option(["cli"], "--report", help=_REPORT_HELP),
    output_dir: Path = typer.Option(
        Path("verdict-report"), "--output-dir", help="Where json/html reports are written."
    ),
    sandbox_backend: str = typer.Option("docker", "--sandbox-backend", help=_SANDBOX_BACKEND_HELP),
    sandbox_image: str = typer.Option(
        "verdict-sandbox:0.1.0", "--sandbox-image", help="Image DockerSandbox runs."
    ),
    sandbox_cpus: float = typer.Option(2.0, "--sandbox-cpus", help="CPU limit passed to DockerSandbox."),
    sandbox_memory_mb: int = typer.Option(
        2048, "--sandbox-memory-mb", help="Memory limit (MB) passed to DockerSandbox."
    ),
    gate_timeout_seconds: int = typer.Option(
        600, "--gate-timeout-seconds", help="Per-gate wall-clock timeout. A hang here is a real PROVEN FAIL."
    ),
    provision_timeout_seconds: int = typer.Option(
        120, "--provision-timeout-seconds", help="How long DockerSandbox waits for `docker run` to come up."
    ),
    install_timeout_seconds: int = typer.Option(
        300, "--install-timeout-seconds", help="Timeout for the dependency-install step."
    ),
    attempt_budget_seconds: int = typer.Option(
        1800,
        "--attempt-budget-seconds",
        help="Global wall-clock ceiling across one whole attempt. 0 disables it.",
    ),
    health_timeout_seconds: int = typer.Option(
        30, "--health-timeout-seconds", help="How long a declared service gets to pass its health check."
    ),
    max_workers: int = typer.Option(
        1,
        "--max-workers",
        help=(
            "How many (config, task) pairs to grade concurrently, each in its own worktree/sandbox. "
            "1 (default) runs serially, in this process, exactly as before Phase 15."
        ),
    ),
    cost_ceiling_usd: float = typer.Option(
        0.0,
        "--cost-ceiling-usd",
        help=(
            "Global spend cap across every (config, task) pair in this run. 0 disables it. "
            "Cooperative, not preemptive — see DESIGN.md's Phase 15 section."
        ),
    ),
    run_cost_ceiling_usd: float = typer.Option(
        0.0,
        "--run-cost-ceiling-usd",
        help=(
            "Hard cap on spend across --max-attempts retries of any ONE (config, task) pair — "
            "distinct from --cost-ceiling-usd's suite-wide total. 0 disables it. An abort here is "
            "marked ERROR (excluded from pass-rate math), never a NOT_DONE agent failure — see "
            "DESIGN.md's Phase 18 section."
        ),
    ),
    store: Path | None = typer.Option(
        None,
        "--store",
        help=(
            "Path to a SQLite history db — if given, this run's results are persisted and checked "
            "for regressions against recorded history (see `verdict history`). Off by default."
        ),
    ),
    commit_sha: str | None = typer.Option(
        None, "--commit-sha", help="Commit this run graded, recorded alongside it. Free-form, optional."
    ),
    run_label: str | None = typer.Option(
        None, "--run-label", help="Free-form label for this run in --store history (e.g. a CI job name)."
    ),
    baseline_window: int = typer.Option(
        DEFAULT_BASELINE_WINDOW,
        "--baseline-window",
        help="How many of the most recent prior recorded runs form the regression baseline.",
    ),
    regression_alpha: float = typer.Option(
        0.05, "--regression-alpha", help="Significance threshold for the historical regression z-test."
    ),
) -> None:
    """Run every --agent against every task in --suite, then print a
    pass-rate-per-dollar leaderboard and a failure-mode breakdown.

    Unlike `verdict run`, this never exits non-zero for a bad score — it's
    a scorecard, not a merge gate; use `verdict gate` in CI for that.
    """
    try:
        tasks = load_suite(suite)
    except SuiteLoadError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    configs = [BenchConfig(label=name, adapter=_build_bench_adapter(name)) for name in agent]
    sandbox_config = _build_sandbox_config(
        sandbox_backend, sandbox_image, sandbox_cpus, sandbox_memory_mb,
        gate_timeout_seconds, provision_timeout_seconds, install_timeout_seconds, attempt_budget_seconds,
        health_timeout_seconds,
    )

    if max_workers < 1:
        raise typer.BadParameter("--max-workers must be >= 1")
    executor = LocalProcessPoolExecutor(max_workers=max_workers) if max_workers > 1 else None

    results = run_suite(
        tasks,
        configs,
        max_attempts=max_attempts,
        sandbox_config=sandbox_config,
        max_error_retries=max_error_retries,
        executor=executor,
        # 0 means "no ceiling" at the CLI layer, same spelling
        # `--attempt-budget-seconds` already uses for its own None-disables knob.
        cost_ceiling_usd=cost_ceiling_usd if cost_ceiling_usd > 0 else None,
        run_cost_ceiling_usd=run_cost_ceiling_usd if run_cost_ceiling_usd > 0 else None,
    )

    _run_id, history = _persist_run(
        _build_store(store), results, commit_sha, run_label, baseline_window, regression_alpha
    )

    if "cli" in report:
        economics.render(results)
        render_failure_modes(results)
    _write_machine_reports(report, output_dir, results, history=history)


@app.command(name="gate")
def gate_cmd(
    repo: Path = typer.Option(
        Path("."), "--repo", help="Path to the repo to grade — graded in place, not isolated."
    ),
    base: str = typer.Option(
        ..., "--base", help="Git ref to diff/attribute against (a PR's base branch or merge-base SHA)."
    ),
    package: str | None = typer.Option(
        None,
        "--package",
        help=(
            "Which package this PR/diff targets in a monorepo — a path relative to --repo, e.g. "
            "'services/api'. Required whenever the repo's shape is ambiguous — see `verdict run "
            "--help`'s --package for the full rule. Omit for a normal single-project repo."
        ),
    ),
    label: str = typer.Option("gate", "--label", help="Label for this run in json/html reports."),
    report: list[str] = typer.Option(["cli"], "--report", help=_REPORT_HELP),
    output_dir: Path = typer.Option(
        Path("verdict-report"), "--output-dir", help="Where json/html reports are written."
    ),
    sandbox_backend: str = typer.Option("docker", "--sandbox-backend", help=_SANDBOX_BACKEND_HELP),
    sandbox_image: str = typer.Option(
        "verdict-sandbox:0.1.0", "--sandbox-image", help="Image DockerSandbox runs."
    ),
    sandbox_cpus: float = typer.Option(2.0, "--sandbox-cpus", help="CPU limit passed to DockerSandbox."),
    sandbox_memory_mb: int = typer.Option(
        2048, "--sandbox-memory-mb", help="Memory limit (MB) passed to DockerSandbox."
    ),
    gate_timeout_seconds: int = typer.Option(
        600, "--gate-timeout-seconds", help="Per-gate wall-clock timeout. A hang here is a real PROVEN FAIL."
    ),
    provision_timeout_seconds: int = typer.Option(
        120, "--provision-timeout-seconds", help="How long DockerSandbox waits for `docker run` to come up."
    ),
    install_timeout_seconds: int = typer.Option(
        300, "--install-timeout-seconds", help="Timeout for the dependency-install step."
    ),
    attempt_budget_seconds: int = typer.Option(
        1800,
        "--attempt-budget-seconds",
        help="Global wall-clock ceiling across one whole attempt. 0 disables it.",
    ),
    health_timeout_seconds: int = typer.Option(
        30, "--health-timeout-seconds", help="How long a declared service gets to pass its health check."
    ),
    allow_test_changes: bool = typer.Option(
        False,
        "--allow-test-changes",
        help=(
            "This PR/diff legitimately edits tests — skip the Phase 12 integrity gate's "
            "test-tampering checks. An operator/CI-supplied flag, deliberately never something a "
            "PR's own verdict.yml can set (that would let a PR author disable its own integrity "
            "gate). Off by default."
        ),
    ),
    store: Path | None = typer.Option(
        None,
        "--store",
        help=(
            "Path to a SQLite history db — if given, this run's results are persisted and checked "
            "for regressions against recorded history (see `verdict history`). Off by default. "
            "A flagged regression is printed but never changes this command's exit code — see "
            "DESIGN.md's Phase 17 section for why only a PROVEN failure on THIS diff does that."
        ),
    ),
    commit_sha: str | None = typer.Option(
        None,
        "--commit-sha",
        help="Commit this run graded, recorded alongside it. Defaults to `git rev-parse HEAD` in --repo.",
    ),
    run_label: str | None = typer.Option(
        None, "--run-label", help="Free-form label for this run in --store history (e.g. a CI job name)."
    ),
    baseline_window: int = typer.Option(
        DEFAULT_BASELINE_WINDOW,
        "--baseline-window",
        help="How many of the most recent prior recorded runs form the regression baseline.",
    ),
    regression_alpha: float = typer.Option(
        0.05, "--regression-alpha", help="Significance threshold for the historical regression z-test."
    ),
) -> None:
    """Grade `--repo` exactly as it's already checked out against `--base`
    — no adapter, no isolation. This is the merge-gate command: a pull
    request's diff already exists as real commits, so there's nothing to
    drive an agent against, just gates/frontend-checks/attribution run
    against what's already there. See DESIGN.md's Phase 6 section for the
    gate policy this enforces: any failing PROVEN signal fails the check;
    JUDGED signals never do.
    """
    sandbox_config = _build_sandbox_config(
        sandbox_backend, sandbox_image, sandbox_cpus, sandbox_memory_mb,
        gate_timeout_seconds, provision_timeout_seconds, install_timeout_seconds, attempt_budget_seconds,
        health_timeout_seconds,
    )
    try:
        verdict = grade_existing_diff(
            repo=repo,
            base_ref=base,
            sandbox_config=sandbox_config,
            allow_test_changes=TestChangeAllowance(allowed=allow_test_changes),
            package=package,
        )
    except (WorktreeError, PackageSelectionError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    task_run = TaskRun(task=verdict.task, agent=label, repo=verdict.repo, attempts=[verdict])
    config_results = [ConfigResult(label=label, task_runs=[task_run])]

    if commit_sha is None and store is not None:
        try:
            commit_sha = rev_parse(repo, "HEAD")
        except WorktreeError:
            commit_sha = None  # not a git repo, or HEAD unresolvable — persist without one

    _run_id, history = _persist_run(
        _build_store(store), config_results, commit_sha, run_label, baseline_window, regression_alpha
    )

    if "cli" in report:
        render_task_run(task_run)
    _write_machine_reports(report, output_dir, config_results, history=history)

    if not verdict.done:
        sys.exit(1)


# Phase 16: "anthropic" is the real integration — provider-agnostic at
# the VisionJudge/VisionModelTransport layer (see vision_judge.py), one
# concrete transport shipped so far. A second provider is a new transport
# class and one new entry here, never a change to how calibration or this
# dict works.
_JUDGES: dict[str, Callable[[], VisionJudge]] = {
    "mock": MockVisionJudge,
    "anthropic": lambda: RealVisionJudge(AnthropicVisionTransport()),
}


def _build_judge(name: str) -> VisionJudge:
    factory = _JUDGES.get(name)
    if factory is None:
        raise typer.BadParameter(f"unknown judge: {name!r} (choices: {', '.join(_JUDGES)})")
    return factory()


@app.command(name="calibrate")
def calibrate_cmd(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Path to a calibration manifest.json (see examples/calibration_dataset).",
    ),
    judge: str = typer.Option(
        "mock",
        "--judge",
        help=(
            f"Which VisionJudge to score: {', '.join(_JUDGES)}. "
            "\"anthropic\" reads its API key from ANTHROPIC_API_KEY — never a CLI flag or "
            "verdict.yml — and makes real, billed API calls."
        ),
    ),
    threshold: float = typer.Option(
        DEFAULT_CONCORDANCE_THRESHOLD, "--threshold", help="Target concordance (fraction, e.g. 0.95)."
    ),
) -> None:
    """Score `--judge` against a human-labeled dataset and report its
    concordance — how often the judge's PASS/FAIL agrees with the human
    label. Never fails the process: this is a diagnostic, not a merge gate,
    so a below-threshold result prints a warning rather than a nonzero exit.
    Examples the judge had no opinion on (a failed/malformed API call) are
    reported separately and excluded from concordance — see
    `CalibrationResult.unavailable`.
    """
    try:
        examples = load_labeled_dataset(dataset)
    except DatasetLoadError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    result = run_calibration(_build_judge(judge), examples, threshold=threshold)
    render_calibration(result)


@app.command(name="ground-truth")
def ground_truth_cmd(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Path to a ground-truth manifest.json (see examples/ground_truth_dataset).",
    ),
    threshold: float = typer.Option(
        DEFAULT_ACCURACY_THRESHOLD,
        "--threshold",
        help="Target accuracy against the human labels (fraction, e.g. 0.8).",
    ),
    sandbox_backend: str = typer.Option("docker", "--sandbox-backend", help=_SANDBOX_BACKEND_HELP),
    sandbox_image: str = typer.Option(
        "verdict-sandbox:0.1.0", "--sandbox-image", help="Image DockerSandbox runs."
    ),
    sandbox_cpus: float = typer.Option(2.0, "--sandbox-cpus", help="CPU limit passed to DockerSandbox."),
    sandbox_memory_mb: int = typer.Option(
        2048, "--sandbox-memory-mb", help="Memory limit (MB) passed to DockerSandbox."
    ),
    gate_timeout_seconds: int = typer.Option(
        600, "--gate-timeout-seconds", help="Per-gate wall-clock timeout. A hang here is a real PROVEN FAIL."
    ),
    provision_timeout_seconds: int = typer.Option(
        120, "--provision-timeout-seconds", help="How long DockerSandbox waits for `docker run` to come up."
    ),
    install_timeout_seconds: int = typer.Option(
        300, "--install-timeout-seconds", help="Timeout for the dependency-install step."
    ),
    attempt_budget_seconds: int = typer.Option(
        1800,
        "--attempt-budget-seconds",
        help="Global wall-clock ceiling across one whole attempt. 0 disables it.",
    ),
    health_timeout_seconds: int = typer.Option(
        30, "--health-timeout-seconds", help="How long a declared service gets to pass its health check."
    ),
) -> None:
    """Replay every (repo, task, patch) in `--dataset` through the real
    `verdict run` pipeline and compare Verdict's own DONE/NOT_DONE/
    UNVERIFIED status to each example's human-assigned label — precision/
    recall/F1 per label, a full confusion matrix, and every individual
    disagreement, so a claim about how trustworthy Verdict is can be
    checked rather than taken on faith. Never fails the process, the same
    diagnostic-not-gate policy `calibrate` already uses: a below-threshold
    accuracy prints a warning, not a nonzero exit.
    """
    sandbox_config = _build_sandbox_config(
        sandbox_backend, sandbox_image, sandbox_cpus, sandbox_memory_mb,
        gate_timeout_seconds, provision_timeout_seconds, install_timeout_seconds, attempt_budget_seconds,
        health_timeout_seconds,
    )
    try:
        examples = load_ground_truth_dataset(dataset)
    except DatasetLoadError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    result = run_ground_truth(examples, sandbox_config=sandbox_config, threshold=threshold)
    render_ground_truth(result)


@app.command(name="flaky")
def flaky_cmd(
    task: str = typer.Option(..., "--task", help="Natural-language description of the work."),
    agent: str = typer.Option(
        ..., "--agent", help=f"Which adapter to drive: mock | {' | '.join(_REAL_AGENTS)}"
    ),
    repo: Path = typer.Option(..., "--repo", help="Path to a git repository to grade against."),
    trials: int = typer.Option(
        DEFAULT_TRIALS, "--trials", help="Independent runs to average over — each its own fresh worktree."
    ),
    compare_to: Path | None = typer.Option(
        None,
        "--compare-to",
        help="A previous `--json` output to compare against via a two-proportion z-test.",
    ),
    json_out: Path | None = typer.Option(
        None, "--json", help="Write this run's FlakinessResult to this path (for a later --compare-to)."
    ),
    sandbox_backend: str = typer.Option("docker", "--sandbox-backend", help=_SANDBOX_BACKEND_HELP),
    sandbox_image: str = typer.Option(
        "verdict-sandbox:0.1.0", "--sandbox-image", help="Image DockerSandbox runs."
    ),
    sandbox_cpus: float = typer.Option(2.0, "--sandbox-cpus", help="CPU limit passed to DockerSandbox."),
    sandbox_memory_mb: int = typer.Option(
        2048, "--sandbox-memory-mb", help="Memory limit (MB) passed to DockerSandbox."
    ),
    gate_timeout_seconds: int = typer.Option(
        600, "--gate-timeout-seconds", help="Per-gate wall-clock timeout. A hang here is a real PROVEN FAIL."
    ),
    provision_timeout_seconds: int = typer.Option(
        120, "--provision-timeout-seconds", help="How long DockerSandbox waits for `docker run` to come up."
    ),
    install_timeout_seconds: int = typer.Option(
        300, "--install-timeout-seconds", help="Timeout for the dependency-install step."
    ),
    attempt_budget_seconds: int = typer.Option(
        1800,
        "--attempt-budget-seconds",
        help="Global wall-clock ceiling across one whole attempt. 0 disables it.",
    ),
    health_timeout_seconds: int = typer.Option(
        30, "--health-timeout-seconds", help="How long a declared service gets to pass its health check."
    ),
) -> None:
    """Run `--agent` on `--task` against `--repo` `--trials` independent
    times and report the pass rate with a Wilson confidence interval. With
    `--compare-to`, also runs a two-proportion z-test against a prior run's
    saved result to say whether a pass-rate change is a real regression or
    just noise from a small sample — see DESIGN.md's Phase 7 section.

    Every real --agent trial is real spend, multiplied by --trials; this
    is a research/CI-diagnostics command, not something to run against a
    paid agent without budgeting for it.
    """
    adapter = _build_adapter(agent, repo)
    sandbox_config = _build_sandbox_config(
        sandbox_backend, sandbox_image, sandbox_cpus, sandbox_memory_mb,
        gate_timeout_seconds, provision_timeout_seconds, install_timeout_seconds, attempt_budget_seconds,
        health_timeout_seconds,
    )
    try:
        result = run_flakiness(
            task=task, repo=repo, adapter=adapter, trials=trials, sandbox_config=sandbox_config
        )
    except _RUN_ERRORS as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if compare_to is not None:
        if not compare_to.exists():
            typer.secho(f"no baseline found at {compare_to}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)
        baseline = FlakinessResult.model_validate_json(compare_to.read_text())
        comparison = compare_flakiness(
            baseline, result, baseline_label=str(compare_to), candidate_label=agent
        )
        render_comparison(comparison)
    else:
        render_flakiness(result, label=agent)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(result.model_dump_json(indent=2))
        typer.echo(f"wrote {json_out}")


@app.command(name="pr-comment")
def pr_comment_cmd(
    report_path: Path = typer.Argument(
        ..., help="Path to a verdict-report.json (from `--report json`) to build the comment from."
    ),
) -> None:
    """Print the advisory PR-comment body (JUDGED signals only) for a
    verdict-report.json to stdout. Used by the GitHub Action to build the
    comment it posts with `gh pr comment` — this command never touches
    GitHub itself, it only builds the text.
    """
    if not report_path.exists():
        typer.secho(f"no report found at {report_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    typer.echo(build_comment_from_file(report_path))


# Phase 17: querying and comparing against persisted history. A separate
# `typer.Typer` sub-app (this codebase's first) rather than two more flat
# top-level commands — "history list"/"history compare" reads as one
# family of operations on a `--store`, the same way a real CLI groups
# subcommands under a noun rather than flattening everything.
history_app = typer.Typer(add_completion=False, help="Query and compare against persisted run history.")
app.add_typer(history_app, name="history")


@history_app.command(name="list")
def history_list_cmd(
    store: Path = typer.Option(..., "--store", help="Path to a SQLite history db."),
    task: str | None = typer.Option(None, "--task", help="Narrow to one task's outcome history over time."),
    agent: str | None = typer.Option(None, "--agent", help="Narrow to one agent — requires --task/--repo."),
    repo: str | None = typer.Option(None, "--repo", help="Narrow to one repo — requires --task/--agent."),
    config_label: str | None = typer.Option(None, "--config-label", help="Narrow to one config label."),
    limit: int = typer.Option(20, "--limit", help="Most-recent N entries to show."),
) -> None:
    """With no --task/--agent/--repo: lists recorded RUNS, most recent
    first. With all three of --task/--agent/--repo given: lists that exact
    task's OUTCOME history over time instead — the same query
    `regression.py` runs internally, exposed directly so a team can see
    the trend a regression flag (or the HTML report's sparkline) is based
    on.
    """
    db = SQLiteStore(store)

    if task is not None or agent is not None or repo is not None:
        if task is None or agent is None or repo is None:
            raise typer.BadParameter("--task, --agent, and --repo must all be given together")
        outcomes = db.history(task, agent, repo, config_label=config_label, limit=limit)
        if not outcomes:
            typer.echo("no recorded history for this (task, agent, repo)")
            return
        table = Table(title=f"History — {agent} × {task[:60]!r}")
        table.add_column("recorded_at")
        table.add_column("config")
        table.add_column("commit")
        table.add_column("status")
        for o in outcomes:
            table.add_row(
                o.recorded_at.isoformat(),
                o.config_label,
                (o.commit_sha or "—")[:12],
                f"[green]{o.status}[/green]" if o.done else f"[red]{o.status}[/red]",
            )
        Console().print(table)
        return

    runs = db.runs(limit=limit)
    if not runs:
        typer.echo("no recorded runs")
        return
    table = Table(title="Recorded runs")
    table.add_column("run_id")
    table.add_column("recorded_at")
    table.add_column("commit")
    table.add_column("label")
    table.add_column("configs")
    for r in runs:
        table.add_row(
            r.run_id, r.recorded_at.isoformat(), (r.commit_sha or "—")[:12], r.label or "—",
            ", ".join(r.config_labels),
        )
    Console().print(table)


@history_app.command(name="compare")
def history_compare_cmd(
    store: Path = typer.Option(..., "--store", help="Path to a SQLite history db."),
    run_id: str = typer.Option(..., "--run-id", help="A previously recorded run to check for regressions."),
    baseline_window: int = typer.Option(
        DEFAULT_BASELINE_WINDOW,
        "--baseline-window",
        help="How many of the most recent prior recorded runs form the regression baseline.",
    ),
    alpha: float = typer.Option(
        0.05, "--regression-alpha", help="Significance threshold for the historical regression z-test."
    ),
) -> None:
    """Re-run regression detection for an already-recorded `--run-id`
    against its own historical baseline — useful for retroactively
    checking any past run without re-grading anything. Reuses exactly the
    same `detect_regressions` this codebase's `bench`/`gate` commands
    already call inline (`_persist_run` in this module) — never a
    separate implementation of the comparison itself.
    """
    db = SQLiteStore(store)
    config_results = db.get_config_results(run_id)
    if not config_results:
        typer.secho(f"no recorded run found for run_id={run_id!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    regressions = detect_regressions(
        db, config_results, baseline_window=baseline_window, alpha=alpha, exclude_run_id=run_id
    )
    if not regressions:
        typer.secho("no regressions found against recorded history", fg=typer.colors.GREEN)
        return
    _render_regressions(regressions)


if __name__ == "__main__":
    app()
