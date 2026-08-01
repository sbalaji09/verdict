from __future__ import annotations

import sys
from pathlib import Path

import typer

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
from verdict.frontend.vision_judge import MockVisionJudge, VisionJudge
from verdict.pr_comment import build_comment_from_file
from verdict.report import render_task_run
from verdict.report_html import render_html
from verdict.report_json import render_json
from verdict.runner import grade_existing_diff, run_with_retries
from verdict.sandbox import ResourceLimits, SandboxConfig
from verdict.sandbox.base import SandboxUnavailableError
from verdict.schema import ConfigResult, TaskRun
from verdict.suite import BenchConfig, SuiteLoadError, load_suite, run_suite
from verdict.worktree import WorktreeError

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


def _write_machine_reports(formats: list[str], output_dir: Path, config_results: list[ConfigResult]) -> None:
    """Writes the `json`/`html` reporters if requested — `cli` is handled
    separately by each command's own rich-based renderer, since that one
    prints to the terminal rather than a file. Shared across `run`/`bench`/
    `gate` so all three reporters (and the merge-gate/scorecard commands
    that produce them) stay in exact lockstep: one `ConfigResult` shape,
    one place that serializes it.
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
        path.write_text(render_html(config_results))
        typer.echo(f"wrote {path}")


@app.command(name="run")
def run_cmd(
    task: str = typer.Option(..., "--task", help="Natural-language description of the work."),
    agent: str = typer.Option(
        ..., "--agent", help=f"Which adapter to drive: mock | {' | '.join(_REAL_AGENTS)}"
    ),
    repo: Path = typer.Option(..., "--repo", help="Path to a git repository to grade against."),
    max_attempts: int = typer.Option(
        1,
        "--max-attempts",
        help=(
            "Retry on failure up to this many times, stopping early on DONE. "
            "Cost is tracked across every attempt, dead ends included. "
            "Each retry re-runs the real agent — with a real --agent that's real spend."
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
    )
    try:
        task_run = run_with_retries(
            task=task, repo=repo, adapter=adapter, max_attempts=max_attempts, sandbox_config=sandbox_config
        )
    except _RUN_ERRORS as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if "cli" in report:
        render_task_run(task_run)
    _write_machine_reports(report, output_dir, [ConfigResult(label=agent, task_runs=[task_run])])

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
    )

    try:
        results = run_suite(tasks, configs, max_attempts=max_attempts, sandbox_config=sandbox_config)
    except _RUN_ERRORS as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if "cli" in report:
        economics.render(results)
        render_failure_modes(results)
    _write_machine_reports(report, output_dir, results)


@app.command(name="gate")
def gate_cmd(
    repo: Path = typer.Option(
        Path("."), "--repo", help="Path to the repo to grade — graded in place, not isolated."
    ),
    base: str = typer.Option(
        ..., "--base", help="Git ref to diff/attribute against (a PR's base branch or merge-base SHA)."
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
    )
    try:
        verdict = grade_existing_diff(repo=repo, base_ref=base, sandbox_config=sandbox_config)
    except WorktreeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    task_run = TaskRun(task=verdict.task, agent=label, repo=verdict.repo, attempts=[verdict])

    if "cli" in report:
        render_task_run(task_run)
    _write_machine_reports(report, output_dir, [ConfigResult(label=label, task_runs=[task_run])])

    if not verdict.done:
        sys.exit(1)


# Only a mock ships today — same honest scope line Phase 4 drew for
# VisionJudge itself: a real vision-model integration is its own project
# (pick a vendor, handle auth, validate against real screenshots), not
# something to fake here.
_JUDGES: dict[str, type[VisionJudge]] = {
    "mock": MockVisionJudge,
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
    judge: str = typer.Option("mock", "--judge", help=f"Which VisionJudge to score: {', '.join(_JUDGES)}"),
    threshold: float = typer.Option(
        DEFAULT_CONCORDANCE_THRESHOLD, "--threshold", help="Target concordance (fraction, e.g. 0.95)."
    ),
) -> None:
    """Score `--judge` against a human-labeled dataset and report its
    concordance — how often the judge's PASS/FAIL agrees with the human
    label. Never fails the process: this is a diagnostic, not a merge gate,
    so a below-threshold result prints a warning rather than a nonzero exit.
    """
    try:
        examples = load_labeled_dataset(dataset)
    except DatasetLoadError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    result = run_calibration(_build_judge(judge), examples, threshold=threshold)
    render_calibration(result)


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


if __name__ == "__main__":
    app()
