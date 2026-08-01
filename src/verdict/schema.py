"""The verdict schema: every signal is tagged PROVEN or JUDGED, never blended.

Phase 0 emitted a single PROVEN signal with a binary passed/failed outcome.
Phase 1 adds three of its own signals (typecheck/build/lint) and most repos
only use a subset of the four available stacks — a Python-only repo has no
tsconfig.json, a script-only repo has no build step — so a binary outcome
can't represent "this gate doesn't apply here" without either lying (calling
it a pass) or being unfairly strict (calling it a failure). GateStatus adds
the third state that was missing. The Judged enum member and the fields
that support it (e.g. Signal.detail carrying a model's rationale) exist now
so Phase 4's vision-intent judge slots in without a schema migration later.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, computed_field


class Provenance(str, Enum):
    """How a signal's pass/fail was decided."""

    PROVEN = "proven"
    """Executed. Deterministic. Reproducible. E.g. a test suite exit code."""

    JUDGED = "judged"
    """An LLM/vision model formed an opinion. Advisory, never load-bearing."""


class GateStatus(str, Enum):
    """The outcome of one signal."""

    PASS = "pass"
    FAIL = "fail"
    NA = "n/a"
    """The gate's stack wasn't detected in this repo (e.g. no tsconfig.json
    for typecheck). Not a failure — a repo property, not an agent defect —
    but also not a pass: it contributes nothing to done/not-done either way.
    """


class Confidence(str, Enum):
    """How much a DONE/NOT_DONE verdict is worth trusting."""

    HIGH = "high"
    LOW = "low"
    """Downgraded when the test gate itself is N/A. Typecheck/build/lint
    catch real defects, but only the test suite exercises *behavior* — a
    repo with no detectable tests can still be reported DONE (every gate
    that did run passed), but that DONE is worth less.
    """


class VerdictStatus(str, Enum):
    """The outcome of a whole Verdict."""

    DONE = "done"
    NOT_DONE = "not_done"
    UNVERIFIED = "unverified"
    """Zero PROVEN gates actually ran (all were N/A, or no signals at all).
    Distinct from DONE: there's nothing executed to ground a claim of
    correctness in, so Verdict refuses to report success by default.
    """

    ERROR = "error"
    """Phase 10's minimal pull-forward of Phase 11's fuller ERROR outcome:
    the attempt couldn't be *evaluated* at all — sandbox provisioning
    failed, a declared service never became healthy, the install step
    hung — as distinct from NOT_DONE (the agent's code was evaluated and
    found wanting) and UNVERIFIED (evaluated, incompletely, but nothing
    observed was wrong). Never derived from `signals`/`_proven_applicable`
    the way the other three statuses are — a `Verdict` is constructed
    directly with `status_override=ERROR` (see `status` below) by whatever
    caught the underlying `SandboxError`, since by definition nothing
    executed to compute a signal-derived status from. Excluded from
    `ConfigResult.pass_rate`/`pass_rate_per_dollar` (see `schema.py`
    further down) — infra noise must never move that number in either
    direction. Full retry policy, a dedicated provenance bucket, and
    richer reporting remain Phase 11's job; this is deliberately just the
    one status value plus the accounting rule.
    """


class FailureLocation(BaseModel):
    """One individual failure within a gate's result — one failing test, one
    typecheck error, one lint violation. A gate's `detail` string is a
    human-readable summary; `failures` is the same underlying data kept
    structured, because Phase 2's attribution needs to re-identify "this
    exact failure" across multiple re-runs at different commits, and text-
    scraping `detail` back apart would just re-derive what the gate parser
    already knew before it got flattened into a string.
    """

    identity: str
    """A stable id for matching the *same* failure across re-runs, even as
    line numbers drift between intermediate bisection states. A pytest node
    id (`tests/test_x.py::test_add`) for test failures; `"{file}:{code}"`
    for typecheck/lint, deliberately excluding the line number.
    """
    file: str | None = None
    line: int | None = None
    code: str | None = None
    message: str = ""


class Signal(BaseModel):
    """One executed check (or, later, one judged opinion) and its outcome."""

    name: str
    provenance: Provenance
    status: GateStatus
    detail: str
    command: str | None = None
    exit_code: int | None = None
    failures: list[FailureLocation] = Field(default_factory=list)
    artifact_path: str | None = None
    """Path to a supporting file too rich for `detail` to hold inline — e.g.
    Phase 4's glitch-scan video recording. Only ever supporting evidence for
    a PROVEN/JUDGED result already decided from other data; nothing reads
    the artifact itself to decide status."""


class AttemptResult(BaseModel):
    """What an agent did during one attempt: its diff and what it cost."""

    diff: str
    files_changed: list[str] = Field(default_factory=list)
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float | None = None
    raw_output: str | None = None


class AttributionKind(str, Enum):
    """How confidently — and whether — a failure was pinned on the agent."""

    REGRESSION = "regression"
    """Bisection isolated exactly one file the agent changed as necessary
    and sufficient to reproduce the failure."""

    PRE_EXISTING = "pre_existing"
    """The failure already reproduced at the worktree's base commit, before
    the agent touched anything — not the agent's fault, full stop."""

    INCONCLUSIVE = "inconclusive"
    """Bisection ran but couldn't cleanly isolate a single culprit (e.g. too
    many untestable intermediate states). Honest non-answer, not a guess."""


class DependencyLink(BaseModel):
    """A static import edge found between the failure's location and the
    culprit file — enrichment only. Absence of a link never weakens the
    Attribution's own proof (that's bisection's job); presence only adds a
    supporting "why" sentence.
    """

    from_file: str
    to_file: str
    depth: int


class Attribution(BaseModel):
    """A causal claim: this failing check, at this location, was caused by
    the agent's change to this file. The *link* (kind, culprit_file, method)
    is always PROVEN — derived from bisection, an executed, repeatable
    procedure — never from an opinion. `explanation` is a template-rendered
    sentence over those same fields; it renders the proof, it doesn't add
    to it.
    """

    provenance: Provenance = Provenance.PROVEN
    kind: AttributionKind
    check_name: str
    failure_id: str
    failure_location: str | None = None
    culprit_file: str | None = None
    method: str
    dependency_link: DependencyLink | None = None
    explanation: str


class Verdict(BaseModel):
    """The end-to-end result of grading one agent attempt at one task."""

    task: str
    agent: str
    repo: str
    attempt: AttemptResult
    signals: list[Signal]
    attributions: list[Attribution] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    budget_exceeded: bool = False
    """Phase 9: True if the global per-attempt wall-clock budget
    (`SandboxConfig.attempt_budget_seconds`) was hit before every gate
    could even be attempted. `signals` in that case is a real but partial
    list — gates that never got to run are simply absent from it, not
    reported as FAIL (Verdict doesn't know they'd have failed) or NA (NA
    means "this repo has no such stack," which isn't what happened). See
    `status` below for how this affects DONE/NOT_DONE/UNVERIFIED, and
    DESIGN.md's Phase 9 section for the full reasoning.
    """
    error: str | None = None
    """Phase 10: set only when this Verdict represents a setup/
    provisioning failure — sandbox never came up, a declared service
    never became healthy, the install step hung. Presence of this field
    is what makes `status` ERROR (see below), overriding the signal-
    derived logic entirely, since by construction there's nothing to
    derive a status *from*: `signals`/`attributions` are empty in this
    case, not partially populated the way a `budget_exceeded` Verdict's
    are.
    """

    def _proven_applicable(self) -> list[Signal]:
        return [
            s
            for s in self.signals
            if s.provenance is Provenance.PROVEN and s.status is not GateStatus.NA
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> VerdictStatus:
        """DONE only if every *applicable* PROVEN gate passed. JUDGED
        signals never affect this — an opinion can flag concern but can't
        override or stand in for executed fact. N/A gates are excluded
        from the check entirely, not treated as passes. If nothing
        executable actually ran, the verdict is UNVERIFIED rather than a
        default DONE — "nothing failed" is not the same claim as "this
        works", and Verdict never lets the former masquerade as the latter.

        A real PROVEN FAIL always wins, budget or not — an agent-caused
        failure that was actually observed doesn't stop being real just
        because the run later ran out of time elsewhere; it still counts
        as NOT_DONE and still counts against the agent in metrics. But a
        `budget_exceeded` run that saw no FAIL is never DONE either: gates
        that never ran can't vouch for the ones that would have, so
        "everything we managed to check passed" is downgraded to
        UNVERIFIED rather than reported as a clean win.

        `error` set takes priority over everything else: a setup failure
        means nothing below was actually evaluated, so there's no FAIL/
        budget/applicable-signals logic left to run — ERROR, full stop.
        """
        if self.error is not None:
            return VerdictStatus.ERROR
        applicable = self._proven_applicable()
        if any(s.status is GateStatus.FAIL for s in applicable):
            return VerdictStatus.NOT_DONE
        if not applicable or self.budget_exceeded:
            return VerdictStatus.UNVERIFIED
        return VerdictStatus.DONE

    @computed_field  # type: ignore[prop-decorator]
    @property
    def done(self) -> bool:
        """Convenience boolean for callers that just need pass/fail (e.g.
        the CLI's exit code). UNVERIFIED counts as not-done: a verdict that
        couldn't verify anything has no basis to report success.
        """
        return self.status is VerdictStatus.DONE

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> Confidence:
        """LOW whenever the test gate didn't actually run (missing, or
        N/A). Typecheck/build/lint gates don't affect this — they catch
        real defects, but only tests exercise behavior, so a DONE verdict
        that never ran a test suite is real but weaker evidence.
        """
        test_signal = next(
            (
                s
                for s in self.signals
                if s.name == "test" and s.provenance is Provenance.PROVEN
            ),
            None,
        )
        if test_signal is None or test_signal.status is GateStatus.NA:
            return Confidence.LOW
        return Confidence.HIGH


class TaskRun(BaseModel):
    """The result of attempting one task, possibly across several retries.
    Cost is accounted across *every* attempt, not just the last — a dead
    end still spent real tokens, and reporting only the winning attempt's
    cost would understate what the task actually took.
    """

    task: str
    agent: str
    repo: str
    attempts: list[Verdict]

    @property
    def final(self) -> Verdict:
        """The attempt that decides done/not-done: the first DONE one if
        there was one, otherwise the last attempt made (run_with_retries
        stops early on DONE, so in practice this is just `attempts[-1]`)."""
        return self.attempts[-1]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def done(self) -> bool:
        return self.final.done

    @computed_field  # type: ignore[prop-decorator]
    @property
    def errored(self) -> bool:
        """True if the attempt that decides this task's outcome couldn't
        be evaluated at all (Phase 10's ERROR status) — infra, never the
        agent's fault. `ConfigResult` excludes errored tasks from
        `pass_rate`'s denominator (see below) so infra noise can't move a
        leaderboard number in either direction.
        """
        return self.final.status is VerdictStatus.ERROR

    @computed_field  # type: ignore[prop-decorator]
    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed_attempt_count(self) -> int:
        return sum(1 for v in self.attempts if not v.done)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens_input(self) -> int:
        return sum(v.attempt.tokens_input for v in self.attempts)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens_output(self) -> int:
        return sum(v.attempt.tokens_output for v in self.attempts)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_cost_usd(self) -> float | None:
        """None if *any* attempt's cost is unknown — summing the attempts
        that do have a figure and calling that "total" would understate
        the real number while looking precise. Better to say "unknown."
        """
        total = 0.0
        for v in self.attempts:
            if v.attempt.cost_usd is None:
                return None
            total += v.attempt.cost_usd
        return total


class ConfigResult(BaseModel):
    """One (agent, config) combination's aggregate result across a set of
    tasks — the unit a leaderboard ranks. "config" is a free-form label the
    caller supplies (e.g. "claude-code / sonnet" vs "claude-code / opus");
    Verdict doesn't need to understand what varies to compare their
    economics.
    """

    label: str
    task_runs: list[TaskRun]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tasks_total(self) -> int:
        return len(self.task_runs)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tasks_done(self) -> int:
        return sum(1 for t in self.task_runs if t.done)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tasks_errored(self) -> int:
        return sum(1 for t in self.task_runs if t.errored)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> float:
        """Errored tasks are excluded from BOTH the numerator (already true
        — `errored` implies not `done`) AND the denominator: a task whose
        sandbox never came up was never actually graded, so it shouldn't
        dilute the rate the way a real observed failure does. `0.0` (not
        `None`) when every task errored — same "report an honest number,
        never a guess" instinct as `total_cost_usd`, just with a
        zero-tasks-graded floor instead of an unknown-cost `None`.
        """
        denominator = self.tasks_total - self.tasks_errored
        if denominator <= 0:
            return 0.0
        return self.tasks_done / denominator

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_cost_usd(self) -> float | None:
        total = 0.0
        for t in self.task_runs:
            cost = t.total_cost_usd
            if cost is None:
                return None
            total += cost
        return total

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens_input(self) -> int:
        return sum(t.total_tokens_input for t in self.task_runs)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens_output(self) -> int:
        return sum(t.total_tokens_output for t in self.task_runs)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate_per_dollar(self) -> float | None:
        """"verdict-pts/$" — one verdict point per task confirmed DONE.
        None when cost is unknown (never divide by a guess) or zero (zero
        spend with zero done tasks is undefined, not infinitely good).
        """
        cost = self.total_cost_usd
        if cost is None or cost <= 0:
            return None
        return self.tasks_done / cost
