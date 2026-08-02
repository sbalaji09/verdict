"""Phase 13: make net-new work verifiable. Passing the visible suite
doesn't prove "add feature X" was done if no visible test ever exercised
X — an agent (or a lazy benchmark task) can look done by construction, not
by having actually built anything. SWE-bench's own answer to this is the
one adopted here: a task can carry HELD-OUT acceptance tests that never
exist in the repo the agent sees, applied only after the agent is finished,
graded against two lists —

- **FAIL_TO_PASS** — tests that fail on the unmodified repo and MUST pass
  after the agent's fix. This is the actual proof of "X was built."
- **PASS_TO_PASS** — tests that already pass and MUST keep passing. The
  regression guard: fixing X by breaking Y doesn't count.

This also resolves Phase 12's open problem for tasks that legitimately
edit tests ("add tests for X"): once acceptance lives in tests the agent
never sees and cannot edit, gaming the *visible* suite stops mattering for
grading purposes. The two mechanisms are complementary, not redundant —
Phase 12's integrity gate still watches the visible suite (worth catching
early, worth a human's attention even when it isn't decisive), while this
module is the actual, ungameable ground truth for tasks that declare it.

## On-disk format

A `SuiteTask` directory MAY add two `task.yml` keys plus a sibling file,
alongside the existing `task`/`repo`/`category`/`allow_test_changes`:

```text
my_suite/add-retry-logic/
  task.yml           # + fail_to_pass: [...], pass_to_pass: [...]
  tests.patch         # a real `git diff`, never copied into repo/
  repo/               # the agent's actual (patch-free) starting point
```

```yaml
task: "Add retry logic with exponential backoff to the HTTP client"
fail_to_pass:
  - "tests/test_http_client.py::test_retries_on_5xx"
  - "tests/test_http_client.py::test_backoff_is_exponential"
pass_to_pass:
  - "tests/test_http_client.py::test_get_returns_body"
```

`tests.patch` is a plain unified diff (`git diff --no-color`, exactly what
`git apply` already accepts) — NOT copied into `repo/`, so the agent's
worktree never contains it and never even gets a chance to read, edit, or
delete the held-out tests it's about to be judged against. Authoring one
is the same motion as authoring the task itself: implement the feature for
real in a scratch checkout, write the tests that prove it, then `git diff`
the test files (and only the test files — a patch that also touches
`repo/`'s source is a task-authoring bug, not a hidden-test one) back out
into `tests.patch`, and revert the source so `repo/` ships unfixed.

A unified-diff FILE was chosen over a "hidden_tests/" directory of whole
files for one concrete reason a directory can't express: applying it can
also patch an EXISTING visible test file (add one more test function to
`tests/test_http_client.py` without duplicating the whole file, or without
the agent's own edits to that same file silently overwriting a same-named
hidden copy). A directory of whole files was the other option seriously
considered; it loses precisely that "patch an existing file" case and
would need its own merge policy for it. SWE-bench's own dataset format
already ships hidden tests as a unified diff for the same reason, which
is the second reason to match it here — a benchmark author converting an
existing SWE-bench-style task doesn't need to reshape anything.

## Grading semantics

Declared node ids are run — and ONLY they, not the whole suite — against
the agent's FINAL commit with `tests.patch` applied on top, inside a
disposable scratch worktree the agent's own worktree never sees. Every
FAIL_TO_PASS and PASS_TO_PASS id must come back PASS; anything else (FAIL,
ERROR, skipped, or never collected at all) fails the whole `acceptance`
signal. Deliberately NOT re-verified that FAIL_TO_PASS actually FAILS on
the unpatched base commit at grading time (only at task-authoring time,
the benchmark author's own responsibility) — SWE-bench's own harness makes
the same simplification, and re-deriving it here would mean a second,
redundant sandboxed run per grading pass for no grading-time benefit.

Emits exactly one PROVEN `Signal` named "acceptance" when the task
declares either list, appended after gates/attribution/integrity/frontend
checks the same way those already are (see `runner.py`) — a `KeyError` in
`gates/registry.py::resolve_gate` is exactly why: bisection only knows the
four `GATE_RUNNERS` names, "acceptance" isn't a fifth. Emits nothing (no
signal at all, not a vacuous PASS) when the task declares neither list —
most tasks still won't have held-out tests, and a `Verdict` shouldn't
carry a phantom signal for a check that was never asked for. `Verdict.
status`'s FAIL check was already gate-name-agnostic before this phase
(confirmed in Phase 12, unchanged here), so an "acceptance" FAIL forces
NOT_DONE through the exact same path a real `test` FAIL already does —
no schema change needed.

Only wired into `runner.run()`/`run_with_retries()` (suite/bench mode,
where a task directory exists to hold `tests.patch` next to `task.yml`) —
NOT into `grade_existing_diff` (`verdict gate`), which grades an arbitrary
PR diff with no benchmark task directory to source a patch from. See
DESIGN.md's own "out of scope" note for this phase.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from verdict.gates.base import exec_command
from verdict.gates.test import _node_id
from verdict.sandbox import Sandbox, SandboxConfig, create_sandbox
from verdict.sandbox.install import run_setup_step
from verdict.schema import FailureLocation, GateStatus, Provenance, Signal
from verdict.worktree import WorktreeError, copy_vendored_dependencies, run_git, scratch_worktree


@dataclass(frozen=True)
class AcceptanceSpec:
    """A task's held-out acceptance criteria — see this module's docstring
    for the on-disk format `suite/loader.py` reads this from. `patch` is
    the raw `tests.patch` text; empty (never applied) when both lists are
    empty, which is also what `declared` is for: most tasks have no
    held-out tests at all, and every caller should check `declared` (or
    just call `check_acceptance`, which already does) rather than assume
    a non-empty `AcceptanceSpec` is always meaningful.
    """

    fail_to_pass: tuple[str, ...] = ()
    pass_to_pass: tuple[str, ...] = ()
    patch: str = ""

    @property
    def declared(self) -> bool:
        return bool(self.fail_to_pass or self.pass_to_pass)


NONE_ACCEPTANCE = AcceptanceSpec()
"""The default every caller resolves a missing `AcceptanceSpec` to — most
tasks don't declare held-out tests, and `check_acceptance(..., NONE_ACCEPTANCE)`
returns `None` (no signal at all) immediately, without touching a
sandbox.
"""


def _parse_testcase_statuses(report_path: str) -> dict[str, GateStatus]:
    """Every `<testcase>` junit reported, PASS or FAIL, keyed by the same
    `pytest <node-id>`-runnable identity `gates/test.py::_node_id` already
    builds for attribution — reused here rather than re-derived, since a
    FAIL_TO_PASS/PASS_TO_PASS entry in `task.yml` is written in exactly
    that same node-id spelling. A `<skipped>` testcase counts as FAIL for
    acceptance purposes: an agent that got a required test skipped
    (`pytest.mark.skip`, a collection error, whatever) hasn't proven
    anything, whatever pytest's own exit code says about it.
    """
    try:
        root = ET.parse(report_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return {}
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return {}

    statuses: dict[str, GateStatus] = {}
    for tc in suite.findall("testcase"):
        node_id = _node_id(tc.get("classname", ""), tc.get("name", ""))
        not_passing = tc.find("failure") is not None or tc.find("error") is not None
        not_passing = not_passing or tc.find("skipped") is not None
        statuses[node_id] = GateStatus.FAIL if not_passing else GateStatus.PASS
    return statuses


def _run_targeted_pytest(
    worktree: Path,
    node_ids: list[str],
    sandbox: Sandbox | None,
    timeout_seconds: int,
    env: dict[str, str] | None,
) -> dict[str, GateStatus]:
    fd, report_path = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    command = ["pytest", "-q", f"--junitxml={report_path}", *node_ids]
    try:
        exec_command(command, cwd=worktree, sandbox=sandbox, timeout_seconds=timeout_seconds, env=env)
        return _parse_testcase_statuses(report_path)
    finally:
        Path(report_path).unlink(missing_ok=True)


def _apply_patch(worktree: Path, patch: str) -> str | None:
    """Applies `patch` (a unified diff) to `worktree`'s working tree.
    Returns an error message on failure, `None` on success — a caller
    turns that message into a real acceptance FAIL, never lets it raise:
    a patch that doesn't apply cleanly against the agent's final commit
    (the agent's edits conflicted with a file the hidden tests touch, or
    the patch itself is stale/malformed) is evaluable, real information
    about this attempt, not an infra failure.
    """
    fd, patch_path = tempfile.mkstemp(suffix=".patch")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(patch)
        run_git("apply", "--whitespace=nowarn", patch_path, cwd=worktree)
    except WorktreeError as exc:
        return str(exc)
    finally:
        Path(patch_path).unlink(missing_ok=True)
    return None


def check_acceptance(
    repo: Path,
    attempt_commit: str,
    spec: AcceptanceSpec,
    sandbox_config: SandboxConfig,
) -> Signal | None:
    """The one entry point. `None` when `spec` declares nothing — most
    tasks — so `runner.py` never appends a phantom signal for a check
    nobody asked for. Otherwise always a real PROVEN `Signal` named
    "acceptance": PASS only if every FAIL_TO_PASS and PASS_TO_PASS id
    came back a real pytest PASS against the patched final commit.

    Runs in its own fresh `scratch_worktree`/sandbox, never the agent's
    own worktree/sandbox — applying `tests.patch` mutates the working
    tree, and doing that to the worktree gates/frontend/integrity checks
    already ran against would contaminate what those checks saw. A
    `SandboxError` provisioning THIS sandbox is deliberately NOT caught
    here: unlike Phase 12's coverage sub-check (explicitly best-effort),
    acceptance is the authoritative signal for a task that declares
    it — degrading it to a silent skip on infra trouble would be exactly
    the "quietly report success anyway" failure mode Phase 11's ERROR
    status exists to prevent. It propagates up to `runner.py`'s ordinary
    `_EVALUATION_ERRORS` handling and the attempt is retried/reported
    ERROR like any other infra failure, never silently treated as PASS.
    """
    if not spec.declared:
        return None

    node_ids = [*spec.fail_to_pass, *spec.pass_to_pass]

    with scratch_worktree(repo, attempt_commit) as wt:
        copy_vendored_dependencies(repo, wt)

        if spec.patch:
            apply_error = _apply_patch(wt, spec.patch)
            if apply_error is not None:
                return Signal(
                    name="acceptance",
                    provenance=Provenance.PROVEN,
                    status=GateStatus.FAIL,
                    detail=f"held-out test patch did not apply cleanly:\n{apply_error}",
                    failures=[
                        FailureLocation(
                            identity="tests_patch_apply_failed",
                            code="tests_patch_apply_failed",
                            message=apply_error,
                        )
                    ],
                )

        setup_env = run_setup_step(wt, sandbox_config)
        with create_sandbox(wt, sandbox_config) as sandbox:
            statuses = _run_targeted_pytest(
                wt, node_ids, sandbox, sandbox_config.gate_timeout_seconds, setup_env
            )

    findings: list[FailureLocation] = []
    for kind, ids in (("fail_to_pass", spec.fail_to_pass), ("pass_to_pass", spec.pass_to_pass)):
        for node_id in ids:
            result = statuses.get(node_id)
            if result is GateStatus.PASS:
                continue
            seen = "never collected" if result is None else "did not pass"
            findings.append(
                FailureLocation(
                    identity=f"{kind}:{node_id}",
                    file=node_id.split("::", 1)[0],
                    code=kind,
                    message=f"{kind} test {seen}: {node_id}",
                )
            )

    status = GateStatus.FAIL if findings else GateStatus.PASS
    if findings:
        detail = f"{len(findings)} held-out acceptance test(s) failed:\n" + "\n".join(
            f"- {f.message}" for f in findings
        )
    else:
        detail = (
            f"all {len(spec.fail_to_pass)} FAIL_TO_PASS and {len(spec.pass_to_pass)} "
            "PASS_TO_PASS held-out tests passed."
        )

    return Signal(
        name="acceptance",
        provenance=Provenance.PROVEN,
        status=status,
        detail=detail,
        failures=findings,
    )
