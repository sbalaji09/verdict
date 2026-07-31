# Verdict — Design

Phase 0's goal was to prove the spine works end to end — isolate, drive an
agent, verify with something *executed*, report a verdict — before any of
the richer phases (causal attribution, cost, frontend truth) got added on
top. Phase 1 expands verification from one gate to four (test/typecheck/
build/lint) and adds the three-state (pass/fail/n-a) model that a repo with
only a subset of those stacks needs. Everything below is scoped to what's
actually built, not the eventual product; sections are marked by phase
where the distinction matters.

## Why a git worktree, not a clone or a copy

Three ways to give an agent an isolated checkout: `cp -r`, `git clone`, or
`git worktree add`. Phase 0 uses a worktree because:

- **Shared object store.** A worktree links back to the same `.git`
  objects as the source repo — no copying history, no duplicating a
  potentially large `.git` directory per attempt. A clone would re-copy the
  whole history; `cp -r` would copy `.git` too, uselessly.
- **A real, independent working directory.** Unlike `cp -r`, a worktree has
  its own git index and `HEAD`, so the agent can `git add`, `git commit`,
  even create its own branches, without any of that touching the source
  repo's index or `HEAD`.
- **It's disposable by construction.** `git worktree add -b verdict/<id>
  <path> HEAD` creates both a scratch directory *and* a scratch branch in
  one step; `git worktree remove --force` plus `git branch -D` guarantee
  no trace is left in the source repo, even if the run crashes — cleanup
  runs in a `finally` block (`src/verdict/worktree.py::isolated_worktree`).

Each attempt gets a fresh worktree off the source repo's current `HEAD` —
not a fixed base commit — so grading always reflects the repo's current
state, matching how a developer would actually hand a task to an agent.

**What "isolated" does *not* mean here:** no container, no network
sandboxing, no filesystem jail. A worktree only isolates the agent's edits
from the *source repo's working tree and index* — it does not stop an agent
process from reading/writing elsewhere on disk or making network calls.
That's a deliberate scope cut for Phase 0, not an oversight: containerized
sandboxing is a separate, later concern once there's something worth
sandboxing more tightly (e.g. running an agent against an untrusted task
suite in CI).

## The Adapter interface

```python
class Adapter(Protocol):
    name: str
    def run(self, task: str, worktree: Path) -> AttemptResult: ...
```

One method, one job: make the agent act on `task` inside `worktree`, then
report what changed (`diff`, `files_changed`) and what it cost
(`tokens_input/output`, `cost_usd`). An adapter must not decide whether the
work is *correct* — that's the gates' job, kept as a separate stage on
purpose. This is what lets `runner.py` swap `MockAdapter` for
`ClaudeCodeAdapter` (or, later, Cursor/Aider/Codex adapters) without
touching worktree isolation, gates, or reporting at all.

Two implementations ship in Phase 0:

- **`MockAdapter`** — writes a fixed `{path: contents}` map into the
  worktree. No LLM, no network, zero cost. Exists so the rest of the
  pipeline is testable and demoable without spending real API tokens.
- **`ClaudeCodeAdapter`** — shells out to the real `claude` CLI:
  `claude -p "<task>" --output-format json --permission-mode acceptEdits`,
  run with `cwd=worktree`. Three choices worth calling out:
  - **Subprocess, not the Python SDK.** The CLI is what a developer already
    has installed and authenticated; a subprocess boundary also means a
    hang or crash in the agent process can't corrupt Verdict's own process.
  - **`--permission-mode acceptEdits`.** Headless runs can't answer
    permission prompts, so edits are auto-accepted; this is scoped to the
    disposable worktree, never the real repo.
  - **`--output-format json`** hands back a `usage` object and
    `total_cost_usd` for free — exactly the fields `AttemptResult` needs,
    with no output-scraping.

Both adapters compute their `diff`/`files_changed` the same way, via
`worktree.diff_against_base()`: `git add -A` then `git diff --cached HEAD`.
Running `add -A` first means the diff captures uncommitted edits *and* any
commits the agent made on its own worktree branch — an agent is free to
`git commit` mid-task and the diff still comes out correct.

## The verdict schema: proven vs. judged, enforced at the type level

```python
class Provenance(str, Enum):
    PROVEN = "proven"   # executed, deterministic, reproducible
    JUDGED = "judged"   # an LLM/vision model's opinion

class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NA = "n/a"           # this gate's stack wasn't detected in the repo

class Signal(BaseModel):
    name: str
    provenance: Provenance
    status: GateStatus
    detail: str
    command: str | None = None
    exit_code: int | None = None

class VerdictStatus(str, Enum):
    DONE = "done"
    NOT_DONE = "not_done"
    UNVERIFIED = "unverified"  # zero PROVEN gates actually ran

class Verdict(BaseModel):
    task: str
    agent: str
    repo: str
    attempt: AttemptResult
    signals: list[Signal]

    @property
    def status(self) -> VerdictStatus:
        applicable = [
            s for s in self.signals
            if s.provenance is Provenance.PROVEN and s.status is not GateStatus.NA
        ]
        if not applicable:
            return VerdictStatus.UNVERIFIED
        if any(s.status is GateStatus.FAIL for s in applicable):
            return VerdictStatus.NOT_DONE
        return VerdictStatus.DONE
```

Every `Signal` — from Phase 0's single test gate, to Phase 1's typecheck/
build/lint, to a vision-intent judge in Phase 4 — carries its `Provenance`
from the moment it's created. There's no separate "trust level" field
bolted on afterward, and no code path that reads a judged opinion as if it
were an executed fact: `Verdict.status` only ever consults `PROVEN`
signals. A `JUDGED` signal can be glowing and it still can't turn a failing
proven check into `DONE`. This is the one property the whole project is
organized around, so it's enforced in the type that every downstream
consumer (CLI, reporter, and — eventually — the CI gate) reads, not just
as a convention.

**Why a third status, added in Phase 1.** Phase 0 had one gate, so a
binary passed/failed was enough. Phase 1 has four, and most repos only use
a subset — a Python-only repo has no `tsconfig.json`, a script-only repo
has no build step. A binary outcome forces a lie either way: call a
missing stack a "pass" and `DONE` stops meaning "every check I ran
succeeded"; call it a "fail" and a plain script repo can never be `DONE`
without adopting tooling it has no use for. `GateStatus.NA` says the true
thing — this gate contributes nothing, in either direction — and
`Verdict.status` excludes `NA` signals from the pass/fail check entirely
rather than defaulting them either way.

**Why `UNVERIFIED` is a third *verdict* status, not just "not done".** If
every gate comes back `NA` (a repo with none of the four stacks
detected, or a fresh directory with no config at all), "no failures" is
technically true but meaningless — nothing executed. Reporting `DONE` in
that case would be exactly the failure mode Verdict exists to prevent: an
absence of evidence read as evidence of correctness. `UNVERIFIED` names
that state instead of silently collapsing it into `DONE`. (`Verdict.done`,
the boolean convenience property the CLI's exit code uses, treats
`UNVERIFIED` as not-done — a verdict with nothing to point to has no basis
to claim success — but the report still shows `UNVERIFIED` distinctly from
`NOT_DONE`, because "nothing ran" and "something ran and failed" are
different facts a reader should be able to tell apart.)

**Why `confidence` is separate from `status`.** Among the four gates, only
`test` exercises *behavior* — typecheck/build/lint catch real defects, but
none of them can tell you the code does the right thing, only that it's
well-formed. So `Verdict.confidence` drops to `LOW` specifically when the
`test` gate didn't run (missing or `NA`), independent of whatever the
other three gates found. A repo with passing typecheck/build/lint and no
tests at all is legitimately `DONE` (nothing failed) but the report labels
it low-confidence `DONE` rather than presenting it identically to a `DONE`
backed by a passing test suite — same status, different evidentiary
weight, and the reader shouldn't have to infer that from the gate list
themselves.

`Signal` and `Verdict` already carry the shape Phase 4 needs (a `JUDGED`
signal is just a `Signal` with `provenance=JUDGED` and a model's rationale
in `detail`) — no schema migration required to add the vision judge later.

## The gate abstraction (Phase 1)

A **gate** (`test`/`typecheck`/`build`/`lint`) is a category; a
**ToolRunner** is one concrete way to satisfy it — pytest and jest both
satisfy `test`, they just apply to different repos:

```python
class ToolRunner(Protocol):
    tool: str     # "pytest", "tsc", "eslint", ...
    gate: str     # "test", "typecheck", "build", "lint"
    def applicable(self, worktree: Path) -> bool: ...
    def run(self, worktree: Path) -> Signal: ...
```

`gates/registry.py` resolves each gate independently, in this order:

1. **`verdict.yml` override**, if the repo's `gates.<name>` key is set —
   run verbatim, no structured parsing (see below).
2. **First applicable autodetected `ToolRunner`**, tried in a fixed
   priority list per gate (e.g. typecheck tries `tsc` then `mypy`).
3. **`GateStatus.NA`** if nothing applied.

| Gate | Runners tried, in order | Applicability check | Structured signal |
|---|---|---|---|
| test | pytest → jest → go test | `pytest.ini`/`[tool.pytest]`/`tests/test_*.py`; `jest.config.*` or a `jest` dependency; `go.mod` | pytest: `--junitxml=<tmp>`, parsed via `xml.etree`; jest: `--json --outputFile=<tmp>`; go: `go test -json ./...` (newline-delimited JSON events) |
| typecheck | tsc → mypy | `tsconfig.json` + `node_modules/.bin/tsc` present; `[tool.mypy]`/`mypy.ini` | tsc: regex on its default `file(line,col): error TSxxxx: msg` text (no native JSON mode exists); mypy: `--output json` (one JSON object per diagnostic — verified against a real mypy install before committing to this design, since it's an easy thing to get wrong) |
| build | npm build script | `package.json` has a `"build"` script | exit code + log tail only — see below for why |
| lint | eslint → ruff | eslint config file/`node_modules/.bin/eslint`; `[tool.ruff]`/`ruff.toml` | eslint: `-f json`; ruff: `--output-format=json` — both native JSON reporters |

**Why `build` is coarser than the other three.** Test/typecheck/lint each
have a tool-independent structured format (junit XML is a real standard;
eslint and ruff both ship JSON reporters). "Build" doesn't — `next build`,
`vite build`, and a hand-rolled webpack script share no schema, and
guessing which bundler is in play to parse its particular output would be
fragile and constantly falling behind new tools. So the build gate does
what the Phase 0 test gate already did when it had no structured format to
lean on: trust the repo's own `npm run build` and grade by exit code, with
the log tail as `detail`. This is a real, executed, `PROVEN` signal — it's
just a coarser one, and that's a property of build tooling in general, not
a shortcut Verdict is taking. There's no autodetected Python build gate at
all, for the same underlying reason at one remove: there's no
cross-project "the build step" convention in Python the way `package.json`
`scripts.build` is for npm. A `verdict.yml` override is the only way to
get a build gate on a non-npm repo.

**Why overrides skip structured parsing.** The autodetected paths above
all work because Verdict controls the exact invocation — it can add
`--output-format=json` because it built the command. A `verdict.yml`
override hands us an arbitrary string; guessing which tool it invokes from
the text (and hoping it happens to also request structured output) is
exactly the kind of fragile, silently-wrong heuristic the rest of this
design avoids elsewhere. So overrides run verbatim through `shell=True`
and are graded purely by exit code, with output tail as detail — same
shape as Phase 0's original test gate, and an intentional trade: exactness
of command vs. richness of parsed detail. The exit code is no less
`PROVEN` for it.

**Why the autodetected commands don't use `shell=True`.** Every command
Verdict itself constructs is built as an argv list and run without a
shell — there is no interpolated content in these commands (no task text,
no filenames from the diff), so there's no injection surface to close, but
building the habit of argv-not-string here is what makes the one
deliberate `shell=True` (the `verdict.yml` override path, which is
genuinely a user-supplied string) legible as a considered exception rather
than an oversight.

## Vendored dependencies and worktree isolation

Phase 0's worktree gives an agent a fresh checkout of every *tracked*
file. That's insufficient for npm-based gates: `node_modules` is (rightly)
gitignored, so a bare `git worktree add` never has it, and `typecheck`/
`build`/`lint` would report `NA` (missing binary) or a misleading `FAIL`
("tsc: command not found") on every single run — not a real defect, an
artifact of how the isolation was built. `runner.py` closes this gap after
creating the worktree: if the source repo has a known vendored-dependency
directory (`node_modules`, so far — the one Phase 1 needed) that the fresh
worktree lacks, it's **copied**, not symlinked, into the worktree before
the adapter or gates run. Copy rather than symlink so an agent that
reinstalls or mutates dependencies mid-task can never affect the source
repo's copy — the same isolation guarantee Phase 0 already gives tracked
files, extended to this one untracked-but-load-bearing directory. This
does not contradict Phase 0's "no filesystem jail" scope note — it doesn't
add any *sandboxing*, it just makes the existing tracked-files-only
isolation actually usable for a stack whose dependencies live outside git.

What Phase 1 does **not** do: run `npm install`/`pip install` fresh inside
the worktree. That would make every run's gate results depend on network
availability and add real, uncapped time per attempt — a heavier design
decision than copying an already-installed directory, and out of scope
here. A repo whose dependencies were never installed anywhere (not even in
the source checkout) will still see accurate `NA`/`FAIL` signals; it just
won't get dependencies materialized from nothing.

## Demo repos

- `examples/sample_repo/` — Python, one seeded bug (`add()` subtracts),
  one failing test. Demonstrates `test` (pytest), `typecheck` (mypy), and
  `lint` (ruff); `build` is honestly `NA` — no cross-project Python build
  convention exists to autodetect.
- `examples/sample_node_repo/` — TypeScript, same seeded bug. Demonstrates
  `typecheck` (tsc) and `build` (`tsc -p tsconfig.json` via npm). Its
  tests use Node's built-in test runner, which Phase 1 doesn't autodetect
  (only pytest/jest/go), so `test` is `NA` here too — a real, useful
  demonstration of `confidence` dropping to `LOW` on an otherwise-passing
  `DONE` verdict, not a gap papered over for the demo.

Both are real, standalone git repos (bootstrap each once with its
`setup.sh`, which for the node repo also runs `npm install` so
`node_modules/.bin/tsc` exists to be vendored into worktrees as described
above) so `--repo` always has something genuine to isolate, rather than a
fixture invented just for the CLI demo.

## What's explicitly out of scope through Phase 1

- No causal attribution of *why* a check failed to a specific edit: Phase 2.
- No cost-to-correct / pass-rate-per-dollar aggregation across attempts —
  `AttemptResult` captures cost per attempt, but nothing rolls it up yet:
  Phase 3.
- No frontend/browser checks at all: Phase 4.
- No retry loop, no multi-attempt orchestration — one task, one agent, one
  attempt, one verdict.
- No fresh dependency installation inside the worktree (see above) — only
  already-installed vendored directories are made available.
- Autodetection covers pytest/jest/go test, tsc/mypy, npm build scripts,
  and eslint/ruff — not every stack (e.g. no Rust/cargo, no Java/Maven).
  `verdict.yml` covers anything not autodetected, at the cost of
  structured parsing.
