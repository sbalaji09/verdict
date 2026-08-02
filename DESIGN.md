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

---

## Phase 2 — Causal Failure Attribution

## The goal, restated

For every failing `PROVEN` gate, answer: *which specific file the agent
changed caused this, and why isn't a guess?* The output shape is the
README's own example: *"agent edited `FILE`, which caused `CHECK` to fail
at `LOCATION`."* The hard constraint carried over from Phase 0's schema
philosophy: the **link** — which file, which check — has to be an executed,
reproducible fact, tagged `PROVEN` like everything else. Only the sentence
wrapping that fact is template-rendered, and even that never states
anything the structured `Attribution` record doesn't already contain.

This was prototyped on paper and reviewed before any code was written (see
the design conversation this section summarizes); two real bugs surfaced
once the paper design met actual git, both fixed and covered by tests
before the fixtures went green — noted inline below, since "designed
correctly on the whiteboard" and "correct once it touches a real repo"
turned out to be different milestones.

## Two prerequisite fixes to Phase 0's worktree code

Attribution needs to answer "did this fail *before* the agent touched
anything?" and "which of the agent's real commits introduced this?" —
neither question was answerable with what Phase 0 built:

1. **`Worktree` didn't remember its own starting point.** `isolated_worktree`
   created a branch off `HEAD` but never recorded that commit anywhere.
   Fixed by adding `Worktree.base_commit`, captured once at creation time.
2. **`diff_against_base` diffed against `HEAD`, not the base commit.** If
   an agent commits mid-run, `HEAD` moves to point at its own commits, and
   `git diff --cached HEAD` would only see trailing uncommitted scraps —
   silently missing everything already committed. Fixed by diffing against
   the recorded `base_commit` explicitly. This was a real, if latent, bug
   in Phase 0/1 code, not something new to Phase 2 — Phase 2 is just what
   finally exercised the code path (a real agent committing as it works)
   that would have tripped over it.

## The core mechanism: two-level `git bisect`

The brief calls for "using git bisect over the agent's commits." Taken
literally, that only works if the agent made multiple commits — the
adapters built so far produce one flat diff, committed or not, all at
once. The design handles both with the *same* mechanism, applied twice:

```text
Level 1 — commit-level bisect (real agent commits, if any)
  base ──▶ C1 ──▶ C2 ──▶ ... ──▶ Cn (= the agent's final state)
  Finds: the first commit where the specific failing check starts failing.
  Skipped (trivially "the one commit") if the agent made ≤1 commit — today's
  default, since neither adapter commits incrementally yet.

Level 2 — file-level bisect, *within* the level-1 culprit commit
  parent(culprit) ──▶ +file_a ──▶ +file_a+file_b ──▶ ... ──▶ culprit
  These are synthetic commits Verdict builds itself: for each file the
  culprit commit touched, `git checkout <culprit> -- <file>` pulls that
  file's *final* content and commits it — cumulative, so the last
  synthetic commit's tree is byte-identical to the culprit commit's tree.
  Bisects again over this ladder to find the exact culprit *file*.
```

Content-based reconstruction (`git checkout <commit> -- <path>`) rather
than hand-applying that commit's diff hunk-by-hunk: patch application can
fail on context-line mismatches; pulling a file's final blob straight from
the commit that produced it cannot fail that way. Because each synthetic
commit adds exactly one file, level 2's answer is always a single,
specific file — never "somewhere in these three."

**Why bisect at all, instead of just testing each changed file one at a
time?** With *k* changed files, reverting-and-testing each one costs *k*
re-runs. Bisection costs `O(log k)` — for a typical change touching 5–10
files, 3–4 re-runs instead of 10. Same reason `git bisect` exists instead
of a linear walk through history.

**Why real `git bisect run`, not a hand-rolled binary search?** Two
reasons. First, it's literally what was asked for. Second — and this is
the reason that actually mattered once it was built — `git bisect run`
comes with a third outcome, `skip` (exit code 125), for states where the
question genuinely can't be answered (e.g. an intermediate synthetic state
where the code doesn't parse because only half of a two-file change has
landed). Reimplementing that correctly by hand would mean re-deriving a
well-tested piece of git for no benefit.

## The check: pass, fail, or skip

A tiny wrapper — `attribution/bisect_cli.py` — is what `git bisect run`
actually invokes at each candidate commit:

```text
python -m verdict.attribution.bisect_cli <gate> [identity]
```

`git bisect` checks out each candidate directly into the current working
directory (it doesn't copy anywhere), so the wrapper just inspects
`Path.cwd()` — no path argument needed. It re-runs the *real* gate
(`gates/registry.py::resolve_gate`) and asks one question: is the specific
already-known failure (`identity` — a `FailureLocation.identity`, e.g. a
pytest node id, or `None` to mean "the gate as a whole," used for `build`)
present in the freshly parsed result?

- Present → **bad** (exit 1)
- Gate ran cleanly, absent → **good** (exit 0)
- Gate's stack isn't even present at this state (`N/A`), or it failed with
  no structured failure list to check identity against (a raw
  `verdict.yml`-override gate) → **skip** (exit 125) — an unopinionated
  "can't tell," never a guess in either direction.

This is the same discipline as `Verdict.status` refusing to report `DONE`
from zero evidence, applied one level down: when the check can't positively
confirm the *exact* original failure, it says so instead of picking a side.

Reusing `resolve_gate` (rather than reimplementing scoped tool invocations)
means bisection automatically inherits every parser Phase 1 already wrote
and tested — no new command construction, no new output parsing. It also
means `FailureLocation.identity` had to be *stable across bisection
states* — see `gates/typecheck.py`'s docstring on why identity is
`"{file}:{code}"`, deliberately excluding the line number, which drifts as
a diff is partially applied.

## Bug #1, found by testing against real git: `HEAD` lies after `bisect run`

The first version of the bisection primitive read the culprit off
`git rev-parse HEAD` after `git bisect run` finished, on the assumption
that git leaves the tree checked out at the answer. Testing against a real
3-commit fixture (base → unrelated change → the actual bug) proved that
assumption wrong: when bisection can conclude the answer *without* needing
to re-test the commit already known to be bad (a common case with few
candidates), git leaves `HEAD` at the last commit it actually *tested* —
which can be the adjacent **good** commit, not the bad one. The real
answer only exists as text: `git bisect run` prints
`"<sha> is the first bad commit"` on success. The fix
(`attribution/bisect.py`) parses that line instead of trusting the final
working-tree state. Left as a comment at the fix site, since it's exactly
the kind of "worked on paper, wrong against real git" trap worth flagging
for the next person who touches this code.

## Bug #2, found the same way: gates contaminate "the agent's diff"

The original design committed the agent's final state (for level-1's
commit ladder) *after* running the gates, on the theory that "commit
whatever's sitting in the worktree" was a simple, late-as-possible step.
Testing against a real repo showed why that's wrong: running pytest leaves
`__pycache__/*.pyc` behind as a side effect, and `git add -A` doesn't know
the difference between "the agent wrote this" and "a tool we ran
afterward wrote this." The result was bisection confidently naming a
compiled `.pyc` file as the culprit. Fixed by moving the commit
(`runner.py`) to immediately after `adapter.run()` returns and *before*
any gate executes — so `attempt_commit` captures exactly what the agent
changed, and whatever gates leave behind afterward is simply never staged
into anything attribution looks at.

Both bugs share a shape worth naming: the paper design was internally
consistent and still wrong, because it made an assumption about git or
about process side effects that only a real repo could contradict. Neither
would have been caught by reasoning about the algorithm harder — only by
running it.

## Step 0: is this even the agent's fault?

Before any bisection, the exact same check the bisector uses runs once
against the worktree's `base_commit` (pre-agent). If it already fails
there, attribution stops: this is `PRE_EXISTING`, not a `REGRESSION`, and
no amount of bisecting the agent's diff will explain a failure the agent
didn't introduce. Cheap (one gate run, no bisection), and it's the
difference between "the agent broke this" and "this was already broken" —
a distinction a raw pass/fail gate result can't make on its own.

## Enrichment: the dependency graph never gets to vote

Once level 2 names a culprit file, `attribution/depgraph.py` builds a
lightweight static import graph (Python: `ast`-parsed `import`/`from`
statements, resolved against every `.py` file's dotted module path; JS/TS:
regex-scanned `import ... from` / `require(...)`, resolved only for
relative specifiers) and checks for a path from the failure's file to the
culprit file, up to 4 hops.

- Found → the sentence gets a supporting clause: *"(`test_login.py`
  depends on `auth.py`.)"*
- Not found → the `Attribution` is reported exactly the same, just without
  that clause.

This is deliberately not a real resolver — no `sys.path`/tsconfig `paths`
awareness, no dynamic-import handling, no distinguishing a package from
what it re-exports. A full resolver is a project of its own, and it isn't
what's load-bearing here: the graph can *only add* a clause, never remove
or override the bisection result. Missing an edge (a real, expected
outcome — dynamic imports, config-driven wiring, and runtime string
references are all invisible to a static scan) means a thinner sentence,
never a wrong attribution. This is the same PROVEN/JUDGED discipline from
Phase 0, recursed one level: the *link* is proven by execution; the *why*
is enrichment, and enrichment doesn't get a vote on whether the link holds.

## Schema additions (all additive — `status`/`done`/`confidence` unchanged)

```python
class FailureLocation(BaseModel):
    identity: str          # stable across bisection states — see typecheck.py
    file: str | None = None
    line: int | None = None
    code: str | None = None
    message: str = ""

class AttributionKind(str, Enum):
    REGRESSION = "regression"      # bisection isolated exactly one culprit file
    PRE_EXISTING = "pre_existing"  # failed before the agent touched anything
    INCONCLUSIVE = "inconclusive"  # bisection ran but couldn't cleanly converge

class DependencyLink(BaseModel):
    from_file: str
    to_file: str
    depth: int

class Attribution(BaseModel):
    provenance: Provenance = Provenance.PROVEN   # the link is always proven
    kind: AttributionKind
    check_name: str
    failure_id: str
    failure_location: str | None = None
    culprit_file: str | None = None
    method: str                                   # "baseline" / "bisect(commit)" / "bisect(file)" / ...
    dependency_link: DependencyLink | None = None
    explanation: str                              # a rendering of the fields above, not a new claim
```

`Signal.failures: list[FailureLocation]` is the other addition — each gate
parser already built this structure internally (junit XML → `failed_names`,
mypy/tsc/eslint/ruff JSON → per-error lists) and previously threw it away
by flattening straight into the `detail` string. Phase 2 needed that
structure back, so the parsers now return it instead of re-deriving it by
parsing `detail` text a second time — reusing Phase 1's investment rather
than duplicating it.

`Attribution.provenance` defaults to `Provenance.PROVEN` — attribution
doesn't invent a parallel trust system, it reuses Phase 0's enum directly.
There is deliberately no `JUDGED` attribution in Phase 2: every
`Attribution` in this codebase is proven by bisection or it doesn't exist.

## Worked example: `forgot-to-update-fixture`

`scale(x)` multiplies by a constant `MULTIPLIER` defined in `settings.py`;
`fixtures/expected.json` is a precomputed answer for a specific input,
checked by a test. The agent's task: change the multiplier. It edits
`settings.py` — correctly — and never touches the now-stale fixture.

1. **Baseline**: the test passes at `base_commit`. Not pre-existing.
2. **Level 1**: one file changed (`settings.py`) → trivially the culprit
   commit, no bisection needed.
3. **Level 2**: that commit touched exactly one file → trivially the
   culprit file, no bisection needed either.
4. **Dependency graph**: `test_calc.py` imports `scale` from `calc.py`,
   which imports `MULTIPLIER` from `settings.py` — a 2-hop edge found.
5. **Result**: `Attribution(kind=REGRESSION, culprit_file="settings.py", ...)`.

This is the fixture that most directly demonstrates a real limitation,
stated plainly rather than papered over: Verdict attributes to files the
agent *changed*, never to files it *should have changed but didn't*. A
human reviewer would say "you also needed to regenerate the fixture" —
that's a fact about what's missing from the diff, not about what's in it,
and attribution has no mechanism for claims about absence. `settings.py`
is still the correct, honest answer to "which edit caused this failure,"
even though it isn't the complete story a human would tell.

## Fixtures

| Fixture | What it exercises |
|---|---|
| `test_broke_a_test_attributes_to_the_edited_file` | Baseline case: one file changed, direct dependency edge, no bisection needed |
| `test_unrelated_file_is_never_blamed` | Two files changed (one relevant, one a decoy); validates bisection precision, not just "blame the diff" |
| `test_forgot_to_update_fixture_blames_the_file_actually_touched` | The limitation above, made concrete and asserted on |
| `test_import_error_is_attributed_not_crashed_on` | A collection-time `ImportError`, not a normal assertion failure — different junit XML shape (`classname=""`, `name=<module>`); required fixing `_node_id` to recognize this case rather than construct a bogus `.py::name` identity |
| `test_pre_existing_failure_is_not_blamed_on_the_agent` | Step 0 correctly refuses to attribute a failure that predates the attempt |

All five run the real pipeline end to end — real git repos, real `git
bisect`, real gate re-runs — not mocked bisection. Slower than a pure unit
test (a few seconds total for all five), but this is exactly the class of
logic (subtle git semantics, process side effects) that looks correct on
paper and needs to be checked against the real thing, as both bugs above
demonstrate.

## What's explicitly out of scope for Phase 2

- **No LLM phrasing pass.** `explanation` is template-rendered, not
  model-generated — a deliberate decision (discussed and confirmed before
  implementation): it's zero-cost, zero-latency, and has no hallucination
  surface, and the brief's own README example sentence is already fully
  reachable from the structured `Attribution` fields with a format string.
  A phrasing pass remains a pure presentation upgrade addable later
  without touching the algorithm.
- **File granularity, not line/hunk granularity.** Level 2 always narrows
  to "this file," never "this specific line." Going finer would mean
  hunk-level synthetic commits — meaningfully more bisection steps for
  marginal gain, since the diff itself already shows which lines in the
  named file changed.
- **No cross-failure batching.** Each individual failure (each failing
  test, each typecheck error) is attributed independently, even when
  several clearly share the same root cause — so a change that breaks 5
  tests the same way triggers 5 independent bisections rather than
  recognizing they'd converge on the same answer. Capped at 5 attributions
  per gate (`MAX_ATTRIBUTIONS_PER_GATE`) to bound worst-case cost on a
  badly broken repo; not silent — the cap is a documented constant, not a
  quietly dropped tail.
- **Dependency graph covers Python and JS/TS only**, and only static,
  relative-import-resolvable edges — matching the two stacks the example
  repos and gates already target.
- **No handling of renamed files as a single unit** — git's diff surfaces
  a rename without `-M` detection as a delete-and-add pair, which
  attribution would treat as two separate files rather than recognizing
  the rename. Uncommon enough in the fixtures this phase targets not to be
  worth the added detection logic yet.

---

## Phase 3 — Economic Scoring

## Why cost-to-correct, and not raw token cost

A dashboard that says "Agent A used 2.1M tokens this month, Agent B used
900K" tells you almost nothing useful, for a reason that's easy to miss:
it silently assumes every token bought the same thing. It doesn't. A cheap
agent that fails twice and needs a third attempt can easily cost *more*
overall than an expensive agent that gets it right the first time — and a
raw token/dollar total actively hides this, because it has no notion of
"got there" at all. It's a measure of *effort spent*, not *value
produced*.

Two specific ways raw cost lies, worth naming because both are easy to
miss if you only look at a total:

1. **It rewards giving up cheaply.** An agent that fails fast and never
   retries posts a *lower* token bill than one that actually solves the
   task — and a raw-cost leaderboard would rank the one that produced
   nothing as more "efficient."
2. **It hides retries as if they were free.** If only the winning
   attempt's cost is counted, three failed attempts before a fourth
   succeeds look identical, cost-wise, to succeeding on the first try.
   They are not identical — the first is 4x the spend for the same
   outcome — and a metric that can't see the difference can't be used to
   choose between agents.

`pass_rate_per_dollar` fixes both: it's *value produced* (tasks
confirmed `DONE` — the same proven, executed fact Phase 0's schema was
built around, not a self-report) divided by *everything it took to get
there*, dead ends included. One verdict point per task confirmed `DONE`;
`pass_rate_per_dollar = tasks_done / total_$_across_every_attempt`. That
number answers the question that actually matters when picking a config:
*for a dollar spent, how much verified-correct work do I get back?* — not
*how many tokens did this burn*.

## Why cost had to become a multi-attempt concept

Every phase through Phase 2 assumed one task → one attempt → one `Verdict`
— Phase 0's own DESIGN.md scoped multi-attempt orchestration out
explicitly. "Track cost across all attempts, including dead ends" doesn't
fit inside that shape: there's no such thing as "dead ends" without
something making — and remembering — more than one attempt. So Phase 3
introduces the first orchestration layer above a single `run()`:

```python
def run_with_retries(task, repo, adapter, max_attempts=1) -> TaskRun:
    attempts = []
    for _ in range(max(max_attempts, 1)):
        verdict = run(task=task, repo=repo, adapter=adapter)
        attempts.append(verdict)
        if verdict.done:
            break
    return TaskRun(task=task, agent=adapter.name, repo=str(repo), attempts=attempts)
```

Every attempt gets kept, not just the winner — that's the entire feature,
and it's a two-line loop because `run()` was already fully self-contained
(its own isolated worktree, its own gates, its own attribution) from
Phase 0 onward. `max_attempts` defaults to **1**: retries are opt-in via
`--max-attempts`, not automatic. With `ClaudeCodeAdapter`, every retry is
real spend — Verdict re-running an agent three times without being asked
would mean silently tripling someone's bill the first time they hit a
flaky failure. Existing single-attempt behavior and cost is unchanged
unless a caller explicitly asks for more.

## Two schema types, one for each level of aggregation

```python
class TaskRun(BaseModel):        # one task, however many attempts it took
    task: str
    agent: str
    repo: str
    attempts: list[Verdict]

    @property
    def total_cost_usd(self) -> float | None:
        total = 0.0
        for v in self.attempts:
            if v.attempt.cost_usd is None:
                return None       # never sum a partial figure and call it "total"
            total += v.attempt.cost_usd
        return total

class ConfigResult(BaseModel):   # one (agent, config) label, across many TaskRuns
    label: str
    task_runs: list[TaskRun]

    @property
    def pass_rate_per_dollar(self) -> float | None:
        cost = self.total_cost_usd
        if cost is None or cost <= 0:
            return None
        return self.tasks_done / cost
```

Both totals share the same discipline that shows up everywhere else in
this schema: **an unknown number is reported as unknown, never silently
treated as zero.** If one attempt's `cost_usd` is `None` (an adapter that
didn't report it, and no `verdict.yml` pricing to fall back on),
`TaskRun.total_cost_usd` is `None` — not "the sum of what we do know,"
which would look precise while quietly being wrong. The same logic
propagates up to `ConfigResult`, and `pass_rate_per_dollar` refuses to
divide by an unknown or zero cost rather than returning a number that
looks meaningful but isn't (division by zero would be `inf`, which sorts
as "best" on a naive leaderboard — exactly backwards for "we don't
actually know what this cost").

`ConfigResult.label` is a free-form string the caller supplies (e.g.
`"claude-code / sonnet"` vs `"claude-code / opus"`) rather than a
structured `{agent, model}` pair — Verdict doesn't need to understand
*what* varies between two configs to compare their economics, only that
they're being compared.

## Where the dollar figure actually comes from

`AttemptResult.cost_usd` was already populated directly by
`ClaudeCodeAdapter` (the `claude` CLI's own `total_cost_usd`, a real
invoice figure) since Phase 0. Phase 3 adds a fallback, applied centrally
in `runner.py` — the same "adapters shouldn't need to know about this"
principle Phase 2 used to move diff computation out of the adapters:

```python
def _apply_pricing_fallback(attempt, config):
    if attempt.cost_usd is not None or config.token_pricing is None:
        return attempt          # adapter's own figure always wins
    computed = config.token_pricing.cost_usd(attempt.tokens_input, attempt.tokens_output)
    return attempt.model_copy(update={"cost_usd": computed})
```

Only used when the adapter itself reported nothing — an adapter's own
number is a real invoice; `verdict.yml`'s `cost.price_per_1k_tokens` is a
configured estimate, and the more authoritative source always wins.

## What "the leaderboard" is, and isn't, in Phase 3

`economics.py` is the ranking *engine*: `rank()` sorts a list of
`ConfigResult`s by `pass_rate_per_dollar` descending, with entries whose
economics are undefined (unknown or zero cost) sorted after every entry
with a real number — never first (that would reward not knowing your own
cost) and never dropped silently (an entry a leaderboard can't rank is
still worth showing, with an honest `—` rather than a fabricated number).
Ties within that undefined group fall back to raw pass rate, since that's
still real signal even without a cost figure to divide by. `render()`
prints it as a table.

What doesn't exist yet: anything that *populates* a list of `ConfigResult`
from a real multi-task benchmark suite automatically. There's no suite
file format, no `verdict bench` command, no persisted run history across
CLI invocations — a `ConfigResult` today is something a caller assembles
by hand (or a test does, as in `test_economics.py`) from `TaskRun`s they
already have. This was a deliberate scope line, confirmed before writing
any code: building a suite format now would mean designing Phase 5
("benchmark suites") under Phase 3's name, before Phase 5 gets to define
what a suite actually looks like. Phase 3 ships the math and the ranking
rule, fully tested; Phase 5 wires a real suite runner to feed it.

## Tests

`test_economics.py` covers the accounting claims directly, not just
end-to-end:

- Cost sums across *every* attempt, including attempts that failed —
  proven with a `TaskRun` built from 2 dead ends + 1 winner, asserting the
  total includes all three, not just the winner's cost.
- Any single unknown attempt cost makes the `TaskRun` total unknown, not a
  partial sum.
- The `pass_rate_per_dollar` formula itself, plus its two "refuse to
  answer" cases: unknown cost, and zero cost (not treated as `inf`).
- `rank()`'s ordering, including the "unknown cost sorts last, tiebroken
  by pass rate" rule.
- `run_with_retries` end-to-end against a real worktree, using a small
  test-only adapter that fails twice before fixing the bug on its third
  call — asserting the resulting `TaskRun` has exactly 3 attempts, 2 of
  them counted failed, and cost summed across all 3 (not just the
  successful one) — the concrete "dead ends get counted" claim the brief
  asked for, checked against real orchestration rather than asserted in
  the abstract.
- The pricing fallback: an adapter-reported cost is never overridden by
  `verdict.yml` pricing, and pricing only fills in when the adapter
  reported nothing at all.

## What's explicitly out of scope for Phase 3

- No suite file format, no `verdict bench` command, no persisted
  cross-invocation run history — see above. `ConfigResult` is a library
  type today, not yet a CLI-driven report over many tasks read from disk.
- No retry *strategy* beyond "try again from scratch, up to N times." No
  backoff, no changing the prompt on retry, no giving the agent visibility
  into why the previous attempt failed. Each retry is an independent,
  identical attempt at the same task.
- `pass_rate` in `ConfigResult` is unweighted — ten easy tasks and one
  hard task count equally. Difficulty-weighting is a suite-design concern
  that belongs with Phase 5's suite format, not this phase's aggregation
  math.

---

## Phase 4 — Frontend Truth

## The goal, restated

Every phase through Phase 3 verifies a repo through its command line —
tests, typecheck, build, lint. None of that touches what an actual user
sees. A change can pass every one of those gates and still ship a CTA
button that's invisible, or a click that goes nowhere: `frontend/` closes
that gap by driving the repo's own dev server in a real headless browser
(Playwright) and grading what's actually rendered, in the same
proven-vs-judged discipline as everything else. The hard constraint,
directly from the brief: **a run that passes every test but fails a proven
frontend check is NOT DONE** — a frontend regression has to be exactly as
disqualifying as a broken test, never a lesser, advisory concern.

## Five checks, in decreasing order of trust

```text
DOM assertion          PROVEN   the intended change reached the rendered DOM
Interaction drive       PROVEN   the real user action produces the real outcome
Perceptual screenshot   PROVEN   before/after render diff, thresholded not raw-pixel
Glitch scan             PROVEN   frame-burst diffing catches transient flicker/never-settled
Vision-intent judge     JUDGED   a vision model's opinion — advisory, never load-bearing
```

Four of these needed **zero schema migration** — exactly as Phase 0's
design promised: every one is just a `Signal` with a
`frontend:<kind>:<check-name>` name, PROVEN for the first four, JUDGED for
the last. `Verdict.status` already only ever consults PROVEN signals (see
Phase 0's section above), so "a passing vision judgment can't rescue a
failing DOM check" and "a failing vision judgment can't sink an otherwise-
clean run" were both already true the moment these signals started
flowing into the same `signals: list[Signal]` every other gate uses — no
special-casing needed in `Verdict.status` for frontend signals
specifically. The one genuinely additive field is `Signal.artifact_path`
(optional, defaults to `None`) — added for the glitch scan's video
recordings, see below; every existing `Signal` construction across every
earlier phase is unaffected.

## Config: `frontend.checks` in `verdict.yml`

```yaml
frontend:
  start: "npm run dev"
  url: "http://localhost:3000"
  viewports: [1440, 375]
  screenshot_threshold: 0.02        # fraction of pixels, not a raw count
  glitch_scan: true                 # on by default — see the Glitch scan section below
  glitch_capture_seconds: 1.5
  glitch_frame_interval_seconds: 0.15
  glitch_diff_threshold: 0.05
  checks:
    - name: cta-visible-and-navigates
      dom:
        selector: "#cta"
        visible: true
        class_contains: "cta"
      interaction:
        click: "#cta"
        expect_url_contains: "/signup"
      vision_intent: "A prominent CTA button is visible above the fold."
```

`DomSpec`/`InteractionSpec`/`FrontendCheckSpec` (`config.py`) are
deliberately narrow — selector existence/visibility/class/attribute/text
for DOM, one click plus an expected URL substring or a newly-visible
selector for interaction. Nothing here reaches into computed layout,
pixel positions, or accessibility trees beyond what Playwright's own
`is_visible()`/`get_attribute()` expose — the same "only what's
mechanically checkable" discipline Phase 1's gate parsers used.

## Where "before" actually comes from

The visual-diff check needs a real pre-agent render to diff against, not a
guess from source. `frontend/runner.py::_capture_before` gets one by
reusing `worktree.py`'s `scratch_worktree(repo, worktree.base_commit)` —
the same detached-checkout primitive Phase 2's bisection already uses to
explore arbitrary commits — to spin up a second, disposable checkout
pinned at the worktree's recorded `base_commit`, vendor its `node_modules`
the same way the main worktree's are (`copy_vendored_dependencies`, moved
into `worktree.py` this phase so both call sites share it), start the
*same* `frontend.start` command against it, and screenshot every
configured viewport. That server is torn down before the "after" server
(against the agent's actual final worktree) starts — same port, so they
can never run concurrently — and its screenshots feed straight into
`visual_diff.perceptual_diff_ratio` against the "after" set. This is why
the diff is trustworthy: both renders come from the same browser, same
code path, same viewport — the only variable is the four extra digits of
git history between `base_commit` and the agent's final commit.

## Why frontend signals are never bisected

`runner.py` appends frontend signals to `signals` *after* calling
`attribute_failures`, not before. `attribution/engine.py`'s bisector calls
`gates/registry.py::resolve_gate(gate, ...)` at each candidate commit, and
`GATE_RUNNERS` only has four keys (`test`/`typecheck`/`build`/`lint`) —
handing it `"frontend:dom:cta"` would be a `KeyError`, not a graceful
"can't attribute this." Rather than teach the bisector a fifth gate
category it doesn't actually know how to re-run standalone (a frontend
check needs a live dev server up, not just a `subprocess.run`), Phase 4
scopes causal attribution to the four gates it already covers and lets
frontend failures show up as ungrouped `Signal`s — still fully counted by
`Verdict.status`, just without a "which file caused this" sentence yet.
Extending bisection to frontend checks is future work, not a correctness
gap in what's shipped: the failure is real and reported either way.

## Glitch scan: catching what a single before/after screenshot can't

The perceptual visual-diff check compares exactly two moments — before the
agent's change, after it. That's blind to anything that happens *between*
those two moments: a flash of unstyled content on load, a modal that
flickers open and immediately shut, a layout that jumps and jumps back,
an animation that gets stuck instead of resolving. A regression that
self-corrects within a second still shipped a visibly broken instant to a
real user, and a single before/after screenshot pair structurally cannot
see it — you'd need to happen to screenshot the exact glitching
millisecond, which is exactly the kind of thing a fixed two-shot diff
misses by construction.

`frontend/capture.py` + `frontend/glitch.py` close that gap with a third,
independent PROVEN check: a short burst of real screenshots (not a decoded
video — see below), taken close together in time, compared frame-to-frame
with the same `perceptual_diff_ratio` the before/after check already uses.
Two burst windows run per check, each driven by `verdict.yml`'s
`glitch_capture_seconds`/`glitch_frame_interval_seconds`/
`glitch_diff_threshold`:

- **Load burst** (`frontend:glitch_scan:load`) — captured immediately
  after navigation, at the primary viewport, via
  `capture.capture_settle_burst`.
- **Interaction burst** (`frontend:glitch_scan:<check-name>`) — one frame
  captured immediately before the click, then `interaction_check`'s click
  and assertions run, then frames continue for the configured window, via
  `capture.capture_action_burst`. The click happens exactly once — inside
  the burst's own action callback — so the glitch scan and the interaction
  check are two views of the *same* real click, not two separate
  simulated ones.

`glitch.py`'s `scan_for_glitches` looks for two distinct shapes a plain
before/after diff can't distinguish, both computed purely from pairwise
`perceptual_diff_ratio` calls over the burst:

- **Flicker** — frame *i* differs sharply from both its neighbors, while
  those neighbors closely resemble *each other*. That asymmetry (endpoints
  agree, middle frame doesn't) is the signature of something that
  appeared and reverted within roughly one capture interval — a diff
  across the whole burst (frame 0 vs. frame N) would show ~0% change and
  miss it entirely, the same blind spot the before/after check has, just
  at a smaller time scale.
- **Never settled** — the last two captured frames still differ beyond
  threshold, meaning the page was still visibly changing at the end of the
  capture window, when a well-behaved render or interaction is expected to
  have reached a stable final state.

Both are still PROVEN: every frame is a screenshot Playwright actually
took, and every finding is arithmetic over those frames — no model, no
opinion, fully reproducible given the same burst.

**Video: real evidence, not synthesized.** Every page opened during a
glitch scan is recorded with Playwright's own `record_video_dir` — a real
`.webm` of the actual browser session, saved via
`Browser.new_page(record_video_dir=...)` and finalized (per Playwright's
own contract) once the page closes. This is deliberately *not* built by
stitching the burst screenshots into a video ourselves — Playwright is
already recording the real thing at its own native frame rate, so
reconstructing a lower-fidelity video from our sparser diagnostic frames
would be strictly worse evidence for a human reviewer, for no benefit to
the automated detection (which only needs the burst, not the video).
`Signal.artifact_path` carries the recording's path so a failing
`frontend:glitch_scan:*` signal points straight at footage of the actual
glitch — this is the one field Phase 4 added to the schema (see above).

**Retention policy: keep evidence only when there's something to explain.**
Recording every page is real disk cost, and most runs are clean. So
`run_frontend_checks` records into one temp directory for the whole run,
then deletes it in a `finally` block *unless* some PROVEN signal in the
run failed — a clean run leaves nothing behind; a failing one keeps the
whole capture directory (every page opened that run, not just the one
specific page that failed) as surrounding context, which is simpler than
cherry-picking a single file and still points a reviewer at real footage.
A `PASS` glitch-scan signal never carries an `artifact_path` — even if its
own recording happened to still exist as of that Signal being
constructed, the run isn't over yet and the file may still be deleted
before the caller ever looks, so the signal doesn't make a promise it
can't keep.

**Why not decode Playwright's own video for the automated detection
instead of a separate screenshot burst?** Decoding a `.webm` back into
frames means a real codec dependency (`opencv-python`, `ffmpeg`) for
something a loop of `page.screenshot()` calls already provides directly,
at the sample rate glitch detection actually needs (a handful of frames
over ~1-2 seconds, not 30fps). The screenshot burst and the video
recording run concurrently on the same page for the same reason DOM/
interaction checks and gates coexist elsewhere in this codebase: each
does the one job it's suited for — the burst feeds the deterministic
PROVEN check, the video is for a human's eyes — without either one having
to serve both purposes.

## Flakiness handling

A browser-driven check has more ways to flake than a `pytest` run —
network timing, port races, font rendering, transient websocket
connections — and "flaky" here would mean the single worst thing a
verdict tool can do: report NOT_DONE (or DONE) by accident. Four concrete
defenses, each scoped to a specific flake source:

1. **Readiness polling, not a fixed sleep.** `frontend/server.py::dev_server`
   polls the configured `url` every 250ms (capped by
   `frontend.ready_timeout_seconds`) rather than sleeping a guessed
   duration before assuming the server is up — a `sleep(2)` either wastes
   time on a fast server or races a slow one. Polling also distinguishes
   two failure shapes that need different messages: the process **exited**
   before answering (a real crash — surfaced with the process's own log
   tail) versus it **never answered in time** (a hang or a wrong URL).
2. **Process-group teardown.** `dev_server` starts the command with
   `start_new_session=True` and kills the whole process group (`SIGTERM`,
   then `SIGKILL` after a grace period) on the way out. A dev server
   launched via `npm run dev` typically forks a real child (node); killing
   only the shell would leave that child bound to the port, silently
   breaking the *next* run's readiness poll with a stale listener. Every
   `dev_server` context guarantees the port is free when it exits, success
   or exception.
3. **`networkidle` is a best-effort grace period, not a requirement.**
   `_goto` navigates with `wait_until="load"`, then tries
   `wait_for_load_state("networkidle", timeout=2000)` and swallows a
   timeout rather than propagating it. A strict `networkidle` wait would
   hang indefinitely against any dev server that keeps a persistent
   connection open (hot-module-reload websockets are the common case) —
   exactly the kind of real-world dev server behavior that would make
   every check against a Vite/webpack-dev-server app flake or hang. Two
   seconds of "let things settle if they're going to" is enough for a
   screenshot to reflect a rendered page without staking correctness on a
   condition modern dev tooling routinely never satisfies.
4. **Perceptual, thresholded visual diff — not raw pixels.** This is the
   biggest flake source a naive implementation would have shipped:
   comparing screenshots byte-for-byte (or even pixel-for-pixel) would
   flake on font hinting and anti-aliasing jitter alone, on a page nobody
   touched. `visual_diff.py` downscales both renders to a fixed small size
   *and* ignores any single pixel's grayscale delta under
   `PIXEL_TOLERANCE` (30/255) before computing the changed-pixel ratio —
   see that module's docstring for the full reasoning. The ratio, not a
   raw count, is what's compared against `frontend.screenshot_threshold`,
   so the same config value is meaningful at 1440px and at 375px. Verified
   directly in `test_visual_diff.py`: identical images diff to exactly
   `0.0`, and a 5/255 delta (representative render noise) also diffs to
   `0.0`, while a real content change (black vs. white) diffs to `~1.0`.
5. **Bounded interaction timeouts that fail loudly.** `checks.py` gives
   clicks and post-click assertions a fixed 5s budget
   (`INTERACTION_TIMEOUT_MS`) — long enough for a real click/navigation,
   short enough that a genuinely broken interaction reports FAIL in
   seconds rather than hanging the whole run.
6. **Glitch-scan timing is honestly best-effort, and its threshold is a
   separate config value from the visual diff's.** `capture.py`'s own
   docstring states plainly that `interval_seconds` is a floor, not a
   promise — each `page.screenshot()` call takes real, variable time on
   top of the requested gap. That's acceptable for what glitch detection
   needs (catching a state change within *roughly* one interval), but it
   would not be acceptable if the same number were being used as a
   precise timing budget elsewhere, which is why `glitch_diff_threshold`
   is its own `verdict.yml` key rather than reusing
   `screenshot_threshold` — frame-to-frame noise over a short interval and
   whole-page noise over an entire edit are different enough
   measurements that one config value tuned for one would either be too
   strict or too loose for the other.

**What Phase 4 deliberately does not do about flakiness.** No automatic
retry-on-failure for frontend checks specifically, and no multi-run
variance measurement — both are the README roadmap's own next item
("Flakiness detection: multi-seed variance, confidence intervals on
pass-rate"), scoped out here for the same reason Phase 3 scoped out a
retry *strategy* beyond "try again from scratch": designing a general
flakiness-confidence model is a project of its own, and folding it into
Phase 4 under a different name would mean building Phase 5-or-later
without it getting to define its own shape. Every defense above targets a
*specific, named* flake source with a concrete mechanism — there's no
blanket "just retry until it passes," which would quietly convert a real
intermittent bug into a passing verdict.

## `VisionJudge`: pluggable, and honestly a stub today

`frontend/vision_judge.py` defines `VisionJudge` as a `Protocol` (`judge(
screenshot_png, intent) -> VisionJudgment`) — the same shape as `Adapter`
in `adapters/__init__.py`: one method, pluggable implementations, so
`runner.py` never needs to know which concrete judge it's holding. Only
`MockVisionJudge` ships, and it says so in its own rationale text rather
than pretending to have inspected anything: it passes unconditionally,
with a `detail` string that states plainly it has no way to actually see
the image. Wiring up a real vision-model API is a real integration project
(pick a vendor, handle auth, validate against real screenshots) — building
an untested stub that pretends to call a real model would be strictly
worse than an honest, documented Mock, the same call Phase 0 made for
`MockAdapter` on the agent side.

## Demo repo

`examples/sample_frontend_repo/` — a zero-npm-dependency static site (a
~40-line `http.createServer` in `server.js`, so `--repo` works with no
`npm install` step at all) with one seeded bug: `public/index.html`'s CTA
link carries a leftover `hidden` CSS class from an earlier draft, so
`#cta` never renders even though clicking it is supposed to navigate to
`/signup.html`. `verdict.yml` configures one `FrontendCheckSpec` combining
all three PROVEN checks plus a `vision_intent` string. `--agent mock`
works out of the box (`cli.py`'s `_MOCK_PATCHES` ships the fix); run
against the unfixed repo, `frontend:dom:cta-visible-and-navigates` and
`frontend:interaction:cta-visible-and-navigates` both FAIL — while
`frontend:visual_diff` still PASSes (a hidden element contributes ~0% to
a perceptual diff) and the JUDGED vision signal still PASSes (`MockVisionJudge`
always does) — and `Verdict.status` is still correctly `NOT_DONE`,
demonstrating the brief's exact scenario: proven checks fail, judged
opinion doesn't matter, done.

## What's explicitly out of scope for Phase 4

- **No causal attribution for frontend failures** — see above; scoped to
  the four gates `attribution/engine.py` already covers.
- **No flakiness/variance detection** — see above; that's the README
  roadmap's own next item, not folded in here under a different name.
- **DOM/interaction checks run once, at the first configured viewport**,
  not once per viewport — only the perceptual screenshot diff is per-
  viewport. Responsive *rendering* is checked at every configured width;
  responsive *interaction behavior* (e.g. a mobile hamburger menu behaving
  differently from a desktop nav) is not, and would need its own
  per-viewport check plumbing.
- **No real vision-model integration** — see `VisionJudge` above.
- **No parallel viewport/browser execution** — screenshots and checks run
  sequentially against one Chromium instance. Real, but bounded, added
  latency per run; parallelizing across viewports is a performance
  optimization that doesn't change what gets verified.
- **A single dev-server command, not a build-then-serve pipeline** —
  `frontend.start` is trusted to bring up something Playwright can hit;
  Verdict doesn't orchestrate a separate build step first. A repo whose
  dev server requires a prior build should say so in its own `start`
  command (e.g. `"npm run build && npm run preview"`).
- **The glitch scan cannot distinguish an intentional animation from a
  real glitch.** A page with a genuine loading spinner, a deliberate CSS
  transition, or a lazy-loaded image is *also* "still visibly changing" by
  the same measurement an actual bug would trigger — `scan_for_glitches`
  has no concept of design intent, only pixel change over time. This is a
  real, known false-positive source, not an oversight: a repo with
  legitimate ongoing motion on load should either raise
  `glitch_diff_threshold`, extend `glitch_capture_seconds` past the
  animation's own duration, or set `glitch_scan: false` for that check.
  Teaching the scan to recognize "intentional" motion would mean either a
  model's opinion (moving it out of the PROVEN bucket entirely) or a
  taxonomy of animation patterns to special-case — neither is in scope
  here.
- **No video decoding for the automated detection** — see the glitch-scan
  section above; the burst and the recording are two independent captures
  of the same session, deliberately not derived from each other.
- **One capture window per check, not adaptive retries.** If a burst
  happens to under-sample a genuine flicker (bad luck on interval timing
  against a very brief glitch), Phase 4 doesn't automatically lengthen the
  window and try again — consistent with "no automatic retry-on-flake"
  above. A repo with known brief glitches should tune
  `glitch_frame_interval_seconds` down rather than rely on Verdict to
  retry until it happens to catch one.

---

## Phase 5 — Benchmark Suites & More Adapters

## The goal, restated

Every phase through Phase 4 answers "is this one agent's one attempt at
one task actually correct" — thoroughly, but one task at a time. Phase 5
doesn't add a new way of *grading* anything; it adds a way of *repeating*
grading across many tasks and many agents, then aggregating the result
into the two things the brief asked for: a ranked pass-rate-per-dollar
leaderboard, and a failure-mode breakdown. The load-bearing design
decision, made before any code: **a suite task is a `verdict run` with its
repo and task text pre-wired, nothing more.** No new acceptance-criteria
format, no new trust bucket, no new way for a check to pass or fail — see
below for why that constraint was non-negotiable.

## The task/acceptance-criteria format

```text
my_suite/
  bug-fix-calculator/
    task.yml          # { task: "...", repo: "repo", category: "bug-fix" }
    repo/              # a real git repo — its own tests, its own verdict.yml
  refactor-tax-calc/
    task.yml
    repo/
```

`task.yml`'s only required key is `task` — the exact natural-language
instruction `--task` would take for a single `verdict run`. `repo`
defaults to `"repo"` (relative to the task directory); `category` is
optional free-form metadata (`"bug-fix"`/`"refactor"`/`"feature-add"` in
the starter suite) used only by the failure-mode breakdown's reporting,
never by scoring.

**The acceptance criteria are never written down as prose in `task.yml` at
all.** They're whatever `repo/`'s own gates (test/typecheck/build/lint)
and `verdict.yml` (frontend checks, gate overrides, cost pricing) already
define — checked the exact same executable way a lone `verdict run` checks
them. This was the one design question worth pausing on before writing
`suite/loader.py`: a rubric-style `acceptance_criteria: ["the button
should be green", "the API should validate input"]` field would have been
easy to add and immediately would have reintroduced exactly the problem
Phase 0 exists to solve — a prose rubric that has to be *interpreted*
(by a human, or worse, by another LLM) instead of *executed*. Making a
suite task nothing more than "a repo + a task string" means every
acceptance criterion a suite task has is, by construction, something
`resolve_gate`/`run_frontend_checks` already knows how to check
mechanically. `suite/loader.py` needed **zero new schema** for this
reason: `SuiteTask` is a thin dataclass (`name`, `task`, `repo`,
`category`), and `run_suite` (`suite/runner.py`) is a loop over
`run_with_retries` — the exact function Phase 3 already built for a single
task's retries — called once per `(config, task)` pair.

## The suite runner: no new scoring logic

```python
@dataclass
class BenchConfig:
    label: str
    adapter: Adapter

def run_suite(tasks, configs, max_attempts=DEFAULT_MAX_ATTEMPTS) -> list[ConfigResult]:
    return [
        ConfigResult(
            label=config.label,
            task_runs=[
                run_with_retries(task=t.task, repo=t.repo, adapter=config.adapter, max_attempts=max_attempts)
                for t in tasks
            ],
        )
        for config in configs
    ]
```

(Written flat above for clarity; the real implementation is the same two
nested loops.) Every `(config, task)` pair gets its own fully isolated
`run_with_retries` call — its own worktree, its own gates, its own
attribution, its own cost accounting — with zero interaction between
configs or between tasks. `ConfigResult`'s `pass_rate`/`total_cost_usd`/
`pass_rate_per_dollar` properties and `economics.rank`/`render` (Phase 3)
needed **zero changes**: a suite run just produces the same
`list[ConfigResult]` a caller could already have hand-assembled from
individually-run `TaskRun`s, at a larger scale. `BenchConfig.label` is
free-form for the same reason `ConfigResult.label` already was — "claude-
code / sonnet" vs. "claude-code / opus" can both point at the same
adapter class configured differently however that adapter exposes it;
Verdict doesn't need to understand what varies between two configs to
rank them.

## Failure-mode breakdown: a tally, not a new classifier

`failure_modes.py`'s `summarize_failure_modes` answers "what kind of thing
does this config keep failing at, across a whole suite" by counting each
`Signal.name` that's `PROVEN` and `FAIL` across every task run that didn't
end `DONE` — nothing more sophisticated than a `Counter`. Two things this
deliberately is **not**:

- **Not a root-cause classifier.** Phase 2's attribution already answers
  "why did this specific failure happen" per task, bisected to a culprit
  file. This answers a different, coarser question at a different
  altitude — which named checks a config fails across *many* tasks — and
  doesn't try to explain any individual failure.
- **Not influenced by JUDGED signals**, for the same reason `Verdict.status`
  isn't: a vision-intent opinion is advisory, so it never counts as a
  "failure mode" a config can be said to actually have — only executed,
  proven failures do. Verified directly in `test_failure_modes.py`.

Rendered as a plain table (`render_failure_modes`) alongside the
leaderboard — e.g. "which agents break responsive layout, which never
update fixtures" (the README's own framing) reduces, honestly, to "which
named `Signal`s fail most often for this config," which is exactly what
gets displayed.

## `verdict bench`

```bash
verdict bench --suite examples/starter_suite --agent mock --agent claude-code
```

Loads the suite, builds one `BenchConfig` per `--agent` (repeatable),
calls `run_suite`, then prints `economics.render` followed by
`render_failure_modes`. One deliberate behavioral difference from
`verdict run`: **`bench` never exits non-zero for a bad score.** `run` is
built to be a CI merge gate — one task, one pass/fail, a nonzero exit
blocks a PR. A suite is a scorecard for *comparing* configs, where "config
X only got 40%" is data to report, not a build to fail; there's no single
correct threshold Verdict could pick on the caller's behalf, so `bench`
just reports and exits 0.

## The starter suite

`examples/starter_suite/` ships three tasks, one per category the brief
named:

| Task | Category | Acceptance criterion |
|---|---|---|
| `bug-fix` | bug-fix | `add()` subtracts instead of adding; existing tests must pass |
| `refactor` | refactor | duplicated validation logic across two tax functions; existing tests must still pass |
| `feature-add` | feature-add | a test already asserts `multiply()` exists; it must be added and pass |

All three are minimal, real, standalone git repos (`setup.sh` bootstraps
all three at once, same convention as every other `examples/*/setup.sh`),
graded by the exact same `test` gate Phase 1 built — no suite-specific
verification machinery exists. `--agent mock` works out of the box via
`SuiteMockAdapter` (see below), so the whole suite is demoable with zero
API keys, same as every earlier example.

**A real limitation the refactor task surfaced immediately, worth stating
plainly rather than glossing over:** a mock adapter given an unrelated,
no-op patch for the refactor task still scores it `DONE`, because "the
existing tests still pass" is genuinely satisfiable by *not touching the
file at all*. This isn't a bug in the suite or the runner — it's an
honest structural fact about executable-acceptance grading for
refactor-shaped tasks specifically: "did not break behavior" and "actually
did the refactor" are different claims, and only the first one is
something a test suite can attest to. A human reviewer would immediately
notice the duplication is still there; Verdict's executable checks
structurally cannot, the same class of limitation Phase 2's
`forgot-to-update-fixture` example already named for attribution ("Verdict
attributes to files the agent changed, never to files it should have
changed but didn't"). Verified directly, not asserted: `test_bench.py`
runs a real no-op `SuiteMockAdapter` against all three starter-suite tasks
and confirms exactly this — the bug-fix and feature-add tasks correctly
fail, and the refactor task's true behavior depends on what "fixing" it
even means executably.

## `SuiteMockAdapter`: one canned patch per task, not one per repo

`cli.py`'s existing `_MOCK_PATCHES` (Phase 0) is keyed by repo *name*,
which works for one repo per demo but breaks down the moment `--agent
mock` needs a *different* patch per task in the same suite run — a single
`MockAdapter` instance is constructed with one fixed patch and can't vary
it per call. `Adapter.run(task, worktree)` only ever receives the task's
*text*, never an id, so `SuiteMockAdapter` (`adapters/mock.py`) looks its
patch up by that text instead — the only per-task identity an `Adapter`
call actually carries. This is a new adapter implementation, not a change
to the `Adapter` Protocol itself: `SuiteMockAdapter.run` still has the
exact same signature, and internally just constructs and delegates to a
plain `MockAdapter` once it's found the right patch.

## Four adapters added, one at a time — the interface never changed

Cursor, Codex, Aider, and OpenHands were implemented in that order, each
as its own standalone module mirroring `ClaudeCodeAdapter`'s shape
(subprocess call to the tool's own CLI, defensive parsing of whatever
usage/cost information it hands back, a dedicated `*AdapterError` for
"the CLI itself couldn't run"). After each one, the same check ran: does
`Adapter` (`name: str` + `run(task, worktree) -> AttemptResult`) still fit
with zero modification? It did, all four times — **`adapters/__init__.py`
was not touched during this phase.** Concretely, per adapter:

- **`CursorAdapter`** (`adapters/cursor.py`) — closest to `ClaudeCodeAdapter`,
  since Cursor's CLI (`cursor-agent`) was deliberately modeled on Claude
  Code's: `-p`/`--print`, `--output-format json`, a force flag to
  auto-accept edits headlessly. Same JSON-payload shape, same defensive
  `.get()`-based field extraction.
- **`CodexAdapter`** (`adapters/codex.py`) — the first real deviation in
  *output* shape, not interface: `codex exec --json` streams
  newline-delimited JSON events rather than one final object, so this
  adapter parses line-by-line and keeps the last event carrying a usable
  `usage` dict. It also doesn't report its own dollar cost the way Claude
  Code's CLI does — `cost_usd` is left `None` rather than guessed, and a
  user who wants a populated `$` column for this adapter configures
  `verdict.yml`'s `cost.price_per_1k_tokens`, which Phase 3's pricing
  fallback (`runner.py::_apply_pricing_fallback`) already exists for. A
  genuinely different CLI output format changed *how this one adapter
  parses stdout*; it didn't touch what `Adapter.run` returns or its
  signature.
- **`AiderAdapter`** (`adapters/aider.py`) — the second deviation: Aider
  has no structured output mode at all, only a human-readable summary line
  (`Tokens: 2.3k sent, 456 received. Cost: $0.01 message, $0.03 session.`).
  Handled with a regex over that line — the exact same move
  `gates/typecheck.py`'s `tsc` runner already made for a tool with no
  native JSON reporter (see Phase 1's table). Still the same `Adapter`
  shape; only the parsing strategy had to fit the tool.
- **`OpenHandsAdapter`** (`adapters/openhands.py`) — the most conservative
  of the four, and deliberately so. OpenHands is typically driven by
  workspace/LLM-provider configuration rather than one self-contained
  invocation, and has no documented structured cost output at all.
  Rather than fabricate a number or half-guess at a config surface Verdict
  doesn't manage, this adapter reports `tokens_input=tokens_output=0`,
  `cost_usd=None`, always — an honest "unknown," not a wrong "zero" dressed
  up as a real figure. `raw_output` still carries the CLI's full stdout so
  a human has something to read even where accounting doesn't exist yet.

**Why the interface held for all four, not by luck:** `Adapter` was
scoped from Phase 0 to the smallest possible contract — take a task and a
worktree, report a diff and a cost, don't judge correctness. Every
CLI-driven coding agent, regardless of its own output format, fits that
shape: it's given a prompt, it edits files, and it may or may not hand
back token/cost accounting on the way out. The differences between these
four tools all turned out to live *inside* `run()`'s implementation (how
to invoke the CLI, how to parse what it prints), never in what `run()`
takes or returns. Each adapter's tests (`test_cursor_adapter.py`,
`test_codex_adapter.py`, `test_aider_adapter.py`,
`test_openhands_adapter.py`) fake `subprocess.run` rather than requiring
the real external binary (none of the four are installed in this
environment) — they verify Verdict's side of the contract: command
construction, defensive parsing, and that a missing binary or a timeout
raises a clear, dedicated error instead of crashing or silently returning
zeros.

## What's explicitly out of scope for Phase 5

- **No suite-level cross-task cost/pass-rate weighting** — `ConfigResult.
  pass_rate` (Phase 3) is still unweighted; a suite of ten easy tasks and
  one hard one still counts them equally. Difficulty-weighting remains a
  suite-*design* concern, same scope line Phase 3 already drew.
- **No persisted run history across `bench` invocations** — every
  `verdict bench` call is a fresh run; nothing is written to disk for a
  later invocation to compare against. A historical trend view is a
  reporting feature layered on top of `ConfigResult`, not something this
  phase's runner needed to build to satisfy "run every config across all
  tasks."
- **No parallel task/config execution** — `run_suite` is two sequential
  loops. Every `(config, task)` pair is fully independent, so this is a
  real, addressable performance gap, not a correctness one; parallelizing
  it doesn't change any task's grade.
- **No live progress reporting mid-suite** — `verdict bench` prints once,
  at the end, after every config has run every task. A long suite gives no
  feedback until it's entirely done.
- **No model-level adapter configuration surface** — `BenchConfig.label`
  lets a caller *label* "claude-code / sonnet" vs. "claude-code / opus,"
  but Verdict doesn't provide a uniform way to actually *select* a model
  for a given adapter; that's left to however each adapter's own
  constructor or environment exposes it (an API-key env var, a CLI flag
  baked into the adapter's own `command` list, etc.).
- **Adapter CLI flags/output shapes are believed-correct as of writing,
  not verified against the real binaries** — none of Cursor/Codex/Aider/
  OpenHands are installed in this environment (only Claude Code's own
  adapter, from Phase 0, has ever been exercised against a real CLI).
  Every new adapter's defensive parsing is designed so a wrong guess about
  exact flags or JSON field names degrades gracefully (0 tokens, `None`
  cost, a clear `*AdapterError` on outright failure) rather than crashing
  — but "degrades gracefully if wrong" is not the same claim as "verified
  correct." A real integration pass against each installed CLI is future
  work, not something a sandboxed test run can honestly claim to have
  done.

---

## Phase 6 — Merge Gate & Reports

## The goal, restated

Every phase through Phase 5 answers "is this agent's attempt correct" for
someone running Verdict by hand or reading a scorecard. Phase 6 puts
Verdict where the brief's own README already said it belongs: "drops into
CI as a merge gate that blocks an agent's PR when the agent's own work
doesn't pass." That's two separable things, built as two separable
pieces — a way to grade a PR that already exists (no agent to drive), and
a way to hand the result to a human or a CI system in a form each can
actually use (a terminal, a script, a browser). Neither piece needed a new
trust model: the gate's pass/fail is still exactly `Verdict.status`, and
every reporter renders the exact same PROVEN/JUDGED-tagged data every
earlier phase already produces.

## Grading a PR that already exists: no adapter, no isolation

Every phase through Phase 5 assumed the shape "hand an agent a task, let
it produce a diff, grade the diff" — `run()`/`run_with_retries()` isolate
a fresh worktree specifically because there's an *agent* about to mutate
it. A pull request breaks that assumption in a way that's actually
simplifying, not complicating: **the diff already exists, as real commits,
before Verdict ever runs.** There's no task to hand anyone and nothing to
isolate — the CI runner's own checkout already *is* the thing to grade.
`runner.py::grade_existing_diff(repo, base_ref)` is the new entry point
built for exactly this:

```python
def grade_existing_diff(repo: Path, base_ref: str) -> Verdict:
    base_commit = rev_parse(repo, base_ref)
    final_commit = rev_parse(repo, "HEAD")
    diff, files_changed = diff_between(repo, base_commit, final_commit)
    attempt = AttemptResult(diff=diff, files_changed=files_changed, cost_usd=None)

    worktree = Worktree(path=repo, branch="HEAD", base_commit=base_commit)
    config = load_config(repo)
    signals = run_all_gates(repo, config)
    attributions = attribute_failures(repo, worktree, final_commit, signals)
    signals = signals + run_frontend_checks(repo, worktree, config, task="pull request diff")

    return Verdict(task="grade existing diff", agent="gate", repo=str(repo),
                    attempt=attempt, signals=signals, attributions=attributions)
```

Three things worth calling out, none of which needed new machinery:

- **`Worktree` is reused as a plain value, not constructed via
  `isolated_worktree()`.** `Worktree` was already just a dataclass
  (`path`, `branch`, `base_commit`) — nothing stops building one that
  points straight at the real checkout instead of a throwaway copy.
  `attribute_failures` only ever *reads* `worktree.path` (for the
  dependency-graph scan) and `worktree.base_commit` (bisection's starting
  point); it never writes through this reference itself.
- **Bisection is safe against the real checkout because it always was.**
  Re-reading `attribution/bisect.py`: `run_bisect` runs `git bisect`
  entirely inside its own `scratch_worktree(repo, bad_sha)` — a disposable
  detached worktree — and `repo` is only ever used as the *source* for
  `git worktree add --detach`, which doesn't touch `repo`'s own HEAD or
  index. This was already true in every earlier phase (`repo` passed to
  `attribute_failures` in `runner.run()` is the original source repo, not
  the throwaway worktree); Phase 6 just relies on a property Phase 2
  already had to get right.
- **`diff_between` (new, in `worktree.py`) is read-only where
  `diff_against_base` isn't.** `diff_against_base` calls `git add -A`
  before diffing, because Phase 0 needed to capture an agent's
  *uncommitted* edits in a worktree Verdict owns. Grading an existing PR
  has nothing uncommitted to catch — `HEAD` already is the final state —
  so `diff_between` runs a plain `git diff base final` and never stages
  anything. Staging in a CI runner's own checkout would be a real,
  unwanted side effect for a mode whose entire point is "grade the repo
  as found." Verified directly in `test_worktree.py`: `git status
  --porcelain` is asserted identical before and after the call.

`AttemptResult.cost_usd` is `None`, not `0` — this diff wasn't produced by
an adapter Verdict drove, so "what it cost" is a question this mode simply
doesn't answer, and `0` would look like a real, known figure where
`Verdict`'s whole schema discipline (Phase 3) says an unknown should never
impersonate a known zero.

## The gate policy

`verdict gate --repo . --base <ref>` exits non-zero exactly when
`Verdict.status` is not `DONE` — i.e. exactly the same rule `verdict run`
already used for its exit code, applied to a `Verdict` this mode computed
differently. Restated because it's the one policy the whole phase exists
to enforce: **any failing PROVEN signal fails the check.** Test, typecheck,
build, lint, DOM assertion, interaction drive, perceptual screenshot diff,
glitch scan — every one of them is load-bearing, none of them softer than
another. `UNVERIFIED` (zero PROVEN gates ran at all) also fails the check,
for the same reason it's not-done everywhere else in this schema: a check
that verified nothing has no basis to wave a PR through.

**JUDGED signals never fail the check, and never appear in its exit code
at all.** They're surfaced a structurally different way — an advisory PR
comment (`verdict pr-comment`, driven by `action.yml`) — specifically so
there's no code path where a vision-model opinion could accidentally
become load-bearing by sharing a decision point with PROVEN signals. This
is the same policy `Verdict.status` has enforced since Phase 0, made
visible at the CI-integration layer: the *check* (pass/fail, blocks
merging) and the *comment* (opinion, informational) are two separate
GitHub API objects, populated by two separate action steps, so there is no
single flag or threshold that could quietly let a glowing JUDGED score
compensate for a failing PROVEN one, or vice versa.

## Reporters: one shape, three renderers

`report_json.py` and `report_html.py` both consume exactly
`list[ConfigResult]` — the same shape `bench`'s leaderboard already
produces (Phase 5), and the same shape `run`/`gate` produce once wrapped
as a single-config, single-task list. One shape, three renderers (`cli`
via `report.py`/`economics.py`, `json`, `html`) means the three can never
disagree about what happened — they're views over the same object graph,
not three independent tallies that could drift.

**Turning `Verdict.status`/`done`/`confidence` (and the `TaskRun`/
`ConfigResult` aggregate properties) into `@computed_field @property`
instead of plain `@property` was the one real schema change this phase
needed**, caught by actually trying to serialize a report rather than by
inspecting the schema on paper: Pydantic's `model_dump`/`model_dump_json`
silently omit plain `@property` values, so the very first `verdict-report.json`
this phase produced was missing `status`/`done`/`pass_rate_per_dollar`
entirely — every number a consumer (a CI script, `pr-comment`, a future
dashboard) would actually want. Marking them `@computed_field` is purely
additive (every existing field, every existing in-Python access pattern
via `verdict.status` is unchanged) and closes the gap once, at the schema
level, rather than having `report_json.py` hand-reconstruct
`Verdict.status`'s PROVEN-only logic itself — which would have been
exactly the kind of duplicated trust-decision logic this whole codebase
has avoided since Phase 0.

**Why JSON is nearly a one-liner.** `Verdict`/`TaskRun`/`ConfigResult` have
been Pydantic models since Phase 0 — `report_json.py` doesn't reshape
anything, it wraps `[cr.model_dump(mode="json") for cr in config_results]`
in a small `{"schema_version": 1, "configs": [...]}` envelope so every
consumer has one stable top-level shape to parse regardless of which
command produced the report.

**The HTML dashboard is one self-contained file — inline CSS, inline JS,
no external request of any kind** (no CDN script, no Google Font, no
separate stylesheet). Three sections: a leaderboard (reusing
`economics.rank`'s ordering, so the HTML and CLI leaderboards can never
disagree about who's ranked where), a failure-mode breakdown (reusing
`failure_modes.summarize_failure_modes` for the same reason), and one
collapsible card per task run showing its PROVEN/JUDGED signals (visually
distinguished — JUDGED gets a dashed border and an "opinion" badge, and
never changes a card's pass/fail color, which comes from `Verdict.status`
alone), its causal analysis, and its cost. The one inline script is a
single "show only failing tasks" checkbox — genuinely useful for a CI
artifact with many tasks, and the only interactivity that felt worth the
one script tag rather than none. **All interpolated text is HTML-escaped**
(`html.escape`, not manual string replacement) — a task description or a
signal's `detail` can contain anything an agent wrote, including literal
`<script>` tags in a prompt or a stack trace, and none of that is trusted
markup. Verified directly in `test_reporters.py`: a task string containing
`<script>alert(1)</script>` renders as the escaped `&lt;script&gt;...`,
never as live markup.

## The GitHub Action

`action.yml` is a composite action (`verdict-ai/action` is what it's
published as) with one job: install Verdict, run the gate, upload the
full report as a CI artifact, post the advisory comment, then fail the
job if the gate failed. The awkward-looking bracket around one step —

```bash
set +e
verdict gate ...
code=$?
set -e
echo "exit_code=$code" >> "$GITHUB_OUTPUT"
```

— exists because composite-action `run:` steps execute under `bash -e`;
without disabling it around exactly this one command, the script would
abort the instant `verdict gate` exits non-zero and never reach the line
that records the exit code for the later "fail the check" step to use.
`continue-on-error: true` on the same step keeps *that* failure from
ending the job immediately — the artifact upload and PR-comment steps
below it (`if: always()`) still need to run even when the gate failed, and
a final step explicitly re-raises the captured exit code once everything
else has run. This is the standard idiom for "run cleanup/reporting steps
after a failing step, but still fail the job overall" — not a workaround,
a normal composite-action pattern once GitHub Actions' step semantics are
taken into account.

**`verdict pr-comment <report.json>` is its own CLI command, not a
standalone script bundled with the action**, so the comment-body logic
(`pr_comment.py::build_comment`) is versioned, tested, and installed
alongside the rest of Verdict rather than living as an untested shell
script the action happens to carry. The action's own comment-posting step
is deliberately thin — `gh pr comment --edit-last --body-file ... || gh pr
comment --body-file ...` — because `gh` (preinstalled on every
GitHub-hosted runner) already does comment upsert correctly; hand-rolling
a GitHub REST API call here would mean a network client this package
doesn't otherwise need, for something a two-line shell idiom already
does better. `--edit-last` means a repeatedly-pushed PR accumulates one
updated comment, not a growing pile of stale ones.

**`install-frontend-extra` is opt-in, not automatic.** Installing
Playwright's Chromium browser costs real setup time on every single CI
run; most repos don't configure `frontend:` checks in `verdict.yml` at
all, so paying that cost by default would slow down every gate run to
benefit only the ones that need it. A repo that does configure frontend
checks sets `install-frontend-extra: true` once in its workflow file.

## End-to-end verification

`tests/test_gate.py` is the closest thing to "run the action" available
without real GitHub infrastructure to drive (no live PR, no `gh` CLI
authenticated against a real repo, in this environment). Two levels:

- **Unit-level**, against synthetic repos built the same way
  `test_worktree.py`/`test_runner.py` already do: a PR that fixes the
  seeded bug grades `DONE`; a PR that changes something unrelated grades
  `NOT_DONE` and attributes the failure `PRE_EXISTING`; a PR whose *first*
  of two real commits fixes the bug and whose *second* reintroduces it
  exercises real bisection (not just "blame the only changed commit") and
  attributes `REGRESSION` to the right file.
- **End-to-end against the real `examples/sample_repo`** — not a synthetic
  fixture, the actual checked-in example: bootstrap it via its own
  `setup.sh` (idempotent, same as a CI workflow's own setup step would
  do), copy it to a scratch directory so the checked-in example is never
  mutated, commit a real fix as a second commit standing in for a PR's own
  commits, then drive the *exact* `verdict gate` CLI invocation
  `action.yml`'s `run:` step uses (via `typer.testing.CliRunner`, not a
  hand-called Python function) — asserting the process exit code, the
  written `verdict-report.json`'s `done`/`status` fields, and that the
  HTML report file exists. A second test runs the same real example with
  an irrelevant change instead of a fix and asserts exit code `1`. This is
  the full path from "a real example repo" to "CLI exit code a CI job
  would act on," the same standard Phase 2's attribution tests held
  themselves to for real `git bisect` rather than mocking it.

## What's explicitly out of scope for Phase 6

- **No real GitHub Actions runner exercised the action itself** — no live
  PR, no real `gh pr comment` call, no real `GITHUB_TOKEN`. `action.yml`'s
  YAML structure and the exact shell idioms it depends on
  (`continue-on-error` + captured exit code, `gh pr comment --edit-last`)
  are standard, well-documented patterns, but "standard pattern, carefully
  reasoned through" is not the same claim as "verified against a real
  workflow run" — matching the same honesty Phase 5 already applied to its
  four adapters' CLI invocations.
- **No comment deduplication beyond `gh pr comment --edit-last`** — this
  matches by *author*, not by the `MARKER` comment embeds. A repo where
  some other tool or bot also comments as the same identity could see
  `--edit-last` overwrite the wrong comment; out of scope to build a
  full search-by-marker upsert (a `gh api` + `jq` round trip) for a
  concern that doesn't arise from the action's own single-identity use.
- **No configurable gate policy** — "any failing PROVEN signal fails the
  check" is not adjustable per-repo (no "allow N failures," no per-gate
  severity levels). This is a deliberate reflection of the project's core
  thesis, not an oversight: a configurable threshold is exactly the kind
  of knob that quietly turns "proven" into "proven, unless we didn't feel
  like blocking on it."
- **No historical trend view across gate runs** — each `verdict gate`
  invocation is independent; nothing is persisted for comparing today's
  PR against yesterday's. Same scope line Phase 5 already drew for
  `bench`.
- **The HTML dashboard has no server-side or client-side data fetching**
  — it's rendered once, fully, at report-generation time. A dashboard that
  updates live or pulls from an API is a different kind of artifact than
  "a single file `actions/upload-artifact` can attach to a run."

---

## Phase 7 — Statistical Rigor

## The goal, restated

Every phase through Phase 6 answers a single yes/no question well: did
this one attempt pass, does this one PR merge. Two questions neither
answers, both flagged explicitly as future work back in Phase 4's own
scope-cut notes: **is the JUDGED bucket's opinion actually any good**, and
**is a pass-rate change real, or is it noise from re-running something that
was never perfectly deterministic in the first place.** Phase 7 answers
both with real, if modest, statistics — `calibration.py` and
`flakiness.py` — kept deliberately outside `Verdict`'s own schema: neither
is a property of one attempt, so neither belongs on `Verdict` itself.

## Judge calibration: concordance, not vibes

`Verdict.status` already refuses to let a JUDGED signal decide DONE/
NOT_DONE — that's been true since Phase 0. But "can't be load-bearing" and
"is worth reading at all" are different claims, and nothing before this
phase measured the second one. A vision judge that agrees with a human
reviewer 55% of the time is barely better than a coin flip; its PR-comment
opinion would still render with exactly the same confident tone as one
that's right 95% of the time; a team would have no way to know which one
they'd wired up.

`calibration.py`'s `run_calibration(judge, examples, threshold=0.95)` is
almost embarrassingly simple by design: for each `LabeledExample`
(a screenshot, the intent it was checked against, and a human's own
true/false verdict), run the judge, compare `judgment.passed` to
`human_label`, and count agreements. `CalibrationResult.concordance` is
`agreements / total` — a `@computed_field`, not a stored value, following
the exact "unknown reported as unknown" rule `Verdict.status` already
established: `concordance` is `None` (not `0.0`, not `1.0`) when the
dataset has zero examples, and `meets_threshold` is correspondingly
`False` rather than treating "nothing measured" as a pass.

**The dataset format is a JSON manifest, not a new gate or a new
Provenance value.** `load_labeled_dataset` reads
`[{"name", "screenshot", "intent", "human_label"}, ...]`, with
`screenshot` resolved relative to the manifest's own directory — same
"co-located, portable" convention `examples/starter_suite/*/task.yml`'s
`repo` field already uses. This is deliberately a standalone tool a team
runs once (or on a schedule) to audit a judge they've configured, not
something wired into `verdict run`/`gate`'s own hot path — calibration
against a fixed labeled set doesn't change per-PR, so re-running it on
every gate invocation would just be wasted judge calls.

**Every individual disagreement is kept, not just the aggregate number.**
`CalibrationResult.disagreements: list[Disagreement]` names exactly which
example the judge got wrong and what it said — a 75% concordance score
alone tells a team "something's off"; the disagreement list tells them
*which screenshots* to go look at, the same "point at the evidence, don't
just report a verdict" instinct behind Phase 2's `Attribution.explanation`
and Phase 4's glitch-scan video retention.

**`verdict calibrate` never exits non-zero.** Consistent with `bench`'s
own policy (Phase 5): a below-threshold concordance is diagnostic
information a team acts on deliberately, not a build to fail — there's no
CI step where "the judge's calibration regressed" should silently block a
merge the way a failing PROVEN gate does. The CLI prints a loud
`[bold red]WARNING[/bold red]` instead, and that's the whole enforcement
mechanism this phase ships.

**Only `MockVisionJudge` ships to calibrate against, for the same reason
Phase 4 never shipped a real vision-model integration.** `examples/
calibration_dataset/` demonstrates the mechanism honestly rather than
pretending to demonstrate a real judge's accuracy: `MockVisionJudge`
always returns `passed=True` (see `frontend/vision_judge.py`), so the
dataset's four hand-labeled examples (three `human_label: true`, one
`human_label: false`) drive its concordance to exactly `75%` — below the
`95%` default target, triggering the warning path on every demo run. This
is real arithmetic over a real (if intentionally simple) judge, not a
canned "success" output; a team that plugs in a real `VisionJudge`
implementation runs the identical command against their own labeled
screenshots.

## Flakiness detection: a real interval, not a raw percentage

The second gap: nothing before this phase distinguished "this agent got
worse" from "I ran it twice and got a different answer, because agents
(and browsers, and dev servers) aren't perfectly deterministic." A raw
pass-rate comparison — "70% last week, 60% this week" — looks like a
regression right up until you ask how much of that ten-point gap a handful
of trials could produce by chance alone, which for small samples is often
"most of it or all of it."

`flakiness.py::run_flakiness(task, repo, adapter, trials=10)` runs the
exact same `(task, repo, adapter)` through `runner.run()` — not
`run_with_retries()` — independently, `trials` times, and reports the raw
pass count plus a **Wilson score interval** around the pass rate. Two
choices worth calling out:

- **`run()`, deliberately not `run_with_retries()`.** Retries exist to
  *get past* a flaky failure on one task attempt (Phase 3); flakiness
  detection exists to *measure how often that failure happens in the
  first place*. Feeding retried outcomes into this measurement would hide
  exactly the variance it's trying to surface — a task that "always
  eventually passes by attempt 3" would report artificially high, when
  the number that actually matters here is "how often does attempt 1
  work."
- **Wilson, not the naive normal-approximation interval.** The textbook
  `p ± z·√(p(1-p)/n)` interval degenerates exactly where flakiness
  detection lives: small `n` (a handful of trials, not thousands) and `p`
  near 0 or 1 (a mostly-reliable agent, the common case). At `p=1.0` after
  3/3 passing trials, the naive interval collapses to `[1.0, 1.0]` — a
  claim of *zero-width certainty* from three data points, which is simply
  false. Wilson's interval still narrows toward 1.0 as `n` grows, but
  never claims that kind of certainty from a small sample — verified
  directly in `test_wilson_interval_never_claims_zero_width_certainty_at_p_one`.
  Computing an arbitrary confidence level's critical `z` would need the
  normal quantile function, which isn't in the standard library and isn't
  worth hand-approximating (a real source of subtle, hard-to-notice
  error); `_Z_FOR_CONFIDENCE` hardcodes the three standard levels
  (90/95/99%) and `wilson_interval` raises for anything else rather than
  quietly using a slightly-wrong number.

## Regression vs. noise: a real two-proportion z-test

`compare_flakiness(baseline, candidate)` is the direct answer to "is this
change real": a pooled two-proportion z-test over the two
`FlakinessResult`s, producing an actual `p_value`, not a hand-wavy "the
number went down." `ComparisonVerdict` is three-way — `REGRESSION`,
`IMPROVEMENT`, `NOISE` — and `NOISE` is the answer whenever `p_value`
doesn't clear `alpha` (default `0.05`), the exact same "an honest 'can't
tell' beats a guess in either direction" discipline Phase 2's bisector uses
for `skip` and Phase 6's `INCONCLUSIVE` attribution kind: a ten-point raw
pass-rate drop across two 10-trial samples is `NOISE` here (verified in
`test_small_sample_gap_is_noise_not_a_regression`), because ten trials
genuinely can't distinguish that gap from chance — reporting it as a
confirmed regression would be a false alarm dressed up as rigor.

The pooled standard error degenerates to exactly `0` when both samples'
outcomes agree completely (`p_pool` collapses to `0` or `1` — every trial
in both samples was the same outcome) — handled as its own case rather
than as a division-by-zero crash, always resolving to `NOISE`, which is
the only mathematically consistent answer when both proportions are
already identical.

## Why neither tool touches `Verdict`'s schema

Both `CalibrationResult` and `FlakinessResult`/`FlakinessComparison` are
Pydantic models (for the same free JSON serialization every schema type in
this codebase gets — `verdict flaky --json` round-trips a `FlakinessResult`
straight through `model_dump_json`/`model_validate_json` for a later
`--compare-to`), but neither is added to `schema.py`. Both are reports
*about* a judge or an agent's variance, generated by running many
`Verdict`s/`run()` calls, never a property any single `Verdict` carries —
the same "the whole codebase is organized around what's PROVEN vs.
JUDGED" boundary just doesn't have anywhere for "how often was this
proof itself proof" to plug into a single verdict's own fields. This
mirrors `failure_modes.py`'s `FailureModeBreakdown` (Phase 5): a tally
computed *over* a `ConfigResult`, kept as its own module-local type rather
than bolted onto `ConfigResult` itself.

## Tests

`test_flakiness.py` and `test_calibration.py` split the same way the
modules do: pure-math claims (Wilson interval shape, the z-test's
regression/improvement/noise classification, including the two
degenerate zero-standard-error cases) are asserted directly against known
inputs; `run_flakiness` gets one real end-to-end test against the shared
`git_repo` fixture with a deliberately-alternating fake adapter (fixes the
bug on odd calls, leaves it broken on even ones) proving trials are
genuinely independent `run()` calls, not a cached or reused result;
`load_labeled_dataset` is tested against every malformed-manifest shape
(missing key, non-boolean label, missing screenshot file, empty dataset)
the same defensive-parsing standard `suite/loader.py` already holds
itself to.

## What's explicitly out of scope for Phase 7

- **No real vision-model judge to calibrate against** — same scope line
  Phase 4 drew for `VisionJudge` itself; `MockVisionJudge` is what
  `examples/calibration_dataset` demonstrates the mechanism against, real
  and honestly imperfect (75% concordance), not a stand-in pretending to
  be a real model's score.
- **No automatic CI gate on calibration or flakiness** — `calibrate` warns,
  `flaky` reports; neither exits non-zero. `verdict gate` (Phase 6)
  remains the only command whose exit code blocks a merge, on purpose —
  see that phase's "no configurable gate policy" note for why blocking on
  a *statistical* signal specifically would be a different, weaker kind of
  promise than blocking on an executed PROVEN check.
- **No persisted history of calibration/flakiness runs across
  invocations** — `--compare-to` reads one prior `--json` file the caller
  supplies; there's no database or trend store keeping every past run.
  Same scope line Phase 5/6 already drew for `bench`/`gate` history.
- **`run_flakiness` has no seed/temperature control over the agent
  itself** — "multi-seed" here means "N independent real attempts," not
  controlling an adapter's own sampling parameters, which `Adapter`'s
  interface (Phase 0) has no hook for and isn't uniform across the five
  adapters Phase 5 added.
- **Only three confidence levels are supported** (90/95/99%) — see the
  Wilson interval section above for why an arbitrary level isn't worth the
  normal-quantile approximation it would require.

## Phase 8 — Sandboxed Execution

## The problem, restated

Phase 0 flagged this and deferred it explicitly: *"no container, no
network sandboxing, no filesystem jail... containerized sandboxing is a
separate, later concern once there's something worth sandboxing more
tightly."* Seven phases later, there's a lot worth sandboxing more
tightly: every gate tool, every coding-agent CLI (`adapters/*.py`), the
frontend dev server, and attribution's bisection re-runs all execute
code that either comes directly from an untrusted repo/PR or was just
written by an autonomous agent — with this process's own filesystem
access, network reachability, and (before this phase) full host
environment, `os.environ` included. That's the single biggest gap in the
product's own trust model: Verdict's whole premise is grading *untrusted*
code by executing it, and until this phase, "executing it" meant running
it with Verdict's own privileges.

## Threat model

| Untrusted input | What could go wrong | Control |
|---|---|---|
| Agent-authored test/build/lint code (`gates/*.py`) | Reads Verdict's own env (API keys passed to adapters), writes/deletes outside the worktree, reaches the network | `Sandbox.exec()`: explicit-only `env`, `--read-only` rootfs + worktree-only writable mount, `--network none` by default |
| A `verdict.yml` gate override / `frontend.start` (shell string sourced from the repo being graded) | Host `shell=True` — arbitrary shell injection with host privileges | Wrapped as `["sh", "-c", command]` and executed *inside* the sandbox — the shell boundary moved, not removed |
| A coding agent's own CLI (`adapters/*.py`) | Same privileges as gates, plus it's the process actually producing the code gates later execute | Routed through the same `Sandbox.exec()` chokepoint — no adapter calls `subprocess` directly anymore |
| The frontend dev server (agent-authored server code) | Binds ports, can fork children that outlive naive teardown, executes arbitrary server-side logic | Runs via `Sandbox.exec_background()`; a container's whole process tree dies with it — no `os.killpg` process-group bookkeeping needed |
| Historical commits re-checked-out during bisection | Same as gates, at N different commits instead of one | `attribution/bisect.py`'s self-driven binary search calls one fresh `Sandbox.exec()` per candidate commit — see below |
| Agent-authored HTML/JS the frontend dev server serves | Arbitrary JS executes in Playwright's Chromium renderer | **Not closed by Phase 8** — see "known gap" below |

## The `Sandbox` abstraction

One Protocol (`sandbox/base.py`), two implementations:

```python
class Sandbox(Protocol):
    def exec(self, cmd, cwd, env=None, timeout_seconds=600,
              limits=None, network=False) -> ExecResult: ...
    def exec_background(self, cmd, cwd, env=None, network=False) -> BackgroundHandle: ...
    def __enter__(self) -> "Sandbox": ...
    def __exit__(self, *exc) -> None: ...
```

`cmd` is always an argv list, never a shell string — a caller holding a
shell command from an untrusted source (`verdict.yml`, `frontend.start`)
wraps it as `["sh", "-c", command]` itself, so the shell interpretation
happens *inside* whichever backend is active, never via a host
`shell=True`. `env` is the *complete* set of extra variables a process
sees, merged with a small fixed baseline — never the caller's full
`os.environ` — so an adapter that genuinely needs a secret (an API key)
must name it explicitly; nothing is inherited implicitly, on either
backend.

**`DockerSandbox`** (the default): one container per `with` block, bound
to one worktree, `docker rm -f`'d unconditionally on exit. `--network
none` unless a caller asks for `network=True` (only the install step and
the dev server do); `--read-only` rootfs with the worktree bind-mounted
read-write and `/tmp` a small tmpfs, so an agent can edit its own checkout
freely but can't write anywhere else in the container. CPU/memory/pids
limits are passed to `docker run` today; *enforcement quality* (OOM
detection, a `killed_reason` beyond `"timeout"`) is Phase 9, not this
phase — the `ResourceLimits` fields exist now so call sites don't need to
change again when that lands.

**`LocalSandbox`** (opt-in only): today's pre-Phase-8 behavior — direct
subprocess execution, no isolation — behind the same interface, so
swapping backends is a config change, never a code change. Constructing
one prints, once, a loud:

```text
⚠️  UNSAFE — no isolation. Agent-generated code will execute with this
    process's full privileges (filesystem, network, credentials). Use
    only for local development on trusted repos.
```

It's the library-level *default* (`SandboxConfig()`, unqualified) — so
embedding code and tests don't silently require a Docker daemon — but the
`verdict` CLI itself defaults `--sandbox-backend` to `docker`, which is
the actual product default this phase is about.

**Sandbox settings never come from the repo being graded.** `SandboxConfig`
is constructed from Verdict's own CLI flags (`--sandbox-backend`,
`--sandbox-image`, `--sandbox-cpus`, `--sandbox-memory-mb`), never parsed
out of `verdict.yml` the way `gate_overrides`/`frontend` are — an
untrusted PR shipping a `verdict.yml` must never be able to turn off its
own sandbox.

## Base image: one fat, pinned image

`Dockerfile` (repo root) builds a single multi-language image — Python
(via `pyenv`), Node (via `nvm`), Go, and Playwright's Chromium — rather
than per-language images selected at runtime. Tradeoff, decided up front
rather than discovered mid-implementation:

- **Bake, don't mount.** A thin base image with host toolchains bind-
  mounted in was rejected: it reintroduces a host-trust dependency
  (whatever's installed on whichever machine runs Verdict) this phase
  exists to remove, and creates version drift between dev machines that a
  baked image doesn't have.
- **One fat image over several slim ones**, for now: simpler to build,
  test, and reason about with one image; a polyglot repo (Node frontend +
  Go backend) needs only one container, not a coordination story across
  several. The tradeoff is a larger pull (~2-3GB) and one team owning
  every language's version currency. `SandboxConfig.image` is a config
  field, not a hardcoded name, specifically so a future per-language
  selection strategy is a config change, not a refactor.
- **Version managers are baked in, unused so far.** `pyenv`/`nvm` are
  installed so a *later* phase can resolve a repo's own `.python-version`/
  `.nvmrc` pin instead of silently running against whatever the image's
  default happens to be — Phase 8 installs the managers and one default
  interpreter of each; it does not yet read or honor a repo's pin. A repo
  that pins Node 18 while the image defaults to Node 20 is a known,
  explicit gap, not something silently misattributed to the agent.
- **Pinned by tag, not yet by digest.** `verdict-sandbox:0.1.0` (not
  `:latest`) is referenced by version, so a given Verdict release always
  grades against the same toolchain — reproducible verdicts across image
  rebuilds. Digest pinning is strictly stronger (a tag can technically be
  overwritten) and is a follow-up hardening step, not done here.

## The network-policy boundary: install vs. gates

Before this phase, Verdict never ran a real dependency install — repos
were expected to pre-vendor `node_modules` (`worktree.py`'s
`copy_vendored_dependencies`). That's still the primary path. Phase 8
adds a minimal, explicitly scoped exception: `sandbox/install.py` detects
an install command (only when the dependency directory looks genuinely
absent — `package.json` with no `node_modules`, etc.) and runs it with
`network=True`, in its **own** sandbox session, separate from the one
gates run in. Gates' sandbox session stays `network=False` for the entire
run, full stop — the two never share a container, because Docker fixes a
container's network mode at creation time, and letting the install step's
network-on session "leak" into the gate-running session would silently
widen the boundary this phase exists to draw.

Deliberately out of scope, deferred to **Phase 10**: broader install-
command autodetection, dependency caching across runs, resolving a repo's
pinned language version via the now-present `pyenv`/`nvm`, and service
dependencies (databases, etc.). A test
(`tests/test_sandbox_docker_adversarial.py`) proves the boundary itself:
gates cannot reach the network under any circumstance; the install step
can.

## Bisection: a self-driven binary search, not `git bisect run`

The old `attribution/bisect.py` shelled out to `git bisect run <cmd>`,
where **git itself** repeatedly checked out a candidate commit and
re-invoked `bisect_cli.py` as a fresh **host** subprocess at each one — a
nested subprocess-of-a-subprocess Verdict couldn't route through a
`Sandbox` individually; the most it could sandbox was the outer `git
bisect` call, not each individual check.

Phase 8 rewrote `run_bisect` to drive its own binary search over
`commits_between(repo, good_sha, bad_sha)` (with a bounded forward nudge
past `SKIP` results, mirroring `git bisect skip`'s own heuristic), so
every candidate commit's check is one `Sandbox.exec()` call this module
controls directly. `bisect_cli.py` — which existed only to be invoked *by*
`git bisect run` — is gone; nothing else called it.

This was scoped as a "how it executes" change, not a "what it concludes"
change: the existing two-level design (commit-level bisect, then a
synthetic per-file ladder — `attribution/synth.py`, untouched) is
unchanged, and every existing Phase 2 attribution fixture
(`tests/test_attribution.py`) still passes unmodified, proving the
rewritten search converges on the same culprit commit/file a real `git
bisect` would have found. A sandbox infra failure mid-search (a
`SandboxUnavailableError`, a Docker daemon hiccup) maps to `Reproduction.
SKIP` via `check_reproduces`'s existing exception handling — the same
taxonomy `bisect_cli.py`'s exit codes used to drive — so an infra blip
can't be misread as "this commit is the culprit."

## Known gap: the browser isn't sandboxed

The dev server itself now runs inside a `Sandbox` (`frontend/server.py`),
but Playwright's Chromium (`frontend/runner.py`) still launches directly
on the host and reaches the dev server the same way it always has. Since
the page served is entirely agent-authored HTML/JS, and that JS executes
in this host-side renderer, this remains real, unsandboxed execution of
untrusted code — the single highest-risk surface flagged during this
phase's design review. Closing it (e.g. running Playwright's own
remote-server mode inside the same container as the dev server, with the
host driving it over CDP) is real, scoped work of its own and is not
implemented here; `frontend/runner.py::run_frontend_checks`'s docstring
carries the same note inline.

## What's explicitly out of scope for Phase 8

- **CPU/memory/pids enforcement quality** — limits are passed to `docker
  run`, but OOM detection and a `killed_reason` beyond `"timeout"` don't
  exist yet. **Phase 9.**
- **Network allowlisting beyond off/on** — `SandboxConfig` has no
  per-host allowlist; a sandbox session can reach nothing or everything
  on its network, no middle ground. **Phase 9.**
- **Digest-pinning the base image**, install-command breadth, dependency
  caching, language-version pin resolution, and service dependencies —
  see the sections above. **Phase 9/10, as noted per item.**
- **Sandboxing the Playwright browser itself** — see "known gap" above.
- **Multi-repo/multi-tenant resource isolation** — this phase's unit is
  one worktree, one container; nothing here addresses running many
  untrusted repos' sandboxes concurrently on shared infrastructure.

## Phase 9 — Timeouts, Kill-Trees, and the Attempt Budget

## The gap Phase 8 left open

Phase 8 gave every command a *timeout parameter* — `exec_command`,
`Sandbox.exec()` all accepted `timeout_seconds` from the start. What Phase
8 did NOT make robust:

- A timeout only ever killed the **one process** `exec()` started. Nothing
  killed what that process itself forked. A hung dev server that spawned
  a real child process, or a hung test that spawned its own subprocess,
  left that child running — an **orphan**, potentially still bound to a
  port — after the "timed-out" parent was gone. `DockerSandbox.
  exec_background`'s original pidfile scheme made this concrete: it
  tracked the wrapped shell's own PID, then `exec`'d into the real
  command, so `terminate()`'s `kill <pid>` only ever reached one process,
  by construction.
- There was no **global** ceiling. Four gates timing out at their
  individual `timeout_seconds` each could still add up to an unbounded
  total, and nothing stopped a run from grinding through install → gate 1
  → gate 2 → attribution → frontend checks indefinitely if each individual
  step stayed just under its own limit.
- CPU/memory/pids limits were passed to `docker run`, but disk had no
  enforcement attempt at all, and there was no distinction anywhere in the
  code between "a gate hung" and "the sandbox itself never came up" — both
  would have produced roughly the same shape of failure.

Phase 9 closes all three, and — per an explicit design decision made
before writing any of it — draws one more line the code didn't draw
before: which of those failures is the **agent's** fault, and which is
**infrastructure's**.

## The kill-tree fix: process groups, not process IDs

Every backend now launches the command it's given as the leader of its
own process group (`start_new_session=True` for `LocalSandbox`'s
`subprocess.Popen`; `setsid` for `DockerSandbox`'s in-container wrapper —
`setsid cmd`'s PID *is* its own process-group id, since `setsid` makes the
process it execs into both a new session leader and, by consequence, a new
process-group leader). Killing on timeout always targets the **group**
(`os.killpg` locally; `kill -TERM -PID` — the leading `-` is what makes it
a group signal — inside the container), not the one PID that was started.
Every child and grandchild the command forked, as long as none of them
independently called `setsid`/`setpgid` to escape the group (ordinary dev
servers and test runners don't), dies with it.

`DockerSandbox.exec()` itself changed shape to make this possible mid-
session: it used to be one blocking `docker exec`, whose host-side
timeout killed the **host's** `docker` CLI process while leaving whatever
was running *inside* the container untouched — the timeout would return,
but the workload kept running in a container about to be reused for the
next gate. Phase 9 makes `exec()` a tracked, polled operation instead:
launch detached (`docker exec -d`) under `setsid`, with the command's own
(group) pid recorded to a pidfile inside the container; poll for an
exit-code marker file up to `timeout_seconds`; on timeout, explicitly
`kill -TERM -PID` then `-KILL` the group before returning. This is the
same mechanism `exec_background` (the dev server) already needed, just
also applied to the blocking gate-execution path — one shared
`_start_tracked`/kill helper serves both.

`tests/test_sandbox_killtree.py` proves the actual claim end to end
against `LocalSandbox`: a script that forks a grandchild which binds a
real port and sleeps, itself hangs forever (the "agent-introduced
infinite loop" shape), gets `exec()`'d with a 2-second timeout, and the
test then independently verifies — not by trusting the killed process's
own exit code, but by trying to re-bind the port and checking the
grandchild's PID directly — that both the port is free and the grandchild
is gone. `test_sandbox_docker_adversarial.py`'s equivalent (Docker-gated)
additionally proves the *session* survives a timed-out `exec()` — a
follow-up command in the same container still runs, since only that one
command's tree was killed, not the whole container.

## Resource limits: what's enforced, what's attempted, what's still open

CPU/memory/pids limits were already passed to `docker run` in Phase 8 and
needed no change — they're session-level (cgroup limits apply to the
whole container, every `exec()` call against it, for its lifetime), which
is exactly the granularity Phase 9's shared-container-across-gates design
needs. Phase 9 adds a best-effort attempt at **disk**: `--storage-opt
size=<disk_mb>m` on `docker run`, which only actually works with specific
storage-driver/backing-filesystem combinations (overlay2 on xfs with
pquota — notably NOT the common ext4-backed default). Rather than fail
every run whose Docker install doesn't support it, `DockerSandbox.
__enter__` tries once with the flag and, on failure, retries once without
it — a storage-driver mismatch degrades to "no disk quota," not "no
sandbox at all." What Phase 9 does NOT add: OOM-specific detection
(`ExecResult.killed_reason` is `"timeout"` or `None` — never `"oom"` yet,
even though `--memory` can and does kill a process today) or real
enforcement of a per-command (as opposed to per-container) resource
ceiling. Both are explicitly deferred — see below.

## The agent-fault / infra-fault distinction

This was a design decision made explicitly before writing any of this
phase's code (see the conversation that scoped it), not something the
implementation backed into. Two failure shapes look superficially similar
— "something didn't finish in time" — but mean opposite things:

- **A gate hangs.** The agent wrote (or left broken) code that a test
  suite, typechecker, linter, or build step never returns on. This is
  real, observed evidence about the agent's own output — exactly as real
  as any other nonzero exit code. It is graded as a normal **PROVEN
  FAIL**, no special status: `exec_command` labels the resulting FAIL's
  detail text with `"timed out after Ns"` so a human reading the report
  understands *why* it failed, but `GateStatus` itself needs no new value
  — timeout already produces exit code 124, and every existing parser
  already treats a nonzero exit as FAIL. `ToolRunner.run()`'s own
  docstring now says this explicitly, so a future gate implementation
  doesn't have to rediscover it.
- **Provisioning hangs.** The sandbox container never comes up
  (`docker run` itself times out), or the dependency-install step never
  finishes (`npm install` stuck against a broken registry). Neither of
  these is something the agent wrote — the agent didn't author the
  install command, and it certainly didn't author Verdict's own container
  startup. These raise `ProvisioningTimeoutError` (a subclass of the
  existing `SandboxUnavailableError`, itself a `RuntimeError`) and **abort
  the whole attempt** — no Verdict is produced, exactly the same way an
  adapter CLI (`claude`, `aider`, ...) hanging already aborted the attempt
  before this phase. `cli.py`'s `_RUN_ERRORS` tuple — already the single
  place every "this attempt couldn't be evaluated" exception type is
  caught and turned into a clean exit-code-2 CLI error — now includes
  `SandboxUnavailableError` (covering both the timeout subclass and the
  pre-existing "daemon unreachable" case, which, before this phase, wasn't
  actually caught there either and would have surfaced as a raw
  traceback).

  `ProvisioningTimeoutError` subclassing `SandboxUnavailableError`
  specifically (rather than being its own unrelated exception) is a
  forward-looking choice: Phase 11 is expected to introduce a real ERROR
  outcome and fold this whole "couldn't evaluate" family into it. Keeping
  every member of that family in one hierarchy now means Phase 11 has one
  ancestor type to catch, not several scattered `except` clauses to find
  and update.

- **The global attempt budget exceeding** is deliberately a *third*
  thing, not folded into either bucket above. It's not the agent's fault
  (nothing it wrote timed out) and it's not quite an infra failure either
  (nothing actually broke — there just wasn't enough time to check
  everything). See the next section.

## The global per-attempt budget

`SandboxConfig.attempt_budget_seconds` (default 1800s / 30 minutes, `None`
disables it) is a single wall-clock ceiling computed once at the start of
`runner.run()`/`grade_existing_diff()` — covering the adapter's own run,
the install step, every gate, attribution's bisection, and the frontend
checks combined, not an independent allowance re-granted per phase. Both
entry points check the same deadline before starting each subsequent
phase (`gates/registry.py::run_all_gates` checks it before each individual
gate too, not just once at the top of the loop) and, once it's passed,
skip everything remaining rather than attempting it.

Skipped gates are simply **absent** from `Verdict.signals`, not present
with some placeholder status — reporting them as FAIL would blame the
agent for something Verdict itself never checked; reporting them as NA
would misrepresent "we didn't get to it" as "this repo has no such stack."
`Verdict.budget_exceeded: bool` carries the fact itself, and `status`'s
logic (see `schema.py`) was extended with one rule, decided explicitly
rather than left implicit:

- A real, observed PROVEN FAIL always wins, budget or not. An agent-caused
  failure that was actually witnessed doesn't stop being real just
  because the run later ran out of time somewhere else — it's still
  NOT_DONE, and it still counts against the agent in `pass_rate`/
  `pass_rate_per_dollar`.
- A `budget_exceeded` run that saw **no** FAIL is never DONE. "Every gate
  we managed to run passed" is a materially weaker claim than "every gate
  passed" when some gates never ran at all — the same reasoning
  `VerdictStatus.UNVERIFIED` already existed for (zero PROVEN gates ran at
  all); Phase 9 just extends it to cover "some ran, some didn't, and none
  of the ones that did found anything wrong" as the same kind of
  incomplete-therefore-unproven claim, rather than inventing a fourth
  status.

## What's explicitly out of scope for Phase 9

- **OOM-specific detection** — `ExecResult.killed_reason` only
  distinguishes `"timeout"` from `None` today; a process killed by
  `--memory`'s cgroup limit currently just looks like any other nonzero
  exit, not a specifically-flagged OOM. Real detection needs reading the
  container's own OOM-kill status (`docker inspect`'s `OOMKilled` field)
  rather than inferring it from an exit code.
- **A real per-command CPU/memory ceiling** — today's limits are
  per-container (shared across every gate run in that session), not
  re-enforced or re-partitioned per individual `exec()` call.
- **Disk quota portability** — the `--storage-opt size=` best-effort
  attempt silently degrades to "no quota" on the common storage-driver
  configurations that don't support it; there's no fallback enforcement
  mechanism (a periodic `du` check and manual kill, for instance) for
  those cases.
- **Folding `ProvisioningTimeoutError`/`SandboxUnavailableError` into a
  real ERROR `Verdict` outcome** — that's Phase 11, as noted above; today
  they abort the attempt (no `Verdict` at all) rather than producing one
  tagged ERROR.
- **Per-phase (rather than global) budgets** — there's no separate "the
  adapter itself gets at most N minutes of the 30" sub-allocation; the
  budget is one shared pool, first-come-first-served across phases in
  execution order.

## Phase 10 — Setup, Services, and a Base-State Cache

## The problem, restated

Phase 8's sandbox and Phase 9's timeouts made execution safe and bounded;
neither made it *correct* for anything beyond a repo that already has its
dependencies vendored and needs nothing else to boot. `sandbox/install.py`
shipped in Phase 8 as an explicitly minimal placeholder — its own
docstring listed exactly what it deferred: *"broader/more accurate
autodetection, dependency caching across runs, resolving a repo's pinned
language version..., and service dependencies (databases, etc.)."* Phase
10 is that list, plus one efficiency problem the codebase had been living
with since Phase 2: `attribution/engine.py`'s baseline check and
`frontend/runner.py`'s before-image each independently re-rendered the
exact same pre-agent `base_commit` from scratch, every single time either
was needed — up to `MAX_ATTRIBUTIONS_PER_GATE` (5) times per gate, per
attempt, for the baseline check alone.

Four decisions were made explicitly, before writing any of this phase's
code, and are recorded here rather than left implicit in the diff:

## Service dependencies: an allowlist, not an arbitrary image

`verdict.yml`'s new `services:` section (parsed by `config.py`'s
`ServiceSpec`) never lets the repo being graded name a raw `image:` for
Verdict to `docker run` — it names a `type` (`postgres`, `redis`, `mysql`,
`mongodb`) and a `version`, looked up against `sandbox/services.py`'s own
`_SERVICE_IMAGES` allowlist, which Verdict's code owns entirely. An
unrecognized type or version is a real, surfaced `SetupError` — never a
silent fallback to some default image, and never something `verdict.yml`
can talk its way around by editing itself. Extending the allowlist (a new
service type, a newer pinned version, a genuinely custom image for some
future use case) is an operator action against Verdict's own code, the
same "sandbox policy never comes from the repo" rule Phase 8 established
for everything else in this module family. `env`/`port` on a `ServiceSpec`
are still repo-controlled — but their blast radius is bounded to
configuring a fresh, ephemeral, Verdict-launched container (a Postgres
password, a port number), not to choosing what code runs.

## Networking: a per-attempt `--internal` Docker network

Gates need to reach a declared service (a test suite hitting Postgres)
without gaining the general internet egress Phase 8/9 close by default —
previously a binary choice (`--network none` or `bridge`) with nothing in
between, and explicitly flagged as a gap in Phase 9's "out of scope" list.
Phase 10 closes it with `docker network create --internal
verdict-attempt-<id>` (`sandbox/docker.py`): Docker itself refuses to
route this network to the public internet, full stop, so joining it is
safe in a way a generic allowlist would need more machinery to guarantee.
Each declared service joins with `--network-alias <name>` (the DNS name
`verdict.yml` gave it); the gate/adapter container joins the same network
*only when any services are declared* — the common, service-free case is
completely unchanged, still `--network none`. `sandbox/services.py`'s
`start_services` tears down whatever it already started on the first
failure (an unrecognized type/version, a service that never passes its
health check), and `runner.py` tears down the network itself in a
`finally` around the whole attempt — plus `sandbox/docker.py::
sweep_leaked_networks`, run once at the start of a batch of work, cleans
up `verdict-attempt-*` networks a previous crashed process never got to
remove (best-effort: `docker network rm` already refuses to remove a
network still attached to a live container, so this only ever reaches
genuine orphans). `tests/test_sandbox_services_docker.py` (Docker-gated)
proves both halves for real: a fixture repo whose test suite needs
Postgres actually passes end to end, and a gate container joined to the
service network still cannot reach the public internet.

Gates block on every declared service's health check (a per-type command
— `pg_isready`, `redis-cli ping`, ...) before running at all. A service
that never becomes healthy is `SetupError`, never a gate FAIL — the same
agent-fault/infra-fault line Phase 9 drew for provisioning timeouts, now
applied to "the database never came up."

## Version pins and a broader, lockfile-aware install

`sandbox/versions.py` reads `.python-version`/`.nvmrc` if present and
resolves them against pyenv/nvm — both already baked into the image since
Phase 8, unused until now. pyenv's shims stay on `PATH` inside the image,
so resolving a Python pin is just `PYENV_VERSION=<pin>` in the env overlay
(installing the version first via `pyenv install --skip-existing` if it's
not already present). nvm has no persistent shim directory on this image's
`PATH` (the Dockerfile hardcodes `PATH` to one baked-in Node version), so
resolving a Node pin means sourcing `nvm.sh` once during setup, installing
if needed, and resolving the concrete `node` binary's directory to prepend
onto `PATH` for everything after — nvm itself is never invoked again past
that point. The resulting overlay (`sandbox/install.py::run_setup_step`'s
return value) is merged into every subsequent gate's own execution env
(`gates/base.py`/`registry.py`'s new `env` parameter, threaded the same
mechanical way Phase 9 threaded `timeout_seconds`) — the pinned version is
what actually runs the gates, not just what a resolution step confirmed
was available. Deliberately NOT resolved: anything short of a literal
pinned-version file (`pyproject.toml`'s `requires-python`, `package.json`'s
`engines.node`, a version *range* rather than a single pin) — see "out of
scope" below.

Install detection (`_detect_install_command`) is now lockfile-aware:
`package-lock.json` → `npm ci`, `pnpm-lock.yaml` → `pnpm install
--frozen-lockfile`, `yarn.lock` → `yarn install --frozen-lockfile`,
`poetry.lock` → `poetry install`, `uv.lock` → `uv sync --frozen`, falling
back to a loose `npm install`/`pip install -r requirements.txt` only when
no lockfile is present. A lockfile-respecting install is preferred
wherever a lockfile exists on purpose: a repo that committed one is asking
for exactly what it locked, and `--frozen`/`ci` fail loudly on a
lockfile/manifest mismatch rather than silently resolving something
adjacent to it.

Version-pin resolution and dependency install both run inside the SAME
network-on sandbox session (`run_setup_step`), before gates' own
network-off session opens — unchanged from Phase 8's separation, just with
more work happening in that one networked step.

## The base-state cache

Keyed on `(base_commit_sha, lockfile_hash, sandbox_image_tag)` —
`sandbox/cache.py::cache_key`. `lockfile_hash` is computed via `git show
{ref}:{lockfile}` for every lockfile name this module knows about, so
computing it needs no worktree checkout at all. The image tag is part of
the key deliberately: a different `verdict-sandbox` image can carry
different tool versions and produce genuinely different real results, so
leaving it out of the key would let a stale cache entry silently survive
an image upgrade.

Two independent artifact kinds live under one cache entry
(`~/.cache/verdict/base-state/<key>/`, overridable via `VERDICT_CACHE_DIR`
— no eviction/TTL this phase, entries simply accumulate, noted as
deferred below rather than silently unbounded-and-unmentioned):

- **`gate_signals.json`** — all four gates' PROVEN signals at
  `base_commit`, computed together on the first miss (not just the one
  gate that happened to trigger it) so the *next* failing signal's
  baseline check, whatever gate it's against, hits the cache too.
  `attribution/engine.py::_reproduces_at` consults this before rendering
  anything; `attribution/reproduce.py::reproduction_from_signal` was
  split out of `check_reproduces` specifically so the cache-hit path can
  reuse the exact same signal-to-Reproduction logic against a *cached*
  Signal, not just a freshly-resolved one.
- **`screenshots/<viewport>.png`** — the before-image
  `frontend/runner.py::_capture_before` renders once per unique
  `base_commit`/lockfile/image combination instead of once per
  `run_frontend_checks` call (previously: once per attempt, even across
  `run_with_retries` attempts that share the same `base_commit`).

A cache MISS still does real, full work — render a scratch worktree,
install dependencies (`run_setup_step`, so a cache population correctly
reflects a repo *with* its deps installed, closing a gap the pre-Phase-10
baseline check had: it never called `copy_vendored_dependencies` or any
install step at all, so it could already be checking against a
dependency-less checkout). A cache HIT skips the scratch worktree and the
sandbox entirely — `tests/test_cache.py`'s
`test_reproduces_at_uses_a_seeded_cache_without_touching_the_sandbox`
proves this concretely, by seeding the cache and pointing
`_reproduces_at` at a Docker backend with no reachable daemon: reaching a
real answer instead of a `SandboxUnavailableError` is the proof the
sandbox was never touched.

Bisection's own intermediate-commit checks (`attribution/bisect.py`) are
deliberately NOT cached — they render a different commit on nearly every
call by construction, so there's no reuse value the way there is for the
one commit (`base_commit`) that's identical across every baseline check
and every before-image in a given attempt.

## The ERROR status: a minimal pull-forward from Phase 11

Phase 9's DESIGN.md assigned "a real ERROR `Verdict` outcome" to Phase 11
and had `ProvisioningTimeoutError` abort the whole attempt with no
`Verdict` produced at all. Phase 10 needs the same infra-not-agent
treatment for its own new failure modes (an unrecognized service
type/version, a service that never becomes healthy, an unresolvable
version pin) — and, per an explicit decision made before coding, pulls
forward a minimal `VerdictStatus.ERROR` now rather than waiting:

- `schema.py`'s `Verdict.error: str | None` field, when set, makes
  `status` ERROR unconditionally — checked first, ahead of the
  FAIL/budget/applicable-signals logic the other three statuses derive
  from, since by construction nothing was actually evaluated.
- `SetupError` (`sandbox/base.py`) is a new sibling of
  `ProvisioningTimeoutError` under the same `SandboxUnavailableError`
  ancestor — one family, one thing Phase 11 will eventually need to catch
  in one place, not two.
- **The catch site is what actually changed, not the raise.** A single
  `verdict run`/`verdict gate` invocation still lets `SandboxError`
  propagate uncaught straight to `cli.py`'s `_RUN_ERRORS` — exit code 2,
  no report, unchanged from Phase 9. `suite/runner.py::run_suite` is the
  one place Phase 10 actually changes behavior: it now catches
  `SandboxError` **per `(config, task)` pair**, records that task as a
  real `Verdict(status=ERROR, error=str(exc))`, and moves on to the next
  task — a suite grading dozens of (agent, task) combinations shouldn't
  lose everything else it already computed because one repo's Postgres
  never came up.
- `ConfigResult.pass_rate` excludes errored tasks from its DENOMINATOR,
  not just the numerator (`tasks_errored`, new) — a task that was never
  actually graded shouldn't dilute the rate the way a real observed
  failure does, in either direction. `0.0` (not `None`) when every task
  errored, matching the "report an honest number, not a guess" instinct
  `total_cost_usd`'s own `None`-on-unknown already established elsewhere
  in this file.
- `report.py`'s `_VERDICT_STYLE` dict (a plain lookup keyed by the three
  pre-Phase-10 `VerdictStatus` members, no default) would have raised
  `KeyError` on any ERROR verdict — caught during this phase's own review
  and fixed alongside adding the status, not left as a latent crash for
  whoever first triggered it for real.

Deliberately minimal, per the decision that scoped it: no retry policy,
no dedicated provenance bucket, no richer reporting beyond the one status
value, the accounting rule, and not crashing the renderer. Phase 11 is
still where the fuller shape of this gets designed.

## What's explicitly out of scope for Phase 10

- **Broader version-pin sources** — `pyproject.toml`'s `requires-python`,
  `package.json`'s `engines.node`, version *ranges* rather than a single
  pinned value. Only a literal `.python-version`/`.nvmrc` is resolved.
- **Dependency caching across DIFFERENT commits sharing a lockfile** —
  the base-state cache reuses gate signals/screenshots for the identical
  `base_commit`, but doesn't yet cache *installed dependency artifacts*
  (a `node_modules`/`.venv` snapshot) keyed on `lockfile_hash` alone, which
  would let two different commits with an unchanged lockfile skip
  re-installing entirely. Real, valuable, and deferred.
- **Cache eviction/TTL** — `~/.cache/verdict/base-state/` has no size cap
  and no expiry; entries accumulate until something (a human, a future
  `verdict cache clear`) removes them.
- **Bisection's own scratch worktrees still don't install dependencies**
  — `attribution/bisect.py::_check_at` renders intermediate commits with
  no `copy_vendored_dependencies`/`run_setup_step` call, a pre-existing
  gap this phase fixed for the baseline check specifically but left alone
  for bisection's many-different-commits case.
- **Adapter CLIs don't see the version-pin env overlay** — `run_setup_step`'s
  overlay is threaded to gates, not to `Adapter.run()`'s own `sandbox.exec()`
  call; an agent that itself shells out to a pinned-version tool during
  its own work sees the image default, not the resolved pin.
- **`verdict flaky`'s trial loop doesn't get suite-style ERROR handling**
  — only `run_suite` catches `SandboxError` per unit of work; a
  `SandboxError` mid-trial still crashes the whole `verdict flaky`
  command, same as a single `verdict run` does.
- **Folding ERROR into a fuller Phase 11 shape** — retry policy, a
  dedicated provenance bucket, richer reporting. See above.

## Phase 11 — The Outcome Taxonomy: ERROR Gets a Retry Policy and a Voice

## The problem, restated

Phase 10 pulled `VerdictStatus.ERROR` forward deliberately minimally: one
status value, one accounting rule (excluded from `pass_rate`'s
denominator), and just enough renderer plumbing not to crash. Its own
DESIGN.md section named exactly what it left undone — "no retry policy,
no dedicated provenance bucket, no richer reporting" — and scoped that to
this phase. The animating complaint behind doing it now: a leaderboard
that can't tell "the agent failed" from "our own sandbox never came up"
is lying about what it measures, and a run that could have succeeded on a
second try but was never given one is wasted signal. Three things had to
be true by the end of this phase: ERROR has to mean the same thing
everywhere (sandbox failures, setup failures, adapter crashes, infra
timeouts — not just the service-health/provisioning cases Phase 10
covered), infra flake has to get a bounded second (and third) chance
automatically, and a legitimate agent NOT_DONE must never get swept into
that same automatic retry — those are different questions with different
answers, and conflating them either wastes an agent's retry budget on
infra Verdict can't fix, or lets Verdict quietly pretend a repeatable
agent failure was just bad luck.

## Widening what counts as ERROR: adapter crashes join sandbox failures

Phase 10's catch site only ever saw `SandboxError` (provisioning, service
health, setup/install). But `Adapter.run`'s own docstring already drew
the line Verdict needed here, before this phase touched a line of code:
"must not raise on the agent merely failing the task — only on the
adapter itself being unable to run." A raised adapter exception was
*already*, by contract, infra-not-agent — Phase 10 just didn't act on
that fact yet, so an adapter CLI crashing (missing binary, malformed
output, an unexpected non-zero exit) still propagated as a bare exception
with no ERROR Verdict at all, in either the single-run or suite path.

Phase 11 gives every adapter error class a shared ancestor —
`verdict.adapters.AdapterError(RuntimeError)` — and every existing
per-adapter class (`ClaudeCodeAdapterError`, `CursorAdapterError`,
`CodexAdapterError`, `AiderAdapterError`, `OpenHandsAdapterError`) now
subclasses it instead of `RuntimeError` directly. This isn't cosmetic:
it's what lets `runner.py` catch "the adapter itself is broken" as one
concept — `_EVALUATION_ERRORS = (SandboxError, AdapterError,
WorktreeError)` — without hardcoding a per-adapter list that would need a
new line added every time a sixth, seventh, ... agent adapter ships.
`WorktreeError` joins the same tuple for the same reason: git worktree
isolation failing is exactly as much "Verdict's own infra broke" as a
sandbox never coming up, and was already excluded from ever becoming a
gate Signal.

## The retry policy: two structurally separate loops, not one generic one

Before this phase, `run_with_retries`'s `max_attempts` loop was the only
retry mechanism that existed, and it only ever "worked" for ERROR by
accident: a `SandboxError` raised mid-loop simply propagated straight out
of `run_with_retries` uncaught, aborting every remaining attempt, not
retrying. Phase 11 replaces that with two loops that never share code:

- **`_run_attempt`** (new, `runner.py`) — retries ONLY on a raised
  `_EVALUATION_ERRORS` exception, up to `max_error_retries` extra times
  (`DEFAULT_MAX_ERROR_RETRIES = 2`, a new constant independent of
  `DEFAULT_MAX_ATTEMPTS`). It never inspects a returned `Verdict`'s
  status at all — structurally, there is no code path in this function
  that could see a NOT_DONE and decide to retry it, because NOT_DONE is
  never raised, only returned. Exhausting the budget without success
  builds a real `Verdict(error=str(exc))` via the same `_error_verdict`
  helper `suite/runner.py` used to build inline before this phase (now
  shared, not duplicated).
- **`run_with_retries`'s own loop** (existing, changed) — retries ONLY a
  returned `Verdict` that isn't `done`, up to `max_attempts` times,
  exactly as before. The one behavior change: it now also stops early on
  `VerdictStatus.ERROR`, not just DONE — once `_run_attempt` has already
  exhausted its own bounded infra-retry budget, handing the same broken
  sandbox back to the agent for another full `max_attempts` round buys
  nothing and would just spend more of the agent's budget (real spend,
  for a real adapter) against something already proven broken.

Both loops append every attempt they make to the same flat
`TaskRun.attempts` list — an ERROR retry is exactly as real an "attempt"
for cost-accounting purposes as an agent retry is, so `total_cost_usd`
already accounts for it correctly with no schema change needed
(`_error_verdict`'s `cost_usd=0.0` is real: by construction, nothing in
`_EVALUATION_ERRORS` fires after the agent has spent anything — every one
of them is either pre-adapter setup or the adapter's own "I never ran"
signal).

**The one behavior change this forces at the `run()` level**: `run()`
itself is UNCHANGED — it still lets `_EVALUATION_ERRORS`-family exceptions
propagate all the way out uncaught, exactly as `test_timeouts.py`'s
`test_provisioning_timeout_raises_and_never_produces_a_signal` continues
to prove. Only `run_with_retries` (and therefore `verdict run`, `verdict
bench` via `run_suite`) catches and retries now. `grade_existing_diff`
(`verdict gate`, the merge-gate command) is deliberately untouched this
phase — see "out of scope" below.

## `suite/runner.py`: the catch site moves down a layer

Phase 10's `run_suite`/`_run_task` had its own inline `except
SandboxError` block, building an ERROR `Verdict` by hand, because
`run_with_retries` didn't handle it. That block is gone: `run_with_retries`
now never lets `_EVALUATION_ERRORS` escape, so `_run_task` is back to a
one-expression wrapper. `run_suite` gained one new parameter,
`max_error_retries` (threaded straight to `run_with_retries`, default
`DEFAULT_MAX_ERROR_RETRIES`) — every `(config, task)` pair gets its own
independent bounded infra-retry budget, same as before, just resolved one
call frame lower than it used to be.

## Economics: audited, not rewritten

`ConfigResult.pass_rate`/`pass_rate_per_dollar` already excluded errored
tasks from their denominators as of Phase 10 — this phase's job was to
audit every OTHER metric path for the same discipline, not re-derive it:

- `failure_modes.py::summarize_failure_modes` — already correct by
  construction: an ERROR `Verdict` has `signals=[]`, so it contributes
  zero entries to the failing-check tally without needing a status check
  at all.
- `economics.py`/`report_html.py`'s leaderboards — correct in the number
  (`pass_rate` was already right), but the *display* wasn't: showing
  `tasks_done/tasks_total` next to a percentage computed over
  `tasks_total - tasks_errored` was internally inconsistent (a reader
  could compute the shown fraction and get a different percentage than
  the one printed next to it). Both now show `tasks_done/<graded>` where
  `<graded> = tasks_total - tasks_errored`, plus a new dedicated
  `errored` column (`N/tasks_total`, or `—` when zero) — the leaderboard
  now states the exclusion instead of leaving a reader to infer it from a
  percentage that doesn't match the fraction beside it.
- `flakiness.py`'s Wilson-interval `pass_rate` — audited and left alone
  on purpose: `run_flakiness` calls `run()` directly (not
  `run_with_retries`), so it was never in scope for this phase's ERROR
  handling to begin with (Phase 10 already noted this as deferred; still
  deferred, see "out of scope" below), and its `pass_rate` isn't the
  metric this phase's economics language ("pass-rate-per-dollar
  denominators") refers to regardless.
- `calibration.py` — never consults `Verdict.status`, nothing to audit.

## Renderers: CLI and HTML get the same `errored` column, JSON already had it

`report.py` (CLI) already rendered ERROR distinctly as of Phase 10 (bold
magenta `VERDICT: ERROR`, the `error` message shown inline) — unchanged
this phase. `report_json.py` needed no change either: `ConfigResult`'s
`tasks_errored` computed field and every `Verdict.status`/`error` field
already round-trip through `model_dump(mode="json")` with no reshaping,
so a consumer parsing the JSON report could already tell ERROR apart from
NOT_DONE before this phase touched anything. The actual gaps were in the
two *aggregate* renderers:

- `economics.py`'s CLI leaderboard table and `report_html.py`'s HTML
  leaderboard table both gained the `errored` column described above.
- The HTML report's "show only failing tasks" checkbox previously kept
  only `.fail`-classed tasks, silently hiding `.error`-classed ones (a
  reader who ticked the box to see what needed attention would have
  ERROR tasks disappear along with the DONE ones) — inverted to hide only
  `.pass`, so both FAIL and ERROR stay visible under the filter, which is
  what a reader ticking that box actually wants to see.

## Tests

`tests/test_error_retry.py` (new) proves the retry-policy split directly:
an adapter that raises on every call is retried exactly
`max_error_retries` times before landing on ERROR; one that raises twice
then succeeds recovers within the budget; a `max_error_retries=0` run
makes exactly one attempt; an ERROR that exhausts its infra-retry budget
stops the outer `max_attempts` loop early even when attempts remain; a
legitimate NOT_DONE from `_AlwaysFailsAdapter` is never touched by the
error-retry path regardless of how generous `max_error_retries` is, and
is bounded by `max_attempts` exactly as before; an `AdapterError` raise is
routed through the same ERROR path as a `SandboxError`; and two
`run_suite`-level tests prove the same bounded-retry-then-ERROR behavior
end-to-end, including a mixed suite where one task errors and is excluded
from `pass_rate`'s denominator while a sibling task's real pass is
unaffected. `tests/test_error_routing.py`'s Phase 10 tests are unchanged
in substance; its module docstring and `tests/test_timeouts.py`'s
provisioning-timeout test docstring were updated to describe the new
`run()` vs. `run_with_retries` split rather than claim (as Phase 10
correctly did, at the time) that both behave identically.

## What's explicitly out of scope for Phase 11

- **`grade_existing_diff`/`verdict gate`** — still lets
  `_EVALUATION_ERRORS`-family exceptions propagate to `cli.py`'s
  `_RUN_ERRORS` (`WorktreeError` only was actually caught there before;
  `SandboxError` from `grade_existing_diff`'s own service/sandbox setup
  was already an uncaught crash pre-Phase-11 and remains one). The
  merge-gate command grades a PR's diff in place, not via an `Adapter` —
  there's no agent-retry concept for it to slot next to, and giving it
  its own bounded infra-retry-then-ERROR treatment is real, valuable, and
  deliberately deferred rather than bolted on asymmetrically with the
  agent-driven commands in the same pass.
- **`verdict flaky`** — same Phase 10 deferral, still true: `run_flakiness`
  calls `run()` directly, so a `SandboxError` mid-trial still crashes the
  whole command. `--trials` runs measuring agent flakiness and this
  phase's infra-flake retry are two different kinds of "flaky" that
  happen to share a word; conflating them into one code path wasn't
  attempted.
- **A dedicated provenance bucket for ERROR** — `Verdict.status` still
  derives from `_proven_applicable()`/`budget_exceeded`/`error` exactly
  as Phase 10 left it; ERROR still short-circuits ahead of all of that
  rather than participating in the PROVEN/JUDGED signal model. Real
  future work (e.g. "which specific gate was mid-flight when the sandbox
  died") stays deferred.
- **Cross-run retry memory** — each `verdict bench`/`verdict run`
  invocation's error-retry budget is independent; there's no persistent
  "this repo's sandbox has failed N times across the last M runs, stop
  trying" circuit breaker. A human (or a wrapper script) still decides
  when a suite's infra is unfixably broken.

## Phase 12 — Defend the Thesis: Test-Integrity Detection

## The problem, restated

Every prior phase's grading logic rests on one unexamined assumption: that
a passing test suite at the final commit means the same thing it meant at
the base commit. It doesn't have to. An agent graded on "did the PROVEN
signals pass" has a strictly easier path to a green `test` signal than
fixing the actual bug — delete the failing assertion, mark the test
`xfail`, delete the test file outright, or just hardcode the literal the
buggy code already returns. None of that is hypothetical or exotic; it's
the single most obvious thing to try once "tests pass" is the metric.
Phase 12 is Verdict checking its own homework: comparing the agent's final
commit against the pre-agent base commit and flagging exactly that family
of moves, as its own PROVEN signal that can force NOT_DONE the same way a
real test failure already does.

## Where this had to be asked, not decided: the allowance trust boundary

Before writing any detection code, one question was raised rather than
answered silently, per explicit instruction: some tasks legitimately edit
tests ("add tests for the new endpoint" is a completely ordinary bug-fix-
adjacent request), and a strict-by-default integrity gate will flag that
task's own, intended work. The answer that came back combined two of the
offered options and added a constraint that shapes the whole
implementation:

1. **A task-declared allowance** (`TestChangeAllowance`, `integrity.py`) —
   implementable now, not deferred to a future phase.
2. **Documented as interim**, not a permanent design: Phase 13's held-out/
   hidden tests are the intended long-term answer to "how do you grade a
   test-writing task safely" — once acceptance lives in tests the agent
   can't see or edit, gaming the *visible* tests stops mattering for
   scoring purposes, and the allowance becomes a convenience for a human
   reading the report (skip an expected FAIL) rather than a security
   control the grade depends on.
3. **A hard trust-boundary rule, load-bearing today**: the allowance may
   ONLY come from a source outside the graded repo's own control —
   `task.yml` (benchmark-author-controlled, read by `suite/loader.py`
   before any agent worktree exists) in suite/bench mode, or an
   operator-supplied CLI flag (`--allow-test-changes`) in single-run/
   merge-gate mode. It must NEVER be settable from `verdict.yml` inside
   the graded repo/worktree, in any mode — an agent (run/bench) or a PR
   author (gate) who could flip a flag in a file they themselves control
   would simply disable their own integrity gate, the exact same failure
   mode Phase 10 already guarded against for the service image allowlist
   (`sandbox/services.py`'s `_SERVICE_IMAGES`, never a raw `image:` the
   repo names). `VerdictConfig`/`config.py` has no `allow_test_changes`
   field, on purpose, and `test_verdict_config_has_no_allow_test_changes_
   field` (`tests/test_integrity.py`) asserts that structurally, not just
   by convention.

`TestChangeAllowance` is a small, frozen dataclass: `allowed: bool` plus
an optional `allowed_paths: tuple[str, ...] | None` (glob patterns) for
scoping a permission to specific files rather than blanket-excusing every
test file in the repo. Default (`DENY_ALL`, a module-level constant) is
strict — `run()`/`grade_existing_diff()`/`run_with_retries()` all resolve
a `None` argument to `DENY_ALL`, never to something lenient, so a caller
that forgets to pass one gets the safe behavior. Scoped vs. blanket
matters for one more thing: a scoped allowance does NOT excuse a drop in
collected test count (see below) — the drop can't be attributed to just
the permitted files — while a blanket one does, since the whole test
surface was explicitly authorized to change.

## What gets checked, and how each reuses existing machinery

`integrity.py::check_test_integrity` runs once per attempt, after gates
and attribution (same slot `run_frontend_checks` already occupies, for
the same reason — see below), and emits exactly one `Signal(name=
"integrity", provenance=PROVEN, ...)`:

- **Modified/deleted test files** — `worktree.changed_files(repo, base,
  final)` (Phase 2's own diff primitive, unchanged) filtered to paths that
  look like test files, then `worktree.file_content_at(repo, ref, path)`
  (new — a thin `git show <ref>:<path>` wrapper) to tell "existed at base,
  gone now" from "existed at base, edited" from "didn't exist at base at
  all" (never flagged — adding a test is never suspicious). No AST
  diffing was invented for this: the codebase already had none (Phase 2's
  own `depgraph.py` is import-resolution only, not content diffing;
  `synth.py`'s docstring explicitly avoids hunk-level diffing as fragile)
  and this phase didn't need one either — whole-file before/after text is
  enough for every check below.
- **A drop in collected test count** — the only check that doesn't rely on
  diffing test-file content at all, which is exactly its point: it also
  catches an agent disabling test discovery *without* touching a
  `test_*.py` file (deleting `pytest.ini`, editing `package.json`'s test
  script) — a vector the file-content checks structurally can't see.
  Needed a new structured field, `Signal.tests_collected: int | None`
  (`schema.py`), because `gates/test.py`'s `PytestRunner`/`JestRunner`
  already computed this number (junit's `tests` attribute, Jest's
  `numTotalTests`) but only ever folded it into `detail`'s free text —
  Phase 12 needed to compare two runs' counts programmatically, not
  re-parse a sentence. `GoTestRunner` stays `None`: `go test`'s event
  stream has no equivalent single "collected" figure, an honest gap, not
  silently patched over. The base commit's count comes from
  `attribution.engine.base_gate_signals` — Phase 10's own base-state-cache
  render, pulled out of `_reproduces_at`'s inline logic into a reusable
  public function specifically so this phase could ask it "what did the
  `test` gate see at the base commit" without re-deriving Phase 10's
  caching from scratch. `_reproduces_at` itself is now three lines calling
  it — a straightforward extraction, not a behavior change (confirmed by
  the full existing attribution/cache test suites still passing unchanged).
- **Newly added skip/xfail markers** — a line-level diff (base file's
  lines as a set, scan the final file's lines for ones NOT in that set)
  against a pattern list covering pytest (`@pytest.mark.skip`,
  `@pytest.mark.xfail`, `pytest.skip(`), unittest (`@unittest.skip`,
  `self.skipTest(`), and JS test runners' `.skip(`/`x*(` conventions
  (`it.skip`, `xit`, `xdescribe`, ...).
- **Weakened or removed assertions** — two independent sub-checks, both
  heuristic and both explicitly documented as such (see below): a net
  DROP in assertion-keyword line count between the two file versions
  (`assert`, `self.assert*`, `expect(`, `.should.`), and a separate
  pattern match for assertions that got REPLACED by something trivially
  true (`assert True`, `assert 1`, `.assertTrue(True)`) — a net-count
  check alone would miss that swap, since a vacuous assertion still
  counts as "an assertion" by keyword.
- **Hardcoded expected outputs** — the one detector this phase couldn't
  make anything but a heuristic, on purpose documented as one in its own
  docstring: a line matching an equality assertion (`== <literal>`) whose
  non-literal prefix is byte-identical between base and final but whose
  literal changed. Catches `assert add(2, 3) == 5` → `assert add(2, 3) ==
  -1` without trying to know whether the new literal is "correct" (that's
  what re-running the `test` gate already checks) — it only flags that
  the *expectation itself* moved, which a legitimate spec change can also
  do. This is precisely why the allowance mechanism exists rather than an
  outright block.
- **A coverage drop** — genuinely best-effort, and the one place this
  phase adds real new execution cost: `measure_pytest_coverage` runs
  `pytest --cov=. --cov-report=term-missing` inside the sandbox and
  parses the `TOTAL ... NN%` line; returns `None` (never raises) on ANY
  failure — pytest-cov not installed, no importable package for `--cov`
  to target, a timeout. Comparison logic (`coverage_regression_finding`,
  a 2-point tolerance for run-to-run noise) is a pure function, decoupled
  from the real I/O on purpose so it's unit-testable without a sandbox at
  all. The base-commit measurement (`_measure_base_coverage`) mirrors
  `base_gate_signals`'s scratch-worktree pattern but is deliberately NOT
  cached — coverage isn't part of `sandbox/cache.py`'s schema, and adding
  it there is real, valuable, out-of-scope work (see below). Skipped
  outright (fast) for non-pytest repos via a cheap `PytestRunner().
  applicable()` check before ever invoking a subprocess.

Every finding is a `FailureLocation` (identity, file, `code` = the finding
kind, `message` = the human explanation) appended to the `integrity`
signal's `failures` list — reusing that existing field rather than
inventing a parallel structure is what makes a FAIL here render correctly
in the CLI/JSON/HTML reporters, the failure-mode breakdown, and (had it
failed) attribution's causal-analysis section, with zero changes to any
of them.

## Why "integrity" can force NOT_DONE with no schema change

`Verdict.status`'s FAIL check (`schema.py`) was already gate-name-agnostic
before this phase — `any(s.status is GateStatus.FAIL for s in applicable)`
never inspected `s.name`. So a `Signal(name="integrity", provenance=
PROVEN, status=FAIL, ...)` automatically forces `NOT_DONE` through the
exact same code path a real `test` FAIL already uses, confirmed rather
than assumed: `test_gutting_tests_forces_not_done_even_though_the_test_
gate_passes` (`tests/test_integrity.py`) runs a fake adapter that deletes
the assertion (the `test` signal genuinely PASSES — nothing left to fail)
through the real `runner.run()` pipeline and asserts `verdict.status is
NOT_DONE`. Nothing in `schema.py` needed to change for this to work — the
name-agnostic FAIL check was already exactly what Phase 12 needed.

## Where it slots into the pipeline, and why there

`check_test_integrity` runs in the same place, and for the same reason,
`run_frontend_checks` already does: AFTER `attribute_failures`, not passed
into it. `attribute_failures` bisects by calling `gates/registry.py`'s
`resolve_gate(gate, ...)`, which only knows the four `GATE_RUNNERS` names
(`test`/`typecheck`/`build`/`lint`) and would `KeyError` on `"integrity"`
— exactly the same reason frontend checks were already appended after
attribution rather than folded into `signals` before it (see that code's
own existing comment, extended this phase to name the integrity check
too). An `integrity` FAIL is real PROVEN evidence for `Verdict.status`
either way; it's just not (yet) bisectable to a culprit file the way
test/typecheck/build/lint failures are — a real, deferred gap (see below),
not a design flaw baked in this phase.

## Tests

`tests/test_integrity.py` — one fixture per detector (deleted test file,
weakened/removed assertion, added xfail marker, added skip marker,
hardcoded literal change, collected-count drop via a `pytest.ini` deletion
that never touches a `test_*.py` file, and a coverage drop both as a pure
function and wired through `check_test_integrity` with `measure_pytest_
coverage` stubbed), plus: the allowance letting a declared edit through
end to end, a path-scoped allowance NOT excusing an out-of-scope file, a
brand-new test file never being flagged, `run()`-level end-to-end proof
that gutting tests forces NOT_DONE even though the `test` gate itself
reports PASS, the same end-to-end proof that a *legitimate* test edit
reaches DONE only with an allowance and is flagged for review without one,
and `suite/loader.py` parsing `allow_test_changes` from `task.yml` in both
its bool and path-list forms (plus rejecting a malformed value, and
defaulting to deny-all without the key).

## What's explicitly out of scope for Phase 12

- **AST-aware or semantic diffing** — every content-based check here is a
  line/regex pattern match over whole-file before/after text, not a
  parsed, structural understanding of what an assertion actually claims.
  A sufficiently indirect rewrite (extracting the same weakened check into
  a helper function, restructuring control flow around a deleted
  assertion) can evade every heuristic here. Real, harder work; Phase
  13's held-out tests are the actual answer to "the agent can't game what
  it can't see," not a smarter diff.
- **Bisecting an `integrity` FAIL** — it's a real PROVEN signal that
  forces NOT_DONE, but `attribute_failures` never runs on it (see above);
  a report never says *which specific edit* caused the integrity flag the
  way it would for a real test regression, only that one exists and why
  (the finding list in `detail`/`failures`).
- **Coverage-drop caching** — every check that wants a coverage number
  pays the real `pytest --cov` cost fresh, at both commits; not folded
  into `sandbox/cache.py`'s base-state cache the way gate signals are.
- **Coverage for Jest/Go** — `measure_pytest_coverage` is pytest-only;
  a JS/TS or Go repo's integrity check simply never gets a coverage
  sub-finding, PASS or FAIL, rather than a wrong one.
- **`verdict gate`'s exit-code/ERROR handling untouched** — `--allow-
  test-changes` was added to `gate_cmd` for symmetry with `run_cmd`, but
  the command's pre-existing gaps (documented in Phase 11's own "out of
  scope") are unchanged: `grade_existing_diff` still doesn't get bounded
  infra-error retry, and an `integrity` FAIL there produces exit 1 exactly
  like any other `verdict.done is False`, no new exit code carved out for
  it specifically.
- **A generic "trusted config source" abstraction** — the allowance's
  trust boundary is enforced by construction (only `suite/loader.py` and
  `cli.py`'s CLI-flag parsing ever construct a non-`DENY_ALL`
  `TestChangeAllowance`; nothing reads one from `VerdictConfig`), not by
  a reusable "which config sources are trusted" framework. If a future
  phase needs a second trusted-vs-repo-controlled setting, it'll likely
  want to generalize this rather than copy it a third time — noted, not
  built.

## Phase 13 — Held-Out Acceptance Tests

## The problem, restated

Phase 12 defended against an agent gaming the tests it can see. It never
addressed the case one layer up: a task whose VISIBLE suite doesn't cover
the thing the task actually asked for. "Add retry logic to the HTTP
client" against a repo whose existing tests never call the retry path can
be "finished" by doing absolutely nothing — every PROVEN signal, integrity
included, reports clean, because there is nothing in the repo that could
possibly report otherwise. Passing the existing suite was never proof
that net-new work happened; it was only ever proof that nothing already-
tested broke. Phase 13 gives Verdict a way to actually assert the
positive claim, the same way SWE-bench does: a task can ship held-out
tests the agent never sees, applied only after it's finished, and grading
requires them to pass.

## The on-disk format, proposed before writing any grading code

Per the phase's own instruction, the format was designed and written down
(`acceptance.py`'s module docstring carries the canonical version) before
any of the grading logic below was implemented. The shape:

```text
my_suite/add-retry-logic/
  task.yml           # + fail_to_pass: [...], pass_to_pass: [...]
  tests.patch         # a real `git diff --no-color`, NOT copied into repo/
  repo/               # the agent's actual, patch-free starting point
```

Two new optional `task.yml` keys, `fail_to_pass`/`pass_to_pass` — each a
list of pytest node ids, spelled exactly the way `gates/test.py`'s own
`_node_id` already reconstructs them for attribution (`path/to/
test_x.py::test_name`), so a benchmark author who already has junit output
in front of them doesn't need to reformat anything. A sibling file,
`tests.patch`, holds a plain unified diff (exactly what `git apply`
already accepts, no custom format invented) that adds or modifies whatever
test files those node ids live in.

**Why a patch file, not a `hidden_tests/` directory of whole files** — the
other option seriously considered before writing code: a directory can't
express "add one more test function to an EXISTING visible test file"
without either duplicating that file's other content or inventing a merge
policy for combining a hidden copy with the agent's own edits to the same
file. A unified diff applies on top of whatever's already there — new
file or existing one — the same way any other patch does, so this case
falls out for free rather than needing special-casing. It also happens to
be exactly what SWE-bench's own dataset format already ships, which was
the second, smaller reason: a benchmark author converting an existing
SWE-bench-style task doesn't need to reshape anything, just copy `test_
patch` over as `tests.patch` and read off `FAIL_TO_PASS`/`PASS_TO_PASS`
into the equivalent `task.yml` keys.

`tests.patch` is read by `suite/loader.py` at suite-LOAD time — before any
agent worktree exists, the identical trust-boundary timing Phase 12's
`allow_test_changes` already established — and is never copied into
`repo/`. The agent's worktree is a checkout of `repo/` alone; it has no
path through which it could read, edit, or delete a test it's about to be
judged against, which is a stronger guarantee than Phase 12's integrity
gate can offer for the visible suite (that one can only *detect*
tampering after the fact; this one makes tampering structurally
impossible by never exposing the target at all).

## Grading semantics: what actually runs, and when

`acceptance.py::check_acceptance` — called once per attempt, in `runner.
run()`, right after Phase 12's integrity check, in its OWN fresh `scratch_
worktree`/sandbox at the agent's `attempt_commit` (never the agent's own
worktree — applying `tests.patch` mutates the working tree, and every
check that already ran against that worktree would be contaminated by a
patch it never expected):

1. Apply `tests.patch` to the scratch worktree. A patch that doesn't apply
   cleanly (the agent's own edits conflicted with a file the hidden tests
   touch, or the patch is stale) is a real, evaluable outcome — a PROVEN
   `acceptance` FAIL with the git error in `detail`, never an ERROR and
   never silently ignored.
2. Run `pytest` scoped to EXACTLY the declared node ids (`FAIL_TO_PASS` +
   `PASS_TO_PASS` together, one invocation) — not the whole suite. Parsing
   reuses `gates/test.py`'s own `_node_id` (imported, not re-derived) to
   map junit's `classname`/`name` back to the same node-id spelling
   `task.yml` was written in.
3. Every declared id must come back a real pytest PASS. FAIL, ERROR,
   `<skipped>`, and "never appeared in the junit report at all" (a typo'd
   node id, or a test the patch was supposed to add but didn't) are all
   treated identically — not-PASS fails the whole signal, no partial
   credit.

**Deliberately NOT re-verified**: that a `FAIL_TO_PASS` id actually fails
against the unpatched base commit at grading time. That's the benchmark
author's responsibility at task-authoring time (the name is a promise
about the dataset, not a runtime assertion); re-deriving it here would
cost a second full sandboxed run per grading pass, for a check whose
only purpose is catching a malformed benchmark task, not grading an
agent. SWE-bench's own harness makes the identical simplification for the
identical reason.

**`SandboxError` while provisioning the scratch sandbox is deliberately
NOT caught** inside `check_acceptance` — the one place this module departs
from Phase 12's coverage sub-check, which IS explicitly best-effort and
swallows failures. Acceptance is the authoritative signal once a task
declares it; degrading silently on infra trouble would mean exactly the
"quietly report success anyway" failure mode Phase 11's `ERROR` status
exists to prevent. It propagates to `runner.py`'s ordinary `_EVALUATION_
ERRORS` handling instead — retried, then reported `ERROR`, like any other
infra failure, never treated as an implicit PASS.

## Why "acceptance" can force NOT_DONE with no schema change, again

Same fact Phase 12 already established and leaned on: `Verdict.status`'s
FAIL check (`schema.py`) has never inspected `Signal.name`, only
`provenance`/`status`. A `Signal(name="acceptance", provenance=PROVEN,
status=FAIL)` forces `NOT_DONE` through the exact same generic path a
real `test` FAIL or Phase 12's `integrity` FAIL already do. Confirmed, not
assumed: `test_visible_suite_passes_but_held_out_fail_to_pass_fails_is_
not_done` (`tests/test_acceptance.py`) builds a repo whose visible suite
is `assert True` — genuinely, structurally unable to fail — runs a
no-op agent through the real `runner.run()` pipeline with a held-out
`FAIL_TO_PASS` test still failing, and asserts `verdict.status is NOT_
DONE` even though the `test` signal itself reports PASS.

## How this resolves Phase 12's open problem

Phase 12's own DESIGN.md section named the tradeoff explicitly: a
strict-by-default integrity gate flags even a legitimate "add tests for
X" task, and the mitigation shipped then (`TestChangeAllowance`, a
task-declared, trust-boundary-guarded exception) was documented as
interim — "once acceptance lives in tests the agent can't see or edit,
gaming the visible tests stops mattering for grading purposes." Phase 13
is that promise kept, not a new mechanism bolted on top of the old one:
a task with real `fail_to_pass`/`pass_to_pass` tests doesn't need
`allow_test_changes` at all, because the tests that actually decide
DONE/NOT_DONE were never in the agent's worktree to edit in the first
place. The two mechanisms stay independent (Phase 12's integrity gate
still watches the visible suite when it exists — worth a human's
attention even when it isn't decisive) rather than one replacing the
other in code; a benchmark author picks whichever fits a given task, or
both.

## What's explicitly out of scope for Phase 13

- **Non-pytest acceptance** — `check_acceptance` only knows how to run
  targeted pytest node ids. A Jest/Go task can't declare `fail_to_pass`/
  `pass_to_pass` yet; doing so is real, valuable follow-on work structured
  the same way `gates/test.py` already handles multiple tools, just not
  built this phase.
- **Re-verifying `FAIL_TO_PASS` fails at the base commit** — see above;
  a benchmark-authoring-time responsibility, not a grading-time check.
- **`verdict gate`/`grade_existing_diff`** — no task directory exists in
  merge-gate mode to source a `tests.patch` from, so acceptance checking
  isn't wired in there at all, the same scoping Phase 12's `allow_test_
  changes` CLI flag drew for that command's other new capability.
- **Caching the scratch-worktree/sandbox acceptance run** — unlike Phase
  10's base-state cache (keyed on a `base_commit` many callers share
  across a run), an acceptance check's inputs (`attempt_commit` +
  `tests.patch`) are unique to one attempt by construction, so there's no
  reuse value the caching machinery elsewhere in this codebase exists to
  capture.
- **Suite-authoring tooling** — no `verdict suite generate-patch` or
  similar helper for producing `tests.patch` from a scratch checkout; a
  benchmark author runs `git diff` themselves, per this phase's own
  format docstring.
