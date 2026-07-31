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
