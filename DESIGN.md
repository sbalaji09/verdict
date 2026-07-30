# Verdict — Phase 0 Design

Phase 0's goal: prove the spine works end to end — isolate, drive an agent,
verify with something *executed*, report a verdict — before any of the
richer phases (causal attribution, cost, frontend truth) get added on top.
Everything below is scoped to what's actually built, not the eventual product.

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

class Signal(BaseModel):
    name: str
    provenance: Provenance
    passed: bool
    detail: str
    command: str | None = None
    exit_code: int | None = None

class Verdict(BaseModel):
    task: str
    agent: str
    repo: str
    attempt: AttemptResult
    signals: list[Signal]

    @property
    def done(self) -> bool:
        proven = [s for s in self.signals if s.provenance is Provenance.PROVEN]
        if not proven:
            return False
        return all(s.passed for s in proven)
```

Every `Signal` — from Phase 0's single test gate today, to typecheck/build/
lint in Phase 1, to a vision-intent judge in Phase 4 — carries its
`Provenance` from the moment it's created. There's no separate "trust
level" field bolted on afterward, and no code path that reads a judged
opinion as if it were an executed fact: `Verdict.done` only ever consults
`PROVEN` signals. A `JUDGED` signal can be glowing and it still can't turn
a failing proven check into `done`, and it can't manufacture `done` on its
own either — a verdict with zero proven signals is *not done*, because
there's nothing executed to ground the claim in. This is the one property
the whole project is organized around, so it's enforced in the type that
every downstream consumer (CLI, reporter, and — eventually — the CI gate)
reads, not just as a convention.

`Signal` and `Verdict` already carry the shape Phase 4 needs (a `JUDGED`
signal is just a `Signal` with `provenance=JUDGED` and a model's rationale
in `detail`) — no schema migration required to add the vision judge later.

## Phase 0's one gate

`gates/test_gate.py` does exactly one thing: run the repo's test command
and turn the exit code into a `PROVEN` `Signal`. Command resolution order:
explicit `--test-cmd` override → `pytest.ini`/`setup.cfg` present →
`pyproject.toml` has a `[tool.pytest...]` table → `package.json` has a
`test` script → a `tests/` directory with `test_*.py` files → give up.
"Give up" is itself reported as a failed, *proven* signal ("no test command
found" is deterministic and reproducible too) rather than silently
skipping the gate.

Phase 1 adds typecheck/build/lint as siblings of this module; Phase 0
deliberately stops at one gate so the spine — worktree → adapter → gate →
schema → report — is fully wired and provably correct before breadth gets
added.

## Demo repo

`examples/sample_repo/` is a two-file Python package with one seeded bug
(`add()` subtracts) and one failing test. It's a real, standalone git repo
(bootstrap once with `examples/sample_repo/setup.sh`) so `--repo` has
something genuine to worktree-isolate, rather than a fixture invented just
for the CLI demo.

## What's explicitly out of scope for Phase 0

- Only one gate (test). Typecheck/build/lint: Phase 1.
- No causal attribution of *why* a test failed to a specific edit: Phase 2.
- No cost-to-correct / pass-rate-per-dollar aggregation across attempts —
  `AttemptResult` captures cost per attempt, but nothing rolls it up yet:
  Phase 3.
- No frontend/browser checks at all: Phase 4.
- No retry loop, no multi-attempt orchestration — one task, one agent, one
  attempt, one verdict.
