<div align="center">

# Verdict

**Does your AI coding agent actually work — or does it just say it does?**

Verdict is an open-source harness that grades AI coding agents on *executable truth*, not opinion.
It runs your repo's real tests, typecheck, build, and lint; verifies the frontend in a real browser;
pinpoints the exact action that caused each failure; and ranks agents by **pass-rate-per-dollar** on *your* codebase.

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Agents](https://img.shields.io/badge/agents-Claude%20Code%20%7C%20Cursor%20%7C%20Codex%20%7C%20Aider%20%7C%20OpenHands-orange)](#supported-agents)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-informational)](#contributing)

</div>

---

## Why Verdict exists

AI coding agents are confident. They report "✅ Done — I fixed the bug and updated the tests," and often they're right. But the industry has a trust problem: only ~29% of developers trust AI output today, down from 40% in 2024. The reason is simple — **"done" is usually self-reported, and self-reports don't compile.**

Most eval tools answer "does the agent's output *look* correct?" by asking another LLM. That's circular: you're trusting an opinion to check an opinion. Verdict answers a harder, more useful question — **"is the agent's work *actually* correct, and if not, what specifically did it break, and what did it cost to get there?"** — by grounding every verdict in code that runs.

Verdict is agent-agnostic, runs locally with one command, and drops into CI as a merge gate that blocks an agent's PR when the agent's own work doesn't pass.

## The core idea: Proven vs. Judged

Every verdict Verdict emits is explicitly labeled by how it was reached. This is the whole philosophy in one table:

| Bucket | How it's decided | Examples | Trust |
| --- | --- | --- | --- |
| **PROVEN** | Executed. Deterministic. Reproducible. | tests pass, build succeeds, typecheck clean, DOM node present, the button actually navigates | Ground truth |
| **JUDGED** | An LLM/vision model formed an opinion | "the refactor reads cleanly," "the CTA looks green and above the fold" | Advisory, and *labeled as such* |

Verdict reserves the model's opinion for the genuinely subjective residue — style, intent, aesthetics — and never lets a *judged* signal masquerade as a *proven* one. When a run passes every test but a vision check says the UI intent wasn't met, Verdict counts it as **not done** — and tells you which half failed.

## Quickstart

```bash
# install
pipx install verdict-eval        # or: npm i -g verdict-eval

# grade a single task, using whatever agent you like
verdict run \
  --task "make the CTA green and move it above the fold" \
  --agent claude-code \
  --repo .

# grade a whole benchmark suite and rank configs by cost-to-correct
verdict bench ./verdict/suite --agents claude-code,cursor,aider
```

Verdict clones your repo into an isolated git worktree per attempt (so nothing touches your working tree), lets the agent do its work, then runs the verification pipeline against the result.

## What a verdict looks like

```text
$ verdict run --task "make the CTA green and move it above the fold" --agent claude-code

  Verdict — run #42   repo: acme/storefront   agent: claude-code   attempts: 3

  PROVEN  (executed)
    ✓ build            next build                       passed
    ✓ typecheck        tsc --noEmit                     passed
    ✓ lint             eslint                           passed
    ✓ unit tests       142/142                          passed
    ✓ dom assertion    <button data-testid="cta">       present · class "btn-primary--green"
    ✓ interaction      click CTA → /checkout            navigation fired
    ✗ e2e regression   header nav @ 375px viewport      FAILED

  JUDGED  (vision model)
    ~ visual intent    0.72   "CTA is green ✓; 'above the fold' met at 1440px,
                               but below the fold at 375px"

  CAUSAL ANALYSIS
    ✗ e2e regression → header nav collapses on mobile
        agent edited  src/components/Hero.tsx  (added `position: absolute` to .cta-wrap),
        which removed the flow height the header relied on → nav overlaps at ≤ 414px.
        first bad commit: a3f9c21 · not covered by any pre-existing test

  COST-TO-CORRECT
    attempts 3 (2 failed, 1 partial) · tokens 214k · cost $1.38
    pass-rate-per-dollar: 0.52 verdict-pts/$   (rank 2 of 4 configs)

  VERDICT:  NOT DONE  — passes all tests, fails responsive layout + partial visual intent
```

That single screen is the product: a proven/judged split, a *causal* explanation instead of a raw log dump, and a cost figure that reflects reality including retries and dead ends.

## How it works

Verdict runs a four-stage pipeline. Every stage tags its outputs `proven` or `judged`.

### 1 · Executable grounding (backend truth)
Verdict discovers and runs your repo's own quality gates — test runner, typecheck, build, lint — inside the isolated worktree, capturing exit codes and structured, parsed output (not just stdout). These are the bedrock `proven` signals. If your repo says it's green, it's green; if a test fails, Verdict has the failure, the stack, and the location.

### 2 · Frontend truth
The failure class that slips through every backend test: the change that compiles, passes CI, and still ships a button that doesn't render or a style that got overridden. Verdict spins up the app in a headless browser (Playwright) and, in order of decreasing trust:

- **DOM assertion** *(proven)* — the intended change actually reached the rendered DOM (the node exists, with the expected attributes/classes).
- **Interaction drive** *(proven)* — Verdict performs the real user action and asserts the real outcome (click the new CTA → the navigation/network call fires).
- **Screenshot diff** *(proven, perceptual)* — before/after render diff to catch *unexpected* regressions elsewhere on the page, using perceptual thresholds (not raw pixels) to stay stable across fonts and anti-aliasing.
- **Visual-intent judge** *(judged)* — a vision model scores the rendered screenshot against the task description ("was the CTA made green and moved above the fold?"). This is explicitly an opinion and is bucketed as such.

A run that passes the tests but fails the DOM/interaction check is counted as **not done** — because it isn't.

### 3 · Causal failure analysis
Verdict doesn't hand you a trace and wish you luck. For each failing `proven` check it intersects three things — the agent's actual diff (files and lines it touched), the failure's location (stack/first-bad-commit via bisection), and the repo's dependency graph — to attribute the failure to a *specific agent action*: *"edited `auth.py` but never updated the login fixture, so `test_login` fails at line 44."* The causal link (which action, which check) is `proven`; the natural-language phrasing is the only place a constrained LLM is allowed to help, and it's grounded in the executed facts, never inventing them.

### 4 · Cost-to-correct
Raw token cost is a vanity metric — it ignores that agents retry, backtrack, and abandon branches. Verdict tracks **total** spend across every attempt (including the failed and abandoned ones) and reframes it as **pass-rate-per-dollar**: how much *verified-correct work* you get per dollar, per agent, per model, per config, on your own repo. That's the number that actually tells you which setup to pay for.

## CI gate

Block a merge when an agent's own work doesn't pass — proven checks are required, judged checks are advisory by default.

```yaml
# .github/workflows/verdict.yml
name: verdict-gate
on: [pull_request]
permissions:
  contents: read
  pull-requests: write   # only needed for the advisory judged-signal comment
jobs:
  verdict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0   # bisection/attribution needs the PR's base ref history
      - uses: verdict-ai/action@v1
        with:
          install-frontend-extra: true   # only if verdict.yml configures frontend checks
```

Any failing PROVEN signal (test/typecheck/build/lint/frontend) fails the check — that's the whole gate policy, and it isn't configurable per-repo (see DESIGN.md's Phase 6 section for why). JUDGED signals never affect the check; they're posted as a separate, advisory PR comment.

## Supported agents

Verdict is agent-agnostic via a small pluggable adapter interface — it drives the agent, captures its diff and token accounting, and grades the result. Bring your own by implementing `Adapter.run(task, worktree) -> AttemptResult`.

| Agent | Status |
| --- | --- |
| Claude Code | ✅ |
| Cursor | ✅ |
| Codex | ✅ |
| Aider | ✅ |
| OpenHands | ✅ |
| *your agent* | 🔌 implement one method |

## Configuration

Verdict autodetects most repos; override anything in `verdict.yml`:

```yaml
# verdict.yml
gates:
  test:      "pytest -q"          # auto-detected if omitted
  typecheck: "tsc --noEmit"
  build:     "next build"
  lint:      "eslint ."

frontend:
  start:     "npm run dev"
  url:       "http://localhost:3000"
  viewports: [1440, 768, 375]     # responsive checks
  vision_model: "gpt-4-class"     # judged bucket only
  glitch_scan: true               # frame-burst flicker/never-settled detection, video on failure

cost:
  price_per_1k_tokens: { input: 0.003, output: 0.015 }
```

Report format is a CLI flag, not a `verdict.yml` key: `--report cli --report json --report html --output-dir verdict-report` (repeatable; any combination). `json`/`html` write `verdict-report.json`/`verdict-report.html` — the HTML file is a single, self-contained dashboard (inline CSS/JS, no external requests) suitable as a CI artifact.

## Benchmark suites

Point Verdict at a folder of tasks to turn it from a one-shot checker into a scorecard. Each task is a real change with executable acceptance criteria; Verdict runs every agent/model/config against all of them and produces a ranked, `pass-rate-per-dollar` leaderboard plus a failure-mode breakdown (which agents hallucinate APIs, which break responsive layout, which never update fixtures). Ships with a starter suite of bug-fix, refactor, and feature-add tasks; add your own by dropping a folder in.

```bash
verdict bench --suite examples/starter_suite --agent mock --agent claude-code
```

A suite task is just a `verdict run` with its repo and task text pre-wired — see `examples/starter_suite/*/task.yml` for the format, and DESIGN.md's Phase 5 section for why acceptance criteria are never written as prose.

## Statistical rigor

Two diagnostics, neither of them a merge gate — see DESIGN.md's Phase 7 section for the full reasoning.

```bash
# How often does the vision judge actually agree with a human reviewer?
verdict calibrate --dataset examples/calibration_dataset/manifest.json --judge mock

# Run the same task N independent times; report pass rate with a Wilson confidence interval.
verdict flaky --task "fix the bug" --agent mock --repo examples/sample_repo --trials 10 --json baseline.json

# Later: is a pass-rate change a real regression, or noise from a small sample?
verdict flaky --task "fix the bug" --agent mock --repo examples/sample_repo --trials 10 --compare-to baseline.json
```

`calibrate` warns (never fails the process) when a judge's concordance with human labels drops below `--threshold` (default 95%). `flaky --compare-to` runs a real two-proportion z-test, not a raw percentage diff, so a ten-point pass-rate swing across a handful of trials gets called `NOISE` rather than a false-alarm `REGRESSION`.

## Roadmap

- [x] Executable grounding: test / typecheck / build / lint in isolated worktrees
- [x] Proven-vs-judged verdict schema
- [x] Causal failure attribution (action → failing check)
- [x] Cost-to-correct / pass-rate-per-dollar leaderboard
- [x] Frontend truth: DOM + interaction + perceptual diff + vision-intent judge
- [x] CI merge gate (GitHub Action)
- [x] Adapters: Claude Code, Cursor, Codex, Aider, OpenHands
- [x] Judge calibration report (concordance vs. human labels, target ≥ 95%)
- [x] Flakiness detection (multi-seed variance, confidence intervals on pass-rate)
- [ ] Hosted dashboard + historical regression tracking

## How Verdict relates to prior work

Execution-based evaluation of coding agents was pioneered by benchmarks like **SWE-bench** — run the real tests to verify a patch — on *frozen* task sets. Verdict brings that rigor to *your* repo as a daily-driver dev and CI tool, and adds three things those benchmarks don't: **causal failure attribution**, a **cost-to-correct** economic metric, and **frontend truth** (most coding-agent evals stop at the backend and never open a browser). The `proven`/`judged` split is Verdict's answer to the LLM-as-judge trust problem: use the model where opinion is unavoidable, execute everywhere else, and always tell the user which is which.

## Contributing

Issues and PRs welcome — especially new agent adapters and benchmark tasks that capture a real failure mode you've hit. See `CONTRIBUTING.md`.

## License

MIT © Siddharth Balaji

<div align="center">
<sub>Verdict — because "✅ Done" isn't a test result.</sub>
</div>