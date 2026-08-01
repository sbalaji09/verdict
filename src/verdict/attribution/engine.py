"""Orchestrates attribution for a finished run: for each failing PROVEN
signal, and each individual failure within it, produce an Attribution.

The algorithm, in order:
  0. Baseline — did this exact failure already reproduce before the agent
     touched anything? If so, it's PRE_EXISTING; nothing below runs.
  1. Level-1 bisect — over the agent's *real* commits (if it made more than
     one), find the first commit where the failure starts reproducing.
  2. Level-2 bisect — within that commit's changed files (synthesized as a
     one-file-per-commit ladder), find the exact culprit file.
  3. Enrich with a static dependency-graph edge, if one exists, between the
     failure's file and the culprit file.
  4. Render a template sentence over the structured result — the sentence
     is a rendering of steps 0-3, never a new claim.

See DESIGN.md for the full walkthrough and why each step is shaped this way.
"""

from __future__ import annotations

from pathlib import Path

from verdict.attribution.bisect import run_bisect
from verdict.attribution.depgraph import build_dependency_graph, find_dependency_depth
from verdict.attribution.reproduce import Reproduction, reproduction_from_signal
from verdict.attribution.synth import build_synthetic_ladder
from verdict.config import load_config
from verdict.gates.registry import run_all_gates
from verdict.sandbox import SandboxConfig
from verdict.sandbox.base import SandboxError
from verdict.sandbox.cache import (
    DEFAULT_CACHE_DIR,
    cache_key,
    compute_lockfile_hash,
    load_gate_signals,
    save_gate_signals,
)
from verdict.sandbox.config import create_sandbox
from verdict.sandbox.install import run_setup_step
from verdict.schema import (
    Attribution,
    AttributionKind,
    DependencyLink,
    FailureLocation,
    GateStatus,
    Provenance,
    Signal,
)
from verdict.worktree import (
    Worktree,
    changed_files,
    commits_between,
    copy_vendored_dependencies,
    parent_commit,
    scratch_worktree,
)

# Bounds cost on a badly-broken repo: attributing 40 failing lint rules is
# rarely more useful than attributing the first several, and each one costs
# real bisection re-runs. Not silent — report.py notes how many were cut.
MAX_ATTRIBUTIONS_PER_GATE = 5


def attribute_failures(
    repo: Path,
    worktree: Worktree,
    final_commit: str,
    signals: list[Signal],
    sandbox_config: SandboxConfig | None = None,
) -> list[Attribution]:
    """`final_commit` must be a commit capturing exactly what the agent
    changed — captured by the caller *before* any gate ran, so gate
    byproducts (e.g. `__pycache__`) never get mistaken for agent edits.
    """
    sandbox_config = sandbox_config or SandboxConfig()
    attributions: list[Attribution] = []
    for signal in signals:
        if signal.provenance is not Provenance.PROVEN or signal.status is not GateStatus.FAIL:
            continue
        targets: list[FailureLocation | None] = (
            list(signal.failures) if signal.failures else [None]
        )
        for target in targets[:MAX_ATTRIBUTIONS_PER_GATE]:
            attributions.append(
                _attribute_one(repo, worktree, final_commit, signal.name, target, sandbox_config)
            )
    return attributions


def _reproduces_at(
    repo: Path, ref: str, gate: str, identity: str | None, sandbox_config: SandboxConfig
) -> Reproduction:
    """Phase 10: consults the base-state cache before rendering anything.
    `ref` is always `worktree.base_commit` in practice — the one thing
    that never changes across however many times this gets called in a
    single `attribute_failures` run (once per failing signal being
    attributed) — so a cache hit here is the common case after the first
    call, not a rare one.

    On a MISS, renders every gate at once (not just `gate`) and caches
    the full set — the next failing signal's baseline check, whatever
    gate it's against, then hits the cache too. Any failure along the way
    (a `SandboxError`, or anything else genuinely unanticipated) degrades
    to SKIP/untestable rather than raising — this baseline check has
    always been resilient by design (a failed baseline check just means
    "fall through to bisection," not "abort the attempt"), and Phase 10
    doesn't change that.
    """
    lockfile_hash = compute_lockfile_hash(repo, ref)
    key = cache_key(ref, lockfile_hash, sandbox_config.image)

    cached = load_gate_signals(DEFAULT_CACHE_DIR, key)
    if cached is not None and gate in cached:
        return reproduction_from_signal(cached[gate], identity)

    try:
        with scratch_worktree(repo, ref) as wt:
            copy_vendored_dependencies(repo, wt)
            setup_env = run_setup_step(wt, sandbox_config)
            config = load_config(wt)
            with create_sandbox(wt, sandbox_config) as sandbox:
                signals, _ = run_all_gates(
                    wt,
                    config,
                    sandbox=sandbox,
                    timeout_seconds=sandbox_config.gate_timeout_seconds,
                    env=setup_env,
                )
    except SandboxError:
        return Reproduction.SKIP
    except Exception:
        return Reproduction.SKIP

    gate_signals = {signal.name: signal for signal in signals}
    save_gate_signals(DEFAULT_CACHE_DIR, key, gate_signals)

    target_signal = gate_signals.get(gate)
    if target_signal is None:
        return Reproduction.SKIP
    return reproduction_from_signal(target_signal, identity)


def _location(target: FailureLocation | None) -> str | None:
    if target is None:
        return None
    if target.file and target.line:
        return f"{target.file}:{target.line}"
    return target.file


def _attribute_one(
    repo: Path,
    worktree: Worktree,
    final_commit: str,
    gate: str,
    target: FailureLocation | None,
    sandbox_config: SandboxConfig,
) -> Attribution:
    identity = target.identity if target else None
    failure_id = identity or gate
    failure_location = _location(target)

    if _reproduces_at(repo, worktree.base_commit, gate, identity, sandbox_config) is Reproduction.BAD:
        return Attribution(
            kind=AttributionKind.PRE_EXISTING,
            check_name=gate,
            failure_id=failure_id,
            failure_location=failure_location,
            method="baseline",
            explanation=(
                f"{gate} ({failure_id}) was already failing before this attempt — "
                "not caused by the agent."
            ),
        )

    commits = commits_between(repo, worktree.base_commit, final_commit)
    culprit_commit: str
    if len(commits) <= 1:
        culprit_commit = commits[0] if commits else final_commit
        method = "single change (no intermediate commits to bisect)"
    else:
        bisected_commit = run_bisect(
            repo, worktree.base_commit, final_commit, gate, identity, sandbox_config
        )
        if bisected_commit is None:
            return _inconclusive(gate, failure_id, failure_location, "bisect(commit)")
        culprit_commit = bisected_commit
        method = "bisect(commit)"

    parent = parent_commit(repo, culprit_commit)
    files = changed_files(repo, parent, culprit_commit)
    if not files:
        return _inconclusive(gate, failure_id, failure_location, method)

    culprit_file: str | None
    if len(files) == 1:
        culprit_file = files[0]
    else:
        with build_synthetic_ladder(repo, parent, culprit_commit, files) as (bad_sha, sha_to_file):
            file_culprit_commit = run_bisect(repo, parent, bad_sha, gate, identity, sandbox_config)
            culprit_file = sha_to_file.get(file_culprit_commit) if file_culprit_commit else None
        if culprit_file is None:
            return _inconclusive(gate, failure_id, failure_location, "bisect(file)")
        method = "bisect(file)"

    dependency_link = None
    if target and target.file:
        graph = build_dependency_graph(worktree.path)
        depth = find_dependency_depth(graph, target.file, culprit_file)
        if depth is not None:
            dependency_link = DependencyLink(from_file=target.file, to_file=culprit_file, depth=depth)

    return Attribution(
        kind=AttributionKind.REGRESSION,
        check_name=gate,
        failure_id=failure_id,
        failure_location=failure_location,
        culprit_file=culprit_file,
        method=method,
        dependency_link=dependency_link,
        explanation=_render_regression(gate, failure_id, culprit_file, dependency_link),
    )


def _inconclusive(gate: str, failure_id: str, failure_location: str | None, method: str) -> Attribution:
    return Attribution(
        kind=AttributionKind.INCONCLUSIVE,
        check_name=gate,
        failure_id=failure_id,
        failure_location=failure_location,
        method=method,
        explanation=f"could not isolate a single cause for {gate} ({failure_id}) failing.",
    )


def _render_regression(
    gate: str, failure_id: str, culprit_file: str, dependency_link: DependencyLink | None
) -> str:
    sentence = f"agent edited {culprit_file}, which caused {gate} to fail at {failure_id}."
    if dependency_link:
        sentence += f" ({dependency_link.from_file} depends on {dependency_link.to_file}.)"
    return sentence
