from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from verdict.adapters.mock import MockAdapter
from verdict.config import config_for_package, load_config
from verdict.monorepo import PackageSelectionError, detect_sibling_candidates, resolve_package
from verdict.runner import grade_existing_diff, run, run_with_retries
from verdict.schema import GateStatus, VerdictStatus


def _init_git(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)


def _signal(verdict, name: str):
    return next(s for s in verdict.signals if s.name == name)


# --- resolve_package unit tests -------------------------------------------------


def test_resolve_package_defaults_to_root_for_a_normal_single_project_repo(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n")
    config = load_config(tmp_path)
    assert resolve_package(tmp_path, config, None) is None


def test_resolve_package_honors_explicit_request(tmp_path: Path) -> None:
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "api").mkdir(parents=True)
    config = load_config(tmp_path)
    assert resolve_package(tmp_path, config, "services/api") == "services/api"


def test_resolve_package_rejects_a_nonexistent_explicit_request(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    with pytest.raises(PackageSelectionError):
        resolve_package(tmp_path, config, "does/not/exist")


def test_resolve_package_auto_selects_a_single_declared_package(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "verdict.yml").write_text("packages:\n  api:\n    gates:\n      test: 'pytest'\n")
    config = load_config(tmp_path)
    assert resolve_package(tmp_path, config, None) == "api"


def test_resolve_package_requires_a_choice_among_declared_packages(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "web").mkdir()
    (tmp_path / "verdict.yml").write_text("packages:\n  api: {}\n  web: {}\n")
    config = load_config(tmp_path)
    with pytest.raises(PackageSelectionError, match="multiple packages"):
        resolve_package(tmp_path, config, None)


def test_resolve_package_flags_an_undeclared_ambiguous_monorepo(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "pyproject.toml").write_text("[tool.pytest]\n")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "package.json").write_text("{}")
    config = load_config(tmp_path)
    assert set(detect_sibling_candidates(tmp_path)) == {"api", "web"}
    with pytest.raises(PackageSelectionError, match="monorepo"):
        resolve_package(tmp_path, config, None)


def test_resolve_package_does_not_guess_a_single_nested_project(tmp_path: Path) -> None:
    # No root markers, exactly one candidate — still not a guess; behaves
    # exactly like a repo with no detected stack (root-scoped, honest NA),
    # not an error and not an auto-selected package.
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "pyproject.toml").write_text("[tool.pytest]\n")
    config = load_config(tmp_path)
    assert resolve_package(tmp_path, config, None) is None


# --- config layering --------------------------------------------------------


def test_config_for_package_layers_package_overrides_over_root(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "verdict.yml").write_text(
        "gates:\n  lint: 'root-lint'\n"
        "packages:\n  api:\n    gates:\n      test: 'pytest -k api'\n"
    )
    root_config = load_config(tmp_path)
    pkg_config = config_for_package(root_config, "api")
    assert pkg_config.override_for("test") == "pytest -k api"
    # the root-wide lint override still applies — a package layers on top,
    # it doesn't replace the root config wholesale.
    assert pkg_config.override_for("lint") == "root-lint"


def test_config_for_package_reads_the_packages_own_directory_verdict_yml(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "verdict.yml").write_text("gates:\n  test: 'pytest -k from-directory'\n")
    (tmp_path / "verdict.yml").write_text("packages:\n  api: {}\n")
    root_config = load_config(tmp_path)
    pkg_config = config_for_package(root_config, "api")
    assert pkg_config.override_for("test") == "pytest -k from-directory"


def test_config_for_package_inline_root_block_wins_over_the_directory_file(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "verdict.yml").write_text("gates:\n  test: 'from-directory'\n")
    (tmp_path / "verdict.yml").write_text(
        "packages:\n  api:\n    gates:\n      test: 'from-inline-block'\n"
    )
    root_config = load_config(tmp_path)
    pkg_config = config_for_package(root_config, "api")
    assert pkg_config.override_for("test") == "from-inline-block"


def test_config_for_package_is_a_no_op_without_a_package(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text("gates:\n  test: 'pytest'\n")
    config = load_config(tmp_path)
    assert config_for_package(config, None) is config


# --- end-to-end: a real monorepo fixture ------------------------------------


@pytest.fixture
def monorepo(tmp_path: Path) -> Path:
    """Two independent Python packages under one repo, no markers at the
    root — the undeclared-ambiguous-monorepo shape `resolve_package` must
    refuse to guess at, and the shape `--package`/`packages:` resolve.
    """
    repo = tmp_path / "repo"
    (repo / "services" / "api").mkdir(parents=True)
    (repo / "services" / "api" / "calculator.py").write_text(
        "def add(a, b):\n    return a - b  # bug\n"
    )
    (repo / "services" / "api" / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "services" / "api" / "pytest.ini").write_text("[pytest]\n")

    (repo / "apps" / "web").mkdir(parents=True)
    (repo / "apps" / "web" / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "apps" / "web" / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "apps" / "web" / "pytest.ini").write_text("[pytest]\n")

    _init_git(repo)
    return repo


def test_ungraded_package_selection_reports_error_not_a_guess(monorepo: Path) -> None:
    # `run()` itself raises PackageSelectionError (see `_EVALUATION_ERRORS`
    # in runner.py); `run_with_retries` is what turns it into a real,
    # reportable ERROR Verdict rather than crashing the caller.
    adapter = MockAdapter(patches={"NOTES.md": "no-op\n"})
    task_run = run_with_retries(task="fix add()", repo=monorepo, adapter=adapter, max_error_retries=0)
    verdict = task_run.final

    assert verdict.status is VerdictStatus.ERROR
    assert "monorepo" in (verdict.error or "")
    assert "services" in (verdict.error or "") and "apps" in (verdict.error or "")


def test_package_flag_grades_only_the_selected_package(monorepo: Path) -> None:
    adapter = MockAdapter(
        patches={"services/api/calculator.py": "def add(a, b):\n    return a + b\n"}
    )
    verdict = run(task="fix api's add()", repo=monorepo, adapter=adapter, package="services/api")

    assert verdict.status is VerdictStatus.DONE
    assert _signal(verdict, "test").status is GateStatus.PASS


def test_package_flag_does_not_leak_the_other_packages_failures(monorepo: Path) -> None:
    # web's calculator is already correct; api's is still broken — grading
    # web only must not see api's failure at all.
    adapter = MockAdapter(patches={"NOTES.md": "no-op\n"})
    verdict = run(task="no-op on web", repo=monorepo, adapter=adapter, package="apps/web")

    assert verdict.status is VerdictStatus.DONE
    assert _signal(verdict, "test").status is GateStatus.PASS


def test_declared_packages_block_lets_a_single_package_repo_resolve_without_the_flag(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "services" / "api").mkdir(parents=True)
    (repo / "services" / "api" / "calculator.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "services" / "api" / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "services" / "api" / "pytest.ini").write_text("[pytest]\n")
    (repo / "verdict.yml").write_text("packages:\n  services/api: {}\n")
    _init_git(repo)

    adapter = MockAdapter(patches={"services/api/calculator.py": "def add(a, b):\n    return a + b\n"})
    verdict = run(task="fix add()", repo=repo, adapter=adapter)

    assert verdict.status is VerdictStatus.DONE


def test_grade_existing_diff_supports_package_selection(monorepo: Path) -> None:
    subprocess.run(
        ["git", "checkout", "-b", "base"], cwd=monorepo, check=True, capture_output=True,
    )
    (monorepo / "services" / "api" / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    subprocess.run(["git", "checkout", "-b", "fix"], cwd=monorepo, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=monorepo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "fix api"], cwd=monorepo, check=True)

    verdict = grade_existing_diff(repo=monorepo, base_ref="base", package="services/api")
    assert verdict.status is VerdictStatus.DONE


# --- end-to-end: custom build system via explicit verdict.yml --------------


@pytest.fixture
def make_build_repo(tmp_path: Path) -> Path:
    """A repo with no cross-project autodetectable build convention
    (no package.json) — the `build` gate can only ever come from an
    explicit `verdict.yml` override, exactly as DESIGN.md documents for
    the npm case, extended here to a Makefile/make-driven build."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calculator.py").write_text("def add(a, b):\n    return a - b  # bug\n")
    (repo / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "pytest.ini").write_text("[pytest]\n")
    (repo / "Makefile").write_text(
        "build:\n\tpython -c \"import calculator\"\n"
    )
    (repo / "verdict.yml").write_text("gates:\n  build: 'make build'\n")
    _init_git(repo)
    return repo


def test_custom_build_system_gate_runs_via_explicit_override(make_build_repo: Path) -> None:
    adapter = MockAdapter(patches={"calculator.py": "def add(a, b):\n    return a + b\n"})
    verdict = run(task="fix add()", repo=make_build_repo, adapter=adapter)

    assert _signal(verdict, "build").status is GateStatus.PASS
    assert _signal(verdict, "build").command == "make build"
    assert verdict.status is VerdictStatus.DONE


def test_custom_build_system_gate_fails_on_a_broken_makefile_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "pytest.ini").write_text("[pytest]\n")
    (repo / "Makefile").write_text("build:\n\texit 1\n")
    (repo / "verdict.yml").write_text("gates:\n  build: 'make build'\n")
    _init_git(repo)

    verdict = run(task="no-op", repo=repo, adapter=MockAdapter(patches={"NOTES.md": "no-op\n"}))
    assert _signal(verdict, "build").status is GateStatus.FAIL
    assert verdict.status is VerdictStatus.NOT_DONE
