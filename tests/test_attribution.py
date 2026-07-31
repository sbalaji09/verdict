"""End-to-end attribution tests: build a small real git repo per scenario,
run the full pipeline (worktree -> mock agent -> gates -> attribution), and
check what got attributed to what. These exercise the actual `git bisect`
machinery, not mocks of it — slower than a unit test, but this is exactly
the kind of logic that looks right on paper and needs to be checked against
real git.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from verdict.adapters.mock import MockAdapter
from verdict.runner import run
from verdict.schema import AttributionKind


def _git_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative_path, content in files.items():
        target = repo / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    return repo


def _attributions_for(verdict, gate: str = "test") -> list:
    return [a for a in verdict.attributions if a.check_name == gate]


# --- broke-a-test ------------------------------------------------------

def test_broke_a_test_attributes_to_the_edited_file(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path, {
        "calculator.py": "def add(a, b):\n    return a + b\n",
        "test_calculator.py": "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        "pytest.ini": "[pytest]\n",
    })
    adapter = MockAdapter(patches={"calculator.py": "def add(a, b):\n    return a - b\n"})
    verdict = run(task="break it", repo=repo, adapter=adapter)

    attrs = _attributions_for(verdict)
    assert len(attrs) == 1
    assert attrs[0].kind is AttributionKind.REGRESSION
    assert attrs[0].culprit_file == "calculator.py"


# --- touched-unrelated-file ---------------------------------------------

def test_unrelated_file_is_never_blamed(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path, {
        "calculator.py": "def add(a, b):\n    return a + b\n",
        "README.md": "a calculator\n",
        "test_calculator.py": "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        "pytest.ini": "[pytest]\n",
    })
    adapter = MockAdapter(patches={
        "calculator.py": "def add(a, b):\n    return a - b\n",
        "README.md": "a calculator (now with docs)\n",
    })
    verdict = run(task="improve docs, also break it", repo=repo, adapter=adapter)

    attrs = _attributions_for(verdict)
    assert len(attrs) == 1
    assert attrs[0].culprit_file == "calculator.py"
    assert attrs[0].culprit_file != "README.md"


# --- forgot-to-update-fixture -------------------------------------------

def test_forgot_to_update_fixture_blames_the_file_actually_touched(tmp_path: Path) -> None:
    # scale() depends on MULTIPLIER; fixtures/expected.json is a stale
    # precomputed answer the agent needed to regenerate but didn't.
    repo = _git_repo(tmp_path, {
        "settings.py": "MULTIPLIER = 2\n",
        "calc.py": "from settings import MULTIPLIER\n\ndef scale(x):\n    return x * MULTIPLIER\n",
        "fixtures/expected.json": '{"result": 10}\n',
        "test_calc.py": (
            "import json\nfrom calc import scale\n\n"
            "def test_scale():\n"
            "    expected = json.load(open('fixtures/expected.json'))\n"
            "    assert scale(5) == expected['result']\n"
        ),
        "pytest.ini": "[pytest]\n",
    })
    # agent changes the multiplier but never regenerates the fixture
    adapter = MockAdapter(patches={"settings.py": "MULTIPLIER = 3\n"})
    verdict = run(task="change the multiplier to 3", repo=repo, adapter=adapter)

    attrs = _attributions_for(verdict)
    assert len(attrs) == 1
    assert attrs[0].kind is AttributionKind.REGRESSION
    # Verdict can only attribute to files the agent actually touched — it
    # never fabricates a claim about the fixture it *didn't* touch, even
    # though a human would say "you also needed to update the fixture."
    assert attrs[0].culprit_file == "settings.py"


# --- import-error ---------------------------------------------------------

def test_import_error_is_attributed_not_crashed_on(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path, {
        "helpers.py": "def double(x):\n    return x * 2\n",
        "calculator.py": "from helpers import double\n\ndef add(a, b):\n    return double(a) + double(b)\n",
        "test_calculator.py": "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 10\n",
        "pytest.ini": "[pytest]\n",
    })
    # typo'd import — breaks collection, not just one assertion
    broken_calculator = "from helper import double\n\ndef add(a, b):\n    return double(a) + double(b)\n"
    adapter = MockAdapter(patches={"calculator.py": broken_calculator})
    verdict = run(task="refactor the import", repo=repo, adapter=adapter)

    assert verdict.status.value == "not_done"
    attrs = _attributions_for(verdict)
    assert len(attrs) == 1
    assert attrs[0].kind is AttributionKind.REGRESSION
    assert attrs[0].culprit_file == "calculator.py"


# --- pre-existing failure ---------------------------------------------------

def test_pre_existing_failure_is_not_blamed_on_the_agent(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path, {
        # already broken at the base commit, before any agent runs
        "calculator.py": "def add(a, b):\n    return a - b\n",
        "README.md": "a calculator\n",
        "test_calculator.py": "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        "pytest.ini": "[pytest]\n",
    })
    adapter = MockAdapter(patches={"README.md": "a calculator (docs update)\n"})
    verdict = run(task="just update the docs", repo=repo, adapter=adapter)

    attrs = _attributions_for(verdict)
    assert len(attrs) == 1
    assert attrs[0].kind is AttributionKind.PRE_EXISTING
    assert attrs[0].culprit_file is None
