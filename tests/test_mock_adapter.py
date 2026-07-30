from __future__ import annotations

from pathlib import Path

import pytest

from verdict.adapters.mock import MockAdapter


def test_mock_adapter_writes_patches_and_reports_diff(tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path
    (repo / "a.py").write_text("old\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    adapter = MockAdapter(patches={"a.py": "new\n"})
    result = adapter.run("some task", repo)

    assert (repo / "a.py").read_text() == "new\n"
    assert result.files_changed == ["a.py"]
    assert "new" in result.diff
    assert result.cost_usd == 0.0


def test_mock_adapter_requires_at_least_one_patch() -> None:
    with pytest.raises(ValueError):
        MockAdapter(patches={})
