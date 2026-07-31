from __future__ import annotations

from pathlib import Path

from verdict.config import load_config


def test_load_config_missing_file_has_no_overrides(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    assert config.override_for("test") is None


def test_load_config_reads_gate_overrides(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text(
        "gates:\n  test: 'pytest -q'\n  lint: 'ruff check .'\n"
    )
    config = load_config(tmp_path)
    assert config.override_for("test") == "pytest -q"
    assert config.override_for("lint") == "ruff check ."
    assert config.override_for("build") is None


def test_load_config_ignores_unknown_keys(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text(
        "gates:\n  test: 'pytest -q'\n  bogus: 'rm -rf /'\n"
        "frontend:\n  url: 'http://localhost:3000'\n"
    )
    config = load_config(tmp_path)
    assert config.override_for("test") == "pytest -q"
    assert config.override_for("bogus") is None
