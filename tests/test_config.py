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
    # `frontend:` is missing `start`, so it's an incomplete section, not a
    # usable config — parsed as absent rather than crashing on it.
    assert config.frontend is None


def test_load_config_parses_frontend_section(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text(
        """
frontend:
  start: "npm run dev"
  url: "http://localhost:3000"
  viewports: [1440, 375]
  screenshot_threshold: 0.05
  vision_model: "gpt-4-class"
  checks:
    - name: cta
      dom:
        selector: "#cta"
        visible: true
        class_contains: "cta"
      interaction:
        click: "#cta"
        expect_url_contains: "/signup"
      vision_intent: "A visible CTA is present."
"""
    )
    config = load_config(tmp_path)
    assert config.frontend is not None
    assert config.frontend.start == "npm run dev"
    assert config.frontend.url == "http://localhost:3000"
    assert config.frontend.viewports == (1440, 375)
    assert config.frontend.screenshot_threshold == 0.05
    assert config.frontend.vision_model == "gpt-4-class"
    assert len(config.frontend.checks) == 1

    check = config.frontend.checks[0]
    assert check.name == "cta"
    assert check.dom is not None
    assert check.dom.selector == "#cta"
    assert check.dom.visible is True
    assert check.dom.class_contains == "cta"
    assert check.interaction is not None
    assert check.interaction.click == "#cta"
    assert check.interaction.expect_url_contains == "/signup"
    assert check.vision_intent == "A visible CTA is present."


def test_load_config_frontend_defaults_when_only_required_fields_given(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text(
        'frontend:\n  start: "npm run dev"\n  url: "http://localhost:3000"\n'
    )
    config = load_config(tmp_path)
    assert config.frontend is not None
    assert config.frontend.viewports == (1440,)
    assert config.frontend.checks == []
    assert config.frontend.vision_model is None


def test_load_config_no_frontend_section_is_none(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text("gates:\n  test: 'pytest -q'\n")
    config = load_config(tmp_path)
    assert config.frontend is None
