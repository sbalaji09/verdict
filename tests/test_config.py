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


def test_load_config_no_services_section_is_empty_list(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text("gates:\n  test: 'pytest -q'\n")
    config = load_config(tmp_path)
    assert config.services == []


def test_load_config_parses_services_section(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text(
        """
services:
  - name: db
    type: postgres
    version: "16"
    env:
      POSTGRES_PASSWORD: verdict
      POSTGRES_DB: app_test
  - name: cache
    type: redis
    version: "7"
    port: 6380
"""
    )
    config = load_config(tmp_path)
    assert len(config.services) == 2

    db = config.services[0]
    assert db.name == "db"
    assert db.type == "postgres"
    assert db.version == "16"
    assert db.env == {"POSTGRES_PASSWORD": "verdict", "POSTGRES_DB": "app_test"}
    assert db.port is None

    cache = config.services[1]
    assert cache.name == "cache"
    assert cache.type == "redis"
    assert cache.port == 6380


def test_load_config_drops_a_service_entry_missing_a_required_field(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text(
        """
services:
  - name: db
    type: postgres
    version: "16"
  - name: incomplete
    type: redis
"""
    )
    config = load_config(tmp_path)
    # Missing `version` on the second entry — dropped, not crashed, same
    # per-item-skip convention `_parse_checks` already uses for frontend
    # checks. It is deliberately NOT the same as an unrecognized
    # type/version, which sandbox/services.py surfaces as a real error
    # rather than silently dropping — see test_services.py.
    assert len(config.services) == 1
    assert config.services[0].name == "db"


def test_load_config_no_backend_section_is_none(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text("gates:\n  test: 'pytest -q'\n")
    config = load_config(tmp_path)
    assert config.backend is None


def test_load_config_backend_missing_required_field_is_none(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text(
        "backend:\n  start: 'python app.py'\n"  # no health_url
    )
    config = load_config(tmp_path)
    assert config.backend is None


def test_load_config_parses_backend_section(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text(
        """
backend:
  start: "python app.py"
  health_url: "http://localhost:8000/healthz"
  ready_timeout_seconds: 45
  migrate: "alembic upgrade head"
  smoke:
    - path: "/api/widgets"
      expect_status: 200
      expect_body_contains: "widgets"
    - path: "/api/widgets"
      method: post
      body: '{"name": "x"}'
      expect_status: 201
      name: "create widget"
"""
    )
    config = load_config(tmp_path)
    assert config.backend is not None
    backend = config.backend
    assert backend.start == "python app.py"
    assert backend.health_url == "http://localhost:8000/healthz"
    assert backend.ready_timeout_seconds == 45
    assert backend.migrate == "alembic upgrade head"
    assert len(backend.smoke) == 2

    get_spec = backend.smoke[0]
    assert get_spec.method == "GET"
    assert get_spec.expect_status == 200
    assert get_spec.expect_body_contains == "widgets"
    assert get_spec.name is None

    post_spec = backend.smoke[1]
    assert post_spec.method == "POST"  # uppercased
    assert post_spec.body == '{"name": "x"}'
    assert post_spec.expect_status == 201
    assert post_spec.name == "create widget"


def test_load_config_backend_defaults_when_only_required_fields_given(tmp_path: Path) -> None:
    (tmp_path / "verdict.yml").write_text(
        'backend:\n  start: "python app.py"\n  health_url: "http://localhost:8000/healthz"\n'
    )
    config = load_config(tmp_path)
    assert config.backend is not None
    assert config.backend.ready_timeout_seconds == 30
    assert config.backend.migrate is None
    assert config.backend.smoke == []
