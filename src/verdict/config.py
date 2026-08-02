"""Loads `verdict.yml` overrides from the repo being graded.

`gates` (Phase 1), `cost` (Phase 3), `frontend` (Phase 4), `services`
(Phase 10), and `backend` (Phase 14) are read. The `report` section from
the README's example config belongs to a later phase and is still ignored
rather than rejected, so a forward-looking config file doesn't break
earlier phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

GATE_NAMES = ("test", "typecheck", "build", "lint")

DEFAULT_VIEWPORTS = (1440,)
DEFAULT_VIEWPORT_HEIGHT = 900
DEFAULT_READY_TIMEOUT_SECONDS = 30
DEFAULT_SCREENSHOT_THRESHOLD = 0.02
"""Fraction of pixels (after the grayscale/tolerance pass in
`frontend/visual_diff.py`) allowed to differ before-vs-after before the
perceptual diff gate reports FAIL. Deliberately a fraction, not a raw pixel
count, so the same config value is meaningful across viewport sizes."""

DEFAULT_GLITCH_SCAN = True
DEFAULT_GLITCH_CAPTURE_SECONDS = 1.5
DEFAULT_GLITCH_FRAME_INTERVAL_SECONDS = 0.15
DEFAULT_GLITCH_DIFF_THRESHOLD = 0.05
"""Same fractional-diff meaning as `screenshot_threshold`, applied
frame-to-frame within a burst rather than before-vs-after — see
`frontend/glitch.py`."""


@dataclass
class DomSpec:
    """One PROVEN assertion against the rendered DOM: a node exists (and,
    optionally, is actually visible / carries an expected class or attribute
    substring) — never more than what's mechanically checkable from the
    accessibility/render tree, on purpose (see DESIGN.md's DOM-assertion
    bucket).
    """

    selector: str
    visible: bool | None = None
    class_contains: str | None = None
    attribute: str | None = None
    attribute_contains: str | None = None
    text_contains: str | None = None


@dataclass
class InteractionSpec:
    """One PROVEN user action: click `click`, then assert a real, observable
    outcome — a URL change and/or another selector becoming visible. This is
    "perform the real user action, assert the real outcome," never a proxy
    for it.
    """

    click: str
    expect_url_contains: str | None = None
    expect_selector_visible: str | None = None


@dataclass
class FrontendCheckSpec:
    name: str
    dom: DomSpec | None = None
    interaction: InteractionSpec | None = None
    vision_intent: str | None = None
    """Free-text description of the intended UI outcome, scored by the
    vision-intent judge (JUDGED bucket) — advisory only, see runner.py."""


@dataclass
class FrontendConfig:
    """Parsed `frontend:` section — see README's Configuration example."""

    start: str
    url: str
    viewports: tuple[int, ...] = DEFAULT_VIEWPORTS
    viewport_height: int = DEFAULT_VIEWPORT_HEIGHT
    ready_timeout_seconds: int = DEFAULT_READY_TIMEOUT_SECONDS
    screenshot_threshold: float = DEFAULT_SCREENSHOT_THRESHOLD
    vision_model: str | None = None
    checks: list[FrontendCheckSpec] = field(default_factory=list)
    glitch_scan: bool = DEFAULT_GLITCH_SCAN
    glitch_capture_seconds: float = DEFAULT_GLITCH_CAPTURE_SECONDS
    glitch_frame_interval_seconds: float = DEFAULT_GLITCH_FRAME_INTERVAL_SECONDS
    glitch_diff_threshold: float = DEFAULT_GLITCH_DIFF_THRESHOLD


DEFAULT_BACKEND_READY_TIMEOUT_SECONDS = 30
DEFAULT_SMOKE_METHOD = "GET"
DEFAULT_SMOKE_EXPECT_STATUS = 200


@dataclass
class SmokeRequestSpec:
    """One literal HTTP request fired at the booted service — the "does
    the API actually answer correctly" half of the backend runtime check,
    as distinct from "is something listening" (that's `health_url`'s
    job). `path` is joined onto `health_url`'s own scheme/host/port
    (`urllib.parse.urljoin`), so a task only ever writes the host once.
    """

    path: str
    method: str = DEFAULT_SMOKE_METHOD
    body: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    expect_status: int = DEFAULT_SMOKE_EXPECT_STATUS
    expect_body_contains: str | None = None
    name: str | None = None
    """Free-form label for the resulting `backend:smoke:<name>` Signal —
    defaults to `"{method} {path}"` when not given (see `backend/runner.py`)."""


@dataclass
class BackendConfig:
    """Parsed `backend:` section — see README's Configuration example.
    Entirely opt-in, the same "no section, no checks" contract
    `frontend:` already established: most repos don't declare one, and
    `run_backend_checks` returns `[]` immediately when `config.backend`
    is `None`.
    """

    start: str
    health_url: str
    ready_timeout_seconds: int = DEFAULT_BACKEND_READY_TIMEOUT_SECONDS
    migrate: str | None = None
    """Optional one-shot command (e.g. `alembic upgrade head`) run BEFORE
    `start` — a migration failure means the service never gets a chance
    to boot against a correct schema, so `backend/runner.py` skips
    boot/smoke checks entirely on a migrate FAIL rather than running them
    against an unmigrated database and reporting a confusing secondary
    failure."""
    smoke: list[SmokeRequestSpec] = field(default_factory=list)


@dataclass
class TokenPricing:
    """$ per 1,000 tokens. Only used as a fallback — see runner.py — when
    an adapter doesn't report its own cost_usd directly."""

    input_per_1k: float
    output_per_1k: float

    def cost_usd(self, tokens_input: int, tokens_output: int) -> float:
        return (tokens_input / 1000) * self.input_per_1k + (tokens_output / 1000) * self.output_per_1k


@dataclass
class ServiceSpec:
    """One declared service dependency (Phase 10) — a Postgres/Redis/etc.
    the repo's own code or test suite needs running to boot. Deliberately
    thin: `type`/`version` name an entry in Verdict's OWN image allowlist
    (`sandbox/services.py`), never a raw `image:` string — a `verdict.yml`
    living in the untrusted repo being graded must never get to pick an
    arbitrary image for Verdict to `docker run`, the same "sandbox policy
    never comes from the repo" rule Phase 8 established for everything
    else in this module. Whether `type`/`version` are actually *known* is
    checked at setup time (`sandbox/services.py`), not here — an unknown
    combination is a real, surfaced ERROR, not something this parser
    should silently drop the way a malformed optional field elsewhere in
    this module would be.
    """

    name: str
    """DNS hostname other containers (the gate container, other services)
    resolve this one by — a `--network-alias`, not a real domain."""
    type: str
    version: str
    env: dict[str, str] = field(default_factory=dict)
    port: int | None = None
    """None uses the service type's own standard port (5432 for postgres,
    6379 for redis, ...) — see `sandbox/services.py`'s allowlist."""


class VerdictConfig:
    def __init__(
        self,
        gate_overrides: dict[str, str],
        token_pricing: TokenPricing | None = None,
        frontend: FrontendConfig | None = None,
        services: list[ServiceSpec] | None = None,
        backend: BackendConfig | None = None,
    ) -> None:
        self.gate_overrides = gate_overrides
        self.token_pricing = token_pricing
        self.frontend = frontend
        self.services = services or []
        self.backend = backend

    def override_for(self, gate: str) -> str | None:
        return self.gate_overrides.get(gate)


def _parse_token_pricing(data: dict[str, object]) -> TokenPricing | None:
    cost = data.get("cost")
    if not isinstance(cost, dict):
        return None
    prices = cost.get("price_per_1k_tokens")
    if not isinstance(prices, dict):
        return None
    try:
        return TokenPricing(
            input_per_1k=float(prices["input"]),
            output_per_1k=float(prices["output"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _parse_dom_spec(data: dict[str, object] | None) -> DomSpec | None:
    if not isinstance(data, dict) or not data.get("selector"):
        return None
    return DomSpec(
        selector=str(data["selector"]),
        visible=bool(data["visible"]) if "visible" in data else None,
        class_contains=str(data["class_contains"]) if data.get("class_contains") else None,
        attribute=str(data["attribute"]) if data.get("attribute") else None,
        attribute_contains=(
            str(data["attribute_contains"]) if data.get("attribute_contains") else None
        ),
        text_contains=str(data["text_contains"]) if data.get("text_contains") else None,
    )


def _parse_interaction_spec(data: dict[str, object] | None) -> InteractionSpec | None:
    if not isinstance(data, dict) or not data.get("click"):
        return None
    return InteractionSpec(
        click=str(data["click"]),
        expect_url_contains=(
            str(data["expect_url_contains"]) if data.get("expect_url_contains") else None
        ),
        expect_selector_visible=(
            str(data["expect_selector_visible"]) if data.get("expect_selector_visible") else None
        ),
    )


def _parse_checks(raw: object) -> list[FrontendCheckSpec]:
    if not isinstance(raw, list):
        return []
    checks: list[FrontendCheckSpec] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        checks.append(
            FrontendCheckSpec(
                name=str(item["name"]),
                dom=_parse_dom_spec(item.get("dom")),
                interaction=_parse_interaction_spec(item.get("interaction")),
                vision_intent=str(item["vision_intent"]) if item.get("vision_intent") else None,
            )
        )
    return checks


def _parse_frontend_config(data: dict[str, object]) -> FrontendConfig | None:
    frontend = data.get("frontend")
    if not isinstance(frontend, dict) or not frontend.get("start") or not frontend.get("url"):
        return None

    viewports_raw = frontend.get("viewports")
    viewports = (
        tuple(int(v) for v in viewports_raw)
        if isinstance(viewports_raw, list) and viewports_raw
        else DEFAULT_VIEWPORTS
    )

    return FrontendConfig(
        start=str(frontend["start"]),
        url=str(frontend["url"]),
        viewports=viewports,
        viewport_height=int(frontend.get("viewport_height", DEFAULT_VIEWPORT_HEIGHT)),
        ready_timeout_seconds=int(
            frontend.get("ready_timeout_seconds", DEFAULT_READY_TIMEOUT_SECONDS)
        ),
        screenshot_threshold=float(
            frontend.get("screenshot_threshold", DEFAULT_SCREENSHOT_THRESHOLD)
        ),
        vision_model=str(frontend["vision_model"]) if frontend.get("vision_model") else None,
        checks=_parse_checks(frontend.get("checks")),
        glitch_scan=bool(frontend.get("glitch_scan", DEFAULT_GLITCH_SCAN)),
        glitch_capture_seconds=float(
            frontend.get("glitch_capture_seconds", DEFAULT_GLITCH_CAPTURE_SECONDS)
        ),
        glitch_frame_interval_seconds=float(
            frontend.get("glitch_frame_interval_seconds", DEFAULT_GLITCH_FRAME_INTERVAL_SECONDS)
        ),
        glitch_diff_threshold=float(
            frontend.get("glitch_diff_threshold", DEFAULT_GLITCH_DIFF_THRESHOLD)
        ),
    )


def _parse_smoke_spec(item: object) -> SmokeRequestSpec | None:
    if not isinstance(item, dict) or not item.get("path"):
        return None
    headers_raw = item.get("headers") or {}
    headers = {str(k): str(v) for k, v in headers_raw.items()} if isinstance(headers_raw, dict) else {}
    return SmokeRequestSpec(
        path=str(item["path"]),
        method=str(item.get("method", DEFAULT_SMOKE_METHOD)).upper(),
        body=str(item["body"]) if item.get("body") is not None else None,
        headers=headers,
        expect_status=int(item.get("expect_status", DEFAULT_SMOKE_EXPECT_STATUS)),
        expect_body_contains=(
            str(item["expect_body_contains"]) if item.get("expect_body_contains") else None
        ),
        name=str(item["name"]) if item.get("name") else None,
    )


def _parse_smoke(raw: object) -> list[SmokeRequestSpec]:
    if not isinstance(raw, list):
        return []
    return [spec for item in raw if (spec := _parse_smoke_spec(item)) is not None]


def _parse_backend_config(data: dict[str, object]) -> BackendConfig | None:
    backend = data.get("backend")
    if not isinstance(backend, dict) or not backend.get("start") or not backend.get("health_url"):
        return None
    return BackendConfig(
        start=str(backend["start"]),
        health_url=str(backend["health_url"]),
        ready_timeout_seconds=int(
            backend.get("ready_timeout_seconds", DEFAULT_BACKEND_READY_TIMEOUT_SECONDS)
        ),
        migrate=str(backend["migrate"]) if backend.get("migrate") else None,
        smoke=_parse_smoke(backend.get("smoke")),
    )


def _parse_services(raw: object) -> list[ServiceSpec]:
    if not isinstance(raw, list):
        return []
    services: list[ServiceSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name, kind, version = item.get("name"), item.get("type"), item.get("version")
        if not name or not kind or not version:
            # Missing a required field entirely is a malformed *entry* —
            # same per-item skip `_parse_checks` already uses elsewhere in
            # this module. An entry that names a real but *unrecognized*
            # type/version is different (a real, satisfiable-looking
            # request Verdict can't fulfill) and is deliberately NOT
            # caught here — see `sandbox/services.py`, which raises a
            # surfaced ERROR for that case rather than silently dropping
            # the service the way this loop drops a malformed entry.
            continue
        env_raw = item.get("env") or {}
        env = {str(k): str(v) for k, v in env_raw.items()} if isinstance(env_raw, dict) else {}
        port = int(item["port"]) if item.get("port") else None
        services.append(ServiceSpec(name=str(name), type=str(kind), version=str(version), env=env, port=port))
    return services


def load_config(worktree: Path) -> VerdictConfig:
    path = worktree / "verdict.yml"
    if not path.exists():
        return VerdictConfig(gate_overrides={})

    data = yaml.safe_load(path.read_text()) or {}
    gates = data.get("gates", {}) or {}
    overrides = {
        name: str(command)
        for name, command in gates.items()
        if name in GATE_NAMES and command
    }
    return VerdictConfig(
        gate_overrides=overrides,
        token_pricing=_parse_token_pricing(data),
        frontend=_parse_frontend_config(data),
        services=_parse_services(data.get("services")),
        backend=_parse_backend_config(data),
    )
