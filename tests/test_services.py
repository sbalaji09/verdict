"""`sandbox/services.py`'s image allowlist — the part of Phase 10's
service-dependency machinery that's testable without a real Docker daemon:
an unrecognized `type`/`version` must be rejected with a clear `SetupError`
BEFORE anything is ever `docker run`, never silently substituted or
skipped. Full container lifecycle (start/health-check/teardown) is
exercised for real in `test_sandbox_docker_adversarial.py`'s Docker-gated
DB-service test.
"""

from __future__ import annotations

import pytest

from verdict.config import ServiceSpec
from verdict.sandbox.base import SetupError
from verdict.sandbox.services import _resolve_image


def test_resolve_image_known_type_and_version() -> None:
    spec = ServiceSpec(name="db", type="postgres", version="16")
    assert _resolve_image(spec) == "postgres:16-alpine"


def test_resolve_image_rejects_unrecognized_type() -> None:
    spec = ServiceSpec(name="db", type="oracle", version="19")
    with pytest.raises(SetupError, match="unrecognized type"):
        _resolve_image(spec)


def test_resolve_image_rejects_unrecognized_version_of_a_known_type() -> None:
    spec = ServiceSpec(name="db", type="postgres", version="9")
    with pytest.raises(SetupError, match="unrecognized postgres version"):
        _resolve_image(spec)


def test_resolve_image_never_lets_verdict_yml_name_an_arbitrary_image() -> None:
    """The whole point of `type`/`version` instead of a raw `image:`
    field: there is no code path anywhere in `_resolve_image` that returns
    anything other than one of the pinned strings in `_SERVICE_IMAGES` —
    confirmed here by checking the return value is always one of the
    allowlisted images for a matrix of known types/versions, never
    influenced by anything else on the spec (e.g. `env`).
    """
    from verdict.sandbox.services import _SERVICE_IMAGES

    for service_type, versions in _SERVICE_IMAGES.items():
        for version, expected_image in versions.items():
            spec = ServiceSpec(
                name="x", type=service_type, version=version, env={"MALICIOUS": "irrelevant"}
            )
            assert _resolve_image(spec) == expected_image
