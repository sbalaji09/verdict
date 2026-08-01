"""Service dependencies (Phase 10): a Postgres/Redis/etc. a repo's own
test suite needs running to boot, declared as `services:` in
`verdict.yml` (see `config.py`'s `ServiceSpec`).

Every service — and the gate container itself, when any services are
declared — joins one per-attempt Docker network created with
`--internal` (`sandbox/docker.py::create_network`): Docker refuses to
route that network to the public internet at all, so declaring a service
widens *exactly* what it needs to (reachability to the declared services,
by the DNS name `verdict.yml` gave them via `--network-alias`) without
reopening the general internet egress Phase 8/9 close by default. No
services declared → nothing here runs, and the gate container stays on
`--network none` exactly as before this phase.

`type`/`version` are looked up against `_SERVICE_IMAGES`, an allowlist
Verdict's own code owns — `verdict.yml` (written by the untrusted repo
being graded) never gets to name an arbitrary `image:` for Verdict to
`docker run`. Extending this allowlist is an operator action, not
something a repo can do by editing its own config; an unrecognized
`type`/`version` is a real, surfaced `SetupError`, not a silent skip.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from verdict.config import ServiceSpec
from verdict.sandbox.base import SetupError
from verdict.sandbox.docker import create_network, remove_network, run_docker_cli

# type -> version -> pinned image tag. The ONLY way a `services:` entry
# turns into something Verdict actually runs.
_SERVICE_IMAGES: dict[str, dict[str, str]] = {
    "postgres": {
        "14": "postgres:14-alpine",
        "15": "postgres:15-alpine",
        "16": "postgres:16-alpine",
    },
    "redis": {
        "6": "redis:6-alpine",
        "7": "redis:7-alpine",
    },
    "mysql": {
        "8": "mysql:8.0",
    },
    "mongodb": {
        "6": "mongo:6",
        "7": "mongo:7",
    },
}

_DEFAULT_PORTS: dict[str, int] = {
    "postgres": 5432,
    "redis": 6379,
    "mysql": 3306,
    "mongodb": 27017,
}

_HEALTH_CHECK_COMMANDS: dict[str, list[str]] = {
    "postgres": ["pg_isready", "-U", "postgres"],
    "redis": ["redis-cli", "ping"],
    "mysql": ["mysqladmin", "ping", "-h", "127.0.0.1"],
    "mongodb": ["mongosh", "--quiet", "--eval", "db.runCommand('ping').ok"],
}

DEFAULT_HEALTH_TIMEOUT_SECONDS = 30
_HEALTH_POLL_INTERVAL_SECONDS = 0.5


def _resolve_image(spec: ServiceSpec) -> str:
    versions = _SERVICE_IMAGES.get(spec.type)
    if versions is None:
        raise SetupError(
            f"service '{spec.name}': unrecognized type {spec.type!r} "
            f"(known: {', '.join(sorted(_SERVICE_IMAGES))}) — extending this "
            "allowlist is an operator action, verdict.yml cannot do it itself"
        )
    image = versions.get(spec.version)
    if image is None:
        raise SetupError(
            f"service '{spec.name}': unrecognized {spec.type} version {spec.version!r} "
            f"(known: {', '.join(sorted(versions))})"
        )
    return image


@dataclass
class ServiceHandle:
    name: str
    container: str
    port: int


def _wait_healthy(spec: ServiceSpec, container: str, timeout_seconds: int) -> None:
    command = _HEALTH_CHECK_COMMANDS[spec.type]
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = run_docker_cli(["exec", container, *command], timeout=10)
        if result.returncode == 0:
            return
        time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)
    raise SetupError(
        f"service '{spec.name}' ({spec.type}:{spec.version}) never became healthy "
        f"within {timeout_seconds}s"
    )


def start_services(
    services: list[ServiceSpec],
    network_name: str,
    health_timeout_seconds: int = DEFAULT_HEALTH_TIMEOUT_SECONDS,
) -> list[ServiceHandle]:
    """Starts every declared service on `network_name` (already created),
    in declaration order, blocking on each one's health check before
    starting the next. On the first unrecognized type/version or the
    first service that never becomes healthy, tears down whatever it
    already started (so the caller only ever has to remove the network
    itself, not chase partially-started containers) and raises
    `SetupError` — never a gate Signal, never NOT_DONE.
    """
    handles: list[ServiceHandle] = []
    try:
        for spec in services:
            image = _resolve_image(spec)
            container = f"verdict-svc-{spec.name}-{uuid.uuid4().hex[:8]}"
            port = spec.port or _DEFAULT_PORTS[spec.type]
            run_args = [
                "run", "-d", "--name", container,
                "--network", network_name, "--network-alias", spec.name,
            ]
            for key, value in spec.env.items():
                run_args += ["-e", f"{key}={value}"]
            run_args += [image]
            result = run_docker_cli(run_args, timeout=60)
            if result.returncode != 0:
                raise SetupError(
                    f"service '{spec.name}' ({spec.type}:{spec.version}) failed to start: "
                    f"{result.stderr.strip()}"
                )
            handles.append(ServiceHandle(name=spec.name, container=container, port=port))
            _wait_healthy(spec, container, health_timeout_seconds)
    except SetupError:
        stop_services(handles)
        raise
    return handles


def stop_services(handles: list[ServiceHandle]) -> None:
    for handle in handles:
        run_docker_cli(["rm", "-f", handle.container], timeout=30)


@dataclass
class ServiceSession:
    """The whole per-attempt service setup — network plus every running
    service — as one handle a caller opens once and tears down once.
    `network_name` is `None` when no services were declared, meaning
    there's nothing for the gate container to join and nothing to tear
    down (the common, unchanged-from-before-Phase-10 case).
    """

    network_name: str | None
    handles: list[ServiceHandle] = field(default_factory=list)


def setup_services(
    services: list[ServiceSpec],
    health_timeout_seconds: int = DEFAULT_HEALTH_TIMEOUT_SECONDS,
) -> ServiceSession:
    if not services:
        return ServiceSession(network_name=None)
    network_name = f"verdict-attempt-{uuid.uuid4().hex[:12]}"
    create_network(network_name)
    try:
        handles = start_services(services, network_name, health_timeout_seconds)
    except SetupError:
        remove_network(network_name)
        raise
    return ServiceSession(network_name=network_name, handles=handles)


def teardown_services(session: ServiceSession) -> None:
    if session.network_name is None:
        return
    stop_services(session.handles)
    remove_network(session.network_name)
