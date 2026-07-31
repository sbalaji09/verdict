"""Loads `verdict.yml` overrides from the repo being graded.

Only the `gates` table is read in Phase 1 — the other sections in the
README's example config (frontend, cost, report) belong to later phases and
are ignored for now rather than rejected, so a forward-looking config file
doesn't break Phase 1.
"""

from __future__ import annotations

from pathlib import Path

import yaml

GATE_NAMES = ("test", "typecheck", "build", "lint")


class VerdictConfig:
    def __init__(self, gate_overrides: dict[str, str]) -> None:
        self.gate_overrides = gate_overrides

    def override_for(self, gate: str) -> str | None:
        return self.gate_overrides.get(gate)


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
    return VerdictConfig(gate_overrides=overrides)
