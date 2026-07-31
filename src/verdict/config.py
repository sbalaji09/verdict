"""Loads `verdict.yml` overrides from the repo being graded.

Only `gates` (Phase 1) and `cost` (Phase 3) are read. The other sections in
the README's example config (frontend, report) belong to later phases and
are ignored for now rather than rejected, so a forward-looking config file
doesn't break earlier phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

GATE_NAMES = ("test", "typecheck", "build", "lint")


@dataclass
class TokenPricing:
    """$ per 1,000 tokens. Only used as a fallback — see runner.py — when
    an adapter doesn't report its own cost_usd directly."""

    input_per_1k: float
    output_per_1k: float

    def cost_usd(self, tokens_input: int, tokens_output: int) -> float:
        return (tokens_input / 1000) * self.input_per_1k + (tokens_output / 1000) * self.output_per_1k


class VerdictConfig:
    def __init__(
        self, gate_overrides: dict[str, str], token_pricing: TokenPricing | None = None
    ) -> None:
        self.gate_overrides = gate_overrides
        self.token_pricing = token_pricing

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
    return VerdictConfig(gate_overrides=overrides, token_pricing=_parse_token_pricing(data))
