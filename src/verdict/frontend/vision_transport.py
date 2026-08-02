"""Concrete `VisionModelTransport` implementations (Phase 16) — the part
of `RealVisionJudge` that's actually specific to one vendor's API. Only
Anthropic ships today; see `vision_judge.py`'s `VisionModelTransport`
Protocol docstring for why adding a second provider later never means
touching `RealVisionJudge` or its prompt, just adding a class here (or in
a sibling module, once there's more than one worth splitting out).

Built on `urllib.request` from the standard library, not a vendor SDK or
`httpx` — Verdict has no HTTP client dependency anywhere else in the
codebase (Phase 14's backend smoke checks made the identical choice, for
the identical reason: one plain JSON-over-HTTPS POST doesn't need a new
dependency to express). Every coding-agent adapter in this codebase shells
out to an installed, already-authenticated CLI instead of calling a
vendor API directly (see `adapters/claude_code.py` et al.) — the vision
judge deliberately does NOT follow that pattern, because a coding CLI's
job is "do open-ended work and hand back a diff," while a judge needs a
validated, machine-readable `{passed, rationale}` back on every call, and
forcing structured tool output through a real HTTP API is a much more
reliable way to get that than parsing a coding agent's freeform stdout.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from verdict.frontend.vision_judge import JUDGMENT_TOOL_SCHEMA, TransportResult, VisionTransportError

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_TOKENS = 512


@dataclass(frozen=True)
class _Pricing:
    input_per_1k: float
    output_per_1k: float

    def cost_usd(self, tokens_input: int, tokens_output: int) -> float:
        return (tokens_input / 1000) * self.input_per_1k + (tokens_output / 1000) * self.output_per_1k


# Indicative, not authoritative — Anthropic's own pricing page is the
# source of truth and these figures can drift out of date; kept here only
# so `Signal.cost_usd` has something better than `None` for the model this
# transport defaults to. An unrecognized `model` (a caller-supplied
# override this table has no entry for) simply reports `cost_usd=None`
# rather than guessing — the same "unknown stays unknown" rule the rest of
# this codebase's cost accounting already follows.
_PRICING_PER_1K: dict[str, _Pricing] = {
    "claude-sonnet-5": _Pricing(input_per_1k=0.003, output_per_1k=0.015),
    "claude-opus-5": _Pricing(input_per_1k=0.015, output_per_1k=0.075),
    "claude-haiku-4-5-20251001": _Pricing(input_per_1k=0.001, output_per_1k=0.005),
}


class AnthropicVisionTransport:
    """Calls the Anthropic Messages API with the image inlined as base64,
    forcing the `submit_judgment` tool call (`tool_choice`) so the
    response can only ever be the validated `{passed, rationale}` shape
    `JUDGMENT_TOOL_SCHEMA` describes — never freeform prose to re-parse.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        # Env-var only, read once at construction — never a CLI flag
        # (shell history / process-list exposure) and never sourced from
        # `verdict.yml` (the graded repo's own file, not a trusted secrets
        # store — the same boundary `sandbox/config.py`'s module docstring
        # draws for sandbox policy generally).
        self._api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self._model = model
        self._timeout_seconds = timeout_seconds

    def complete(self, screenshot_png: bytes, system_prompt: str, user_text: str) -> TransportResult:
        if not self._api_key:
            raise VisionTransportError("ANTHROPIC_API_KEY is not set")

        body = {
            "model": self._model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(screenshot_png).decode("ascii"),
                            },
                        },
                    ],
                }
            ],
            "tools": [JUDGMENT_TOOL_SCHEMA],
            # Forces the model to answer via the tool call, not prose —
            # the structural half of the injection defense described in
            # vision_judge.py's module docstring.
            "tool_choice": {"type": "tool", "name": JUDGMENT_TOOL_SCHEMA["name"]},
        }
        request = urllib_request.Request(
            _API_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": _API_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read())
        except urllib_error.HTTPError as exc:
            raise VisionTransportError(f"Anthropic API returned HTTP {exc.code}") from exc
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            raise VisionTransportError(f"could not reach the Anthropic API: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise VisionTransportError(f"Anthropic API returned a non-JSON response: {exc}") from exc

        return self._parse_response(payload)

    def _parse_response(self, payload: Any) -> TransportResult:
        try:
            tool_use = next(
                block
                for block in payload["content"]
                if block.get("type") == "tool_use" and block.get("name") == JUDGMENT_TOOL_SCHEMA["name"]
            )
            passed = bool(tool_use["input"]["passed"])
            rationale = str(tool_use["input"]["rationale"])
            usage = payload.get("usage", {})
            tokens_input = int(usage.get("input_tokens", 0))
            tokens_output = int(usage.get("output_tokens", 0))
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            raise VisionTransportError(
                f"Anthropic response didn't match the expected judgment schema: {exc}"
            ) from exc

        pricing = _PRICING_PER_1K.get(self._model)
        cost_usd = pricing.cost_usd(tokens_input, tokens_output) if pricing else None
        return TransportResult(
            passed=passed,
            rationale=rationale,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
        )
