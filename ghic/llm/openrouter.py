"""OpenRouter provider.

Fallback/secondary choice by default (see settings.py) -- the default free
model (nvidia/nemotron-3-ultra-550b-a55b:free) measured ~17s per call in
practice, well over the webhook's realistic response budget. Kept as the
documented default because it's what was specified; GroqProvider is the
fast primary. See models/LLM_ANALYSIS_CARD.md for the measured comparison.
"""
from __future__ import annotations

from typing import Any

from ._chat_completions import ChatCompletionsProvider

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


class OpenRouterProvider(ChatCompletionsProvider):
    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = DEFAULT_MODEL
    PROVIDER_NAME = "openrouter"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, *,
                 reasoning_effort: str = "low", **kwargs: Any) -> None:
        super().__init__(api_key, model, **kwargs)
        self.reasoning_effort = reasoning_effort

    def _extra_payload_fields(self) -> dict[str, Any]:
        return {"reasoning_effort": self.reasoning_effort}
