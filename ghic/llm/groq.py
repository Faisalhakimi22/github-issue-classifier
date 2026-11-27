"""Groq provider.

Primary choice by default (see settings.py) -- measured ~1.9s per call for
openai/gpt-oss-120b against the same prompt that took OpenRouter's default
model ~17s, comfortably inside the webhook's response budget. See
models/LLM_ANALYSIS_CARD.md for the measured comparison and how it was run.
"""
from __future__ import annotations

from typing import Any

from ._chat_completions import ChatCompletionsProvider

DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqProvider(ChatCompletionsProvider):
    BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = DEFAULT_MODEL
    PROVIDER_NAME = "groq"

    def _extra_payload_fields(self) -> dict[str, Any]:
        # gpt-oss reasons before answering, and the reasoning counts against
        # max_tokens. At Groq's default ("medium") it routinely spent enough
        # of the budget to cut the JSON answer off mid-string. "low" keeps the
        # answer intact and spends fewer of the free tier's tokens per minute.
        # Groq accepts the field only for gpt-oss models; sending it to any
        # other model would fail every request, hence the check.
        if "gpt-oss" in (self.model or ""):
            return {"reasoning_effort": "low"}
        return {}
