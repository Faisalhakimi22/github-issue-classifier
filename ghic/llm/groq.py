"""Groq provider.

Primary choice by default (see settings.py) -- measured ~1.9s per call for
openai/gpt-oss-120b against the same prompt that took OpenRouter's default
model ~17s, comfortably inside the webhook's response budget. See
models/LLM_ANALYSIS_CARD.md for the measured comparison and how it was run.
"""
from __future__ import annotations

from ._chat_completions import ChatCompletionsProvider

DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqProvider(ChatCompletionsProvider):
    BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = DEFAULT_MODEL
    PROVIDER_NAME = "groq"
