"""Abstract provider interface.

New providers (OpenAI, Anthropic, Gemini, Ollama) implement this one method
and nothing in service.py, prompts.py, or the webhook handler needs to
change -- `LLMService` is constructed with whichever `LLMProvider` the
caller wires up (see settings.py's `llm_provider`), never with a provider
name string it branches on internally.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .models import IssueAnalysis, IssueContext


class LLMProvider(ABC):
    """One provider = one way to turn an IssueContext into an IssueAnalysis.

    Implementations raise LLMTimeoutError / LLMProviderError / LLMResponseError
    (see exceptions.py) on failure -- they never return a partial or guessed
    IssueAnalysis, and they never raise anything else. Synchronous by design:
    the webhook handler that calls this is itself synchronous end to end
    (feature extraction, GitHub API calls), so an async provider would just
    need blocking-on-the-loop glue at the one call site for no benefit.
    """

    @abstractmethod
    def analyze(self, context: IssueContext) -> IssueAnalysis:
        raise NotImplementedError
