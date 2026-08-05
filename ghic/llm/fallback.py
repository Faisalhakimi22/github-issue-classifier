"""A priority chain of providers, itself an LLMProvider.

LLMService doesn't need to know fallback exists -- it calls .analyze() on
whatever LLMProvider it was constructed with, and this one just happens to
try several real providers in order before giving up. Any LLMError from one
provider moves to the next; if every provider fails, the last error is
raised (so the caller's `except LLMError` still degrades to the ML-only
comment exactly as it would for a single provider).
"""
from __future__ import annotations

from .. import utils
from .exceptions import LLMError
from .models import IssueAnalysis, IssueContext
from .provider import LLMProvider

logger = utils.get_logger(__name__)


class FallbackLLMProvider(LLMProvider):
    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            raise ValueError("FallbackLLMProvider needs at least one provider")
        self.providers = providers

    def analyze(self, context: IssueContext) -> IssueAnalysis:
        last_error: LLMError | None = None
        for i, provider in enumerate(self.providers):
            try:
                return provider.analyze(context)
            except LLMError as e:
                last_error = e
                remaining = len(self.providers) - i - 1
                logger.warning(
                    "llm provider %d/%d (%s) failed (%s: %s)%s",
                    i + 1, len(self.providers), type(provider).__name__,
                    type(e).__name__, e,
                    f"; trying next of {remaining}" if remaining else "; no more fallbacks",
                )
        assert last_error is not None  # unreachable: the loop always sets it before exhausting
        raise last_error
