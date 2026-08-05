"""LLM-assisted issue analysis: reasoning and user-facing explanation on top
of the (already-decided) ML actionability prediction.

The ML classifier's probability is never recomputed or overridden here --
see llm/prompts.py. This package only adds a second, independent layer of
judgment: category, priority, severity, a plain-language summary, and
suggested labels, all clearly framed as an AI opinion rather than a
statistically validated model (see models/LLM_ANALYSIS_CARD.md for why that
distinction matters in this project specifically).
"""
from .consistency import detect_disagreement
from .exceptions import LLMError, LLMProviderError, LLMResponseError, LLMTimeoutError
from .models import IssueAnalysis, IssueContext
from .provider import LLMProvider
from .service import LLMService

__all__ = [
    "LLMError",
    "LLMProviderError",
    "LLMResponseError",
    "LLMTimeoutError",
    "IssueAnalysis",
    "IssueContext",
    "LLMProvider",
    "LLMService",
    "detect_disagreement",
]
