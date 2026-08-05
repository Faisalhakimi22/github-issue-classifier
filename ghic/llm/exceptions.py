"""Exception hierarchy for the LLM layer.

Every failure mode a provider can hit collapses to one of these three, so
callers (LLMService, the webhook handler) can catch `LLMError` once and
always fall back to the existing ML-only comment -- the "never fail the
webhook" contract lives in that single catch, not scattered per call site.
"""
from __future__ import annotations


class LLMError(Exception):
    """Base class for every LLM-layer failure."""


class LLMTimeoutError(LLMError):
    """The provider didn't respond within the configured timeout budget."""


class LLMProviderError(LLMError):
    """The provider's API itself failed (HTTP error, auth failure, rate limit)."""


class LLMResponseError(LLMError):
    """The provider responded, but the content wasn't valid per the schema
    (malformed JSON, missing/wrong-typed fields, out-of-range values)."""
