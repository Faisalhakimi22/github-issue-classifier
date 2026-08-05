"""Orchestration: cache lookup -> provider call -> cache write, with the
"never fail the webhook" contract enforced in exactly one place.

Caching reuses the project's existing content-addressed JSON cache
(utils.cache_get/cache_put -- the same one ghic.collect uses for GitHub API
responses) rather than inventing a second cache mechanism, adding only the
TTL check that primitive doesn't have. On a deploy with no writable/
persistent disk for it (Vercel -- see docs/DEPLOYMENT.md's ledger section
for the same tradeoff), a cache I/O failure is caught and logged, never
raised: every request just runs uncached rather than the feature breaking.
"""
from __future__ import annotations

import time
from typing import Any

from .. import utils
from .exceptions import LLMError, LLMResponseError
from .models import IssueAnalysis, IssueContext, parse_issue_analysis
from .provider import LLMProvider

logger = utils.get_logger(__name__)

DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
CACHE_NAMESPACE = "llm_analysis"


class LLMService:
    def __init__(
        self,
        provider: LLMProvider,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        cache_namespace: str = CACHE_NAMESPACE,
    ) -> None:
        self.provider = provider
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_namespace = cache_namespace

    def analyze_issue(self, context: IssueContext) -> IssueAnalysis | None:
        """None means "no analysis available" -- caller falls back to the
        ML-only comment. Never raises: every LLM/cache failure is caught
        and logged here."""
        key = self._cache_key(context)

        cached = self._cache_get(key)
        if cached is not None:
            logger.info("llm analysis cache hit for %s", key)
            return cached

        try:
            analysis = self.provider.analyze(context)
        except LLMError as e:
            logger.warning("llm analysis failed (%s: %s); falling back to ML-only comment",
                          type(e).__name__, e)
            return None

        self._cache_put(key, analysis)
        return analysis

    def _cache_key(self, context: IssueContext) -> str:
        return utils.cache_key(
            "issue-analysis", context.repo, context.title, context.body,
            round(context.ml_probability, 4),
        )

    def _cache_get(self, key: str) -> IssueAnalysis | None:
        try:
            raw = utils.cache_get(self.cache_namespace, key)
        except OSError as e:
            logger.debug("llm cache read failed (%s); treating as a miss", e)
            return None
        if raw is None:
            return None
        cached_at = raw.get("_cached_at", 0)
        if time.time() - cached_at > self.cache_ttl_seconds:
            return None
        try:
            return parse_issue_analysis(raw["analysis"])
        except (LLMResponseError, KeyError, TypeError):
            return None  # corrupted/stale-shape cache entry -- ignore, don't crash

    def _cache_put(self, key: str, analysis: IssueAnalysis) -> None:
        payload: dict[str, Any] = {"_cached_at": time.time(), "analysis": analysis.as_dict()}
        try:
            utils.cache_put(self.cache_namespace, key, payload)
        except OSError as e:
            logger.debug("llm cache write failed (%s); continuing without caching this result", e)
