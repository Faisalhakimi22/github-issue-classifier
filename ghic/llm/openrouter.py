"""OpenRouter provider: calls the chat completions API, validates the JSON
response against the schema in models.py.

Retry policy is deliberately narrow: only transient failures (connection
errors, 429, 5xx) are retried, using the project's existing
utils.retry_with_backoff. A timeout is NOT retried by default -- this
provider is called synchronously from inside the webhook handler, which has
to answer GitHub within its ~10s delivery window, so a single slow request
already spent most of that budget; retrying it risks blowing past the
window for a comment that gets discarded by the fallback path anyway. A 4xx
(bad request, invalid API key) is never retried either -- it won't succeed
on attempt two.
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .. import utils
from .exceptions import LLMProviderError, LLMResponseError, LLMTimeoutError
from .models import IssueAnalysis, IssueContext, parse_issue_analysis
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .provider import LLMProvider

logger = utils.get_logger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


class _RetryableProviderError(Exception):
    """Internal marker for a transient failure worth retrying (429/5xx/transport)."""


class OpenRouterProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = OPENROUTER_BASE_URL,
        timeout: float = 8.0,
        max_retries: int = 2,
        retry_base_delay: float = 0.3,
        retry_max_delay: float = 2.0,
        reasoning_effort: str = "low",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouterProvider requires a non-empty api_key")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client()

    def analyze(self, context: IssueContext) -> IssueAnalysis:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(context)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }

        retrying_request = utils.retry_with_backoff(
            max_attempts=self.max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
            exceptions=(_RetryableProviderError,),
            logger=logger,
        )(self._raw_request)

        started = time.perf_counter()
        try:
            response = retrying_request(payload)
        except _RetryableProviderError as e:
            # Retries exhausted -- surface as the public LLMError type. Left
            # as the internal marker, this would escape analyze() uncaught
            # (it isn't an LLMError), defeating the "never fail the webhook"
            # contract LLMService/the caller rely on.
            raise LLMProviderError(f"OpenRouter request failed after retries: {e}") from e
        elapsed_ms = (time.perf_counter() - started) * 1000

        data = response.json()
        usage = data.get("usage") or {}
        logger.info(
            "openrouter analyze: model=%s ms=%.0f prompt_tokens=%s completion_tokens=%s",
            self.model, elapsed_ms,
            usage.get("prompt_tokens"), usage.get("completion_tokens"),
        )

        content = _extract_content(data)
        raw = _parse_json_content(content)
        return parse_issue_analysis(raw)

    def _raw_request(self, payload: dict[str, Any]) -> httpx.Response:
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"OpenRouter request timed out after {self.timeout}s") from e
        except httpx.TransportError as e:
            raise _RetryableProviderError(f"transport error: {e}") from e

        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableProviderError(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise LLMProviderError(
                f"OpenRouter returned HTTP {response.status_code}: {response.text[:200]}"
            )
        return response


def _extract_content(data: dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMResponseError(f"unexpected OpenRouter response shape: {e}") from e


def _parse_json_content(content: str) -> Any:
    text = content.strip()
    # Some models wrap JSON in a fenced code block despite instructions --
    # strip it rather than fail on formatting the prompt already forbade.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"response was not valid JSON: {e}") from e
