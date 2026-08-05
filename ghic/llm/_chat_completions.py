"""Shared HTTP/retry/parsing logic for OpenAI-compatible chat completions
APIs (OpenRouter, Groq, and most other providers use this exact shape).

OpenRouterProvider and GroqProvider both subclass ChatCompletionsProvider,
overriding only what actually differs between them: base URL, default
model, and any extra request parameters a specific provider supports. This
exists so a third OpenAI-compatible provider is a ~15-line subclass, not a
copy-paste of the retry/timeout/parsing logic.
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


class _RetryableProviderError(Exception):
    """Internal marker for a transient failure worth retrying (429/5xx/transport)."""


class ChatCompletionsProvider(LLMProvider):
    """Base for any provider exposing an OpenAI-shaped POST /chat/completions.

    Retry policy: only transient failures (429, 5xx, connection errors)
    retry, via the project's existing utils.retry_with_backoff. A timeout is
    never retried, and neither is a 4xx -- see openrouter.py's module
    docstring for why (this runs synchronously inside the webhook's GitHub
    delivery window).
    """

    #: Set by subclasses.
    BASE_URL: str = ""
    DEFAULT_MODEL: str = ""
    PROVIDER_NAME: str = ""

    def __init__(
        self,
        api_key: str,
        model: str = "",
        *,
        base_url: str = "",
        timeout: float = 8.0,
        max_retries: int = 2,
        retry_base_delay: float = 0.3,
        retry_max_delay: float = 2.0,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(f"{type(self).__name__} requires a non-empty api_key")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = (base_url or self.BASE_URL).rstrip("/")
        self._client = http_client or httpx.Client()

    def _extra_payload_fields(self) -> dict[str, Any]:
        """Provider-specific request fields beyond model/messages/temperature/
        max_tokens. Overridden by subclasses that support extras (e.g.
        OpenRouter's reasoning_effort); empty by default since not every
        OpenAI-compatible API accepts the same optional fields."""
        return {}

    def analyze(self, context: IssueContext) -> IssueAnalysis:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(context)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self._extra_payload_fields(),
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
            raise LLMProviderError(
                f"{self.PROVIDER_NAME} request failed after retries: {e}"
            ) from e
        elapsed_ms = (time.perf_counter() - started) * 1000

        data = response.json()
        usage = data.get("usage") or {}
        logger.info(
            "%s analyze: model=%s ms=%.0f prompt_tokens=%s completion_tokens=%s",
            self.PROVIDER_NAME, self.model, elapsed_ms,
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
            raise LLMTimeoutError(
                f"{self.PROVIDER_NAME} request timed out after {self.timeout}s"
            ) from e
        except httpx.TransportError as e:
            raise _RetryableProviderError(f"transport error: {e}") from e

        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableProviderError(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise LLMProviderError(
                f"{self.PROVIDER_NAME} returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        return response


def _extract_content(data: dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMResponseError(f"unexpected chat completions response shape: {e}") from e


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
