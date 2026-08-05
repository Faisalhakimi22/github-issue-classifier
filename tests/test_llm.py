"""Tests for the ghic.llm package: schema validation, prompt building, the
OpenRouter provider (mocked HTTP via httpx.MockTransport -- no extra
dependency needed), the caching service layer, and the webhook's fallback
behavior when LLM analysis is unavailable or fails.
"""
from __future__ import annotations

import json

import httpx
import pytest

from ghic.llm.exceptions import LLMError, LLMProviderError, LLMResponseError, LLMTimeoutError
from ghic.llm.fallback import FallbackLLMProvider
from ghic.llm.groq import GroqProvider
from ghic.llm.models import IssueContext, parse_issue_analysis
from ghic.llm.openrouter import OpenRouterProvider
from ghic.llm.prompts import build_user_prompt
from ghic.llm.service import LLMService

VALID_RAW = {
    "category": "bug",
    "priority": "high",
    "severity": "medium",
    "confidence": 0.92,
    "summary": "The app crashes on save.",
    "reasoning": "Clear reproduction steps and a stack trace are present.",
    "missing_information": ["Application version"],
    "recommended_labels": ["bug", "backend"],
}


def make_context(**overrides) -> IssueContext:
    defaults = dict(
        repo="acme/widgets", title="Crash on save", body="Steps to reproduce...",
        labels=[], author="reporter", metadata={}, ml_probability=0.82,
        ml_predicted_label=1,
    )
    defaults.update(overrides)
    return IssueContext(**defaults)


# ---------------------------------------------------------------------------
# models.py -- schema validation
# ---------------------------------------------------------------------------
class TestParseIssueAnalysis:
    def test_valid_response_parses(self):
        analysis = parse_issue_analysis(VALID_RAW)
        assert analysis.category == "bug"
        assert analysis.priority == "high"
        assert analysis.confidence == 0.92
        assert analysis.missing_information == ["Application version"]

    def test_priority_is_case_normalized(self):
        raw = {**VALID_RAW, "priority": "HIGH"}
        assert parse_issue_analysis(raw).priority == "high"

    def test_not_a_dict_rejected(self):
        with pytest.raises(LLMResponseError, match="expected a JSON object"):
            parse_issue_analysis(["not", "a", "dict"])

    def test_missing_required_field_rejected(self):
        raw = {k: v for k, v in VALID_RAW.items() if k != "summary"}
        with pytest.raises(LLMResponseError, match="summary"):
            parse_issue_analysis(raw)

    def test_empty_string_field_rejected(self):
        raw = {**VALID_RAW, "category": "   "}
        with pytest.raises(LLMResponseError, match="category"):
            parse_issue_analysis(raw)

    def test_invalid_priority_value_rejected(self):
        raw = {**VALID_RAW, "priority": "urgent"}
        with pytest.raises(LLMResponseError, match="priority"):
            parse_issue_analysis(raw)

    def test_invalid_severity_value_rejected(self):
        raw = {**VALID_RAW, "severity": "catastrophic"}
        with pytest.raises(LLMResponseError, match="severity"):
            parse_issue_analysis(raw)

    def test_confidence_out_of_range_rejected(self):
        raw = {**VALID_RAW, "confidence": 1.5}
        with pytest.raises(LLMResponseError, match="confidence"):
            parse_issue_analysis(raw)

    def test_confidence_wrong_type_rejected(self):
        raw = {**VALID_RAW, "confidence": "high"}
        with pytest.raises(LLMResponseError, match="confidence"):
            parse_issue_analysis(raw)

    def test_confidence_bool_rejected(self):
        # bool is a subclass of int in Python -- must not silently pass as a number.
        raw = {**VALID_RAW, "confidence": True}
        with pytest.raises(LLMResponseError, match="confidence"):
            parse_issue_analysis(raw)

    def test_missing_information_must_be_string_list(self):
        raw = {**VALID_RAW, "missing_information": [1, 2]}
        with pytest.raises(LLMResponseError, match="missing_information"):
            parse_issue_analysis(raw)

    def test_missing_information_and_labels_optional(self):
        raw = {k: v for k, v in VALID_RAW.items()
               if k not in ("missing_information", "recommended_labels")}
        analysis = parse_issue_analysis(raw)
        assert analysis.missing_information == []
        assert analysis.recommended_labels == []

    def test_as_dict_round_trips(self):
        analysis = parse_issue_analysis(VALID_RAW)
        d = analysis.as_dict()
        assert d["category"] == "bug"
        assert d["confidence"] == 0.92


# ---------------------------------------------------------------------------
# prompts.py
# ---------------------------------------------------------------------------
class TestBuildUserPrompt:
    def test_includes_core_fields(self):
        ctx = make_context(title="Crash on save", ml_probability=0.82, ml_predicted_label=1)
        prompt = build_user_prompt(ctx)
        assert "acme/widgets" in prompt
        assert "Crash on save" in prompt
        assert "0.82" in prompt
        assert "actionable bug" in prompt

    def test_never_recompute_instruction_present(self):
        prompt = build_user_prompt(make_context())
        assert "do not recompute" in prompt.lower()

    def test_long_body_is_truncated(self):
        ctx = make_context(body="x" * 10_000)
        prompt = build_user_prompt(ctx)
        assert "[truncated]" in prompt
        assert len(prompt) < 6000

    def test_empty_labels_render_gracefully(self):
        prompt = build_user_prompt(make_context(labels=[]))
        assert "(none)" in prompt


# ---------------------------------------------------------------------------
# openrouter.py -- mocked HTTP via httpx.MockTransport
# ---------------------------------------------------------------------------
def _client_returning(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _chat_response(content: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
    )


class TestOpenRouterProvider:
    def test_successful_call_returns_analysis(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer test-key"
            return _chat_response(json.dumps(VALID_RAW))

        provider = OpenRouterProvider(api_key="test-key", http_client=_client_returning(handler))
        analysis = provider.analyze(make_context())
        assert analysis.category == "bug"

    def test_strips_markdown_code_fence(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _chat_response(f"```json\n{json.dumps(VALID_RAW)}\n```")

        provider = OpenRouterProvider(api_key="test-key", http_client=_client_returning(handler))
        analysis = provider.analyze(make_context())
        assert analysis.category == "bug"

    def test_malformed_json_raises_response_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _chat_response("this is not json")

        provider = OpenRouterProvider(api_key="test-key", http_client=_client_returning(handler))
        with pytest.raises(LLMResponseError):
            provider.analyze(make_context())

    def test_unexpected_response_shape_raises_response_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": "shape"})

        provider = OpenRouterProvider(api_key="test-key", http_client=_client_returning(handler))
        with pytest.raises(LLMResponseError):
            provider.analyze(make_context())

    def test_401_raises_immediately_no_retry(self):
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(401, json={"error": "invalid api key"})

        provider = OpenRouterProvider(
            api_key="bad-key", http_client=_client_returning(handler),
            max_retries=3, retry_base_delay=0.01, retry_max_delay=0.02,
        )
        with pytest.raises(LLMProviderError):
            provider.analyze(make_context())
        assert calls["count"] == 1  # a 4xx must never be retried

    def test_429_is_retried_then_succeeds(self):
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            if calls["count"] < 3:
                return httpx.Response(429, json={"error": "rate limited"})
            return _chat_response(json.dumps(VALID_RAW))

        provider = OpenRouterProvider(
            api_key="test-key", http_client=_client_returning(handler),
            max_retries=3, retry_base_delay=0.01, retry_max_delay=0.02,
        )
        analysis = provider.analyze(make_context())
        assert analysis.category == "bug"
        assert calls["count"] == 3

    def test_persistent_5xx_exhausts_retries_and_raises_provider_error(self):
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(503, text="service unavailable")

        provider = OpenRouterProvider(
            api_key="test-key", http_client=_client_returning(handler),
            max_retries=2, retry_base_delay=0.01, retry_max_delay=0.02,
        )
        # The internal retry marker must never escape as a public exception
        # type -- LLMService (and anything else catching LLMError) has to
        # be able to catch this.
        with pytest.raises(LLMProviderError):
            provider.analyze(make_context())
        assert calls["count"] == 2

    def test_timeout_raises_immediately_no_retry(self):
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            raise httpx.TimeoutException("timed out")

        provider = OpenRouterProvider(
            api_key="test-key", http_client=_client_returning(handler),
            max_retries=3, retry_base_delay=0.01, retry_max_delay=0.02,
        )
        with pytest.raises(LLMTimeoutError):
            provider.analyze(make_context())
        assert calls["count"] == 1  # timeouts are not retried in the webhook path

    def test_empty_api_key_rejected_at_construction(self):
        with pytest.raises(ValueError):
            OpenRouterProvider(api_key="")


# ---------------------------------------------------------------------------
# service.py -- caching + the "never raises" contract
# ---------------------------------------------------------------------------
class _FakeProvider:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def analyze(self, context):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class TestLLMService:
    def test_successful_analysis_is_cached_across_identical_contexts(self, tmp_path, monkeypatch):
        from ghic import utils

        monkeypatch.setattr(utils, "DATA_RAW", tmp_path)
        provider = _FakeProvider(result=parse_issue_analysis(VALID_RAW))
        service = LLMService(provider)
        ctx = make_context()

        first = service.analyze_issue(ctx)
        second = service.analyze_issue(ctx)

        assert first is not None and second is not None
        assert first.category == second.category == "bug"
        assert provider.calls == 1  # second call served from cache

    def test_different_issue_is_not_a_cache_hit(self, tmp_path, monkeypatch):
        from ghic import utils

        monkeypatch.setattr(utils, "DATA_RAW", tmp_path)
        provider = _FakeProvider(result=parse_issue_analysis(VALID_RAW))
        service = LLMService(provider)

        service.analyze_issue(make_context(title="Issue A"))
        service.analyze_issue(make_context(title="Issue B"))
        assert provider.calls == 2

    def test_provider_error_returns_none_not_raise(self, tmp_path, monkeypatch):
        from ghic import utils

        monkeypatch.setattr(utils, "DATA_RAW", tmp_path)
        provider = _FakeProvider(error=LLMProviderError("boom"))
        service = LLMService(provider)

        result = service.analyze_issue(make_context())
        assert result is None  # never raises -- caller falls back

    def test_expired_cache_entry_is_refetched(self, tmp_path, monkeypatch):
        from ghic import utils

        monkeypatch.setattr(utils, "DATA_RAW", tmp_path)
        provider = _FakeProvider(result=parse_issue_analysis(VALID_RAW))
        service = LLMService(provider, cache_ttl_seconds=-1)  # already expired
        ctx = make_context()

        service.analyze_issue(ctx)
        service.analyze_issue(ctx)
        assert provider.calls == 2

    def test_cache_write_failure_does_not_raise(self, tmp_path, monkeypatch):
        from ghic import utils

        def broken_cache_put(namespace, key, value):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(utils, "DATA_RAW", tmp_path)
        monkeypatch.setattr(utils, "cache_put", broken_cache_put)
        provider = _FakeProvider(result=parse_issue_analysis(VALID_RAW))
        service = LLMService(provider)

        result = service.analyze_issue(make_context())
        assert result is not None  # the analysis itself still succeeds


# ---------------------------------------------------------------------------
# groq.py -- thin subclass of the shared base; light coverage is enough,
# the retry/timeout/parsing matrix is already exercised against OpenRouter
# above (both go through the same ChatCompletionsProvider).
# ---------------------------------------------------------------------------
class TestGroqProvider:
    def test_uses_groq_base_url_and_default_model(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return _chat_response(json.dumps(VALID_RAW))

        provider = GroqProvider(api_key="test-key", http_client=_client_returning(handler))
        analysis = provider.analyze(make_context())

        assert analysis.category == "bug"
        assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
        assert captured["body"]["model"] == "openai/gpt-oss-120b"

    def test_malformed_response_raises_response_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return _chat_response("not json")

        provider = GroqProvider(api_key="test-key", http_client=_client_returning(handler))
        with pytest.raises(LLMResponseError):
            provider.analyze(make_context())


# ---------------------------------------------------------------------------
# fallback.py -- priority chain
# ---------------------------------------------------------------------------
class TestFallbackLLMProvider:
    def test_primary_success_never_calls_secondary(self):
        primary = _FakeProvider(result=parse_issue_analysis(VALID_RAW))
        secondary = _FakeProvider(result=parse_issue_analysis(VALID_RAW))
        chain = FallbackLLMProvider([primary, secondary])

        chain.analyze(make_context())
        assert primary.calls == 1
        assert secondary.calls == 0

    def test_primary_failure_falls_through_to_secondary(self):
        primary = _FakeProvider(error=LLMTimeoutError("too slow"))
        secondary = _FakeProvider(result=parse_issue_analysis(VALID_RAW))
        chain = FallbackLLMProvider([primary, secondary])

        result = chain.analyze(make_context())
        assert result.category == "bug"
        assert primary.calls == 1
        assert secondary.calls == 1

    def test_all_providers_failing_raises_the_last_error(self):
        primary = _FakeProvider(error=LLMTimeoutError("primary timed out"))
        secondary = _FakeProvider(error=LLMProviderError("secondary rejected"))
        chain = FallbackLLMProvider([primary, secondary])

        with pytest.raises(LLMProviderError, match="secondary rejected"):
            chain.analyze(make_context())

    def test_empty_provider_list_rejected_at_construction(self):
        with pytest.raises(ValueError):
            FallbackLLMProvider([])

    def test_works_transparently_inside_llm_service(self, tmp_path, monkeypatch):
        """The whole point of the abstraction: LLMService doesn't need to
        know a fallback chain is involved at all."""
        from ghic import utils

        monkeypatch.setattr(utils, "DATA_RAW", tmp_path)
        primary = _FakeProvider(error=LLMProviderError("down"))
        secondary = _FakeProvider(result=parse_issue_analysis(VALID_RAW))
        service = LLMService(FallbackLLMProvider([primary, secondary]))

        result = service.analyze_issue(make_context())
        assert result is not None
        assert result.category == "bug"

    def test_isinstance_of_llm_error_hierarchy(self):
        # Sanity check the test fixtures themselves use real LLMError types,
        # since FallbackLLMProvider only catches LLMError.
        assert issubclass(LLMTimeoutError, LLMError)
        assert issubclass(LLMProviderError, LLMError)
