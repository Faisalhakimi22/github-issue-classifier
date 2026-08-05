"""Tests for idempotency (ghic.service.idempotency) and the QStash async
queue (ghic.service.qstash) -- the two pieces async webhook processing adds
on top of the existing synchronous path. Webhook-integration coverage
(duplicate deliveries, queued vs synchronous processing, the
/internal/process-issue endpoint) lives in test_service.py alongside the
rest of the webhook tests.
"""
from __future__ import annotations

import base64
import hashlib
import time

import jwt
import pytest

from ghic.service.idempotency import (
    FileIdempotencyStore,
    _InMemoryIdempotencyStore,
    build_idempotency_store,
)
from ghic.service.qstash import QStashError, publish, verify_signature


# ---------------------------------------------------------------------------
# idempotency.py
# ---------------------------------------------------------------------------
class TestInMemoryIdempotencyStore:
    def test_first_seen_returns_true(self):
        store = _InMemoryIdempotencyStore()
        assert store.mark_if_new("abc") is True

    def test_second_seen_returns_false(self):
        store = _InMemoryIdempotencyStore()
        store.mark_if_new("abc")
        assert store.mark_if_new("abc") is False

    def test_different_keys_independent(self):
        store = _InMemoryIdempotencyStore()
        assert store.mark_if_new("a") is True
        assert store.mark_if_new("b") is True


class TestFileIdempotencyStore:
    def test_dedup_within_one_instance(self, tmp_path):
        store = FileIdempotencyStore(tmp_path / "seen.json")
        assert store.mark_if_new("delivery-1") is True
        assert store.mark_if_new("delivery-1") is False

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "seen.json"
        store1 = FileIdempotencyStore(path)
        store1.mark_if_new("delivery-1")

        store2 = FileIdempotencyStore(path)  # simulated restart
        assert store2.mark_if_new("delivery-1") is False
        assert store2.mark_if_new("delivery-2") is True

    def test_missing_file_starts_empty(self, tmp_path):
        store = FileIdempotencyStore(tmp_path / "does-not-exist-yet.json")
        assert store.mark_if_new("anything") is True

    def test_corrupted_file_degrades_to_empty_not_crash(self, tmp_path):
        path = tmp_path / "seen.json"
        path.write_text("not valid json{{{", encoding="utf-8")
        store = FileIdempotencyStore(path)
        assert store.mark_if_new("delivery-1") is True  # didn't crash on bad file


class TestBuildIdempotencyStore:
    def test_no_config_gives_in_memory(self):
        store = build_idempotency_store()
        assert isinstance(store, _InMemoryIdempotencyStore)

    def test_file_path_gives_file_backend(self, tmp_path):
        store = build_idempotency_store(file_path=tmp_path / "seen.json")
        assert isinstance(store, FileIdempotencyStore)


# ---------------------------------------------------------------------------
# qstash.py -- publish()
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json = json_body or {}
        self.text = text

    def json(self):
        return self._json


class TestPublish:
    def test_successful_publish_returns_message_id(self, monkeypatch):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _FakeResponse(200, {"messageId": "msg-123"})

        import requests

        monkeypatch.setattr(requests, "post", fake_post)
        message_id = publish(
            "https://example.com/internal/process-issue", {"a": 1}, "test-token",
        )
        assert message_id == "msg-123"
        assert captured["headers"]["Authorization"] == "Bearer test-token"
        assert "qstash-us-east-1.upstash.io" in captured["url"]
        assert "https://example.com/internal/process-issue" in captured["url"]

    def test_network_failure_raises_qstash_error(self, monkeypatch):
        import requests

        def fake_post(*a, **kw):
            raise requests.ConnectionError("dns failure")

        monkeypatch.setattr(requests, "post", fake_post)
        with pytest.raises(QStashError):
            publish("https://example.com/x", {}, "token")

    def test_non_ok_status_raises_qstash_error(self, monkeypatch):
        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResponse(401, text="bad token"))
        with pytest.raises(QStashError):
            publish("https://example.com/x", {}, "token")

    def test_missing_message_id_raises_qstash_error(self, monkeypatch):
        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResponse(200, {}))
        with pytest.raises(QStashError):
            publish("https://example.com/x", {}, "token")


# ---------------------------------------------------------------------------
# qstash.py -- verify_signature()
# ---------------------------------------------------------------------------
SIGNING_KEY = "test-signing-key-long-enough-for-hs256-32-bytes"
DESTINATION_URL = "https://example.com/internal/process-issue"


def _make_signature(body: bytes, key: str = SIGNING_KEY, url: str = DESTINATION_URL,
                    exp_delta: int = 300, iss: str = "Upstash") -> str:
    body_hash = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("=")
    now = int(time.time())
    claims = {
        "iss": iss, "sub": url, "exp": now + exp_delta, "iat": now, "nbf": now - 1,
        "jti": "test-jti", "body": body_hash,
    }
    return jwt.encode(claims, key, algorithm="HS256")


class TestVerifySignature:
    def test_valid_signature_accepted(self):
        body = b'{"repo": "acme/widgets"}'
        sig = _make_signature(body)
        assert verify_signature(sig, body, [SIGNING_KEY], DESTINATION_URL) is True

    def test_wrong_key_rejected(self):
        body = b'{"a": 1}'
        sig = _make_signature(body, key="wrong-key-also-long-enough-for-hs256")
        assert verify_signature(sig, body, [SIGNING_KEY], DESTINATION_URL) is False

    def test_tampered_body_rejected(self):
        sig = _make_signature(b'{"a": 1}')
        assert verify_signature(sig, b'{"a": 2}', [SIGNING_KEY], DESTINATION_URL) is False

    def test_wrong_destination_url_rejected(self):
        body = b'{"a": 1}'
        sig = _make_signature(body, url="https://attacker.example/other")
        assert verify_signature(sig, body, [SIGNING_KEY], DESTINATION_URL) is False

    def test_wrong_issuer_rejected(self):
        body = b'{"a": 1}'
        sig = _make_signature(body, iss="NotUpstash")
        assert verify_signature(sig, body, [SIGNING_KEY], DESTINATION_URL) is False

    def test_expired_token_rejected(self):
        body = b'{"a": 1}'
        sig = _make_signature(body, exp_delta=-60)  # expired a minute ago
        assert verify_signature(sig, body, [SIGNING_KEY], DESTINATION_URL) is False

    def test_missing_header_rejected(self):
        assert verify_signature(None, b"body", [SIGNING_KEY], DESTINATION_URL) is False

    def test_no_signing_keys_configured_rejected(self):
        body = b'{"a": 1}'
        sig = _make_signature(body)
        assert verify_signature(sig, body, [], DESTINATION_URL) is False

    def test_key_rotation_next_key_accepted(self):
        """A request signed under the previous (current) key must still
        verify during a rotation window if the caller passes both keys."""
        body = b'{"a": 1}'
        old_key = "old-signing-key-long-enough-for-hs256-too"
        sig = _make_signature(body, key=old_key)
        # caller configured [new_key, old_key] -- old one is second in line
        new_key = "new-signing-key-long-enough-for-hs256-too"
        assert verify_signature(sig, body, [new_key, old_key], DESTINATION_URL) is True
