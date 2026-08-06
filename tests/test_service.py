"""Tests for the webhook service: signature verification, event routing,
the issues.opened flow (dry-run and write mode), and the predict API.

The model and GitHub client are stubbed — these tests exercise the service
logic, not sklearn. End-to-end inference against a real .joblib is covered
by test_inference_smoke (skipped when no trained model is present).
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ghic.service.app import create_app, verify_signature  # noqa: E402
from ghic.service.inference import Prediction, format_comment, format_llm_comment  # noqa: E402
from ghic.service.settings import ServiceSettings  # noqa: E402

SECRET = "test-secret"


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class StubPredictor:
    model_name = "stub"

    @property
    def cfg(self):
        from ghic.config import get_config

        return get_config(require_token=False)

    def __init__(self, proba: float = 0.9, threshold: float = 0.5) -> None:
        self.proba = proba
        self.threshold = threshold
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> Prediction:
        self.calls.append(kwargs)
        return Prediction(
            repo=kwargs["repo_full_name"],
            issue_number=kwargs["issue_number"],
            proba=self.proba,
            threshold=self.threshold,
            predicted_label=int(self.proba >= self.threshold),
            model_name=self.model_name,
        )


class StubGitHub:
    def __init__(self) -> None:
        self.comments: list[tuple[str, int, str]] = []
        self.labels: list[tuple[str, int, list[str]]] = []
        self.project_items: list[tuple[str, str]] = []

    def get_user(self, login: str, installation_id: int) -> dict[str, Any]:
        return {"created_at": "2020-01-01T00:00:00Z", "public_repos": 5, "followers": 2}

    def get_latest_release_date(self, full_name: str, installation_id: int) -> str:
        return "2024-06-01T00:00:00Z"

    def post_comment(self, full_name, issue_number, body, installation_id) -> None:
        self.comments.append((full_name, issue_number, body))

    def add_labels(self, full_name, issue_number, labels, installation_id) -> None:
        self.labels.append((full_name, issue_number, labels))

    def add_issue_to_project(self, project_node_id, issue_node_id, installation_id) -> None:
        self.project_items.append((project_node_id, issue_node_id))


def make_settings(**overrides: Any) -> ServiceSettings:
    defaults: dict[str, Any] = dict(
        model_path=Path("unused.joblib"),
        webhook_secret=SECRET,
        dry_run=True,
        suggest_related=False,   # tests must not depend on models/dup_index.joblib
        suggest_category=False,  # ...nor on models/category.joblib
        estimate_effort=False,   # ...nor on models/effort.joblib
        suggest_assignees=False,  # ...nor on data/processed/assignments.json
    )
    defaults.update(overrides)
    return ServiceSettings(**defaults)


def make_client(settings: ServiceSettings, predictor=None, gh=None) -> TestClient:
    app = create_app(settings, predictor=predictor or StubPredictor(), gh_client=gh)
    return TestClient(app)


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def issue_opened_payload(**issue_overrides: Any) -> dict[str, Any]:
    issue = {
        "number": 42,
        "title": "App crashes on startup",
        "body": "Steps to reproduce: 1. open app 2. crash. Stack trace attached.",
        "created_at": "2024-07-01T10:00:00Z",
        "user": {"login": "alice"},
    }
    issue.update(issue_overrides)
    return {
        "action": "opened",
        "issue": issue,
        "repository": {"full_name": "acme/widgets"},
        "installation": {"id": 123},
    }


def post_webhook(client: TestClient, payload: dict[str, Any], event: str = "issues",
                 secret: str = SECRET):
    body = json.dumps(payload).encode()
    return client.post(
        "/webhook",
        content=body,
        headers={
            "X-GitHub-Event": event,
            "X-Hub-Signature-256": sign(body, secret),
            "Content-Type": "application/json",
        },
    )


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------
class TestSignature:
    def test_valid_signature(self):
        body = b'{"a": 1}'
        assert verify_signature(SECRET, body, sign(body))

    def test_wrong_secret_rejected(self):
        body = b'{"a": 1}'
        assert not verify_signature(SECRET, body, sign(body, "other-secret"))

    def test_tampered_body_rejected(self):
        assert not verify_signature(SECRET, b'{"a": 2}', sign(b'{"a": 1}'))

    def test_missing_or_malformed_header_rejected(self):
        assert not verify_signature(SECRET, b"x", None)
        assert not verify_signature(SECRET, b"x", "sha1=deadbeef")

    def test_webhook_rejects_bad_signature(self):
        client = make_client(make_settings())
        body = json.dumps(issue_opened_payload()).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-GitHub-Event": "issues",
                     "X-Hub-Signature-256": sign(body, "wrong")},
        )
        assert resp.status_code == 401

    def test_no_secret_and_no_allow_unsigned_is_unavailable(self):
        client = make_client(make_settings(webhook_secret=""))
        resp = client.post("/webhook", content=b"{}",
                           headers={"X-GitHub-Event": "ping"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Idempotency: same X-GitHub-Delivery must never process twice
# ---------------------------------------------------------------------------
def post_webhook_with_delivery(client: TestClient, payload: dict[str, Any], delivery_id: str,
                               event: str = "issues", secret: str = SECRET):
    body = json.dumps(payload).encode()
    return client.post(
        "/webhook",
        content=body,
        headers={
            "X-GitHub-Event": event,
            "X-GitHub-Delivery": delivery_id,
            "X-Hub-Signature-256": sign(body, secret),
            "Content-Type": "application/json",
        },
    )


class TestIdempotency:
    def test_duplicate_delivery_id_is_ignored(self):
        gh = StubGitHub()
        app = create_app(make_settings(dry_run=False, post_comment=True),
                         predictor=StubPredictor(), gh_client=gh)
        client = TestClient(app)

        first = post_webhook_with_delivery(client, issue_opened_payload(), "delivery-abc")
        second = post_webhook_with_delivery(client, issue_opened_payload(), "delivery-abc")

        assert first.status_code == 200
        assert first.json().get("duplicate") is not True
        assert second.status_code == 200
        assert second.json() == {"ok": True, "duplicate": True}
        assert len(gh.comments) == 1  # not 2 -- the whole point

    def test_different_delivery_ids_both_process(self):
        gh = StubGitHub()
        app = create_app(make_settings(dry_run=False, post_comment=True),
                         predictor=StubPredictor(), gh_client=gh)
        client = TestClient(app)

        post_webhook_with_delivery(client, issue_opened_payload(number=1), "delivery-1")
        post_webhook_with_delivery(client, issue_opened_payload(number=2), "delivery-2")
        assert len(gh.comments) == 2

    def test_requests_without_delivery_header_are_never_deduped(self):
        """Backward compatibility: a caller (or test) that doesn't send
        X-GitHub-Delivery gets the pre-idempotency behavior unchanged --
        no header means no dedup key to check."""
        client = make_client(make_settings())
        first = post_webhook(client, issue_opened_payload())
        second = post_webhook(client, issue_opened_payload())
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json().get("duplicate") is not True


# ---------------------------------------------------------------------------
# Async processing via QStash
# ---------------------------------------------------------------------------
class StubQueuePublisher:
    """Records what would have been published; lets tests assert the
    webhook returned fast without doing any ML/LLM/GitHub work inline."""

    def __init__(self, message_id="msg-test", error=None):
        self.message_id = message_id
        self.error = error
        self.published: list[tuple[str, dict]] = []

    def __call__(self, destination_url, payload, token, region="us-east-1"):
        if self.error:
            raise self.error
        self.published.append((destination_url, payload))
        return self.message_id


class TestAsyncProcessing:
    def _async_settings(self, **overrides):
        return make_settings(
            dry_run=False, post_comment=True,
            use_async_processing=True, qstash_token="test-token",
            qstash_current_signing_key="test-signing-key-long-enough-for-hs256",
            public_base_url="https://example.com",
            **overrides,
        )

    def test_queued_when_async_configured(self, monkeypatch):
        # The handler does `from .qstash import publish` at call time, so
        # patching the attribute on the source module is what actually
        # takes effect.
        import ghic.service.qstash as qstash_module

        stub_publish = StubQueuePublisher()
        monkeypatch.setattr(qstash_module, "publish", stub_publish)

        gh = StubGitHub()
        app = create_app(self._async_settings(), predictor=StubPredictor(), gh_client=gh)
        client = TestClient(app)

        resp = post_webhook_with_delivery(client, issue_opened_payload(), "delivery-async-1")

        assert resp.status_code == 200
        assert resp.json()["queued"] is True
        assert resp.json()["message_id"] == "msg-test"
        assert len(stub_publish.published) == 1
        assert gh.comments == []  # nothing posted synchronously -- it's queued

    def test_publish_failure_falls_back_to_synchronous(self, monkeypatch):
        from ghic.service.qstash import QStashError
        import ghic.service.qstash as qstash_module

        stub_publish = StubQueuePublisher(error=QStashError("upstash is down"))
        monkeypatch.setattr(qstash_module, "publish", stub_publish)

        gh = StubGitHub()
        app = create_app(self._async_settings(), predictor=StubPredictor(), gh_client=gh)
        client = TestClient(app)

        resp = post_webhook_with_delivery(client, issue_opened_payload(), "delivery-async-2")

        assert resp.status_code == 200
        assert "queued" not in resp.json()
        assert len(gh.comments) == 1  # processed inline instead

    def test_async_not_configured_processes_synchronously(self):
        """Default behavior, unchanged: no QStash config means every issue
        is still scored and commented on inline, same as before this
        feature existed."""
        gh = StubGitHub()
        app = create_app(make_settings(dry_run=False, post_comment=True),
                         predictor=StubPredictor(), gh_client=gh)
        client = TestClient(app)
        resp = post_webhook_with_delivery(client, issue_opened_payload(), "delivery-sync-1")
        assert "queued" not in resp.json()
        assert len(gh.comments) == 1


# ---------------------------------------------------------------------------
# POST /internal/process-issue -- the QStash callback target
# ---------------------------------------------------------------------------
class TestProcessIssueCallback:
    def _signed_request(self, secret_key: str, url: str, payload: dict):
        import base64
        import hashlib
        import time

        import jwt

        body = json.dumps(payload).encode()
        body_hash = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("=")
        now = int(time.time())
        claims = {"iss": "Upstash", "sub": url, "exp": now + 300, "iat": now,
                  "nbf": now - 1, "jti": "test", "body": body_hash}
        return body, jwt.encode(claims, secret_key, algorithm="HS256")

    def test_valid_signature_processes_and_returns_200(self):
        signing_key = "test-signing-key-long-enough-for-hs256"
        settings = make_settings(
            dry_run=False, post_comment=True,
            use_async_processing=True, qstash_token="t",
            qstash_current_signing_key=signing_key,
            public_base_url="https://example.com",
        )
        gh = StubGitHub()
        app = create_app(settings, predictor=StubPredictor(), gh_client=gh)
        client = TestClient(app)

        payload = issue_opened_payload()
        body, sig = self._signed_request(
            signing_key, "https://example.com/internal/process-issue", payload,
        )
        resp = client.post("/internal/process-issue", content=body,
                           headers={"Upstash-Signature": sig, "Content-Type": "application/json"})
        assert resp.status_code == 200
        assert len(gh.comments) == 1

    def test_invalid_signature_rejected(self):
        settings = make_settings(
            use_async_processing=True, qstash_token="t",
            qstash_current_signing_key="test-signing-key-long-enough-for-hs256",
            public_base_url="https://example.com",
        )
        app = create_app(settings, predictor=StubPredictor())
        client = TestClient(app)

        resp = client.post("/internal/process-issue", content=b'{"a": 1}',
                           headers={"Upstash-Signature": "not-a-real-jwt"})
        assert resp.status_code == 401

    def test_not_configured_returns_503(self):
        app = create_app(make_settings(), predictor=StubPredictor())
        client = TestClient(app)
        resp = client.post("/internal/process-issue", content=b"{}")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Event routing
# ---------------------------------------------------------------------------
class TestRouting:
    def test_healthz(self):
        client = make_client(make_settings())
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["model"] == "stub"

    def test_ping_event(self):
        client = make_client(make_settings())
        resp = post_webhook(client, {"zen": "Keep it logically awesome."}, event="ping")
        assert resp.status_code == 200
        assert resp.json()["pong"] == "Keep it logically awesome."

    def test_non_issue_event_ignored(self):
        client = make_client(make_settings())
        resp = post_webhook(client, {"action": "created"}, event="issue_comment")
        assert resp.status_code == 200
        assert "ignored" in resp.json()

    def test_issue_reopened_action_ignored(self):
        client = make_client(make_settings())
        payload = issue_opened_payload()
        payload["action"] = "reopened"
        resp = post_webhook(client, payload)
        assert "ignored" in resp.json()

    def test_issue_closed_without_tracked_prediction(self):
        """closed events feed the online-evaluation loop even when we never
        scored the issue — the outcome is computed but matches nothing."""
        client = make_client(make_settings())
        payload = issue_opened_payload()
        payload["action"] = "closed"
        payload["issue"]["labels"] = []
        payload["issue"]["state_reason"] = "not_planned"
        resp = post_webhook(client, payload)
        data = resp.json()
        assert data["outcome"] == 0
        assert data["matched_prediction"] is False

    def test_bot_author_ignored(self):
        predictor = StubPredictor()
        client = make_client(make_settings(), predictor=predictor)
        resp = post_webhook(client, issue_opened_payload(user={"login": "dependabot[bot]"}))
        assert "ignored" in resp.json()
        assert predictor.calls == []


# ---------------------------------------------------------------------------
# issues.opened flow
# ---------------------------------------------------------------------------
class TestIssueOpened:
    def test_dry_run_scores_but_never_writes(self):
        gh = StubGitHub()
        client = make_client(
            make_settings(dry_run=True, post_comment=True, apply_label=True), gh=gh
        )
        resp = post_webhook(client, issue_opened_payload())
        data = resp.json()
        assert resp.status_code == 200
        assert data["dry_run"] is True
        assert data["actions"] == []
        assert data["prediction"]["predicted_class"] == "actionable-bug"
        assert gh.comments == [] and gh.labels == []

    def test_write_mode_comments_and_labels_positive(self):
        gh = StubGitHub()
        client = make_client(
            make_settings(dry_run=False, post_comment=True, apply_label=True), gh=gh
        )
        resp = post_webhook(client, issue_opened_payload())
        data = resp.json()
        assert data["actions"] == ["comment", "label"]
        assert len(gh.comments) == 1
        assert gh.labels == [("acme/widgets", 42, ["predicted:actionable-bug"])]

    def test_negative_prediction_never_labeled(self):
        gh = StubGitHub()
        client = make_client(
            make_settings(dry_run=False, post_comment=False, apply_label=True),
            predictor=StubPredictor(proba=0.1),
            gh=gh,
        )
        resp = post_webhook(client, issue_opened_payload())
        assert resp.json()["actions"] == []
        assert gh.labels == []

    def test_enrichment_feeds_predictor(self):
        predictor = StubPredictor()
        client = make_client(make_settings(), predictor=predictor, gh=StubGitHub())
        post_webhook(client, issue_opened_payload())
        call = predictor.calls[0]
        assert call["author_created_at"] == "2020-01-01T00:00:00Z"
        assert call["author_public_repos"] == 5
        assert call["latest_release_iso"] == "2024-06-01T00:00:00Z"

    def test_enrichment_disabled_degrades_to_none(self):
        predictor = StubPredictor()
        client = make_client(make_settings(enrich=False), predictor=predictor,
                             gh=StubGitHub())
        post_webhook(client, issue_opened_payload())
        call = predictor.calls[0]
        assert call["author_created_at"] is None
        assert call["latest_release_iso"] is None

    def test_malformed_payload_is_422(self):
        client = make_client(make_settings())
        resp = post_webhook(client, {"action": "opened", "issue": {}})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/predict
# ---------------------------------------------------------------------------
class TestPredictApi:
    def test_predict_requires_token(self):
        client = make_client(make_settings())
        resp = client.post("/api/predict", json={"title": "crash"})
        assert resp.status_code == 401

    def test_predict_scores_with_token(self):
        client = make_client(make_settings())
        resp = client.post("/api/predict", json={"title": "crash on save"},
                           headers={"X-GHIC-Token": SECRET})
        assert resp.status_code == 200
        assert resp.json()["proba_actionable_bug"] == 0.9

    def test_predict_empty_issue_rejected(self):
        client = make_client(make_settings())
        resp = client.post("/api/predict", json={},
                           headers={"X-GHIC-Token": SECRET})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Per-repo thresholds + /stats
# ---------------------------------------------------------------------------
class TestThresholds:
    def test_parse_repo_thresholds(self):
        from ghic.service.settings import parse_repo_thresholds

        assert parse_repo_thresholds("") == {}
        assert parse_repo_thresholds("a/b=0.35, c/d=0.6") == {"a/b": 0.35, "c/d": 0.6}
        with pytest.raises(ValueError):
            parse_repo_thresholds("a/b")

    def test_threshold_for_prefers_repo_override(self):
        s = make_settings(threshold=0.5, repo_thresholds={"acme/widgets": 0.3})
        assert s.threshold_for("acme/widgets") == 0.3
        assert s.threshold_for("other/repo") == 0.5

    def test_handler_passes_repo_threshold_to_predictor(self):
        predictor = StubPredictor()
        client = make_client(
            make_settings(repo_thresholds={"acme/widgets": 0.3}), predictor=predictor
        )
        post_webhook(client, issue_opened_payload())
        assert predictor.calls[0]["threshold"] == 0.3


class TestStats:
    def test_stats_requires_token(self):
        client = make_client(make_settings())
        assert client.get("/stats").status_code == 401

    def test_stats_accumulates_predictions(self):
        client = make_client(make_settings())
        for n in (1, 2, 3):
            post_webhook(client, issue_opened_payload(number=n))
        resp = client.get("/stats", headers={"X-GHIC-Token": SECRET})
        data = resp.json()
        assert data["scored"] == 3
        assert data["predicted_actionable"] == 3          # stub proba 0.9 >= 0.5
        assert data["positive_rate"] == 1.0
        assert len(data["recent"]) == 3
        assert data["recent"][-1]["issue"] == 3


# ---------------------------------------------------------------------------
# Backtest calibration math
# ---------------------------------------------------------------------------
class TestCalibration:
    def test_best_threshold_finds_separating_cut(self):
        import numpy as np

        from ghic.backtest import best_threshold

        # positives cluster at 0.4, negatives at 0.2 — best F1 needs t <= 0.4,
        # which the default 0.5 misses entirely.
        y = np.array([0] * 50 + [1] * 50)
        p = np.array([0.2] * 50 + [0.4] * 50)
        t = best_threshold(y, p)
        assert t <= 0.4

    def test_best_threshold_degenerate_slice_defaults(self):
        import numpy as np

        from ghic.backtest import best_threshold

        assert best_threshold(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7])) == 0.5

    def test_calibrate_reports_per_repo_and_env_line(self):
        from ghic.backtest import calibrate, format_report

        records = [
            {"repo": "a/b", "number": i, "created_at": f"2024-01-{i:02d}",
             "y_true": i % 2, "proba": 0.8 if i % 2 else 0.1}
            for i in range(1, 21)
        ]
        result = calibrate(records)
        assert "a/b" in result["repos"]
        assert result["overall"]["at_default"]["n"] == 20
        report = format_report(result)
        assert "GHIC_REPO_THRESHOLDS=a/b=" in report


# ---------------------------------------------------------------------------
# Online evaluation: predictions graded at close time
# ---------------------------------------------------------------------------
def issue_closed_payload(number: int = 42, labels: list[str] | None = None,
                         state_reason: str | None = "not_planned") -> dict[str, Any]:
    return {
        "action": "closed",
        "issue": {
            "number": number,
            "user": {"login": "alice"},
            "locked": False,
            "state_reason": state_reason,
            "labels": [{"name": name} for name in (labels or [])],
        },
        "repository": {"full_name": "acme/widgets"},
        "installation": {"id": 123},
    }


class TestOnlineEvaluation:
    def test_tracker_confusion_math(self, tmp_path):
        from ghic.service.tracking import PredictionTracker

        t = PredictionTracker(ledger_path=tmp_path / "ledger.jsonl")
        t.record_prediction("a/b", 1, 0.9, 1)   # predicted bug, truth bug -> tp
        t.record_prediction("a/b", 2, 0.8, 1)   # predicted bug, truth non  -> fp
        t.record_prediction("a/b", 3, 0.1, 0)   # predicted non, truth bug  -> fn
        assert t.record_outcome("a/b", 1, 1)
        assert t.record_outcome("a/b", 2, 0)
        assert t.record_outcome("a/b", 3, 1)
        assert not t.record_outcome("a/b", 99, 0)   # never scored
        s = t.summary()
        assert s["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 0}
        assert s["live_precision"] == 0.5
        assert s["live_recall_lower_bound"] == 0.5

    def test_tracker_ledger_survives_restart(self, tmp_path):
        from ghic.service.tracking import PredictionTracker

        ledger = tmp_path / "ledger.jsonl"
        t1 = PredictionTracker(ledger_path=ledger)
        t1.record_prediction("a/b", 1, 0.9, 1)
        t1.record_outcome("a/b", 1, 1)
        t1.record_prediction("a/b", 2, 0.7, 1)      # still awaiting outcome

        t2 = PredictionTracker(ledger_path=ledger)  # simulated restart
        assert t2.summary()["confusion"]["tp"] == 1
        assert t2.summary()["awaiting_outcome"] == 1
        assert t2.record_outcome("a/b", 2, 0)
        assert t2.summary()["confusion"]["fp"] == 1

    def test_closed_event_grades_earlier_prediction(self):
        client = make_client(make_settings())
        post_webhook(client, issue_opened_payload(number=7))          # scored 1 (stub 0.9)
        resp = post_webhook(client, issue_closed_payload(number=7,
                                                         state_reason="not_planned"))
        data = resp.json()
        assert data["outcome"] == 0                # NOT_PLANNED -> non-actionable
        assert data["matched_prediction"] is True
        stats = client.get("/stats", headers={"X-GHIC-Token": SECRET}).json()
        assert stats["online_evaluation"]["confusion"]["fp"] == 1

    def test_closed_event_bug_label_completed_is_class1(self):
        client = make_client(make_settings())
        post_webhook(client, issue_opened_payload(number=8))
        resp = post_webhook(client, issue_closed_payload(
            number=8, labels=["bug"], state_reason="completed"))
        assert resp.json()["outcome"] == 1
        stats = client.get("/stats", headers={"X-GHIC-Token": SECRET}).json()
        assert stats["online_evaluation"]["confusion"]["tp"] == 1

    def test_closed_question_labeled_issue_is_ignored(self):
        client = make_client(make_settings())
        resp = post_webhook(client, issue_closed_payload(number=9, labels=["question"]))
        assert "ignored" in resp.json()


# ---------------------------------------------------------------------------
# Pluggable ledger backend (JSONL vs Postgres, for deploys with no
# persistent disk — see ghic/service/pg_ledger.py). The in-memory rebuild
# logic (record_*/summary/analytics) is backend-agnostic, so an injected
# fake backend should behave identically to the JSONL default.
# ---------------------------------------------------------------------------
class _FakeLedgerBackend:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def append(self, record: dict) -> None:
        self.records.append(record)

    def replay(self):
        return iter(list(self.records))


class TestLedgerBackend:
    def test_injected_backend_persists_across_simulated_restart(self):
        from ghic.service.tracking import PredictionTracker

        backend = _FakeLedgerBackend()
        t1 = PredictionTracker(backend=backend)
        t1.record_prediction("a/b", 1, 0.9, 1)
        t1.record_outcome("a/b", 1, 1)
        t1.record_prediction("a/b", 2, 0.7, 1)
        assert len(backend.records) == 3

        t2 = PredictionTracker(backend=backend)  # same backend, fresh instance
        assert t2.summary()["confusion"]["tp"] == 1
        assert t2.summary()["awaiting_outcome"] == 1

    def test_database_url_takes_precedence_over_ledger_path(self, tmp_path, monkeypatch):
        from ghic.service import pg_ledger, tracking

        class _FakePostgresBackend(_FakeLedgerBackend):
            def __init__(self, database_url: str) -> None:
                super().__init__()
                self.database_url = database_url

        monkeypatch.setattr(pg_ledger, "PostgresLedgerBackend", _FakePostgresBackend)
        ledger = tmp_path / "ledger.jsonl"
        t = tracking.PredictionTracker(ledger_path=ledger, database_url="postgres://fake/db")
        t.record_prediction("a/b", 1, 0.9, 1)
        assert isinstance(t._backend, _FakePostgresBackend)
        assert not ledger.exists()  # nothing written to the file backend

    def test_no_path_or_url_is_in_memory_only(self):
        from ghic.service.tracking import PredictionTracker, _NullBackend

        t = PredictionTracker()
        assert isinstance(t._backend, _NullBackend)
        t.record_prediction("a/b", 1, 0.9, 1)  # must not raise
        assert t.summary()["awaiting_outcome"] == 1

    def test_settings_database_url_env_precedence(self, monkeypatch):
        from ghic.service.settings import load_settings

        for var in ("GHIC_DATABASE_URL", "DATABASE_URL", "POSTGRES_URL"):
            monkeypatch.delenv(var, raising=False)

        monkeypatch.setenv("POSTGRES_URL", "postgres://from-postgres-url")
        assert load_settings().database_url == "postgres://from-postgres-url"

        monkeypatch.setenv("DATABASE_URL", "postgres://from-database-url")
        assert load_settings().database_url == "postgres://from-database-url"

        monkeypatch.setenv("GHIC_DATABASE_URL", "postgres://from-ghic-database-url")
        assert load_settings().database_url == "postgres://from-ghic-database-url"


# ---------------------------------------------------------------------------
# Walk-forward CV + explanation unwrapping (champion protocol pieces)
# ---------------------------------------------------------------------------
class TestChampionProtocol:
    def test_walk_forward_folds_respect_time(self):
        import pandas as pd

        from ghic.train import walk_forward_folds

        frame = pd.DataFrame({
            "repo_name": ["a/b"] * 50 + ["c/d"] * 50,
            "created_at": [f"2024-01-01T{h:02d}:{m:02d}:00Z"
                           for h in range(10) for m in range(10)],
            "label": [0, 1] * 50,
        })
        folds = walk_forward_folds(frame, n_folds=3, val_fraction=0.1)
        assert len(folds) == 3
        for tr, va in folds:
            assert len(tr) and len(va)
            for repo in ("a/b", "c/d"):
                tr_r = tr[tr.repo_name == repo]
                va_r = va[va.repo_name == repo]
                assert tr_r["created_at"].max() <= va_r["created_at"].min()
        # expanding window: later folds train on strictly more data
        assert len(folds[0][0]) < len(folds[1][0]) < len(folds[2][0])

    def test_unwrap_pipeline_looks_through_calibration(self):
        from ghic.evaluate import unwrap_pipeline

        class FakePipe:
            named_steps = {"pre": None, "clf": None}

        class FakeFrozen:
            estimator = FakePipe()

        class FakeCalibrated:
            calibrated_classifiers_ = [type("CC", (), {"estimator": FakeFrozen()})()]

        assert unwrap_pipeline(FakePipe()) is not None
        assert unwrap_pipeline(FakeCalibrated()) is not None
        assert unwrap_pipeline(object()) is None


# ---------------------------------------------------------------------------
# Duplicate surfacing (assistive candidates on issues.opened)
# ---------------------------------------------------------------------------
class StubDupIndex:
    meta = [{"repo": "acme/widgets"}]

    def query(self, repo, title, body, k=3, min_sim=0.55):
        return [{"number": 7, "title": "Old crash on save", "similarity": 0.83}]


class TestRelatedIssues:
    def test_related_candidates_in_response_and_comment(self):
        gh = StubGitHub()
        app = create_app(make_settings(dry_run=False, post_comment=True,
                                       suggest_related=True),
                         predictor=StubPredictor(), gh_client=gh,
                         dup_index=StubDupIndex())
        client = TestClient(app)
        resp = post_webhook(client, issue_opened_payload())
        assert resp.json()["related_issues"][0]["number"] == 7
        assert "#7" in gh.comments[0][2]
        assert "please verify" in gh.comments[0][2]

    def test_dup_index_failure_never_blocks_prediction(self):
        class BrokenIndex:
            meta = []

            def query(self, *a, **k):
                raise RuntimeError("index corrupt")

        app = create_app(make_settings(suggest_related=True),
                         predictor=StubPredictor(), dup_index=BrokenIndex())
        resp = post_webhook(TestClient(app), issue_opened_payload())
        assert resp.status_code == 200
        assert resp.json()["related_issues"] == []


# ---------------------------------------------------------------------------
# issues.edited / label events / Projects v2 (Phase 18 API surface)
# ---------------------------------------------------------------------------
class TestIssueEdited:
    def edited_payload(self, **issue_overrides: Any) -> dict[str, Any]:
        p = issue_opened_payload(**issue_overrides)
        p["action"] = "edited"
        p["issue"]["state"] = "open"
        return p

    def test_edit_rescores_and_updates_pending_prediction(self):
        predictor = StubPredictor(proba=0.2)
        app = create_app(make_settings(), predictor=predictor)
        client = TestClient(app)
        post_webhook(client, issue_opened_payload())
        predictor.proba = 0.9            # the reporter added repro steps
        resp = post_webhook(client, self.edited_payload())
        assert resp.json()["rescored"] is True
        assert resp.json()["prediction"]["proba_actionable_bug"] == 0.9
        # the pending entry graded at close must reflect the re-score
        assert app.state.tracker.pending[("acme/widgets", 42)]["proba"] == 0.9
        assert app.state.totals["rescored"] == 1
        assert app.state.totals["scored"] == 1   # edits don't inflate 'scored'

    def test_edit_never_posts_even_when_comments_enabled(self):
        gh = StubGitHub()
        app = create_app(make_settings(dry_run=False, post_comment=True,
                                       apply_label=True),
                         predictor=StubPredictor(), gh_client=gh)
        post_webhook(TestClient(app), self.edited_payload())
        assert gh.comments == []
        assert gh.labels == []

    def test_edit_on_closed_issue_ignored(self):
        app = create_app(make_settings(), predictor=StubPredictor())
        payload = self.edited_payload()
        payload["issue"]["state"] = "closed"
        resp = post_webhook(TestClient(app), payload)
        assert "ignored" in resp.json()


class TestLabelEvents:
    def label_payload(self, action: str, label: str) -> dict[str, Any]:
        p = issue_opened_payload()
        p["action"] = action
        p["label"] = {"name": label}
        return p

    def test_labeled_event_recorded(self):
        app = create_app(make_settings(), predictor=StubPredictor())
        client = TestClient(app)
        resp = post_webhook(client, self.label_payload("labeled", "bug"))
        assert resp.json()["recorded"] == "+bug"
        resp = post_webhook(client, self.label_payload("unlabeled", "bug"))
        assert resp.json()["recorded"] == "-bug"
        assert app.state.tracker.label_events == 2

    def test_label_events_survive_ledger_replay(self, tmp_path):
        from ghic.service.tracking import PredictionTracker

        ledger = tmp_path / "ledger.jsonl"
        t = PredictionTracker(ledger_path=ledger)
        t.record_label_event("acme/widgets", 1, "*duplicate", True)
        t2 = PredictionTracker(ledger_path=ledger)
        assert t2.label_events == 1
        assert t2.summary()["label_events_observed"] == 1


class TestProjectsV2:
    def test_positive_prediction_added_to_project(self):
        gh = StubGitHub()
        app = create_app(make_settings(dry_run=False, project_id="PVT_abc"),
                         predictor=StubPredictor(proba=0.9), gh_client=gh)
        payload = issue_opened_payload(node_id="I_node123")
        resp = post_webhook(TestClient(app), payload)
        assert gh.project_items == [("PVT_abc", "I_node123")]
        assert "project" in resp.json()["actions"]

    def test_negative_prediction_not_added(self):
        gh = StubGitHub()
        app = create_app(make_settings(dry_run=False, project_id="PVT_abc"),
                         predictor=StubPredictor(proba=0.1), gh_client=gh)
        post_webhook(TestClient(app), issue_opened_payload(node_id="I_node123"))
        assert gh.project_items == []

    def test_dry_run_never_touches_project(self):
        gh = StubGitHub()
        app = create_app(make_settings(dry_run=True, project_id="PVT_abc"),
                         predictor=StubPredictor(proba=0.9), gh_client=gh)
        post_webhook(TestClient(app), issue_opened_payload(node_id="I_node123"))
        assert gh.project_items == []


# ---------------------------------------------------------------------------
# Dashboard analytics: the six facets, from real ledger data
# ---------------------------------------------------------------------------
class TestDashboardAnalytics:
    def _client_with_activity(self) -> TestClient:
        app = create_app(make_settings(suggest_related=True),
                         predictor=StubPredictor(proba=0.9),
                         dup_index=StubDupIndex())
        client = TestClient(app)
        post_webhook(client, issue_opened_payload(number=1))   # has dup candidates
        app.state.dup_index = None                             # second one won't
        post_webhook(client, issue_opened_payload(number=2))
        post_webhook(client, issue_closed_payload(number=1, state_reason="not_planned"))
        label_event = issue_opened_payload(number=2)
        label_event["action"] = "labeled"
        label_event["label"] = {"name": "comp:editor"}
        post_webhook(client, label_event)
        return client

    def test_all_six_facets_report_real_numbers(self):
        client = self._client_with_activity()
        a = client.get("/stats", headers={"X-GHIC-Token": SECRET}).json()["analytics"]
        # 1. issue trends: both predictions land on today's date bucket
        assert sum(a["issue_trends"]["predictions_per_day"].values()) == 2
        # 2. duplicate rate: exactly one prediction had candidates
        assert a["duplicate_rate"]["predictions_with_related_candidates"] == 1
        assert a["duplicate_rate"]["rate"] == 0.5
        # 3. resolution analytics: one graded close, one pending
        assert a["resolution_analytics"]["resolved_non_actionable"] == 1
        assert a["resolution_analytics"]["awaiting_outcome"] == 1
        # 4. confidence metrics: histogram mass equals predictions
        assert sum(a["confidence_metrics"]["proba_histogram_deciles"]) == 2
        assert a["confidence_metrics"]["proba_histogram_deciles"][9] == 2  # 0.9s
        # 5. label stats
        assert a["label_stats"]["top_labels_added"] == {"comp:editor": 1}
        # 6. component analytics
        assert a["component_analytics"]["acme/widgets"]["scored"] == 2
        assert a["component_analytics"]["acme/widgets"]["positive_rate"] == 1.0

    def test_analytics_rebuilt_from_ledger_on_restart(self, tmp_path):
        from ghic.service.tracking import PredictionTracker

        ledger = tmp_path / "ledger.jsonl"
        t1 = PredictionTracker(ledger_path=ledger)
        t1.record_prediction("a/b", 1, 0.85, 1, related_count=2)
        t1.record_label_event("a/b", 1, "*duplicate", True)
        t1.record_outcome("a/b", 1, 1)

        t2 = PredictionTracker(ledger_path=ledger)
        a = t2.analytics()
        assert sum(a["issue_trends"]["predictions_per_day"].values()) == 1
        assert a["duplicate_rate"]["predictions_with_related_candidates"] == 1
        assert a["duplicate_rate"]["duplicate_labels_observed_live"] == 1
        assert a["resolution_analytics"]["resolved_actionable"] == 1

    def test_old_ledger_lines_without_timestamps_still_replay(self, tmp_path):
        from ghic.service.tracking import PredictionTracker

        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text(
            '{"type": "prediction", "repo": "a/b", "number": 1, "proba": 0.7, "predicted": 1}\n',
            encoding="utf-8",
        )
        t = PredictionTracker(ledger_path=ledger)
        a = t.analytics()
        assert a["issue_trends"]["predictions_per_day"] == {}   # no timestamp -> no trend point
        assert a["component_analytics"]["a/b"]["scored"] == 1   # still counted everywhere else

    def test_dashboard_page_renders_all_facets(self):
        client = make_client(make_settings())
        html = client.get("/dashboard").text
        for element_id in ("trends", "duprate", "resolutions", "confhist",
                           "labelstats", "components"):
            assert f'id="{element_id}"' in html


# ---------------------------------------------------------------------------
# Category suggestion (assistive, never auto-labeled)
# ---------------------------------------------------------------------------
class StubCategoryPredictor:
    classes = ["bug", "feature", "question"]

    def predict_frame(self, feats):
        return {"predicted": "bug", "confidence": 0.72,
                "proba": {"bug": 0.72, "feature": 0.18, "question": 0.10}}


class TestCategorySuggestion:
    def test_category_in_response_and_comment(self):
        gh = StubGitHub()
        app = create_app(make_settings(dry_run=False, post_comment=True),
                         predictor=StubPredictor(), gh_client=gh,
                         category_predictor=StubCategoryPredictor())
        resp = post_webhook(TestClient(app), issue_opened_payload())
        assert resp.json()["category"]["predicted"] == "bug"
        assert "Suggested category" in gh.comments[0][2]
        assert "**bug**" in gh.comments[0][2]

    def test_category_absent_when_disabled(self):
        client = make_client(make_settings())
        resp = post_webhook(client, issue_opened_payload())
        assert resp.json()["category"] is None

    def test_category_failure_never_blocks_prediction(self):
        class Broken:
            classes = []

            def predict_frame(self, feats):
                raise RuntimeError("model corrupt")

        app = create_app(make_settings(), predictor=StubPredictor(),
                         category_predictor=Broken())
        resp = post_webhook(TestClient(app), issue_opened_payload())
        assert resp.status_code == 200
        assert resp.json()["category"] is None
        assert resp.json()["prediction"]["proba_actionable_bug"] == 0.9

    def test_no_category_label_is_ever_applied(self):
        gh = StubGitHub()
        app = create_app(make_settings(dry_run=False, apply_label=True),
                         predictor=StubPredictor(), gh_client=gh,
                         category_predictor=StubCategoryPredictor())
        post_webhook(TestClient(app), issue_opened_payload())
        assert gh.labels == [("acme/widgets", 42, ["predicted:actionable-bug"])]


# ---------------------------------------------------------------------------
# LLM-assisted analysis, wired into the issues.opened flow
# ---------------------------------------------------------------------------
class StubLLMService:
    """Mirrors ghic.llm.LLMService's contract: analyze_issue never raises,
    returns None to signal "fall back to the ML-only comment"."""

    def __init__(self, analysis=None, calls_log=None):
        self.analysis = analysis
        self.calls = calls_log if calls_log is not None else []

    def analyze_issue(self, context):
        self.calls.append(context)
        return self.analysis


def make_llm_analysis(**overrides):
    from ghic.llm.models import IssueAnalysis

    defaults = dict(
        category="bug", priority="high", severity="medium", confidence=0.92,
        summary="The app crashes on save.",
        reasoning=["Clear reproduction steps and a stack trace are present."],
        recommended_action="Investigate immediately -- this affects a core workflow.",
        risk_level="high",
        risk_reasons=["Blocks a core workflow", "Reproducible"],
        business_impact=["Users may lose unsaved work"],
        missing_information=["Application version", "Operating system"],
        recommended_labels=["bug", "backend"],
    )
    defaults.update(overrides)
    return IssueAnalysis(**defaults)


class StubRepoIntelligence:
    """Mirrors RepositoryIntelligenceService's contract: get_context never
    raises and returns a RepositoryContext (possibly empty)."""

    def __init__(self, context=None, raises=None):
        self.context = context
        self.raises = raises
        self.calls = []

    def get_context(self, repo, title, body, **kwargs):
        self.calls.append((repo, title, body, kwargs))
        if self.raises:
            raise self.raises
        return self.context


def make_repo_context(**overrides):
    from ghic.repository_intelligence.models import (
        CodeChunk,
        RepositoryContext,
        RepositoryMetadata,
        RetrievedChunk,
    )

    chunk = CodeChunk(
        repo="acme/widgets", path="src/csv_parser.py", language="Python",
        text="def parse_csv(path):\n    return open(path).read()",
        start_line=10, end_line=11, kind="function", symbol="parse_csv",
    )
    defaults = dict(
        repo="acme/widgets",
        metadata=RepositoryMetadata(repo="acme/widgets", primary_language="Python",
                                    frameworks=["FastAPI"]),
        chunks=[RetrievedChunk(chunk=chunk, score=0.72)],
        indexed=True,
    )
    defaults.update(overrides)
    return RepositoryContext(**defaults)


class TestRepositoryIntelligenceIntegration:
    def test_evidence_section_lists_retrieved_files(self):
        pred = Prediction(repo="acme/widgets", issue_number=1, proba=0.9,
                          threshold=0.5, predicted_label=1, model_name="stub")
        comment = format_llm_comment(
            pred, make_llm_analysis(), repository_context=make_repo_context(),
        )
        assert "### Repository Evidence" in comment
        assert "`src/csv_parser.py`" in comment
        assert "parse_csv" in comment
        assert "lines 10-11" in comment

    def test_no_evidence_section_when_engine_is_off(self):
        pred = Prediction(repo="acme/widgets", issue_number=1, proba=0.9,
                          threshold=0.5, predicted_label=1, model_name="stub")
        comment = format_llm_comment(pred, make_llm_analysis(), repository_context=None)
        assert "Repository Evidence" not in comment

    def test_unindexed_repo_renders_no_section_rather_than_an_empty_one(self):
        from ghic.repository_intelligence.models import RepositoryContext

        pred = Prediction(repo="acme/widgets", issue_number=1, proba=0.9,
                          threshold=0.5, predicted_label=1, model_name="stub")
        comment = format_llm_comment(
            pred, make_llm_analysis(),
            repository_context=RepositoryContext(repo="acme/widgets", indexed=False),
        )
        assert "Repository Evidence" not in comment

    def test_indexed_but_nothing_found_says_so_explicitly(self):
        from ghic.repository_intelligence.models import EMPTY_CONTEXT_NOTE, RepositoryContext

        pred = Prediction(repo="acme/widgets", issue_number=1, proba=0.9,
                          threshold=0.5, predicted_label=1, model_name="stub")
        comment = format_llm_comment(
            pred, make_llm_analysis(),
            repository_context=RepositoryContext(
                repo="acme/widgets", indexed=True, note=EMPTY_CONTEXT_NOTE
            ),
        )
        assert "### Repository Evidence" in comment
        assert EMPTY_CONTEXT_NOTE in comment

    def test_webhook_passes_repository_context_to_the_llm(self):
        repo_ctx = make_repo_context()
        llm = StubLLMService(analysis=make_llm_analysis())
        app = create_app(
            make_settings(dry_run=True), predictor=StubPredictor(),
            llm_service=llm, repo_intelligence=StubRepoIntelligence(repo_ctx),
        )
        post_webhook(TestClient(app), issue_opened_payload())
        assert llm.calls[0].repository_context is repo_ctx

    def test_webhook_response_includes_repository_context(self):
        app = create_app(
            make_settings(dry_run=True), predictor=StubPredictor(),
            repo_intelligence=StubRepoIntelligence(make_repo_context()),
        )
        resp = post_webhook(TestClient(app), issue_opened_payload())
        assert resp.json()["repository_context"]["relevant_files"] == ["src/csv_parser.py"]

    def test_absent_when_not_configured(self):
        app = create_app(make_settings(dry_run=True), predictor=StubPredictor())
        resp = post_webhook(TestClient(app), issue_opened_payload())
        assert resp.status_code == 200
        assert resp.json()["repository_context"] is None

    def test_repo_intelligence_failure_never_breaks_the_webhook(self):
        """get_context is contractually non-raising, but the webhook must
        survive even a broken implementation that violates that."""
        gh = StubGitHub()
        app = create_app(
            make_settings(dry_run=False, post_comment=True), predictor=StubPredictor(),
            gh_client=gh, llm_service=StubLLMService(analysis=make_llm_analysis()),
            repo_intelligence=StubRepoIntelligence(raises=RuntimeError("index exploded")),
        )
        resp = post_webhook(TestClient(app), issue_opened_payload())
        assert resp.status_code == 200
        assert gh.comments  # the comment still went out, without evidence
        assert "Repository Evidence" not in gh.comments[0][2]


class TestLLMAnalysisComment:
    def test_llm_comment_used_when_analysis_available(self):
        gh = StubGitHub()
        llm = StubLLMService(analysis=make_llm_analysis())
        app = create_app(make_settings(dry_run=False, post_comment=True),
                         predictor=StubPredictor(), gh_client=gh, llm_service=llm)
        resp = post_webhook(TestClient(app), issue_opened_payload())

        assert resp.status_code == 200
        assert resp.json()["llm_analysis"]["category"] == "bug"
        comment = gh.comments[0][2]
        assert "GHIC Analysis" in comment
        assert "High" in comment  # priority
        assert "Application version" in comment
        assert "`bug`" in comment and "`backend`" in comment
        # The whole point: no raw sklearn/ML internals in the public comment.
        assert "numeric__" not in comment
        assert "tfidf__" not in comment
        assert "feature importance" not in comment.lower()
        # v2: no implementation-flavored naming anywhere in the comment.
        assert "ML Actionability Score" not in comment
        assert "prompt" not in comment.lower()
        assert "token" not in comment.lower()
        assert "temperature" not in comment.lower()
        assert "provider" not in comment.lower()

    def test_falls_back_to_ml_comment_when_llm_unavailable(self):
        gh = StubGitHub()
        llm = StubLLMService(analysis=None)  # disabled, missing key, or failed
        app = create_app(make_settings(dry_run=False, post_comment=True),
                         predictor=StubPredictor(), gh_client=gh, llm_service=llm)
        resp = post_webhook(TestClient(app), issue_opened_payload())

        assert resp.status_code == 200
        assert resp.json()["llm_analysis"] is None
        comment = gh.comments[0][2]
        assert "Issue triage prediction" in comment  # the original format_comment header

    def test_no_llm_service_configured_behaves_like_before(self):
        client = make_client(make_settings())
        resp = post_webhook(client, issue_opened_payload())
        assert resp.status_code == 200
        assert resp.json()["llm_analysis"] is None

    def test_llm_context_carries_the_ml_probability_unmodified(self):
        llm = StubLLMService(analysis=make_llm_analysis())
        app = create_app(make_settings(dry_run=True), predictor=StubPredictor(proba=0.73),
                         llm_service=llm)
        post_webhook(TestClient(app), issue_opened_payload())
        assert llm.calls[0].ml_probability == 0.73
        assert llm.calls[0].ml_predicted_label == 1

    def test_missing_info_draft_skipped_when_llm_analysis_present(self):
        """Avoids a redundant second LLM call and a duplicate section when
        both GHIC_DRAFT_MISSING_INFO and GHIC_USE_LLM_ANALYSIS are on."""
        gh = StubGitHub()
        llm = StubLLMService(analysis=make_llm_analysis())
        app = create_app(
            make_settings(dry_run=False, post_comment=True, draft_missing_info=True),
            predictor=StubPredictor(), gh_client=gh, llm_service=llm,
        )
        resp = post_webhook(TestClient(app), issue_opened_payload())
        assert resp.json()["missing_info"] is None

    def test_disagreement_renders_warning_instead_of_a_verdict(self):
        pred = Prediction(
            repo="acme/widgets", issue_number=1, proba=0.12, threshold=0.5,
            predicted_label=0, model_name="stub",
        )
        comment = format_llm_comment(pred, make_llm_analysis(), disagreement=True)
        assert "Needs Maintainer Review" in comment
        assert "Likely Not Actionable" not in comment
        assert "Likely Actionable" not in comment
        # Never expose the underlying mechanism as "the models disagree".
        assert "disagree" not in comment.lower()
        assert "reached different conclusions" in comment

    def test_no_disagreement_renders_the_plain_verdict(self):
        pred = Prediction(
            repo="acme/widgets", issue_number=1, proba=0.12, threshold=0.5,
            predicted_label=0, model_name="stub",
        )
        comment = format_llm_comment(pred, make_llm_analysis(), disagreement=False)
        assert "Likely Not Actionable" in comment
        assert "Needs Maintainer Review" not in comment
        assert "reached different conclusions" not in comment

    def test_executive_summary_leads_the_comment(self):
        """The old '### Summary' heading is gone -- analysis.summary is now
        an unheaded lead paragraph right under the title, doubling as the
        executive summary a maintainer reads first."""
        pred = Prediction(repo="acme/widgets", issue_number=1, proba=0.5,
                           threshold=0.5, predicted_label=1, model_name="stub")
        comment = format_llm_comment(pred, make_llm_analysis())
        assert "### Summary" not in comment
        title_idx = comment.index("GHIC Analysis")
        summary_idx = comment.index("The app crashes on save.")
        table_idx = comment.index("| **Classification**")
        assert title_idx < summary_idx < table_idx

    def test_statistical_risk_score_replaces_ml_actionability_score(self):
        pred = Prediction(repo="acme/widgets", issue_number=1, proba=0.42,
                           threshold=0.5, predicted_label=0, model_name="stub")
        comment = format_llm_comment(pred, make_llm_analysis())
        assert "**Statistical Risk Score** | 42%" in comment
        assert "ML Actionability Score" not in comment
        assert "based on historical issue patterns" in comment

    def test_risk_assessment_section_renders_level_and_reasons(self):
        pred = Prediction(repo="acme/widgets", issue_number=1, proba=0.5,
                           threshold=0.5, predicted_label=1, model_name="stub")
        comment = format_llm_comment(
            pred, make_llm_analysis(risk_level="critical",
                                     risk_reasons=["Blocks a production workflow", "Reproducible"]),
        )
        assert "### Risk Assessment" in comment
        assert "**Critical**" in comment
        assert "- Blocks a production workflow" in comment
        assert "- Reproducible" in comment

    def test_business_impact_section_shown_only_when_present(self):
        pred = Prediction(repo="acme/widgets", issue_number=1, proba=0.5,
                           threshold=0.5, predicted_label=1, model_name="stub")
        with_impact = format_llm_comment(
            pred, make_llm_analysis(business_impact=["Data import unavailable"]),
        )
        assert "### Business Impact" in with_impact
        assert "Data import unavailable" in with_impact

        without_impact = format_llm_comment(pred, make_llm_analysis(business_impact=[]))
        assert "### Business Impact" not in without_impact

    def test_low_confidence_adds_explicit_disclaimer(self):
        pred = Prediction(repo="acme/widgets", issue_number=1, proba=0.5,
                           threshold=0.5, predicted_label=1, model_name="stub")
        low = format_llm_comment(pred, make_llm_analysis(confidence=0.2))
        assert "insufficient to reach a high-confidence assessment" in low

        high = format_llm_comment(pred, make_llm_analysis(confidence=0.9))
        assert "insufficient to reach a high-confidence assessment" not in high

    def test_missing_information_uses_framed_intro_and_is_capped(self):
        pred = Prediction(repo="acme/widgets", issue_number=1, proba=0.5,
                           threshold=0.5, predicted_label=1, model_name="stub")
        comment = format_llm_comment(
            pred, make_llm_analysis(missing_information=["Server stack trace", "Browser version"]),
        )
        assert "To speed up investigation, consider adding:" in comment
        assert "- Server stack trace" in comment

    def test_analysis_details_footer_hides_internals_and_shows_metadata(self):
        from datetime import datetime, timezone

        pred = Prediction(repo="acme/widgets", issue_number=1, proba=0.5,
                           threshold=0.5, predicted_label=1, model_name="stub")
        fixed = datetime(2026, 8, 6, 18, 42, tzinfo=timezone.utc)
        comment = format_llm_comment(pred, make_llm_analysis(), generated_at=fixed)

        assert "### Analysis Details" in comment
        assert "AI Analysis Completed" in comment
        assert "Machine Learning + AI Review" in comment
        assert "2026-08-06 18:42 UTC" in comment
        assert "feature importance" not in comment.lower()
        assert "prompt" not in comment.lower()
        assert "temperature" not in comment.lower()

    def test_webhook_end_to_end_surfaces_disagreement_in_response_and_comment(self):
        """ML calls it non-actionable, but the LLM's own priority/severity/
        reasoning strongly implies an actionable bug -- detect_disagreement()
        should catch it and the webhook should both flag it in the JSON
        response and render it in the posted comment."""
        gh = StubGitHub()
        llm = StubLLMService(analysis=make_llm_analysis(
            priority="critical", severity="critical",
            reasoning=["The report includes a reproducible crash with a stack trace."],
        ))
        app = create_app(
            make_settings(dry_run=False, post_comment=True),
            predictor=StubPredictor(proba=0.1, threshold=0.5),  # -> predicted_label=0
            gh_client=gh, llm_service=llm,
        )
        resp = post_webhook(TestClient(app), issue_opened_payload())

        assert resp.status_code == 200
        assert resp.json()["llm_ml_disagreement"] is True
        comment = gh.comments[0][2]
        assert "Needs Maintainer Review" in comment


class TestCategoryDerivation:
    def test_repo_conventions_normalize(self):
        from ghic.category import derive_category

        assert derive_category(["bug"]) == "bug"
        assert derive_category(["type:bug"]) == "bug"
        assert derive_category(["Type: Bug"]) == "bug"
        assert derive_category(["type:support"]) == "question"
        assert derive_category(["feature-request"]) == "feature"

    def test_priority_resolves_conflicts(self):
        from ghic.category import derive_category

        assert derive_category(["bug", "*duplicate"]) == "duplicate"
        assert derive_category(["feature-request", "*question"]) == "question"

    def test_regression_maps_to_bug(self):
        from ghic.category import derive_category

        assert derive_category(["regression"]) == "bug"

    def test_unmapped_labels_yield_none(self):
        from ghic.category import derive_category

        assert derive_category(["stale", "comp:lite", "tf 2.16"]) is None
        assert derive_category([]) is None


# ---------------------------------------------------------------------------
# Assignment suggestions (similarity mechanism; response-only, never assigned)
# ---------------------------------------------------------------------------
class TestAssignmentSuggestion:
    class StubRecommender:
        def recommend(self, repo, title, body, k=3):
            return [{"login": "alice", "score": 2.1}, {"login": "bob", "score": 0.9}]

    def test_suggestions_in_response_never_in_comment(self):
        gh = StubGitHub()
        app = create_app(make_settings(dry_run=False, post_comment=True),
                         predictor=StubPredictor(), gh_client=gh,
                         assignment_recommender=self.StubRecommender())
        resp = post_webhook(TestClient(app), issue_opened_payload())
        assert resp.json()["suggested_assignees"][0]["login"] == "alice"
        # never in the public comment: suggesting people publicly pings them
        assert "alice" not in gh.comments[0][2]
        # and no assignment/label action of any kind resulted from it
        assert gh.labels == []

    def test_absent_when_disabled(self):
        resp = post_webhook(make_client(make_settings()), issue_opened_payload())
        assert resp.json()["suggested_assignees"] == []

    def test_recommender_failure_never_blocks_prediction(self):
        class Broken:
            def recommend(self, *a, **k):
                raise RuntimeError("map corrupt")

        app = create_app(make_settings(), predictor=StubPredictor(),
                         assignment_recommender=Broken())
        resp = post_webhook(TestClient(app), issue_opened_payload())
        assert resp.status_code == 200
        assert resp.json()["suggested_assignees"] == []


# ---------------------------------------------------------------------------
# Effort head (ships only because its run met the pre-declared bar)
# ---------------------------------------------------------------------------
class TestEffortEstimate:
    class StubEffort:
        def predict_frame(self, feats):
            return {"bucket": "1–3 weeks", "basis": "test"}

    def test_bucket_boundaries(self):
        from ghic.effort import bucket_for_days

        assert bucket_for_days(0.5) == "a few days"
        assert bucket_for_days(3.0) == "a few days"
        assert bucket_for_days(10) == "1–3 weeks"
        assert bucket_for_days(45) == "1–3 months"
        assert bucket_for_days(400) == "3+ months"

    def test_estimate_in_response_but_never_in_comment(self):
        gh = StubGitHub()
        app = create_app(make_settings(dry_run=False, post_comment=True),
                         predictor=StubPredictor(), gh_client=gh,
                         effort_predictor=self.StubEffort())
        resp = post_webhook(TestClient(app), issue_opened_payload())
        assert resp.json()["estimated_resolution"]["bucket"] == "1–3 weeks"
        # deliberate: a public time estimate reads as a commitment
        assert "1–3 weeks" not in gh.comments[0][2]

    def test_absent_when_disabled(self):
        resp = post_webhook(make_client(make_settings()), issue_opened_payload())
        assert resp.json()["estimated_resolution"] is None

    def test_effort_failure_never_blocks_prediction(self):
        class Broken:
            def predict_frame(self, feats):
                raise RuntimeError("boom")

        app = create_app(make_settings(), predictor=StubPredictor(),
                         effort_predictor=Broken())
        resp = post_webhook(TestClient(app), issue_opened_payload())
        assert resp.status_code == 200
        assert resp.json()["estimated_resolution"] is None


# ---------------------------------------------------------------------------
# Missing-information drafting (trigger is the load-bearing part)
# ---------------------------------------------------------------------------
class TestDrafting:
    def test_vague_issue_triggers(self):
        from ghic.service.drafting import needs_more_info

        assert needs_more_info("app broken", "it doesnt work pls fix")

    def test_detailed_report_does_not_trigger(self):
        from ghic.service.drafting import needs_more_info

        body = ("Steps to reproduce:\n1. open\n2. crash\n\n```\nTraceback (most recent "
                "call last)\n```\nExpected: no crash. Actual: crash on every launch "
                "since upgrading to v2.1 on Windows 11 with Python 3.12.")
        assert not needs_more_info("Crash on save", body)

    def test_template_fallback_lists_concrete_gaps(self):
        from ghic.service.drafting import draft_missing_info

        result = draft_missing_info("broken", "fix pls")
        assert result is not None
        assert result["source"] in ("template", "llm")
        assert "steps to reproduce" in result["missing"]
        assert "steps to reproduce" in result["draft"] or result["source"] == "llm"

    def test_well_specified_issue_returns_none(self):
        from ghic.service.drafting import draft_missing_info

        body = ("Steps to reproduce: 1. run `foo --bar` 2. observe error\n"
                "```\nValueError: bad input\n```\n"
                "Expected the command to complete; instead it raises on every "
                "run with version 3.2 on Ubuntu 24.04. Happy to add more detail.")
        assert draft_missing_info("ValueError in foo", body) is None

    def test_handler_attaches_draft_when_enabled(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = make_client(make_settings(draft_missing_info=True))
        resp = post_webhook(client, issue_opened_payload(
            title="app broken", body="pls fix"))
        info = resp.json()["missing_info"]
        assert info is not None and len(info["missing"]) >= 2

    def test_handler_skips_draft_when_disabled(self):
        client = make_client(make_settings())
        resp = post_webhook(client, issue_opened_payload(
            title="app broken", body="pls fix"))
        assert resp.json()["missing_info"] is None


# ---------------------------------------------------------------------------
# Observability: latency percentiles, dashboard, OpenAPI export
# ---------------------------------------------------------------------------
class TestObservability:
    def test_stats_reports_latency_percentiles(self):
        client = make_client(make_settings())
        for n in range(3):
            post_webhook(client, issue_opened_payload(number=n + 1))
        stats = client.get("/stats", headers={"X-GHIC-Token": SECRET}).json()
        lat = stats["latency_ms"]["/webhook"]
        assert lat["n"] == 3 and lat["p50"] >= 0 and lat["p99"] >= lat["p50"]

    def test_dashboard_serves_html_without_token(self):
        client = make_client(make_settings())
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "Issue Triage Bot" in resp.text
        assert "X-GHIC-Token" in resp.text     # data still requires the token

    def test_openapi_spec_covers_all_endpoints(self, tmp_path):
        import json

        from ghic.service.app import export_openapi

        path = export_openapi(tmp_path / "openapi.json")
        spec = json.loads(path.read_text(encoding="utf-8"))
        for route in ("/webhook", "/healthz", "/stats", "/api/predict", "/dashboard"):
            assert route in spec["paths"], route


# ---------------------------------------------------------------------------
# Comment formatting
# ---------------------------------------------------------------------------
class TestComment:
    def test_comment_mentions_probability_and_disclaimer(self):
        pred = Prediction(repo="a/b", issue_number=1, proba=0.87, threshold=0.5,
                          predicted_label=1, model_name="rf_balanced")
        text = format_comment(pred)
        assert "87%" in text
        assert "actionable bug" in text
        assert "can be wrong" in text

    def test_comment_shows_confidence_bar_not_raw_features(self):
        pred = Prediction(
            repo="a/b", issue_number=1, proba=0.9, threshold=0.5, predicted_label=1,
            model_name="rf_balanced", signed_contributions=False,
            top_features=[
                {"feature": "numeric__has_code_block", "value": 0.31},
                {"feature": "numeric__has_repro_steps", "value": 0.22},
                {"feature": "text__crash", "value": 0.10},
            ],
        )
        text = format_comment(pred)
        assert "█" in text and "░" in text  # confidence bar rendered
        # The raw sklearn column names must not appear outside the
        # collapsed technical-details block -- the main comment body is
        # everything before that block.
        main_body = text.split("<details>")[0]
        assert "numeric__" not in main_body
        assert "text__" not in main_body
        assert "including a code block" in text
        assert "including reproduction steps" in text

    def test_comment_omits_explanation_when_no_features(self):
        pred = Prediction(repo="a/b", issue_number=1, proba=0.6, threshold=0.5,
                          predicted_label=1, model_name="ensemble", top_features=[])
        text = format_comment(pred)
        assert "Technical details" not in text
        assert "influential" not in text


# ---------------------------------------------------------------------------
# End-to-end inference smoke test against the real trained model
# ---------------------------------------------------------------------------
class TestInferenceSmoke:
    def test_real_model_scores_a_bug_report(self):
        from ghic import utils
        from ghic.service.inference import IssuePredictor

        model = utils.PROJECT_ROOT / "models" / "rf_balanced.joblib"
        if not model.exists():
            pytest.skip("no trained model present (run python -m ghic.train)")
        predictor = IssuePredictor(model, threshold=0.5)
        pred = predictor.predict(
            repo_full_name="acme/widgets",
            issue_number=1,
            title="Crash when opening settings panel",
            body="Steps to reproduce:\n1. open settings\n2. crash\n\n"
                 "```\nTraceback (most recent call last): ...\n```\nExpected: no crash. Actual: crash.",
            created_at="2024-07-01T10:00:00Z",
        )
        assert 0.0 <= pred.proba <= 1.0
        assert pred.predicted_label in (0, 1)
        assert pred.top_features  # explanation produced
