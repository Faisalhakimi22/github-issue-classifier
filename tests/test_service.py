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
from ghic.service.inference import Prediction, format_comment  # noqa: E402
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

    def get_user(self, login: str, installation_id: int) -> dict[str, Any]:
        return {"created_at": "2020-01-01T00:00:00Z", "public_repos": 5, "followers": 2}

    def get_latest_release_date(self, full_name: str, installation_id: int) -> str:
        return "2024-06-01T00:00:00Z"

    def post_comment(self, full_name, issue_number, body, installation_id) -> None:
        self.comments.append((full_name, issue_number, body))

    def add_labels(self, full_name, issue_number, labels, installation_id) -> None:
        self.labels.append((full_name, issue_number, labels))


def make_settings(**overrides: Any) -> ServiceSettings:
    defaults: dict[str, Any] = dict(
        model_path=Path("unused.joblib"),
        webhook_secret=SECRET,
        dry_run=True,
        suggest_related=False,   # tests must not depend on models/dup_index.joblib
        suggest_category=False,  # ...nor on models/category.joblib
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

    def test_issue_edited_action_ignored(self):
        client = make_client(make_settings())
        payload = issue_opened_payload()
        payload["action"] = "edited"
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
        assert "0.87" in text
        assert "actionable bug" in text
        assert "can be wrong" in text


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
