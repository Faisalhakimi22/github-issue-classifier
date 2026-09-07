"""The usage gate on the webhook path, and the ways around it.

The tests that matter here are not "does the limit stop an analysis" -- that
is `test_usage_limits.py`. They are: does it stop it *before* anything
expensive runs, does a failure give the slot back, and does anything the
caller controls move the meter.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from ghic.service import usage
from tests.test_service import (
    SECRET,
    StubGitHub,
    StubPredictor,
    issue_opened_payload,
    make_settings,
    post_webhook,
)

from fastapi.testclient import TestClient

from ghic.service.app import create_app as _create_app


class FakeMeter:
    """The usage module's surface, in memory, with the calls recorded."""

    def __init__(self, limit=None, used=0, plan="starter", enforced=True):
        self.limit = limit
        self.used = used
        self.plan = plan
        self.enforced = enforced
        self.reserved: list[tuple] = []
        self.released: list[tuple] = []
        self.recorded: list[tuple] = []
        self.first_refusal_used = False

    def _plan(self):
        return usage.Plan(
            plan=self.plan,
            max_issues_per_period=self.limit,
            enforced=self.enforced,
        )

    def reserve(self, database_url, workspace_id, repo, issue_number, now=None):
        self.reserved.append((workspace_id, repo, issue_number))
        if not self.enforced:
            return usage.UsageDecision(
                False, "usage_unavailable", usage.UNAVAILABLE, "2026-08"
            )
        if self.limit is None:
            return usage.UsageDecision(True, "unmetered", self._plan(), "2026-08")
        if self.used >= self.limit:
            first = not self.first_refusal_used
            self.first_refusal_used = True
            return usage.UsageDecision(
                False, "issue_quota_exhausted", self._plan(), "2026-08",
                self.used, self.limit, first_refusal=first,
            )
        self.used += 1
        return usage.UsageDecision(
            True, "counted", self._plan(), "2026-08", self.used, self.limit,
            reserved=True,
        )

    def release(self, database_url, workspace_id, period, repo, issue_number,
                reason="analysis_failed"):
        self.released.append((workspace_id, period, repo, issue_number, reason))
        self.used = max(0, self.used - 1)
        return True

    def record(self, database_url, workspace_id, period, repo, issue_number,
               outcome, reason=None):
        self.recorded.append((workspace_id, period, repo, issue_number, outcome, reason))
        return True

    def usage_summary(self, database_url, workspace_id, now=None):
        return {
            "plan": self.plan,
            "period": "2026-08",
            "used": self.used,
            "limit": self.limit,
            "remaining": None if self.limit is None else max(0, self.limit - self.used),
            "enforced": self.enforced,
            "available": True,
            "outcomes": {},
        }

    def has_capacity(self, database_url, workspace_id, now=None):
        if not self.enforced or self.limit is None:
            return True
        return self.used < self.limit

    def period_key(self, period="month", now=None):
        return "2026-08"


def authorize_into(workspace_id):
    def gate(database_url, installation_id, repo_full_name, github_client):
        return {
            "authorized": True,
            "installation_id": int(installation_id),
            "repo": repo_full_name,
            "workspace_id": workspace_id,
        }

    return gate


@contextmanager
def lease(database_url, installation_id, repo_full_name):
    yield {"authorized": True, "installation_id": int(installation_id),
           "repo": repo_full_name, "workspace_id": "ws-1"}


def build(meter, predictor=None, gh=None, **settings):
    # No database_url: the meter is injected, so a real connection would only
    # be the ledger backend trying to reach Postgres from a unit test.
    defaults = dict()
    defaults.update(settings)
    app = _create_app(
        make_settings(**defaults),
        predictor=predictor or StubPredictor(),
        gh_client=gh,
        authorization_gate=authorize_into("ws-1"),
        authorization_lease=lease,
        usage_meter=meter,
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# The gate runs before the work
# ---------------------------------------------------------------------------
def test_an_exhausted_workspace_never_reaches_the_model():
    predictor = StubPredictor()
    gh = StubGitHub()
    meter = FakeMeter(limit=1, used=1)
    client = build(meter, predictor=predictor, gh=gh)

    response = post_webhook(client, issue_opened_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["skipped"] is True
    assert body["quota"] == "exhausted"
    # The gate exists to stop the bill, so the expensive things must not have
    # happened: no enrichment call, no model run, no LLM, no comment.
    assert predictor.calls == []


def test_an_allowed_issue_is_analysed_and_charged():
    predictor = StubPredictor()
    meter = FakeMeter(limit=500, used=0)
    client = build(meter, predictor=predictor)

    response = post_webhook(client, issue_opened_payload())

    assert response.status_code == 200
    assert len(predictor.calls) == 1
    assert meter.reserved == [("ws-1", "acme/widgets", 42)]
    assert meter.used == 1


def test_the_workspace_charged_is_the_one_authorization_decided():
    # Not the one the payload claims. The webhook body is attacker-controlled
    # in the sense that matters here: whoever can reach the endpoint chooses
    # its contents, and a workspace id read from it would let one tenant
    # spend another's quota.
    meter = FakeMeter(limit=500)
    client = build(meter)
    payload = issue_opened_payload()
    payload["_workspace_id"] = "ws-someone-else"
    payload["workspace_id"] = "ws-someone-else"
    payload["plan"] = "enterprise"

    post_webhook(client, payload)

    assert meter.reserved == [("ws-1", "acme/widgets", 42)]


def test_an_unlimited_enterprise_plan_analyses_as_before():
    predictor = StubPredictor()
    meter = FakeMeter(limit=None, plan="enterprise", enforced=True)
    client = build(meter, predictor=predictor)

    assert post_webhook(client, issue_opened_payload()).status_code == 200
    assert len(predictor.calls) == 1


# ---------------------------------------------------------------------------
# Failures are not billed
# ---------------------------------------------------------------------------
def test_a_failed_analysis_gives_the_slot_back():
    class ExplodingPredictor(StubPredictor):
        def predict(self, **kwargs):
            raise RuntimeError("model file corrupt")

    meter = FakeMeter(limit=500)
    client = build(meter, predictor=ExplodingPredictor())

    with pytest.raises(RuntimeError):
        post_webhook(client, issue_opened_payload())

    assert len(meter.released) == 1
    assert meter.released[0][0] == "ws-1"
    assert meter.released[0][4] == "RuntimeError"
    assert meter.used == 0


def test_a_revoked_lease_mid_job_gives_the_slot_back():
    @contextmanager
    def revoked_lease(database_url, installation_id, repo_full_name):
        yield {"authorized": False, "reason": "revoked_installation"}

    meter = FakeMeter(limit=500)
    app = _create_app(
        make_settings(dry_run=False, post_comment=True),
        predictor=StubPredictor(),
        gh_client=StubGitHub(),
        authorization_gate=authorize_into("ws-1"),
        authorization_lease=revoked_lease,
        usage_meter=meter,
    )
    response = post_webhook(TestClient(app), issue_opened_payload())

    assert response.json()["authorization"] == "rejected"
    # Access was revoked between the reservation and the write, so nothing
    # was delivered. No delivery, no charge.
    assert len(meter.released) == 1
    assert meter.used == 0


# ---------------------------------------------------------------------------
# The refusal notice
# ---------------------------------------------------------------------------
def test_a_quota_refusal_never_writes_to_github():
    gh = StubGitHub()
    meter = FakeMeter(limit=1, used=1)
    client = build(meter, gh=gh, dry_run=False, post_comment=True)

    first = post_webhook(client, issue_opened_payload(number=1))
    second = post_webhook(client, issue_opened_payload(number=2))

    assert first.json()["notified"] is False
    assert second.json()["notified"] is False
    assert gh.comments == []


def test_a_dry_run_deployment_announces_nothing():
    gh = StubGitHub()
    meter = FakeMeter(limit=1, used=1)
    client = build(meter, gh=gh, dry_run=True, post_comment=True)

    assert post_webhook(client, issue_opened_payload()).json()["notified"] is False
    assert gh.comments == []


def test_quota_refusal_does_not_call_the_github_client():
    class FailingGitHub(StubGitHub):
        def post_comment(self, *args, **kwargs):
            pytest.fail("quota refusal must not perform a GitHub write")

    meter = FakeMeter(limit=1, used=1)
    client = build(meter, gh=FailingGitHub(), dry_run=False, post_comment=True)

    response = post_webhook(client, issue_opened_payload())
    assert response.status_code == 200
    assert response.json()["notified"] is False


def test_usage_outage_stops_processing_without_a_github_write():
    class UnavailableMeter(FakeMeter):
        def reserve(self, database_url, workspace_id, repo, issue_number, now=None):
            return usage.UsageDecision(
                False,
                "usage_unavailable",
                usage.UNAVAILABLE,
                "2026-08-27",
            )

    predictor = StubPredictor()
    gh = StubGitHub()
    response = post_webhook(
        build(UnavailableMeter(), predictor=predictor, gh=gh),
        issue_opened_payload(),
    )

    assert response.json()["quota"] == "unavailable"
    assert predictor.calls == []
    assert gh.comments == []


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------
def edited_payload(**overrides):
    payload = issue_opened_payload(**overrides)
    payload["action"] = "edited"
    return payload


def test_a_rescore_is_not_charged_again():
    predictor = StubPredictor()
    meter = FakeMeter(limit=500, used=1)
    client = build(meter, predictor=predictor)

    response = post_webhook(client, edited_payload())

    assert response.json()["rescored"] is True
    assert len(predictor.calls) == 1
    # The customer paid for this issue when it was opened. A month boundary
    # falling between the open and the edit must not bill it twice.
    assert meter.reserved == []
    assert meter.used == 1


def test_an_exhausted_workspace_does_not_get_free_rescores():
    predictor = StubPredictor()
    meter = FakeMeter(limit=1, used=1)
    client = build(meter, predictor=predictor)

    response = post_webhook(client, edited_payload())

    assert response.json()["quota"] == "exhausted"
    assert predictor.calls == []
    assert meter.recorded[0][4] == "skipped"


def test_usage_outage_stops_an_edited_issue_before_prediction_or_github_write():
    class UnavailableMeter(FakeMeter):
        def usage_summary(self, database_url, workspace_id, now=None):
            return {
                "available": False,
                "enforced": False,
                "plan": "unavailable",
                "period": "",
                "used": 0,
                "limit": None,
                "remaining": None,
                "outcomes": {},
            }

    predictor = StubPredictor()
    gh = StubGitHub()
    response = post_webhook(
        build(UnavailableMeter(), predictor=predictor, gh=gh),
        edited_payload(),
    )

    assert response.status_code == 200
    assert response.json()["quota"] == "unavailable"
    assert response.json()["rescored"] is False
    assert predictor.calls == []
    assert gh.comments == []


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def test_a_redelivered_webhook_is_discarded_before_the_meter():
    import json

    meter = FakeMeter(limit=500)
    client = build(meter)
    payload = issue_opened_payload()
    body = json.dumps(payload).encode()

    import hashlib
    import hmac

    headers = {
        "X-GitHub-Event": "issues",
        "X-GitHub-Delivery": "delivery-1",
        "X-Hub-Signature-256": "sha256=" + hmac.new(
            SECRET.encode(), body, hashlib.sha256
        ).hexdigest(),
        "Content-Type": "application/json",
    }
    client.post("/webhook", content=body, headers=headers)
    client.post("/webhook", content=body, headers=headers)

    # Two layers stop the double charge, and this is the outer one: the
    # delivery id never reaches the meter twice. The partial unique index in
    # ghic_usage_events is the inner one, for retries that arrive with a
    # fresh delivery id.
    assert len(meter.reserved) == 1
    assert meter.used == 1
