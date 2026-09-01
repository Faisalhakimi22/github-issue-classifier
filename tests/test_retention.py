"""Retention: the 90-day promise, and its blast radius.

The privacy policy says prediction records are deleted after 90 days. The
risk in making that true is not that the sweep fails to run -- it is that it
deletes something nobody promised to delete. So most of what is asserted
here is about what survives.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from ghic.service import retention
from ghic.service.app import create_app
from tests.test_service import StubPredictor, make_settings


class FakeRetentionDatabase:
    def __init__(self, existing=("ghic_ledger", "ghic_idempotency"), rows=None):
        self.existing = set(existing)
        self.rows = rows if rows is not None else {
            "ghic_ledger": 12, "ghic_idempotency": 5,
        }
        self.deletes: list[tuple[str, int]] = []

    def run(self, sql, **params):
        flat = " ".join(sql.split())
        if flat.startswith("SELECT to_regclass"):
            name = params["q"].split(".", 1)[1]
            return [[name in self.existing]]
        if "DELETE FROM" in flat:
            table = flat.split("DELETE FROM ", 1)[1].split(" ", 1)[0]
            self.deletes.append((table, params["days"]))
            return [[self.rows.get(table, 0)]]
        raise AssertionError(f"unexpected SQL: {flat}")

    def close(self):
        pass


@pytest.fixture
def db(monkeypatch):
    fake = FakeRetentionDatabase()

    @contextmanager
    def fake_session(_url):
        yield fake

    monkeypatch.setattr(retention, "session", fake_session)
    return fake


def test_the_sweep_deletes_the_ledger_and_the_idempotency_keys(db):
    result = retention.sweep("postgres://test", 90)
    assert result["ok"] is True
    assert result["total"] == 17
    assert sorted(table for table, _ in db.deletes) == [
        "ghic_idempotency", "ghic_ledger",
    ]


def test_nothing_else_is_touched(db):
    retention.sweep("postgres://test", 90)
    swept = {table for table, _ in db.deletes}
    # Billing records a customer needs to dispute a charge, the repository
    # index, the connection state that makes authorization work, and every
    # account and workspace row: none of these were promised to anyone as
    # deleted, and a retention sweep that took them would be data loss
    # dressed up as a privacy feature.
    for survivor in (
        "ghic_usage_events",
        "ghic_plans",
        "ghic_repo_chunks",
        "ghic_repository_state",
        "ghic_github_installations",
        "ghic_github_repositories",
        "ghic_users",
        "ghic_workspaces",
    ):
        assert survivor not in swept


def test_the_window_is_the_configured_one(db):
    retention.sweep("postgres://test", 30)
    assert {days for _, days in db.deletes} == {30}


def test_the_window_is_never_zero_or_negative(db):
    # A window of zero would mean "delete everything", which is not a
    # retention policy and not what any misconfiguration should produce.
    retention.sweep("postgres://test", 0)
    assert {days for _, days in db.deletes} == {1}


def test_an_absent_table_is_skipped_not_an_error(monkeypatch):
    fake = FakeRetentionDatabase(existing=("ghic_ledger",))

    @contextmanager
    def fake_session(_url):
        yield fake

    monkeypatch.setattr(retention, "session", fake_session)
    result = retention.sweep("postgres://test", 90)
    assert result["ok"] is True
    assert [table for table, _ in fake.deletes] == ["ghic_ledger"]


def test_no_database_is_reported_rather_than_crashed():
    assert retention.sweep("", 90)["ok"] is False


def test_the_window_is_never_interpolated_into_sql(db):
    # `days` arrives as a bound parameter. An injected window would be the
    # one place a number becomes SQL in a statement that deletes rows.
    retention.sweep("postgres://test", 90)
    assert all(isinstance(days, int) for _, days in db.deletes)


# ---------------------------------------------------------------------------
# The scheduled endpoint
# ---------------------------------------------------------------------------
def client(database_url="", **settings):
    """An app whose settings name a database the retention sweep will reach.

    Built without `database_url` and then given one, so that constructing
    the app does not also build a Postgres ledger backend and try to open a
    real connection from a unit test.
    """
    from dataclasses import replace

    app = create_app(make_settings(**settings), predictor=StubPredictor())
    if database_url:
        app.state.settings = replace(app.state.settings, database_url=database_url)
    return TestClient(app)


def test_the_endpoint_refuses_without_a_configured_secret():
    response = client().get("/internal/retention")
    # Not 401. There is nothing to authenticate against, and an
    # unauthenticated deletion endpoint reachable from the internet is worse
    # than a retention promise that is late.
    assert response.status_code == 503


def test_the_endpoint_refuses_a_wrong_secret():
    c = client(cron_secret="s3cret")
    assert c.get("/internal/retention").status_code == 401
    assert c.get(
        "/internal/retention", headers={"Authorization": "Bearer wrong"}
    ).status_code == 401


def test_the_endpoint_sweeps_with_the_right_secret(monkeypatch, db):
    c = client(cron_secret="s3cret", database_url="postgres://test", retention_days=90)
    response = c.get("/internal/retention", headers={"Authorization": "Bearer s3cret"})
    assert response.status_code == 200
    assert response.json()["retention_days"] == 90


def test_retention_can_be_disabled_without_disabling_the_endpoint(db):
    c = client(cron_secret="s3cret", database_url="postgres://test", retention_days=0)
    response = c.get("/internal/retention", headers={"Authorization": "Bearer s3cret"})
    assert response.json()["deleted"] == {}
    assert db.deletes == []
