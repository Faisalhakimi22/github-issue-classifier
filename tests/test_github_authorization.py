"""Focused tests for the persisted GitHub installation authorization gate."""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from ghic.service import github_connection


class FakeConnection:
    def __init__(self, installation_rows, repository_rows):
        self.installation_rows = installation_rows
        self.repository_rows = repository_rows

    def run(self, sql, **params):
        if "to_regclass" in sql:
            return [[True]]
        if "ghic_github_installations" in sql and sql.lstrip().startswith("SELECT"):
            return self.installation_rows
        if "ghic_github_repositories" in sql and sql.lstrip().startswith("SELECT"):
            return self.repository_rows
        raise AssertionError(f"unexpected SQL: {sql}")


class FakeGitHub:
    def __init__(self, status="active"):
        self.status = status
        self.calls = []

    def verify_installation(self, installation_id):
        self.calls.append(installation_id)
        return {"status": self.status}


@contextmanager
def fake_session(connection):
    yield connection


def authorize(monkeypatch, installation_rows, repository_rows, status="active"):
    conn = FakeConnection(installation_rows, repository_rows)
    github = FakeGitHub(status)
    monkeypatch.setattr(
        github_connection,
        "session",
        lambda _database_url: fake_session(conn),
    )
    result = github_connection.authorize_issue(
        "postgres://test", 123, "acme/widgets", github
    )
    return result, github


def active_rows():
    return [[123, "workspace-a", None, "connected"]], [[123, "workspace-a", True]]


def test_unknown_installation_is_skipped(monkeypatch):
    result, github = authorize(monkeypatch, [], [])
    assert result == {"authorized": False, "reason": "unclaimed_installation"}
    assert github.calls == []


def test_invalid_installation_candidate_is_unknown(monkeypatch):
    monkeypatch.setattr(github_connection, "session", pytest.fail)
    result = github_connection.authorize_issue("postgres://test", "bad", "acme/widgets")
    assert result == {"authorized": False, "reason": "unknown_installation"}


def test_revoked_installation_is_skipped(monkeypatch):
    result, github = authorize(
        monkeypatch,
        [[123, "workspace-a", "2026-01-01", "connected"]],
        [[123, "workspace-a", True]],
    )
    assert result["reason"] == "revoked_installation"
    assert github.calls == []


def test_suspended_installation_is_skipped(monkeypatch):
    installations, repositories = active_rows()
    result, github = authorize(monkeypatch, installations, repositories, "suspended")
    assert result["reason"] == "suspended_installation"
    assert github.calls == [123]


@pytest.mark.parametrize(
    ("repositories", "reason"),
    [([], "repository_not_connected"),
     ([[456, "workspace-a", True]], "installation_repository_mismatch"),
     ([[123, "workspace-a", False]], "repository_removed")],
)
def test_repository_must_be_connected_to_this_active_installation(
    monkeypatch, repositories, reason
):
    result, github = authorize(
        monkeypatch, [[123, "workspace-a", None, "connected"]], repositories
    )
    assert result["reason"] == reason
    assert github.calls == []


def test_valid_installation_and_repository_are_authorized(monkeypatch):
    installations, repositories = active_rows()
    result, github = authorize(monkeypatch, installations, repositories)
    assert result == {
        "authorized": True,
        "installation_id": 123,
        "repo": "acme/widgets",
        "workspace_id": "workspace-a",
    }
    assert github.calls == [123]


@pytest.mark.parametrize("status", ["", "pending", "error", "unknown", "future"])
def test_unknown_lifecycle_statuses_fail_closed(monkeypatch, status):
    installations, repositories = active_rows()
    installations[0][3] = status
    result, github = authorize(monkeypatch, installations, repositories)
    assert result["authorized"] is False
    assert result["reason"] == "connection_state_unavailable"
    assert result["workspace_id"] == "workspace-a"
    assert github.calls == []


def test_missing_live_app_verification_fails_closed(monkeypatch):
    installations, repositories = active_rows()
    conn = FakeConnection(installations, repositories)
    monkeypatch.setattr(
        github_connection, "session", lambda _database_url: fake_session(conn)
    )
    result = github_connection.authorize_issue("postgres://test", 123, "acme/widgets")
    assert result["reason"] == "github_installation_unavailable"


def test_missing_workspace_metadata_fails_closed(monkeypatch):
    result, github = authorize(
        monkeypatch, [[123, None, None, "connected"]], [[123, None, True]]
    )
    assert result["reason"] == "connection_state_unavailable"
    assert github.calls == []


def test_legacy_row_shape_fails_closed(monkeypatch):
    result, github = authorize(monkeypatch, [[123, None, "connected"]], [[123, True]])
    assert result["reason"] == "connection_state_unavailable"
    assert github.calls == []
