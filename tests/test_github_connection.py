"""GitHub App installation lifecycle sync.

These deliveries are what keep the dashboard from showing repositories
GHIC can no longer read. The properties worth pinning down are less about
happy-path SQL than about what the handler refuses to do: it must never
invent a workspace connection from an unauthenticated payload, and it must
never turn a bookkeeping failure into a webhook error that GitHub retries.
"""
from __future__ import annotations

import contextlib
from typing import Any

import pytest

from ghic.service import github_connection


class FakeConn:
    """Records statements; answers the two SELECTs the module makes."""

    def __init__(self, *, tables_exist: bool = True, connected: bool = True):
        self.tables_exist = tables_exist
        self.connected = connected
        self.statements: list[tuple[str, dict[str, Any]]] = []

    def run(self, sql: str, **kwargs: Any):
        self.statements.append((" ".join(sql.split()), kwargs))
        if "to_regclass" in sql:
            return [[self.tables_exist]]
        if sql.startswith("SELECT 1 FROM ghic_github_installations"):
            return [[1]] if self.connected else []
        return []

    def writes(self) -> list[str]:
        return [s for s, _ in self.statements if s.startswith(("INSERT", "UPDATE"))]


@pytest.fixture
def patched(monkeypatch):
    """Install a fake `session` and hand back the connection it yields."""

    def install(conn: FakeConn) -> FakeConn:
        @contextlib.contextmanager
        def fake_session(_url: str):
            yield conn

        monkeypatch.setattr(github_connection, "session", fake_session)
        return conn

    return install


URL = "postgresql://example/db"


def installation_payload(action: str, installation_id: int = 100, **extra):
    return {"action": action, "installation": {"id": installation_id}, **extra}


class TestRevocation:
    def test_uninstalling_the_app_revokes_the_connection(self, patched):
        conn = patched(FakeConn())
        result = github_connection.handle_installation_event(
            URL, "installation", installation_payload("deleted")
        )
        assert result["ok"] is True
        assert result["connection_sync"] == "revoked:deleted"
        writes = conn.writes()
        assert any("SET revoked_at = now()" in w for w in writes)
        # Repositories are deactivated too: an installation row marked
        # revoked while its repositories still read active would leave the
        # dashboard listing repositories GHIC cannot read.
        assert any("SET active = false" in w for w in writes)

    def test_suspension_is_treated_as_loss_of_access(self, patched):
        conn = patched(FakeConn())
        result = github_connection.handle_installation_event(
            URL, "installation", installation_payload("suspend")
        )
        assert result["connection_sync"] == "revoked:suspend"
        assert any("SET active = false" in w for w in conn.writes())

    def test_revocation_applies_even_to_an_unclaimed_installation(self, patched):
        # Losing access is true regardless of whether a user ever claimed
        # the installation in GHIC, so this path must not be gated on it.
        conn = patched(FakeConn(connected=False))
        result = github_connection.handle_installation_event(
            URL, "installation", installation_payload("deleted")
        )
        assert result["connection_sync"] == "revoked:deleted"
        assert conn.writes()

    def test_unsuspending_restores_the_connection(self, patched):
        conn = patched(FakeConn(connected=False))
        result = github_connection.handle_installation_event(
            URL,
            "installation",
            installation_payload(
                "unsuspend",
                repositories=[{"id": 1, "full_name": "acme/widgets", "private": False}],
            ),
        )
        assert result["connection_sync"] == "unsuspended"
        writes = conn.writes()
        assert any("SET revoked_at = NULL" in w for w in writes)
        assert any(w.startswith("INSERT INTO ghic_github_repositories") for w in writes)


class TestUnclaimedInstallations:
    def test_a_new_installation_is_not_recorded_from_a_webhook(self, patched):
        # `installation/created` carries no proof of who owns the GHIC
        # workspace. Recording it here would let anyone who installs the
        # App materialise a connection nobody authenticated for.
        conn = patched(FakeConn(connected=False))
        result = github_connection.handle_installation_event(
            URL,
            "installation",
            installation_payload(
                "created",
                repositories=[{"id": 1, "full_name": "acme/widgets"}],
            ),
        )
        assert result["connection_sync"] == "not_connected"
        assert conn.writes() == []

    def test_repository_additions_to_an_unclaimed_installation_are_ignored(
        self, patched
    ):
        conn = patched(FakeConn(connected=False))
        result = github_connection.handle_installation_event(
            URL,
            "installation_repositories",
            installation_payload(
                "added",
                repositories_added=[{"id": 2, "full_name": "acme/other"}],
            ),
        )
        assert result["connection_sync"] == "not_connected"
        assert conn.writes() == []


class TestRepositorySelection:
    def test_removing_a_repository_deactivates_only_that_repository(self, patched):
        conn = patched(FakeConn())
        result = github_connection.handle_installation_event(
            URL,
            "installation_repositories",
            installation_payload(
                "removed",
                repositories_removed=[{"id": 1, "full_name": "acme/widgets"}],
            ),
        )
        assert result["connection_sync"] == "repositories"
        assert result["removed"] == 1
        deactivations = [
            (sql, params)
            for sql, params in conn.statements
            if "SET active = false" in sql
        ]
        assert len(deactivations) == 1
        assert deactivations[0][1]["repo"] == "acme/widgets"

    def test_adding_a_repository_activates_it(self, patched):
        conn = patched(FakeConn())
        result = github_connection.handle_installation_event(
            URL,
            "installation_repositories",
            installation_payload(
                "added",
                repositories_added=[
                    {"id": 2, "full_name": "acme/other", "private": True}
                ],
            ),
        )
        assert result["added"] == 1
        insert = next(
            (sql, params)
            for sql, params in conn.statements
            if sql.startswith("INSERT INTO ghic_github_repositories")
        )
        assert insert[1]["repo"] == "acme/other"
        assert insert[1]["priv"] is True
        assert "ON CONFLICT (repo_full_name) DO UPDATE" in insert[0]

    def test_a_deleted_repository_is_deactivated(self, patched):
        conn = patched(FakeConn())
        result = github_connection.handle_installation_event(
            URL,
            "repository",
            installation_payload(
                "deleted", repository={"full_name": "acme/widgets"}
            ),
        )
        assert result["connection_sync"] == "repository:deleted"
        assert any("SET active = false" in w for w in conn.writes())


class TestFailureIsNeverFatal:
    def test_a_missing_database_url_is_acknowledged(self):
        result = github_connection.handle_installation_event(
            "", "installation", installation_payload("deleted")
        )
        assert result == {"ok": True, "connection_sync": "no_database"}

    def test_absent_tables_are_acknowledged(self, patched):
        conn = patched(FakeConn(tables_exist=False))
        result = github_connection.handle_installation_event(
            URL, "installation", installation_payload("deleted")
        )
        assert result["connection_sync"] == "tables_absent"
        assert conn.writes() == []

    def test_a_payload_without_an_installation_id_is_acknowledged(self):
        result = github_connection.handle_installation_event(
            URL, "installation", {"action": "deleted"}
        )
        assert result["connection_sync"] == "no_installation_id"

    def test_a_database_error_is_swallowed_so_github_does_not_retry(
        self, monkeypatch
    ):
        @contextlib.contextmanager
        def exploding_session(_url: str):
            raise RuntimeError("connection refused")
            yield  # pragma: no cover

        monkeypatch.setattr(github_connection, "session", exploding_session)
        result = github_connection.handle_installation_event(
            URL, "installation", installation_payload("deleted")
        )
        assert result == {"ok": True, "connection_sync": "error"}


class TestWebhookRouting:
    """The events have to actually reach the handler through /webhook."""

    def test_installation_events_are_routed_to_the_sync_handler(self, monkeypatch):
        import json

        from tests.test_service import SECRET, make_client, make_settings, sign

        seen: dict[str, Any] = {}

        def fake_handler(database_url, event, payload):
            seen["event"] = event
            seen["action"] = payload.get("action")
            return {"ok": True, "connection_sync": "stub"}

        monkeypatch.setattr(
            github_connection, "handle_installation_event", fake_handler
        )

        client = make_client(make_settings())
        payload = {"action": "deleted", "installation": {"id": 100}}
        body = json.dumps(payload).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "installation",
                "X-Hub-Signature-256": sign(body, SECRET),
                "Content-Type": "application/json",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["connection_sync"] == "stub"
        assert seen == {"event": "installation", "action": "deleted"}

    def test_an_unrelated_event_is_still_ignored(self):
        import json

        from tests.test_service import SECRET, make_client, make_settings, sign

        client = make_client(make_settings())
        payload = {"action": "created", "comment": {}}
        body = json.dumps(payload).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "issue_comment",
                "X-Hub-Signature-256": sign(body, SECRET),
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 200
        assert "ignored" in resp.json()
