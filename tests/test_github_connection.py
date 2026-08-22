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
    """Records statements; answers the SELECTs the module makes."""

    def __init__(
        self,
        *,
        tables_exist: bool = True,
        connected: bool = True,
        repositories=None,
    ):
        self.tables_exist = tables_exist
        self.connected = connected
        # (repo_full_name, workspace_id) rows the purge walks. Defaults to one
        # so the content-deletion path is exercised rather than skipped.
        self.repositories = (
            [("acme/widgets", "ws-1")] if repositories is None else repositories
        )
        self.statements: list[tuple[str, dict[str, Any]]] = []

    def run(self, sql: str, **kwargs: Any):
        self.statements.append((" ".join(sql.split()), kwargs))
        if "to_regclass" in sql:
            return [[self.tables_exist]]
        if sql.startswith("SELECT 1 FROM ghic_github_installations"):
            return [[1]] if self.connected else []
        if sql.startswith("SELECT repo_full_name, workspace_id"):
            repo = kwargs.get("repo")
            if repo is not None:
                return [[n, w] for n, w in self.repositories if n == repo]
            return [[n, w] for n, w in self.repositories]
        return []

    def writes(self) -> list[str]:
        return [s for s, _ in self.statements if s.startswith(("INSERT", "UPDATE"))]

    def deletes(self):
        return [(s, p) for s, p in self.statements if s.startswith("DELETE")]

    def deleted_tables(self):
        return [s.split()[2] for s, _ in self.deletes()]


class FakeGitHub:
    def __init__(self, repos=None):
        self.repos = repos or [{"id": 1, "full_name": "acme/widgets", "private": False}]

    def list_installation_repositories(self, installation_id):
        return self.repos


class FailingPurgeConn(FakeConn):
    """Fails the first content deletion, to prove the purge rolls back."""

    def run(self, sql: str, **kwargs: Any):
        result = super().run(sql, **kwargs)
        if sql.startswith("DELETE FROM ghic_repo_chunks"):
            raise RuntimeError("chunk delete failed")
        return result


class FailingRepositoryUpdateConn(FakeConn):
    def run(self, sql: str, **kwargs: Any):
        result = super().run(sql, **kwargs)
        if sql.startswith("UPDATE ghic_github_repositories"):
            raise RuntimeError("repository update failed")
        return result


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
    def test_uninstalling_the_app_purges_everything_it_owned(self, patched):
        conn = patched(FakeConn())
        result = github_connection.handle_installation_event(
            URL, "installation", installation_payload("deleted")
        )
        assert result["ok"] is True
        assert result["connection_sync"] == "purged:deleted"
        # Uninstall withdraws the mandate to hold the content, so this is a
        # deletion rather than the status change it used to be.
        assert conn.deleted_tables() == [
            "ghic_repo_chunks",
            "ghic_repository_state",
            "ghic_ledger",
            "ghic_github_repositories",
            "ghic_ledger",
            "ghic_github_installations",
        ]

    def test_the_purge_deletes_children_before_their_parents(self, patched):
        conn = patched(FakeConn())
        github_connection.handle_installation_event(
            URL, "installation", installation_payload("deleted")
        )
        order = conn.deleted_tables()
        # Nothing in this schema cascades, so a parent removed first would
        # either fail on the foreign key or strand its children.
        assert order.index("ghic_repo_chunks") < order.index("ghic_github_repositories")
        assert order.index("ghic_repository_state") < order.index(
            "ghic_github_repositories"
        )
        assert order.index("ghic_github_repositories") < order.index(
            "ghic_github_installations"
        )

    def test_the_purge_is_scoped_to_the_owning_workspace(self, patched):
        conn = patched(FakeConn())
        github_connection.handle_installation_event(
            URL, "installation", installation_payload("deleted")
        )
        for sql, params in conn.deletes():
            if "ghic_repo_chunks" in sql or "ghic_repository_state" in sql:
                assert params["ws"] == "ws-1"
                assert params["repo"] == "acme/widgets"

    def test_a_repository_without_a_workspace_keeps_its_content(self, patched):
        # Unattributable content is left alone: guessing an owner is how one
        # workspace's data ends up deleted by another's uninstall.
        conn = patched(FakeConn(repositories=[("acme/widgets", "")]))
        github_connection.handle_installation_event(
            URL, "installation", installation_payload("deleted")
        )
        assert "ghic_repo_chunks" not in conn.deleted_tables()
        assert "ghic_github_installations" in conn.deleted_tables()

    def test_the_purge_runs_in_one_transaction(self, patched):
        conn = patched(FakeConn())
        github_connection.handle_installation_event(
            URL, "installation", installation_payload("deleted")
        )
        statements = [sql for sql, _ in conn.statements]
        deletes = [i for i, s in enumerate(statements) if s.startswith("DELETE")]
        assert statements.index("BEGIN") < deletes[0]
        assert deletes[-1] < statements.index("COMMIT")

    def test_uninstalling_twice_is_harmless(self, patched):
        # GitHub redelivers, and the reset path may be run by hand. The second
        # pass must delete nothing and still report success.
        patched(FakeConn(repositories=[]))
        result = github_connection.handle_installation_event(
            URL, "installation", installation_payload("deleted")
        )
        assert result["ok"] is True
        assert result["connection_sync"] == "purged:deleted"
        assert result["deleted"]["chunks"] == 0
        assert result["deleted"]["repositories"] == 0

    def test_a_failed_purge_is_reported_as_retryable(self, patched):
        conn = patched(FailingPurgeConn())
        result = github_connection.handle_installation_event(
            URL, "installation", installation_payload("deleted")
        )
        # Never silently "ok": the caller has to know the data is still there.
        assert result["ok"] is False
        assert result["connection_sync"] == "purge_failed"
        assert result["retryable"] is True
        assert any(sql == "ROLLBACK" for sql, _ in conn.statements)
        assert not any(sql == "COMMIT" for sql, _ in conn.statements)

    def test_suspension_is_treated_as_loss_of_access(self, patched):
        conn = patched(FakeConn())
        result = github_connection.handle_installation_event(
            URL, "installation", installation_payload("suspend")
        )
        assert result["connection_sync"] == "suspended"
        installation_update = next(
            w for w in conn.writes() if "connection_status = 'suspended'" in w
        )
        repository_update = next(w for w in conn.writes() if "SET active = false" in w)
        assert "connection_status = 'connected'" in installation_update
        assert "revoked_at IS NULL" in installation_update
        assert "connection_status = 'suspended'" in repository_update
        statements = [sql for sql, _ in conn.statements]
        assert statements.index("BEGIN") < statements.index(installation_update)
        assert statements.index(repository_update) < statements.index("COMMIT")

    def test_lifecycle_write_failure_rolls_back_the_transition(self, patched):
        conn = patched(FailingRepositoryUpdateConn())
        result = github_connection.handle_installation_event(
            URL, "installation", installation_payload("suspend")
        )
        assert result["connection_sync"] == "error"
        assert any(sql == "ROLLBACK" for sql, _ in conn.statements)
        assert not any(sql == "COMMIT" for sql, _ in conn.statements)

    def test_revocation_applies_even_to_an_unclaimed_installation(self, patched):
        # Losing access is true regardless of whether a user ever claimed
        # the installation in GHIC, so this path must not be gated on it.
        conn = patched(FakeConn(connected=False))
        result = github_connection.handle_installation_event(
            URL, "installation", installation_payload("deleted")
        )
        assert result["connection_sync"] == "purged:deleted"
        assert conn.deletes()

    def test_unsuspending_restores_the_connection(self, patched):
        conn = patched(FakeConn(connected=True))
        result = github_connection.handle_installation_event(
            URL,
            "installation",
            installation_payload(
                "unsuspend",
                repositories=[{"id": 1, "full_name": "acme/widgets", "private": False}],
            ),
            FakeGitHub(),
        )
        assert result["connection_sync"] == "unsuspended"
        writes = conn.writes()
        assert any("SET revoked_at = NULL" in w for w in writes)
        assert any(w.startswith("INSERT INTO ghic_github_repositories") for w in writes)
        statements = [sql for sql, _ in conn.statements]
        begin = statements.index("BEGIN")
        commit = statements.index("COMMIT")
        restore = next(
            index for index, sql in enumerate(statements)
            if "SET revoked_at = NULL" in sql
        )
        insert = next(
            index for index, sql in enumerate(statements)
            if sql.startswith("INSERT INTO ghic_github_repositories")
        )
        assert begin < restore < insert < commit

    def test_unsuspend_cannot_restore_a_non_suspended_connection(self, patched):
        conn = patched(FakeConn(connected=False))
        result = github_connection.handle_installation_event(
            URL, "installation", installation_payload("unsuspend"), FakeGitHub()
        )
        assert result["connection_sync"] == "ignored_unsuspend"
        assert conn.writes() == []


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

    def test_unknown_lifecycle_state_is_not_treated_as_connected(self, patched):
        conn = patched(FakeConn(connected=False))
        result = github_connection.handle_installation_event(
            URL,
            "installation_repositories",
            installation_payload(
                "removed",
                repositories_removed=[{"id": 1, "full_name": "acme/widgets"}],
            ),
        )
        assert result["connection_sync"] == "not_connected"
        connected_query = next(
            sql for sql, _ in conn.statements
            if sql.startswith("SELECT 1 FROM ghic_github_installations")
        )
        assert "connection_status = 'connected'" in connected_query
        assert "COALESCE" not in connected_query
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
        # Only the named repository is purged; the installation survives.
        assert "ghic_github_installations" not in conn.deleted_tables()
        for sql, params in conn.deletes():
            assert params.get("repo", "acme/widgets") == "acme/widgets"

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
            FakeGitHub([
                {"id": 2, "full_name": "acme/other", "private": True}
            ]),
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
        # A repository deleted on GitHub can never be re-verified, so its
        # index goes with it.
        assert "ghic_repo_chunks" in conn.deleted_tables()
        assert "ghic_github_repositories" in conn.deleted_tables()
        assert "ghic_github_installations" not in conn.deleted_tables()

    def test_an_archived_repository_keeps_its_index(self, patched):
        # Archiving is reversible and the code is still readable, so paying to
        # re-embed on unarchive would be waste.
        conn = patched(FakeConn())
        result = github_connection.handle_installation_event(
            URL,
            "repository",
            installation_payload(
                "archived", repository={"full_name": "acme/widgets"}
            ),
        )
        assert result["connection_sync"] == "repository:archived"
        assert not conn.deletes()
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

        def fake_handler(database_url, event, payload, github_client=None):
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

    def test_a_failed_purge_answers_500_and_frees_the_delivery_for_retry(
        self, monkeypatch
    ):
        import json

        from tests.test_service import SECRET, make_client, make_settings, sign

        monkeypatch.setattr(
            github_connection,
            "handle_installation_event",
            lambda *a, **k: {
                "ok": False,
                "connection_sync": "purge_failed",
                "retryable": True,
            },
        )

        client = make_client(make_settings())
        payload = {"action": "deleted", "installation": {"id": 100}}
        body = json.dumps(payload).encode()
        headers = {
            "X-GitHub-Event": "installation",
            "X-Hub-Signature-256": sign(body, SECRET),
            "X-GitHub-Delivery": "delivery-purge-1",
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body, headers=headers)

        # A purge that failed must not be acknowledged as done.
        assert resp.status_code == 500

        store = client.app.state.idempotency
        # The delivery is marked consumed before the handler runs, so unless
        # the key is released the retry below would be dropped as a duplicate
        # and the data would never be cleaned up.
        assert store.mark_if_new("delivery-purge-1") is True

    def test_a_successful_purge_still_consumes_the_delivery(self, monkeypatch):
        import json

        from tests.test_service import SECRET, make_client, make_settings, sign

        monkeypatch.setattr(
            github_connection,
            "handle_installation_event",
            lambda *a, **k: {"ok": True, "connection_sync": "purged:deleted"},
        )

        client = make_client(make_settings())
        payload = {"action": "deleted", "installation": {"id": 100}}
        body = json.dumps(payload).encode()
        headers = {
            "X-GitHub-Event": "installation",
            "X-Hub-Signature-256": sign(body, SECRET),
            "X-GitHub-Delivery": "delivery-purge-2",
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body, headers=headers)

        assert resp.status_code == 200
        # Released only on failure: a successful purge keeps its replay guard.
        store = client.app.state.idempotency
        assert store.mark_if_new("delivery-purge-2") is False

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
