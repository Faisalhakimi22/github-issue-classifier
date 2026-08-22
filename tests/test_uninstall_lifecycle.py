"""Uninstall cleanup: the full lifecycle, and what must survive it.

The fake below is a small relational stand-in rather than a statement
recorder. Asserting that a DELETE was *issued* proves nothing about which
rows it would remove, and the whole risk in this feature is deleting a row
belonging to somebody else. So the fake actually stores rows and applies the
predicates, and the tests assert on what is left afterwards.
"""
from __future__ import annotations

from typing import Any

import pytest

from ghic.service import github_connection

URL = "postgresql://example/db"


class TinyDB:
    """Interprets the handful of statements the purge issues."""

    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {
            "ghic_github_installations": [],
            "ghic_github_repositories": [],
            "ghic_repo_chunks": [],
            "ghic_repository_state": [],
            "ghic_ledger": [],
            "ghic_users": [],
            "ghic_workspaces": [],
        }
        self.depth = 0
        self.committed = False

    # -- helpers ---------------------------------------------------------
    def install(self, installation_id, workspace_id, repos):
        self.tables["ghic_github_installations"].append(
            {"installation_id": installation_id, "workspace_id": workspace_id,
             "connection_status": "connected", "revoked_at": None}
        )
        for repo in repos:
            self.tables["ghic_github_repositories"].append(
                {"repo_full_name": repo, "installation_id": installation_id,
                 "workspace_id": workspace_id, "active": True}
            )

    def add_activity(self, repo, workspace_id, *, chunks=3, ledger=2):
        for i in range(chunks):
            self.tables["ghic_repo_chunks"].append(
                {"repo": repo, "workspace_id": workspace_id, "path": f"f{i}.py"}
            )
        self.tables["ghic_repository_state"].append(
            {"repo": repo, "workspace_id": workspace_id, "state": "ready"}
        )
        for i in range(ledger):
            self.tables["ghic_ledger"].append(
                {"workspace_id": workspace_id,
                 "data": {"repo": repo, "type": "prediction"}}
            )

    def count(self, table, **where):
        return len([r for r in self.rows(table, **where)])

    def rows(self, table, **where):
        out = []
        for row in self.tables[table]:
            if all(row.get(k) == v for k, v in where.items()):
                out.append(row)
        return out

    # -- the driver surface ----------------------------------------------
    def run(self, sql: str, **kw: Any):
        flat = " ".join(sql.split())

        if flat == "BEGIN":
            self.depth += 1
            return []
        if flat == "COMMIT":
            self.depth -= 1
            self.committed = True
            return []
        if flat == "ROLLBACK":
            self.depth -= 1
            return []
        if "to_regclass" in flat:
            return [[True]]

        if flat.startswith("SELECT 1 FROM ghic_github_installations"):
            return [[1] for _ in self.rows(
                "ghic_github_installations", installation_id=kw["iid"])]

        if flat.startswith("SELECT repo_full_name, workspace_id"):
            found = self.rows(
                "ghic_github_repositories", installation_id=kw["iid"])
            if "repo" in kw:
                found = [r for r in found if r["repo_full_name"] == kw["repo"]]
            return [[r["repo_full_name"], r["workspace_id"]] for r in found]

        if flat.startswith("DELETE FROM ghic_repo_chunks"):
            return self._delete("ghic_repo_chunks",
                                lambda r: r["workspace_id"] == kw["ws"]
                                and r["repo"] == kw["repo"])
        if flat.startswith("DELETE FROM ghic_repository_state"):
            return self._delete("ghic_repository_state",
                                lambda r: r["workspace_id"] == kw["ws"]
                                and r["repo"] == kw["repo"])
        if flat.startswith("DELETE FROM ghic_ledger"):
            if "repo" in kw:
                return self._delete("ghic_ledger",
                                    lambda r: r["workspace_id"] == kw["ws"]
                                    and r["data"].get("repo") == kw["repo"])
            return self._delete(
                "ghic_ledger",
                lambda r: r["workspace_id"] == kw["ws"]
                and str(r["data"].get("installation_id")) == kw["iid"])
        if flat.startswith("DELETE FROM ghic_github_repositories"):
            return self._delete("ghic_github_repositories",
                                lambda r: r["installation_id"] == kw["iid"])
        if flat.startswith("DELETE FROM ghic_github_installations"):
            return self._delete("ghic_github_installations",
                                lambda r: r["installation_id"] == kw["iid"])
        return []

    def _delete(self, table, predicate):
        kept, removed = [], 0
        for row in self.tables[table]:
            if predicate(row):
                removed += 1
            else:
                kept.append(row)
        self.tables[table] = kept
        return [[1]] * removed


@pytest.fixture
def db(monkeypatch):
    store = TinyDB()

    class _Session:
        def __enter__(self):
            return store

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(github_connection, "session", lambda url: _Session())
    monkeypatch.setattr(github_connection, "_tables_exist", lambda conn: True)
    return store


def uninstall(installation_id=100):
    return github_connection.handle_installation_event(
        URL, "installation",
        {"action": "deleted", "installation": {"id": installation_id}},
    )


class TestFullLifecycle:
    def test_install_create_uninstall_clean(self, db):
        db.install(100, "ws-a", ["acme/widgets"])
        db.add_activity("acme/widgets", "ws-a")
        assert db.count("ghic_repo_chunks") == 3

        result = uninstall()

        assert result["ok"] is True
        assert result["deleted"] == {
            "chunks": 3, "repository_state": 1, "ledger": 2,
            "repositories": 1, "installations": 1,
        }
        for table in ("ghic_repo_chunks", "ghic_repository_state",
                      "ghic_ledger", "ghic_github_repositories",
                      "ghic_github_installations"):
            assert db.tables[table] == [], table

    def test_reinstalling_starts_clean(self, db):
        db.install(100, "ws-a", ["acme/widgets"])
        db.add_activity("acme/widgets", "ws-a")
        uninstall()

        # Same account, new installation id, as GitHub issues on reinstall.
        db.install(101, "ws-a", ["acme/widgets"])
        assert db.count("ghic_repo_chunks") == 0
        assert db.count("ghic_ledger") == 0
        assert db.count("ghic_repository_state") == 0
        # Nothing from the previous installation is still attached.
        assert db.count("ghic_github_repositories", installation_id=100) == 0

    def test_uninstalling_twice_changes_nothing_further(self, db):
        db.install(100, "ws-a", ["acme/widgets"])
        db.add_activity("acme/widgets", "ws-a")
        first = uninstall()
        second = uninstall()
        assert first["deleted"]["chunks"] == 3
        assert second["ok"] is True
        assert second["deleted"] == {
            "chunks": 0, "repository_state": 0, "ledger": 0,
            "repositories": 0, "installations": 0,
        }


class TestUnrelatedDataSurvives:
    def test_another_installation_in_the_same_workspace_is_untouched(self, db):
        db.install(100, "ws-a", ["acme/widgets"])
        db.add_activity("acme/widgets", "ws-a")
        db.install(200, "ws-a", ["acme/other"])
        db.add_activity("acme/other", "ws-a")

        uninstall(100)

        assert db.count("ghic_github_installations", installation_id=200) == 1
        assert db.count("ghic_repo_chunks", repo="acme/other") == 3
        assert db.count("ghic_repository_state", repo="acme/other") == 1
        assert db.count("ghic_repo_chunks", repo="acme/widgets") == 0

    def test_another_workspace_is_untouched(self, db):
        db.install(100, "ws-a", ["acme/widgets"])
        db.add_activity("acme/widgets", "ws-a")
        db.install(300, "ws-b", ["beta/thing"])
        db.add_activity("beta/thing", "ws-b")

        uninstall(100)

        assert db.count("ghic_repo_chunks", workspace_id="ws-b") == 3
        assert db.count("ghic_ledger", workspace_id="ws-b") == 2
        assert db.count("ghic_github_installations", workspace_id="ws-b") == 1

    def test_the_account_and_workspace_records_are_never_touched(self, db):
        db.tables["ghic_users"].append({"firebase_uid": "u1"})
        db.tables["ghic_workspaces"].append({"id": "ws-a"})
        db.install(100, "ws-a", ["acme/widgets"])
        db.add_activity("acme/widgets", "ws-a")

        uninstall(100)

        # Uninstalling an App is not account deletion. The workspace survives
        # having nothing connected to it.
        assert db.count("ghic_users") == 1
        assert db.count("ghic_workspaces") == 1

    def test_unattributed_ledger_rows_are_left_alone(self, db):
        db.install(100, "ws-a", ["acme/widgets"])
        db.tables["ghic_ledger"].append(
            {"workspace_id": None, "data": {"repo": "acme/widgets"}}
        )

        uninstall(100)

        # Rows with no workspace belong to no installation, so no
        # installation's removal may claim them.
        assert db.count("ghic_ledger", workspace_id=None) == 1


class TestPartialRemoval:
    def test_removing_one_repository_keeps_the_others(self, db):
        db.install(100, "ws-a", ["acme/widgets", "acme/keeper"])
        db.add_activity("acme/widgets", "ws-a")
        db.add_activity("acme/keeper", "ws-a")

        github_connection.handle_installation_event(
            URL, "installation_repositories",
            {"action": "removed", "installation": {"id": 100},
             "repositories_removed": [{"full_name": "acme/widgets"}]},
        )

        assert db.count("ghic_repo_chunks", repo="acme/widgets") == 0
        assert db.count("ghic_repo_chunks", repo="acme/keeper") == 3
        # The App is still installed.
        assert db.count("ghic_github_installations", installation_id=100) == 1
