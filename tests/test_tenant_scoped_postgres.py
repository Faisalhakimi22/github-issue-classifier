from __future__ import annotations

from contextlib import contextmanager

import pytest

from ghic.repository_intelligence.config import RepositoryIntelligenceConfig
from ghic.repository_intelligence.repository_service import RepositoryIntelligenceService
from ghic.repository_intelligence.state import PostgresStateStore
from ghic.repository_intelligence.vector_pg import PostgresVectorStore


def test_tenantless_vector_store_cannot_be_created():
    with pytest.raises(ValueError, match="require workspace_id"):
        PostgresVectorStore("postgresql://test", "acme/demo", 1536)


def test_tenantless_vector_replacement_cannot_write():
    with pytest.raises(ValueError, match="require workspace_id"):
        PostgresVectorStore.replace_repository(
            "postgresql://test", "acme/demo", 1536, [], []
        )


def test_tenantless_state_reads_and_deletes_fail_before_database_access():
    store = PostgresStateStore.__new__(PostgresStateStore)
    store.database_url = "postgresql://test"

    with pytest.raises(ValueError, match="requires workspace_id"):
        store.get("acme/demo")
    with pytest.raises(ValueError, match="requires workspace_id"):
        store.delete("acme/demo")
    with pytest.raises(ValueError, match="requires workspace_id"):
        store.list()


def test_postgres_state_write_requires_workspace_before_database_access():
    store = PostgresStateStore.__new__(PostgresStateStore)
    store.database_url = "postgresql://test"

    with pytest.raises(ValueError, match="requires workspace_id"):
        store.put(None)  # type: ignore[arg-type]


def test_production_indexing_rejects_missing_workspace_before_clone():
    cfg = RepositoryIntelligenceConfig(
        database_url="postgresql://test",
        vector_provider="postgres",
    )
    service = RepositoryIntelligenceService(cfg)
    service.repo_cache.ensure = lambda *args, **kwargs: pytest.fail("clone must not run")

    with pytest.raises(ValueError, match="workspace context is required"):
        service.index_repository("acme/demo")


def test_postgres_workspace_resolution_requires_active_connected_ownership(monkeypatch):
    class Connection:
        def __init__(self):
            self.statements = []

        def run(self, sql, **params):
            self.statements.append((sql, params))
            return [["workspace-a"]]

    connection = Connection()

    @contextmanager
    def session():
        yield connection

    monkeypatch.setattr(PostgresStateStore, "_session", lambda self: session())
    store = PostgresStateStore.__new__(PostgresStateStore)
    store.database_url = "postgresql://test"

    assert store.resolve_workspace("acme/demo") == "workspace-a"
    sql, params = connection.statements[0]
    assert "JOIN ghic_github_installations" in sql
    assert "i.workspace_id = r.workspace_id" in sql
    assert "r.active = true" in sql
    assert "i.connection_status = 'connected'" in sql
    assert "i.revoked_at IS NULL" in sql
    assert params["repo"] == "acme/demo"


def test_explicit_workspace_cannot_override_authoritative_repository_ownership():
    class State:
        requires_workspace = True

        @staticmethod
        def resolve_workspace(repo):
            assert repo == "acme/demo"
            return "workspace-b"

    cfg = RepositoryIntelligenceConfig(
        database_url="postgresql://test",
        vector_provider="postgres",
    )
    service = RepositoryIntelligenceService(cfg, state_store=State())
    service.repo_cache.ensure = lambda *args, **kwargs: pytest.fail("clone must not run")

    with pytest.raises(ValueError, match="does not own active repository"):
        service.index_repository("acme/demo", workspace_id="workspace-a")


def test_scoped_vector_queries_cannot_become_global(monkeypatch):
    class Connection:
        def __init__(self):
            self.statements = []

        def run(self, sql, **params):
            self.statements.append((sql, params))
            if sql.startswith("SELECT format_type"):
                return [["vector(1536)"]]
            if sql.startswith("SELECT count(*)"):
                return [[0]]
            return []

    connection = Connection()

    @contextmanager
    def session():
        yield connection

    monkeypatch.setattr(PostgresVectorStore, "_session", lambda self: session())
    store = PostgresVectorStore(
        "postgresql://test", "acme/demo", 1536, workspace_id="workspace-a"
    )
    assert store.size == 0
    count_sql, params = next(
        item for item in connection.statements if item[0].startswith("SELECT count(*)")
    )
    assert "workspace_id = :workspace_id" in count_sql
    assert ":workspace_id IS NULL" not in count_sql
    assert params["workspace_id"] == "workspace-a"
