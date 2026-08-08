from __future__ import annotations

import copy
import re
from contextlib import contextmanager

import numpy as np
import pytest

from ghic.repository_intelligence.models import CodeChunk
from ghic.repository_intelligence.vector_pg import (
    EmbeddingDimensionMismatchError,
    PostgresVectorStore,
    UnsafeEmbeddingMigrationError,
)


class FakeConnection:
    def __init__(
        self,
        dimensions: int | None = 512,
        repositories: dict[str, int] | None = None,
        *,
        extension_available: bool = True,
        fail_on_insert: bool = False,
    ) -> None:
        self.dimensions = dimensions
        self.repositories = dict(repositories or {})
        self.json_rows = {repo: 0 for repo in self.repositories}
        self.extension_available = extension_available
        self.fail_on_insert = fail_on_insert
        self.statements: list[str] = []
        self._snapshot: tuple[int | None, dict[str, int], dict[str, int]] | None = None

    def run(self, sql: str, **params):
        statement = " ".join(sql.split())
        self.statements.append(statement)

        if statement == "CREATE EXTENSION IF NOT EXISTS vector":
            if not self.extension_available:
                raise PermissionError("extension unavailable")
            return []
        if statement.startswith("SELECT format_type"):
            return [] if self.dimensions is None else [[f"vector({self.dimensions})"]]
        if statement.startswith("SELECT repo, count(*), count(embedding)"):
            return [
                [repo, count, count, self.dimensions, self.dimensions, self.json_rows[repo]]
                for repo, count in sorted(self.repositories.items())
                if count
            ]
        if statement == "BEGIN":
            self._snapshot = (
                self.dimensions,
                copy.deepcopy(self.repositories),
                copy.deepcopy(self.json_rows),
            )
            return []
        if statement == "COMMIT":
            self._snapshot = None
            return []
        if statement == "ROLLBACK":
            if self._snapshot is not None:
                self.dimensions, self.repositories, self.json_rows = self._snapshot
                self._snapshot = None
            return []
        if statement.startswith("DELETE FROM ghic_repo_chunks"):
            repo = params["repo"]
            self.repositories[repo] = 0
            self.json_rows[repo] = 0
            return []
        if statement == "ALTER TABLE ghic_repo_chunks DROP COLUMN embedding":
            self.dimensions = None
            return []
        if statement.startswith("ALTER TABLE ghic_repo_chunks ADD COLUMN embedding vector("):
            match = re.search(r"vector\((\d+)\)", statement)
            assert match
            self.dimensions = int(match.group(1))
            return []
        if statement.startswith("INSERT INTO ghic_repo_chunks"):
            if self.fail_on_insert:
                self.fail_on_insert = False
                raise RuntimeError("insert failed")
            repo = params["repo"]
            self.repositories[repo] = self.repositories.get(repo, 0) + 1
            if "embedding_json" in statement:
                self.json_rows[repo] = self.json_rows.get(repo, 0) + 1
            return []
        return []


@contextmanager
def fake_session(connection: FakeConnection):
    yield connection


def chunk(repo: str = "acme/demo") -> CodeChunk:
    return CodeChunk(
        repo=repo,
        path="src/app.py",
        language="Python",
        text="def verify_webhook(): pass",
        start_line=1,
        end_line=1,
        kind="function",
        symbol="verify_webhook",
    )


def use_connection(monkeypatch, connection: FakeConnection) -> None:
    monkeypatch.setattr(
        PostgresVectorStore,
        "_session",
        lambda self: fake_session(connection),
    )


def test_regular_store_rejects_incompatible_pgvector_dimension(monkeypatch):
    connection = FakeConnection(512, {"acme/demo": 1})
    use_connection(monkeypatch, connection)

    with pytest.raises(EmbeddingDimensionMismatchError, match=r"vector\(512\)"):
        PostgresVectorStore("postgresql://test", "acme/demo", 1536)

    assert not any("embedding_json" in sql and sql.startswith("INSERT")
                   for sql in connection.statements)


def test_pgvector_search_uses_sqrt_list_probe_count(monkeypatch):
    connection = FakeConnection(1536, {"acme/demo": 1})
    use_connection(monkeypatch, connection)
    store = PostgresVectorStore("postgresql://test", "acme/demo", 1536)

    assert store.search(np.zeros(1536, dtype=np.float32), top_k=5) == []

    probe = "SET ivfflat.probes = 10"
    search = next(
        sql for sql in connection.statements
        if "ORDER BY embedding <=> CAST(:q AS vector)" in sql
    )
    assert probe in connection.statements
    assert connection.statements.index(probe) < connection.statements.index(search)


def test_full_replace_requires_explicit_dimension_migration(monkeypatch):
    connection = FakeConnection(512, {"acme/demo": 1})
    use_connection(monkeypatch, connection)

    with pytest.raises(EmbeddingDimensionMismatchError, match="forced full rebuild"):
        PostgresVectorStore.replace_repository(
            "postgresql://test",
            "acme/demo",
            1536,
            np.zeros((1, 1536), dtype=np.float32),
            [chunk()],
        )

    assert "BEGIN" not in connection.statements


def test_dimension_migration_refuses_other_repository_rows(monkeypatch):
    connection = FakeConnection(512, {"acme/demo": 1, "other/repo": 2})
    use_connection(monkeypatch, connection)

    with pytest.raises(UnsafeEmbeddingMigrationError, match="other/repo"):
        PostgresVectorStore.replace_repository(
            "postgresql://test",
            "acme/demo",
            1536,
            np.zeros((1, 1536), dtype=np.float32),
            [chunk()],
            allow_dimension_migration=True,
        )

    assert connection.dimensions == 512
    assert connection.repositories == {"acme/demo": 1, "other/repo": 2}
    assert "BEGIN" not in connection.statements


def test_dimension_migration_refuses_mixed_storage(monkeypatch):
    connection = FakeConnection(512, {"acme/demo": 2})
    connection.json_rows["acme/demo"] = 1
    use_connection(monkeypatch, connection)

    with pytest.raises(UnsafeEmbeddingMigrationError, match="mixed vector storage"):
        PostgresVectorStore.replace_repository(
            "postgresql://test",
            "acme/demo",
            1536,
            np.zeros((1, 1536), dtype=np.float32),
            [chunk()],
            allow_dimension_migration=True,
        )

    assert connection.dimensions == 512
    assert connection.repositories == {"acme/demo": 2}
    assert "BEGIN" not in connection.statements


def test_forced_full_replace_atomically_migrates_512_to_1536(monkeypatch):
    connection = FakeConnection(512, {"acme/demo": 2})
    use_connection(monkeypatch, connection)

    store = PostgresVectorStore.replace_repository(
        "postgresql://test",
        "acme/demo",
        1536,
        np.zeros((1, 1536), dtype=np.float32),
        [chunk()],
        allow_dimension_migration=True,
    )

    assert store._pgvector is True
    assert connection.dimensions == 1536
    assert connection.repositories == {"acme/demo": 1}
    assert connection.json_rows == {"acme/demo": 0}
    assert "LOCK TABLE ghic_repo_chunks IN ACCESS EXCLUSIVE MODE" in connection.statements
    assert "DROP INDEX IF EXISTS ghic_repo_chunks_embedding_idx" in connection.statements
    assert "ALTER TABLE ghic_repo_chunks DROP COLUMN embedding" in connection.statements
    assert "ALTER TABLE ghic_repo_chunks ADD COLUMN embedding vector(1536)" in connection.statements
    assert connection.statements.index("BEGIN") < connection.statements.index("COMMIT")


def test_failed_migration_insert_rolls_back_old_schema_and_rows(monkeypatch):
    connection = FakeConnection(
        512,
        {"acme/demo": 2},
        fail_on_insert=True,
    )
    use_connection(monkeypatch, connection)

    with pytest.raises(RuntimeError, match="insert failed"):
        PostgresVectorStore.replace_repository(
            "postgresql://test",
            "acme/demo",
            1536,
            np.zeros((1, 1536), dtype=np.float32),
            [chunk()],
            allow_dimension_migration=True,
        )

    assert connection.dimensions == 512
    assert connection.repositories == {"acme/demo": 2}
    assert "ROLLBACK" in connection.statements
    assert "COMMIT" not in connection.statements


def test_json_fallback_still_replaces_repository_without_pgvector(monkeypatch):
    connection = FakeConnection(
        None,
        {"acme/demo": 2},
        extension_available=False,
    )
    use_connection(monkeypatch, connection)

    store = PostgresVectorStore.replace_repository(
        "postgresql://test",
        "acme/demo",
        1536,
        np.zeros((1, 1536), dtype=np.float32),
        [chunk()],
    )

    assert store._pgvector is False
    assert connection.repositories == {"acme/demo": 1}
    assert connection.json_rows == {"acme/demo": 1}
    assert "COMMIT" in connection.statements
