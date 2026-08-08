"""Postgres-backed vector store: the persistence fix for serverless.

`NumpyVectorStore` keeps everything in one process and one directory, which
is exactly wrong on a platform where the filesystem is ephemeral and the
process is recycled between requests. This backend puts chunks and their
embeddings in Postgres -- the database this deploy already has for the
ledger and idempotency -- so an index survives deploys, restarts, and cold
starts, and is shared by every worker rather than rebuilt per container.

Two search paths, chosen automatically at connect time:

  **pgvector** (`CREATE EXTENSION vector`): similarity runs in the database
  via the `<=>` cosine-distance operator, with an IVFFlat index. This is
  the production path -- it scales to millions of chunks because the
  vectors never leave the server.

  **Python fallback**: when the extension isn't installed (a managed
  Postgres that doesn't offer it), embeddings are stored as JSON and scored
  in the application. Correct but O(n) in transferred bytes, so it is
  scoped to one repository's chunks per query and logged loudly. It exists
  so the feature *works* on any Postgres rather than hard-failing on the
  ones that lack an extension, not because it's a good idea at scale.

Deliberately not implemented here: Qdrant, Pinecone, Milvus, Chroma. Each
is a client library and a running service this codebase cannot exercise in
its test suite, and shipping unverified integration code that "looks right"
is worse than shipping the interface plus a documented contract. `VectorStore`
(indexer.py) is that interface -- see REPOSITORY_INTELLIGENCE_CARD.md for
what a new backend must satisfy.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .. import utils
from .indexer import VectorStore
from .models import CodeChunk, RetrievedChunk

logger = utils.get_logger(__name__)

_CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS ghic_repo_chunks (
    id BIGSERIAL PRIMARY KEY,
    repo TEXT NOT NULL,
    path TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL DEFAULT '',
    parent_symbol TEXT NOT NULL DEFAULT '',
    start_line INTEGER NOT NULL DEFAULT 0,
    end_line INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    embedding_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

# Per-repo and per-file lookups are the two access patterns: search scopes
# to a repo, incremental re-indexing deletes by (repo, path).
_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS ghic_repo_chunks_repo_idx ON ghic_repo_chunks (repo)",
    "CREATE INDEX IF NOT EXISTS ghic_repo_chunks_repo_path_idx "
    "ON ghic_repo_chunks (repo, path)",
)

_IVFFLAT_LISTS = 100
# pgvector's default is one probe, which searches roughly one list and can
# return a short, unstable candidate set before application reranking. Its
# documented starting point is sqrt(lists): 10 probes for this 100-list index.
_IVFFLAT_PROBES = 10

_CREATE_VECTOR_INDEX = (
    "CREATE INDEX IF NOT EXISTS ghic_repo_chunks_embedding_idx "
    "ON ghic_repo_chunks USING ivfflat (embedding vector_cosine_ops) "
    f"WITH (lists = {_IVFFLAT_LISTS})"
)


class EmbeddingDimensionMismatchError(RuntimeError):
    """The configured embedder cannot query the table's fixed-width vectors."""


class UnsafeEmbeddingMigrationError(RuntimeError):
    """A table-wide dimension change would invalidate another repository."""


class PostgresVectorStore(VectorStore):
    """Vectors in Postgres, scoped to one repository per instance.

    Scoping to a repo is what keeps this practical: a query never scans
    another repository's chunks, so one shared table serves thousands of
    repositories without their indexes interfering.
    """

    def __init__(self, database_url: str, repo: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        self.database_url = database_url
        self.repo = repo
        self.dimensions = dimensions
        self._pgvector = False
        self._pending: list[tuple[np.ndarray, CodeChunk]] = []
        self._ensure_schema()

    # -- schema ---------------------------------------------------------
    def _session(self):
        from .._pg import session

        return session(self.database_url)

    def _ensure_schema(self) -> None:
        with self._session() as conn:
            conn.run(_CREATE_CHUNKS_TABLE)
            for statement in _CREATE_INDEXES:
                conn.run(statement)
            self._pgvector = self._try_enable_pgvector(conn)
        logger.info(
            "postgres vector store ready for %s (pgvector=%s, dim=%d)",
            self.repo, self._pgvector, self.dimensions,
        )

    def _try_enable_pgvector(self, conn: Any) -> bool:
        """True when the `vector` extension is usable.

        Enabling requires privileges a managed instance may not grant, so
        a failure here is expected and non-fatal -- it selects the Python
        fallback rather than breaking indexing.
        """
        try:
            conn.run("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as e:
            logger.info("pgvector extension unavailable (%s); using in-Python scoring", e)
            return False
        try:
            current_dimensions = self._embedding_column_dimensions(conn)
            if current_dimensions is None:
                conn.run(
                    "ALTER TABLE ghic_repo_chunks ADD COLUMN "
                    f"embedding vector({self.dimensions})"
                )
            elif current_dimensions != self.dimensions:
                raise EmbeddingDimensionMismatchError(
                    "ghic_repo_chunks.embedding is "
                    f"vector({current_dimensions}), but the configured embedding provider "
                    f"requires vector({self.dimensions}); run an explicit forced full rebuild"
                )
            # IVFFlat needs training data to be worth building; with few
            # rows a sequential scan is faster anyway, so a failure here is
            # also non-fatal.
            self._create_vector_index(conn)
            return True
        except EmbeddingDimensionMismatchError:
            raise
        except Exception as e:
            logger.warning("could not prepare pgvector column (%s); using in-Python scoring", e)
            return False

    @staticmethod
    def _embedding_column_dimensions(conn: Any) -> int | None:
        rows = conn.run(
            "SELECT format_type(a.atttypid, a.atttypmod) "
            "FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname = 'ghic_repo_chunks' "
            "AND a.attname = 'embedding' AND a.attnum > 0 AND NOT a.attisdropped"
        )
        if not rows:
            return None
        type_name = str(rows[0][0])
        if not (type_name.startswith("vector(") and type_name.endswith(")")):
            raise RuntimeError(
                "ghic_repo_chunks.embedding must use a fixed-width vector(n) type; "
                f"found {type_name}"
            )
        try:
            return int(type_name[7:-1])
        except ValueError as e:
            raise RuntimeError(f"could not parse pgvector type {type_name}") from e

    @staticmethod
    def _create_vector_index(conn: Any) -> None:
        try:
            conn.run(_CREATE_VECTOR_INDEX)
        except Exception as e:
            logger.debug("ivfflat index not created (%s); sequential scan is fine", e)

    @classmethod
    def replace_repository(
        cls,
        database_url: str,
        repo: str,
        dimensions: int,
        vectors: np.ndarray,
        chunks: list[CodeChunk],
        *,
        allow_dimension_migration: bool = False,
    ) -> PostgresVectorStore:
        """Atomically replace one repository, migrating vector width only when forced.

        The pgvector column is table-wide. A dimension migration is therefore
        permitted only when every stored row belongs to ``repo``. All new
        embeddings are already in memory before this method begins, and the
        old rows/schema are restored by Postgres if any insert fails.
        """
        store = cls.__new__(cls)
        store.database_url = database_url
        store.repo = repo
        store.dimensions = dimensions
        store._pgvector = False
        store._pending = []
        store._replace_all(
            vectors,
            chunks,
            allow_dimension_migration=allow_dimension_migration,
        )
        return store

    def _replace_all(
        self,
        vectors: np.ndarray,
        chunks: list[CodeChunk],
        *,
        allow_dimension_migration: bool,
    ) -> None:
        matrix = self._validate_vectors(vectors, chunks)
        with self._session() as conn:
            conn.run(_CREATE_CHUNKS_TABLE)
            for statement in _CREATE_INDEXES:
                conn.run(statement)
            try:
                conn.run("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as e:
                logger.info("pgvector extension unavailable (%s); using in-Python scoring", e)
                self._replace_json(conn, matrix, chunks)
                return

            current_dimensions = self._embedding_column_dimensions(conn)
            if current_dimensions is None:
                conn.run(
                    "ALTER TABLE ghic_repo_chunks ADD COLUMN "
                    f"embedding vector({self.dimensions})"
                )
                current_dimensions = self.dimensions
            elif current_dimensions != self.dimensions:
                if not allow_dimension_migration:
                    raise EmbeddingDimensionMismatchError(
                        "refusing to replace vector dimensions without an explicit forced "
                        f"full rebuild (stored={current_dimensions}, configured={self.dimensions})"
                    )
                self._validate_dimension_migration(conn, current_dimensions)

            self._replace_pgvector(
                conn,
                matrix,
                chunks,
                current_dimensions=current_dimensions,
            )
            self._pgvector = True
            self._create_vector_index(conn)

    def _validate_dimension_migration(self, conn: Any, current_dimensions: int) -> None:
        rows = conn.run(
            "SELECT repo, count(*), count(embedding), "
            "min(vector_dims(embedding)), max(vector_dims(embedding)), "
            "count(embedding_json) "
            "FROM ghic_repo_chunks GROUP BY repo ORDER BY repo"
        )
        other_repositories = [str(row[0]) for row in rows if row[0] != self.repo]
        if other_repositories:
            raise UnsafeEmbeddingMigrationError(
                "refusing a table-wide embedding dimension migration while other "
                f"repositories have vectors: {', '.join(other_repositories)}"
            )
        for row in rows:
            total = int(row[1])
            vector_rows = int(row[2])
            minimum = int(row[3]) if row[3] is not None else None
            maximum = int(row[4]) if row[4] is not None else None
            json_rows = int(row[5])
            if (
                vector_rows != total
                or json_rows != 0
                or minimum != current_dimensions
                or maximum != current_dimensions
            ):
                raise UnsafeEmbeddingMigrationError(
                    "refusing to migrate an index with incomplete or mixed vector storage"
                )

    def _replace_pgvector(
        self,
        conn: Any,
        vectors: np.ndarray,
        chunks: list[CodeChunk],
        *,
        current_dimensions: int,
    ) -> None:
        migrating = current_dimensions != self.dimensions
        conn.run("BEGIN")
        try:
            conn.run("LOCK TABLE ghic_repo_chunks IN ACCESS EXCLUSIVE MODE")
            locked_dimensions = self._embedding_column_dimensions(conn)
            if locked_dimensions != current_dimensions:
                raise UnsafeEmbeddingMigrationError(
                    "embedding schema changed while the migration was waiting for its lock"
                )
            if migrating:
                self._validate_dimension_migration(conn, current_dimensions)
                conn.run("DROP INDEX IF EXISTS ghic_repo_chunks_embedding_idx")

            conn.run("DELETE FROM ghic_repo_chunks WHERE repo = :repo", repo=self.repo)
            if migrating:
                conn.run("ALTER TABLE ghic_repo_chunks DROP COLUMN embedding")
                conn.run(
                    "ALTER TABLE ghic_repo_chunks ADD COLUMN "
                    f"embedding vector({self.dimensions})"
                )
            self._insert_pgvector_rows(conn, vectors, chunks)
            conn.run("COMMIT")
        except Exception:
            try:
                conn.run("ROLLBACK")
            except Exception:
                pass
            raise

    def _replace_json(
        self, conn: Any, vectors: np.ndarray, chunks: list[CodeChunk]
    ) -> None:
        conn.run("BEGIN")
        try:
            conn.run("DELETE FROM ghic_repo_chunks WHERE repo = :repo", repo=self.repo)
            self._insert_json_rows(conn, vectors, chunks)
            conn.run("COMMIT")
        except Exception:
            try:
                conn.run("ROLLBACK")
            except Exception:
                pass
            raise

    def _validate_vectors(
        self, vectors: np.ndarray, chunks: list[CodeChunk]
    ) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if len(matrix) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        if matrix.ndim != 2 or matrix.shape[1] != self.dimensions:
            raise ValueError(
                f"expected a 2D embedding matrix with {self.dimensions} columns"
            )
        return matrix

    # -- VectorStore interface -------------------------------------------
    @property
    def size(self) -> int:
        try:
            with self._session() as conn:
                rows = conn.run(
                    "SELECT count(*) FROM ghic_repo_chunks WHERE repo = :repo", repo=self.repo
                )
            return int(rows[0][0])
        except Exception as e:
            logger.warning("could not count chunks for %s: %s", self.repo, e)
            return 0

    def add(self, vectors: np.ndarray, chunks: list[CodeChunk]) -> None:
        """Insert vectors through one shared database session."""
        vectors = self._validate_vectors(vectors, chunks)
        if not len(chunks):
            return

        with self._session() as conn:
            if self._pgvector:
                self._insert_pgvector_rows(conn, vectors, chunks)
            else:
                self._insert_json_rows(conn, vectors, chunks)

    def _insert_pgvector_rows(
        self, conn: Any, vectors: np.ndarray, chunks: list[CodeChunk]
    ) -> None:
        for vector, chunk in zip(vectors, chunks):
            conn.run(
                "INSERT INTO ghic_repo_chunks "
                "(repo, path, language, kind, symbol, parent_symbol, "
                " start_line, end_line, text, embedding) "
                "VALUES (:repo, :path, :language, :kind, :symbol, :parent, "
                ":start, :end, :text, CAST(:embedding AS vector))",
                repo=chunk.repo or self.repo, path=chunk.path,
                language=chunk.language, kind=chunk.kind, symbol=chunk.symbol,
                parent=chunk.parent_symbol, start=chunk.start_line,
                end=chunk.end_line, text=chunk.text,
                embedding=json.dumps([float(v) for v in vector]),
            )

    def _insert_json_rows(
        self, conn: Any, vectors: np.ndarray, chunks: list[CodeChunk]
    ) -> None:
        for vector, chunk in zip(vectors, chunks):
            conn.run(
                "INSERT INTO ghic_repo_chunks "
                "(repo, path, language, kind, symbol, parent_symbol, "
                " start_line, end_line, text, embedding_json) "
                "VALUES (:repo, :path, :language, :kind, :symbol, :parent, "
                ":start, :end, :text, :embedding)",
                repo=chunk.repo or self.repo, path=chunk.path,
                language=chunk.language, kind=chunk.kind, symbol=chunk.symbol,
                parent=chunk.parent_symbol, start=chunk.start_line,
                end=chunk.end_line, text=chunk.text,
                embedding=json.dumps([float(v) for v in vector]),
            )

    def search(self, query: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        try:
            if self._pgvector:
                return self._search_pgvector(query, top_k)
            return self._search_python(query, top_k)
        except Exception as e:
            logger.warning("vector search failed for %s: %s", self.repo, e)
            return []

    def _search_pgvector(self, query: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        # `<=>` is cosine *distance*; similarity is 1 - distance. Vectors
        # are unit-normalized by the embedding provider, so this matches
        # NumpyVectorStore's dot product exactly.
        with self._session() as conn:
            # A search connection is dedicated to this operation, so a
            # session-scoped setting cannot leak into another request.
            conn.run(f"SET ivfflat.probes = {_IVFFLAT_PROBES}")
            rows = conn.run(
                "SELECT path, language, kind, symbol, parent_symbol, start_line, "
                "       end_line, text, 1 - (embedding <=> CAST(:q AS vector)) AS score "
                "FROM ghic_repo_chunks "
                "WHERE repo = :repo AND embedding IS NOT NULL "
                "ORDER BY embedding <=> CAST(:q AS vector) "
                "LIMIT :k",
                q=json.dumps([float(v) for v in query]), repo=self.repo, k=top_k,
            )
        return [self._row_to_retrieved(row, float(row[8])) for row in rows]

    def _search_python(self, query: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        with self._session() as conn:
            rows = conn.run(
                "SELECT path, language, kind, symbol, parent_symbol, start_line, "
                "       end_line, text, embedding_json "
                "FROM ghic_repo_chunks WHERE repo = :repo AND embedding_json IS NOT NULL",
                repo=self.repo,
            )
        if not rows:
            return []
        scored: list[tuple[float, Any]] = []
        query_vector = np.asarray(query, dtype=np.float32)
        for row in rows:
            raw = row[8]
            values = json.loads(raw) if isinstance(raw, str) else raw
            vector = np.asarray(values, dtype=np.float32)
            if vector.shape != query_vector.shape:
                continue  # stale row from a different embedding provider
            scored.append((float(vector @ query_vector), row))
        scored.sort(key=lambda pair: -pair[0])
        return [self._row_to_retrieved(row, score) for score, row in scored[:top_k]]

    def _row_to_retrieved(self, row: Any, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk=CodeChunk(
                repo=self.repo, path=row[0], language=row[1] or "", text=row[7],
                start_line=int(row[5] or 0), end_line=int(row[6] or 0),
                kind=row[2] or "module", symbol=row[3] or "", parent_symbol=row[4] or "",
            ),
            score=score,
        )

    # -- incremental maintenance ------------------------------------------
    def delete_paths(self, paths: list[str]) -> int:
        """Remove every chunk for the given files. The delete half of an
        incremental re-index: changed and deleted files are dropped, then
        the changed ones are re-added."""
        if not paths:
            return 0
        removed = 0
        with self._session() as conn:
            for path in paths:
                rows = conn.run(
                    "DELETE FROM ghic_repo_chunks WHERE repo = :repo AND path = :path "
                    "RETURNING id",
                    repo=self.repo, path=path,
                )
                removed += len(rows or [])
        return removed

    def clear(self) -> int:
        with self._session() as conn:
            rows = conn.run(
                "DELETE FROM ghic_repo_chunks WHERE repo = :repo RETURNING id", repo=self.repo
            )
        return len(rows or [])

    def indexed_paths(self) -> set[str]:
        with self._session() as conn:
            rows = conn.run(
                "SELECT DISTINCT path FROM ghic_repo_chunks WHERE repo = :repo", repo=self.repo
            )
        return {row[0] for row in rows}

    # -- persistence is the database; these are no-ops by design ----------
    def save(self, directory: Any) -> None:
        """Rows are written on `add`. Nothing to flush -- the whole point
        of this backend is that there is no local artifact to persist."""

    @classmethod
    def load(cls, directory: Any) -> VectorStore:
        raise NotImplementedError(
            "PostgresVectorStore is constructed with (database_url, repo, dimensions), "
            "not loaded from a directory"
        )
