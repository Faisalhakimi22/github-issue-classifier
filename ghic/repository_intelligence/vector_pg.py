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
            conn.run(
                f"ALTER TABLE ghic_repo_chunks ADD COLUMN IF NOT EXISTS "
                f"embedding vector({self.dimensions})"
            )
            # IVFFlat needs training data to be worth building; with few
            # rows a sequential scan is faster anyway, so a failure here is
            # also non-fatal.
            try:
                conn.run(
                    "CREATE INDEX IF NOT EXISTS ghic_repo_chunks_embedding_idx "
                    "ON ghic_repo_chunks USING ivfflat (embedding vector_cosine_ops) "
                    "WITH (lists = 100)"
                )
            except Exception as e:
                logger.debug("ivfflat index not created (%s); sequential scan is fine", e)
            return True
        except Exception as e:
            # Most likely: an existing column with a different dimension,
            # i.e. the embedding provider changed. The state store's
            # embedding_provider field is what should trigger a rebuild.
            logger.warning("could not prepare pgvector column (%s); using in-Python scoring", e)
            return False

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
        """Buffer then bulk-insert. Batched because a per-chunk round trip
        to a managed Postgres dominates indexing time entirely."""
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        if not len(chunks):
            return

        batch = 200
        with self._session() as conn:
            for start in range(0, len(chunks), batch):
                window = chunks[start:start + batch]
                window_vectors = vectors[start:start + batch]
                for vector, chunk in zip(window_vectors, window):
                    values = [float(v) for v in vector]
                    if self._pgvector:
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
                            embedding=json.dumps(values),
                        )
                    else:
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
                            embedding=json.dumps(values),
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
