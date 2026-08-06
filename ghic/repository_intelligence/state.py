"""Persistent repository state: lifecycle, metadata, and retrieval stats.

Phase 1 inferred everything from the filesystem -- "is there an index
directory?" was the whole state model. That works on one box and fails
everywhere else: it can't express *queued* or *failed*, it can't tell a
cold start from a never-indexed repo, and it disappears entirely on an
ephemeral filesystem.

This module makes state explicit and storable. `RepositoryRecord` is the
durable row; `RepositoryStateStore` is the interface; three backends ship:

  `PostgresStateStore` -- production. Reuses the same DATABASE_URL and the
  same pg8000 driver the ledger and idempotency store already use, so a
  deploy that already has Postgres needs no new infrastructure.
  `FileStateStore`    -- single-box Docker/Fly with a volume.
  `MemoryStateStore`  -- tests and local runs; explicitly not durable.

Backward compatible by construction: nothing here is required. A service
built without a state store behaves exactly as Phase 1 did (see
repository_service.py), which keeps the "favour extension over
replacement" constraint honest rather than nominal.
"""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from .. import utils

logger = utils.get_logger(__name__)


class RepositoryState(str, Enum):
    """Lifecycle of one repository's index.

    str-valued so a record serializes to JSON and lands in a Postgres text
    column without a custom adapter, and so `state == "ready"` works for
    dashboard code that never imports this module.
    """
    NOT_INDEXED = "not_indexed"
    QUEUED = "queued"
    INDEXING = "indexing"
    READY = "ready"
    UPDATING = "updating"        # re-indexing while a usable index still exists
    FAILED = "failed"
    ARCHIVED = "archived"        # repo deleted/uninstalled; index retained briefly

    @property
    def is_searchable(self) -> bool:
        """Whether retrieval should attempt to use this repo's index.

        UPDATING is searchable on purpose: an incremental re-index leaves
        the previous index in place and usable, so an issue arriving
        mid-update gets slightly stale evidence rather than none.
        """
        return self in (RepositoryState.READY, RepositoryState.UPDATING)

    @property
    def is_in_flight(self) -> bool:
        """Whether a job is already queued/running, so don't queue another."""
        return self in (RepositoryState.QUEUED, RepositoryState.INDEXING,
                        RepositoryState.UPDATING)


# How long a QUEUED/INDEXING record is trusted before another job may be
# queued. A worker that dies mid-index would otherwise wedge a repository in
# INDEXING forever, and nothing would ever retry it.
STALE_IN_FLIGHT_SECONDS = 60 * 60


@dataclass(frozen=True)
class RepositoryRecord:
    """One repository's durable state. `metadata_json` holds the
    RepositoryMetadata produced by the indexer (language, frameworks,
    readme summary...) so the dashboard can render it without loading a
    vector index."""
    repo: str
    state: RepositoryState = RepositoryState.NOT_INDEXED
    owner: str = ""
    name: str = ""
    default_branch: str = ""
    indexed_commit_sha: str = ""
    index_version: int = 0
    embedding_provider: str = ""
    vector_backend: str = ""
    chunk_count: int = 0
    file_count: int = 0
    primary_language: str = ""
    frameworks: list[str] = field(default_factory=list)
    readme_summary: str = ""
    architecture_summary: str = ""
    indexed_at: float = 0.0
    last_accessed_at: float = 0.0
    updated_at: float = 0.0
    retrieval_count: int = 0
    retrieval_hit_count: int = 0      # retrievals that returned >=1 chunk
    index_duration_seconds: float = 0.0
    last_error: str = ""
    metadata_json: dict[str, Any] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.retrieval_hit_count / self.retrieval_count if self.retrieval_count else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "state": self.state.value,
            "owner": self.owner,
            "name": self.name,
            "default_branch": self.default_branch,
            "indexed_commit_sha": self.indexed_commit_sha,
            "index_version": self.index_version,
            "embedding_provider": self.embedding_provider,
            "vector_backend": self.vector_backend,
            "chunk_count": self.chunk_count,
            "file_count": self.file_count,
            "primary_language": self.primary_language,
            "frameworks": self.frameworks,
            "readme_summary": self.readme_summary,
            "architecture_summary": self.architecture_summary,
            "indexed_at": self.indexed_at,
            "last_accessed_at": self.last_accessed_at,
            "updated_at": self.updated_at,
            "retrieval_count": self.retrieval_count,
            "retrieval_hit_count": self.retrieval_hit_count,
            "hit_rate": round(self.hit_rate, 4),
            "index_duration_seconds": round(self.index_duration_seconds, 3),
            "last_error": self.last_error,
            "metadata": self.metadata_json,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RepositoryRecord:
        owner, _, name = str(raw.get("repo", "")).partition("/")
        try:
            state = RepositoryState(raw.get("state", "not_indexed"))
        except ValueError:
            # An unknown state from a newer version: treat as unusable
            # rather than crashing an older worker that reads the same row.
            state = RepositoryState.NOT_INDEXED
        return cls(
            repo=raw["repo"],
            state=state,
            owner=raw.get("owner") or owner,
            name=raw.get("name") or name,
            default_branch=raw.get("default_branch", ""),
            indexed_commit_sha=raw.get("indexed_commit_sha", ""),
            index_version=int(raw.get("index_version", 0)),
            embedding_provider=raw.get("embedding_provider", ""),
            vector_backend=raw.get("vector_backend", ""),
            chunk_count=int(raw.get("chunk_count", 0)),
            file_count=int(raw.get("file_count", 0)),
            primary_language=raw.get("primary_language", ""),
            frameworks=list(raw.get("frameworks") or []),
            readme_summary=raw.get("readme_summary", ""),
            architecture_summary=raw.get("architecture_summary", ""),
            indexed_at=float(raw.get("indexed_at", 0) or 0),
            last_accessed_at=float(raw.get("last_accessed_at", 0) or 0),
            updated_at=float(raw.get("updated_at", 0) or 0),
            retrieval_count=int(raw.get("retrieval_count", 0)),
            retrieval_hit_count=int(raw.get("retrieval_hit_count", 0)),
            index_duration_seconds=float(raw.get("index_duration_seconds", 0) or 0),
            last_error=raw.get("last_error", ""),
            metadata_json=dict(raw.get("metadata") or {}),
        )


class RepositoryStateStore(ABC):
    """Durable repository state. Implementations must never raise on a read
    -- callers treat "unknown" as "not indexed" and continue."""

    @abstractmethod
    def get(self, repo: str) -> RepositoryRecord | None:
        raise NotImplementedError

    @abstractmethod
    def put(self, record: RepositoryRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def list(self, *, limit: int = 100) -> list[RepositoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, repo: str) -> None:
        raise NotImplementedError

    @property
    def durable(self) -> bool:
        """False for stores that don't survive a restart. Surfaced in
        /healthz so an operator can see at a glance that a deploy is
        running without real persistence rather than discovering it when
        every cold start re-indexes."""
        return True

    # -- shared helpers, identical across backends -------------------------
    def mark(
        self, repo: str, state: RepositoryState, *, error: str = "", **fields: Any
    ) -> RepositoryRecord:
        """Transition `repo` to `state`, creating the record if needed."""
        owner, _, name = repo.partition("/")
        current = self.get(repo) or RepositoryRecord(repo=repo, owner=owner, name=name)
        record = replace(
            current, state=state, last_error=error, updated_at=time.time(), **fields
        )
        self.put(record)
        return record

    def record_retrieval(self, repo: str, *, hit: bool) -> None:
        """Bump retrieval counters. Best-effort: a stats write must never
        interfere with serving a webhook."""
        try:
            current = self.get(repo)
            if current is None:
                return
            self.put(replace(
                current,
                retrieval_count=current.retrieval_count + 1,
                retrieval_hit_count=current.retrieval_hit_count + int(hit),
                last_accessed_at=time.time(),
            ))
        except Exception as e:
            logger.debug("could not record retrieval stats for %s: %s", repo, e)


class MemoryStateStore(RepositoryStateStore):
    """Process-local. Correct for tests, explicitly not durable."""

    def __init__(self) -> None:
        self._records: dict[str, RepositoryRecord] = {}
        self._lock = threading.Lock()

    @property
    def durable(self) -> bool:
        return False

    def get(self, repo: str) -> RepositoryRecord | None:
        with self._lock:
            return self._records.get(repo)

    def put(self, record: RepositoryRecord) -> None:
        with self._lock:
            self._records[record.repo] = record

    def list(self, *, limit: int = 100) -> list[RepositoryRecord]:
        with self._lock:
            records = sorted(self._records.values(), key=lambda r: -r.updated_at)
        return records[:limit]

    def delete(self, repo: str) -> None:
        with self._lock:
            self._records.pop(repo, None)


class FileStateStore(RepositoryStateStore):
    """One JSON file, rewritten atomically. Fine for a single worker with a
    volume; not safe across machines, which is what Postgres is for."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.path)          # atomic on POSIX and Windows
        except OSError as e:
            logger.warning("could not persist repository state (%s)", e)

    def get(self, repo: str) -> RepositoryRecord | None:
        with self._lock:
            raw = self._read().get(repo)
        return RepositoryRecord.from_dict(raw) if raw else None

    def put(self, record: RepositoryRecord) -> None:
        with self._lock:
            data = self._read()
            data[record.repo] = record.as_dict()
            self._write(data)

    def list(self, *, limit: int = 100) -> list[RepositoryRecord]:
        with self._lock:
            data = self._read()
        records = [RepositoryRecord.from_dict(r) for r in data.values()]
        return sorted(records, key=lambda r: -r.updated_at)[:limit]

    def delete(self, repo: str) -> None:
        with self._lock:
            data = self._read()
            if data.pop(repo, None) is not None:
                self._write(data)


_CREATE_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS ghic_repository_state (
    repo TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    data JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


class PostgresStateStore(RepositoryStateStore):
    """Production backend. Survives deploys, restarts, and cold starts, and
    is shared across every worker -- which is what makes "don't queue a
    second index for a repo already indexing" work at more than one replica.

    Uses pg8000 (pure Python) and the same connection helper style as
    pg_ledger.py/idempotency.py rather than introducing a second driver;
    compiled wheels are what the Vercel bundle limit punishes.
    """

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        self.database_url = database_url
        self._ensure_table()

    def _session(self):
        from .._pg import session

        return session(self.database_url)

    def _ensure_table(self) -> None:
        with self._session() as conn:
            conn.run(_CREATE_STATE_TABLE)

    def get(self, repo: str) -> RepositoryRecord | None:
        try:
            with self._session() as conn:
                rows = conn.run(
                    "SELECT data FROM ghic_repository_state WHERE repo = :repo", repo=repo
                )
        except Exception as e:
            logger.warning("repository state read failed for %s: %s", repo, e)
            return None
        if not rows:
            return None
        raw = rows[0][0]
        if isinstance(raw, str):
            raw = json.loads(raw)
        return RepositoryRecord.from_dict(raw)

    def put(self, record: RepositoryRecord) -> None:
        try:
            with self._session() as conn:
                conn.run(
                    "INSERT INTO ghic_repository_state (repo, state, data, updated_at) "
                    "VALUES (:repo, :state, :data, now()) "
                    "ON CONFLICT (repo) DO UPDATE SET "
                    "state = EXCLUDED.state, data = EXCLUDED.data, updated_at = now()",
                    repo=record.repo, state=record.state.value,
                    data=json.dumps(record.as_dict()),
                )
        except Exception as e:
            logger.warning("repository state write failed for %s: %s", record.repo, e)

    def list(self, *, limit: int = 100) -> list[RepositoryRecord]:
        try:
            with self._session() as conn:
                rows = conn.run(
                    "SELECT data FROM ghic_repository_state "
                    "ORDER BY updated_at DESC LIMIT :limit", limit=limit,
                )
        except Exception as e:
            logger.warning("repository state list failed: %s", e)
            return []
        out: list[RepositoryRecord] = []
        for (raw,) in rows:
            if isinstance(raw, str):
                raw = json.loads(raw)
            out.append(RepositoryRecord.from_dict(raw))
        return out

    def delete(self, repo: str) -> None:
        try:
            with self._session() as conn:
                conn.run("DELETE FROM ghic_repository_state WHERE repo = :repo", repo=repo)
        except Exception as e:
            logger.warning("repository state delete failed for %s: %s", repo, e)


def build_state_store(
    *, database_url: str = "", file_path: Path | None = None
) -> RepositoryStateStore:
    """Best available backend, in the same precedence order the ledger and
    idempotency store already use: Postgres, then a file, then memory."""
    if database_url:
        try:
            return PostgresStateStore(database_url)
        except Exception as e:
            logger.warning("Postgres state store unavailable (%s); falling back", e)
    if file_path is not None:
        return FileStateStore(file_path)
    logger.info("repository state is in-memory only; it will not survive a restart")
    return MemoryStateStore()
