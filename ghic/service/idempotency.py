"""Idempotency: has this GitHub webhook delivery already been processed?

GitHub reuses the same `X-GitHub-Delivery` ID when it retries a delivery
(a webhook that didn't get a timely 2xx, or one your own async pipeline
queued and later replayed) — so it's the natural dedup key. Three backends,
same shape as tracking.py's ledger split:

  - Postgres (Vercel, no persistent disk): a dedicated table with a UNIQUE
    key column, deduped atomically via `INSERT ... ON CONFLICT DO NOTHING
    RETURNING key` -- this is the one that actually matters under real
    concurrency, since a serverless deploy can run many invocations at once.
  - File (Docker/Fly, persistent disk, single-writer assumption -- same
    caveat as the JSONL ledger backend): a lock-guarded JSON set on disk.
  - In-memory: correct only within one process's lifetime, same meaning as
    "no ledger configured" has everywhere else in this codebase.

Known limitation, not hidden: none of these prune old keys. A busy
long-running deploy accumulates one row/line per delivery forever. Matches
the online-evaluation ledger's own growth characteristics; revisit together
if it ever matters in practice.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from .. import utils

logger = utils.get_logger(__name__)


class IdempotencyStore(Protocol):
    def mark_if_new(self, key: str) -> bool:
        """True if `key` hadn't been seen before (and is now recorded) --
        the caller should proceed. False means it's a duplicate -- skip."""
        ...

    def release(self, key: str) -> None:
        """Forget `key` so a redelivery is processed instead of skipped.

        A delivery is marked consumed before it is handled, which is right
        for issue events: a retry must not post a second comment. It is
        wrong for a handler whose work must actually complete. If such a
        handler fails, the delivery was recorded as done while nothing was
        done, and GitHub's retry would be discarded as a duplicate.

        Releasing the key on that failure path is what makes a non-2xx
        response mean anything. Only use it for work that is safe to repeat.
        """
        ...


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ghic_idempotency (
    key TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def _connect(database_url: str) -> Any:
    import pg8000.native  # lazy import: only needed when DATABASE_URL is set

    u = urlparse(database_url)
    return pg8000.native.Connection(
        user=u.username,
        password=u.password,
        host=u.hostname,
        port=u.port or 5432,
        database=(u.path or "/").lstrip("/"),
        ssl_context=True,
    )


class PostgresIdempotencyStore:
    def __init__(self, database_url: str) -> None:
        self._conn = _connect(database_url)
        self._conn.run(_CREATE_TABLE)

    def mark_if_new(self, key: str) -> bool:
        rows = self._conn.run(
            "INSERT INTO ghic_idempotency (key) VALUES (:key) "
            "ON CONFLICT (key) DO NOTHING RETURNING key",
            key=key,
        )
        return len(rows) > 0

    def release(self, key: str) -> None:
        self._conn.run("DELETE FROM ghic_idempotency WHERE key = :key", key=key)


class FileIdempotencyStore:
    """Single-writer assumption, same as tracking.py's JSONL ledger backend --
    fine for a single Docker/Fly instance, not a safety net against multiple
    concurrent writers to the same file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        if path.exists():
            try:
                self._seen = set(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("idempotency file %s unreadable (%s); starting empty", path, e)

    def mark_if_new(self, key: str) -> bool:
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(sorted(self._seen)), encoding="utf-8")
            return True

    def release(self, key: str) -> None:
        with self._lock:
            self._seen.discard(key)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(sorted(self._seen)), encoding="utf-8")


class _InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: set[str] = set()

    def mark_if_new(self, key: str) -> bool:
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            return True

    def release(self, key: str) -> None:
        with self._lock:
            self._seen.discard(key)


def build_idempotency_store(
    database_url: str = "", file_path: Path | None = None,
) -> IdempotencyStore:
    if database_url:
        return PostgresIdempotencyStore(database_url)
    if file_path:
        return FileIdempotencyStore(file_path)
    return _InMemoryIdempotencyStore()
