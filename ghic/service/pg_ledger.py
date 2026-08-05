"""Postgres-backed ledger storage, for deploys with no persistent disk (Vercel).

The JSONL ledger (see tracking.py) assumes a writable local filesystem that
survives process restarts. Serverless platforms don't offer that — each
invocation may be a fresh container, and any local write is gone by the next
cold start. Vercel (via its Postgres/Neon integration) does offer a database
that survives across invocations, so this module gives PredictionTracker a
second backend with the same shape: append a record, replay them all back on
startup.

pg8000 (pure Python, no compiled extension) is used instead of psycopg2 to
avoid fighting Vercel's function bundle size limit with compiled wheels.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

from .. import utils

logger = utils.get_logger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ghic_ledger (
    id BIGSERIAL PRIMARY KEY,
    data JSONB NOT NULL,
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


class PostgresLedgerBackend:
    """Same append/replay contract as the JSONL backend, backed by a table.

    One row per ledger record (the same dicts tracking.py already builds for
    prediction/outcome/action/label_event), stored as JSONB and replayed back
    in insertion order on startup — mirrors the JSONL file's append+replay
    semantics exactly, so PredictionTracker's in-memory rebuild logic is
    unchanged.
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._conn = _connect(database_url)
        self._conn.run(_CREATE_TABLE)

    def append(self, record: dict[str, Any]) -> None:
        self._conn.run(
            "INSERT INTO ghic_ledger (data) VALUES (:data)",
            data=json.dumps(record, ensure_ascii=False),
        )

    def replay(self) -> Iterator[dict[str, Any]]:
        rows = self._conn.run("SELECT data FROM ghic_ledger ORDER BY id")
        for (data,) in rows:
            yield data if isinstance(data, dict) else json.loads(data)
