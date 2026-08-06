"""Shared Postgres connection helper.

`pg_ledger.py` and `idempotency.py` each grew their own identical
`_connect()`; this is that function, extracted so a third consumer (the
repository state store and the pgvector backend) doesn't make it four.
Those two are deliberately left alone -- they work, and rewriting working
storage code to import a helper is churn, not improvement. New code uses
this one.

pg8000 (pure Python, no compiled extension) throughout, for the reason
pg_ledger.py documents: compiled wheels are what serverless bundle-size
limits punish.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse


def connect(database_url: str) -> Any:
    """A pg8000 native Connection. Caller closes it (or uses `session()`)."""
    import pg8000.native  # lazy: only needed when a DATABASE_URL is set

    u = urlparse(database_url)
    return pg8000.native.Connection(
        user=u.username,
        password=u.password,
        host=u.hostname,
        port=u.port or 5432,
        database=(u.path or "/").lstrip("/"),
        ssl_context=True,
    )


@contextmanager
def session(database_url: str):
    """Connection as a context manager, always closed.

    A connection per operation rather than a long-lived one: serverless
    invocations are short and a pooler (Neon, PgBouncer) handles the reuse.
    A cached connection across invocations is the classic way to exhaust a
    Postgres connection limit from a platform that scales horizontally.
    """
    conn = connect(database_url)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:  # already closed / network gone -- nothing to do
            pass
