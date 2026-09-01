"""Data retention: deleting what the privacy policy says is deleted.

The marketing site and the privacy policy both make a specific promise --
"prediction records are retained for 90 days, then deleted". Until this
module existed that promise was false, and a false retention claim is not a
missing feature, it is a statement to customers that does not hold.

What is in scope is exactly what was promised: the prediction ledger. Not
the billing records, which a customer needs to dispute a charge and which
say nothing about the content of an issue; not the repository index, which
is derived from code the App can re-read at any time and is replaced
wholesale on reindex; not the connection state, which is what makes
authorization work.

The idempotency keys go too, for a different reason. `idempotency.py`
documents that nothing prunes them and that they grow one row per delivery
forever. A key older than the retention window cannot suppress a real
redelivery -- GitHub gives up retrying long before then -- so pruning them
costs nothing and stops a table growing without limit.

Deletion is by age and nothing else. No workspace filter, no repository
filter: a retention window that applies to some tenants and not others is
not a retention window.
"""
from __future__ import annotations

from typing import Any

from .. import utils
from .._pg import session

logger = utils.get_logger(__name__)

DEFAULT_RETENTION_DAYS = 90

#: Ledger record types the policy covers. Everything the tracker writes is a
#: record about an issue, so the sweep is by age rather than by type -- but
#: the list is here so a future record type that must survive is a visible
#: decision rather than an accident.
LEDGER_TABLE = "ghic_ledger"
IDEMPOTENCY_TABLE = "ghic_idempotency"


def _delete_older_than(conn: Any, table: str, days: int) -> int:
    """Delete rows older than `days`, returning how many went.

    The interval is built from an integer parameter rather than interpolated
    text: `make_interval` takes a bound value, so the number never becomes
    part of the SQL.
    """
    rows = conn.run(
        f"WITH removed AS ("  # noqa: S608 -- table names are module constants
        f"  DELETE FROM {table} "
        f"  WHERE created_at < now() - make_interval(days => :days) "
        f"  RETURNING 1"
        f") SELECT count(*) FROM removed",
        days=days,
    )
    return int(rows[0][0]) if rows else 0


def _table_exists(conn: Any, name: str) -> bool:
    rows = conn.run("SELECT to_regclass(:q) IS NOT NULL", q=f"public.{name}")
    return bool(rows and rows[0][0])


def sweep(database_url: str, days: int = DEFAULT_RETENTION_DAYS) -> dict[str, Any]:
    """Run one retention pass. Returns what it removed.

    Safe to run repeatedly and safe to run concurrently: a second pass finds
    nothing left to delete, and two overlapping passes delete disjoint sets
    because each DELETE takes its own row locks.
    """
    if not database_url:
        return {"ok": False, "reason": "no_database", "deleted": {}}
    window = max(1, int(days))
    deleted: dict[str, int] = {}
    with session(database_url) as conn:
        for table in (LEDGER_TABLE, IDEMPOTENCY_TABLE):
            if not _table_exists(conn, table):
                continue
            deleted[table] = _delete_older_than(conn, table, window)
    total = sum(deleted.values())
    if total:
        logger.info("retention swept %d rows older than %d days: %s",
                    total, window, deleted)
    return {"ok": True, "retention_days": window, "deleted": deleted, "total": total}
