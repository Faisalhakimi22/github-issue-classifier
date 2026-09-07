"""Plan limits and usage accounting for the analysis path.

The limits themselves are not written here. They live in `ghic_plans`, in
the database, because two services enforce them: the Hub refuses a
repository connection, this one refuses an issue analysis. The same numbers
written twice in two languages drift, and the direction they drift in is a
customer charged for one thing and served another. The Hub's reader is
`api/_lib/plans.mjs`; this is its counterpart, deliberately shaped the same.

What this module adds beyond reading a number is the accounting, and the
accounting has one rule that shapes everything else: **a quota is spent
before the work runs, not after.** Checking first and recording afterwards
leaves a window in which two concurrent deliveries both see one remaining
slot and both proceed, and leaves a crash between the work and the record as
a free analysis. So an analysis reserves its slot, and gives the slot back
if the work fails.

Not billing for failures is the reason `release()` exists. It mirrors the
idempotency store's `release()` for the same reason: something was marked
done before it was done, and the marking has to be undone when it wasn't.

Idempotency is the schema's job, not this module's care:

    CREATE UNIQUE INDEX ghic_usage_counted_once_idx
      ON ghic_usage_events (workspace_id, period, repo, issue_number)
      WHERE outcome = 'counted'

A redelivered webhook, a QStash retry, or a re-score after an issue edit
cannot bill twice however carelessly the caller behaves. That index is the
guarantee; the code below merely reads it correctly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .. import utils
from .._pg import session

logger = utils.get_logger(__name__)

# SQLSTATE 42P01: relation does not exist. The plan tables have not been
# migrated onto this deployment yet.
UNDEFINED_TABLE = "42P01"

COUNTED = "counted"
FAILED = "failed"
SKIPPED = "skipped"
LIMITED = "limited"


@dataclass(frozen=True)
class Plan:
    """A workspace's plan, as the database has it.

    `max_issues_per_period is None` means unlimited, and it is not the same
    as zero. `int(None)` raises but `Number(null)` is 0 on the Hub side, and
    a coercion in either direction turns the most expensive plan into the
    most restrictive one -- so the None has to survive into every reader.

    `enforced` is False only when the plan could not be read. Enforcement
    paths reject that state; read-only UI may still explain the outage.
    """

    plan: str = "unknown"
    max_repositories: int | None = None
    max_issues_per_period: int | None = None
    period: str = "month"
    enforced: bool = False

    @property
    def unlimited_issues(self) -> bool:
        return self.max_issues_per_period is None


@dataclass(frozen=True)
class UsageDecision:
    """The answer to "may this analysis run, and was it charged for?"."""

    allowed: bool
    reason: str
    plan: Plan
    period: str
    used: int = 0
    limit: int | None = None
    #: True when this call took the slot, and so is the call responsible for
    #: giving it back if the work fails. False when the issue was already
    #: counted (a retry) or when nothing is being enforced.
    reserved: bool = False
    #: True when this is the first refusal of the period, which is the only
    #: one worth telling the maintainer about.
    first_refusal: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "plan": self.plan.plan,
            "period": self.period,
            "used": self.used,
            "limit": self.limit,
            "enforced": self.plan.enforced,
        }


UNAVAILABLE = Plan(plan="unavailable")


class UsageUnavailableError(RuntimeError):
    """The workspace plan cannot be established authoritatively."""


def _sqlstate(error: Exception) -> str:
    """The SQLSTATE of a pg8000 error, or "" if it is not one.

    pg8000 raises `DatabaseError({'S': 'ERROR', 'C': '42P01', ...})`.
    """
    args = getattr(error, "args", None)
    if args and isinstance(args[0], dict):
        return str(args[0].get("C") or "")
    return ""


def period_key(period: str = "month", now: datetime | None = None) -> str:
    """The billing period an instant falls in, as a sortable string.

    UTC, always. A calendar month that starts at the customer's local
    midnight is a month whose boundary moves with a timezone database, and
    two services would have to agree on which one.
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    if period == "day":
        return moment.strftime("%Y-%m-%d")
    return moment.strftime("%Y-%m")


def _plan_from_row(row: Any) -> Plan:
    plan, max_repositories, max_issues, period = row[0], row[1], row[2], row[3]
    return Plan(
        plan=str(plan),
        max_repositories=None if max_repositories is None else int(max_repositories),
        max_issues_per_period=None if max_issues is None else int(max_issues),
        period=str(period or "month"),
        enforced=True,
    )


_PLAN_SQL = (
    "SELECT p.plan, p.max_repositories, p.max_issues_per_period, p.period "
    "FROM ghic_workspaces w JOIN ghic_plans p ON p.plan = w.plan "
    "WHERE w.id = :workspace_id"
)


def _read_plan(conn: Any, workspace_id: str) -> Plan:
    rows = conn.run(_PLAN_SQL, workspace_id=workspace_id)
    if not rows:
        raise UsageUnavailableError("workspace has no valid plan")
    return _plan_from_row(rows[0])


def plan_for_workspace(database_url: str, workspace_id: Any) -> Plan:
    """Return the authoritative plan, or fail instead of granting free work."""
    identifier = str(workspace_id or "").strip()
    if not identifier or not database_url:
        raise UsageUnavailableError("usage requires database and workspace context")
    try:
        with session(database_url) as conn:
            return _read_plan(conn, identifier)
    except Exception as error:
        if isinstance(error, UsageUnavailableError):
            raise
        if _sqlstate(error) == UNDEFINED_TABLE:
            logger.error("plan tables are unavailable")
        else:
            logger.warning("plan lookup failed for %s: %s", identifier, error)
        raise UsageUnavailableError("workspace plan is unavailable") from error


def _count(conn: Any, workspace_id: str, period: str, outcome: str) -> int:
    rows = conn.run(
        "SELECT count(*) FROM ghic_usage_events "
        "WHERE workspace_id = :workspace_id AND period = :period "
        "AND outcome = :outcome",
        workspace_id=workspace_id,
        period=period,
        outcome=outcome,
    )
    return int(rows[0][0]) if rows else 0


def _insert(
    conn: Any,
    workspace_id: str,
    period: str,
    repo: str,
    issue_number: int,
    outcome: str,
    reason: str | None,
) -> None:
    conn.run(
        "INSERT INTO ghic_usage_events "
        "(workspace_id, period, repo, issue_number, outcome, reason) "
        "VALUES (:workspace_id, :period, :repo, :issue_number, :outcome, :reason) "
        "ON CONFLICT DO NOTHING",
        workspace_id=workspace_id,
        period=period,
        repo=repo,
        issue_number=issue_number,
        outcome=outcome,
        reason=reason,
    )


def reserve(
    database_url: str,
    workspace_id: Any,
    repo: str,
    issue_number: Any,
    now: datetime | None = None,
) -> UsageDecision:
    """Take this issue's quota slot, or refuse it.

    Runs as one serialized transaction per (workspace, period). The advisory
    lock is what makes "read the count, then insert if it is under the
    limit" mean anything: under READ COMMITTED two concurrent deliveries
    would each read a count that does not yet include the other's
    uncommitted row, and a plan with one slot left would sell it twice.

    Returns `reserved=True` only when this call is the one that took the
    slot -- and so the only one that should hand it back via `release()`.
    An issue already counted this period (a redelivery, a QStash retry, a
    re-score after an edit) is allowed through without being charged again
    and without owning the reservation.
    """
    identifier = str(workspace_id or "").strip()
    repo_name = str(repo or "")
    try:
        number = int(issue_number)
    except (TypeError, ValueError):
        number = 0
    if not identifier or not database_url or not repo_name or number <= 0:
        return UsageDecision(False, "usage_unavailable", UNAVAILABLE, period_key())

    try:
        with session(database_url) as conn:
            conn.run("BEGIN")
            try:
                plan = _read_plan(conn, identifier)
                period = period_key(plan.period, now)
                if plan.unlimited_issues:
                    conn.run("COMMIT")
                    return UsageDecision(
                        True, "unlimited", plan, period,
                    )

                conn.run(
                    "SELECT pg_advisory_xact_lock("
                    "hashtextextended(:key, 0))",
                    key=f"ghic.usage.{identifier}.{period}",
                )

                already = conn.run(
                    "SELECT 1 FROM ghic_usage_events "
                    "WHERE workspace_id = :workspace_id AND period = :period "
                    "AND repo = :repo AND issue_number = :issue_number "
                    "AND outcome = :outcome",
                    workspace_id=identifier,
                    period=period,
                    repo=repo_name,
                    issue_number=number,
                    outcome=COUNTED,
                )
                used = _count(conn, identifier, period, COUNTED)
                limit = plan.max_issues_per_period

                if already:
                    conn.run("COMMIT")
                    return UsageDecision(
                        True, "already_counted", plan, period, used, limit
                    )

                if used >= limit:
                    first = _count(conn, identifier, period, LIMITED) == 0
                    _insert(
                        conn, identifier, period, repo_name, number,
                        LIMITED, "issue_quota_exhausted",
                    )
                    conn.run("COMMIT")
                    return UsageDecision(
                        False,
                        "issue_quota_exhausted",
                        plan,
                        period,
                        used,
                        limit,
                        first_refusal=first,
                    )

                _insert(conn, identifier, period, repo_name, number, COUNTED, None)
                conn.run("COMMIT")
                return UsageDecision(
                    True, "counted", plan, period, used + 1, limit, reserved=True
                )
            except Exception:
                try:
                    conn.run("ROLLBACK")
                except Exception:  # connection already gone
                    pass
                raise
    except Exception as error:
        if _sqlstate(error) == UNDEFINED_TABLE:
            logger.error("usage tables are unavailable")
        else:
            logger.warning("usage reservation failed for %s: %s", identifier, error)
        return UsageDecision(False, "usage_unavailable", UNAVAILABLE, period_key())


def release(
    database_url: str,
    workspace_id: Any,
    period: str,
    repo: str,
    issue_number: Any,
    reason: str = "analysis_failed",
) -> bool:
    """Give back a reserved slot, and record why.

    An analysis that errored is not something to bill for. The counted row
    is deleted and a `failed` audit row replaces it, so the attempt is still
    visible -- "why was I not analysed" and "why was I charged" have to be
    answerable from the same table, and deleting without trace answers
    neither.

    Failure here is logged and swallowed. This runs on an error path, and a
    second exception raised from it would replace the real one.
    """
    identifier = str(workspace_id or "").strip()
    try:
        number = int(issue_number)
    except (TypeError, ValueError):
        return False
    if not identifier or not database_url or not period:
        return False
    try:
        with session(database_url) as conn:
            conn.run(
                "DELETE FROM ghic_usage_events "
                "WHERE workspace_id = :workspace_id AND period = :period "
                "AND repo = :repo AND issue_number = :issue_number "
                "AND outcome = :outcome",
                workspace_id=identifier,
                period=period,
                repo=str(repo or ""),
                issue_number=number,
                outcome=COUNTED,
            )
            _insert(
                conn, identifier, period, str(repo or ""), number, FAILED, reason
            )
        return True
    except Exception as error:
        logger.warning("could not release usage reservation: %s", error)
        return False


def record(
    database_url: str,
    workspace_id: Any,
    period: str,
    repo: str,
    issue_number: Any,
    outcome: str,
    reason: str | None = None,
) -> bool:
    """Write one audit row. Never raises -- accounting must not break analysis."""
    identifier = str(workspace_id or "").strip()
    try:
        number = int(issue_number)
    except (TypeError, ValueError):
        return False
    if not identifier or not database_url or outcome == COUNTED:
        # `counted` is written only by `reserve`, under its lock. Letting any
        # caller write one would put the billing row outside the one place
        # that checks the limit before writing it.
        return False
    try:
        with session(database_url) as conn:
            _insert(
                conn, identifier, period, str(repo or ""), number, outcome, reason
            )
        return True
    except Exception as error:
        logger.warning("could not record usage event: %s", error)
        return False


def has_capacity(
    database_url: str, workspace_id: Any, now: datetime | None = None
) -> bool:
    """A cheap, non-binding "is there room left?".

    Read-only, and deliberately not the enforcement point -- `reserve()` is,
    and it is the one that serializes. This exists for the two callers where
    spending money to find out would be silly: publishing to QStash an issue
    that the callback will refuse, and re-scoring an edit whose workspace is
    already out of quota.

    Unreadable means no. This remains only an optimization; `reserve()` is the
    serialized enforcement point.
    """
    summary = usage_summary(database_url, workspace_id, now)
    if not summary.get("available"):
        return False
    limit = summary.get("limit")
    if not summary.get("enforced") or limit is None:
        return True
    return int(summary.get("used") or 0) < int(limit)


def usage_summary(
    database_url: str, workspace_id: Any, now: datetime | None = None
) -> dict[str, Any]:
    """What a workspace has used this period, for display.

    Read-only and best-effort: this feeds a dashboard panel, and a panel that
    cannot render is not a reason to fail a request.
    """
    identifier = str(workspace_id or "").strip()
    empty = {
        "plan": UNAVAILABLE.plan,
        "period": period_key(),
        "used": 0,
        "limit": None,
        "remaining": None,
        "enforced": False,
        "available": False,
        "outcomes": {},
    }
    if not identifier or not database_url:
        return empty
    try:
        with session(database_url) as conn:
            plan = _read_plan(conn, identifier)
            period = period_key(plan.period, now)
            rows = conn.run(
                "SELECT outcome, count(*) FROM ghic_usage_events "
                "WHERE workspace_id = :workspace_id AND period = :period "
                "GROUP BY outcome",
                workspace_id=identifier,
                period=period,
            )
            outcomes = {str(row[0]): int(row[1]) for row in rows}
            used = outcomes.get(COUNTED, 0)
            limit = plan.max_issues_per_period
            return {
                "plan": plan.plan,
                "period": period,
                "used": used,
                "limit": limit,
                "remaining": None if limit is None else max(0, limit - used),
                "enforced": plan.enforced,
                "available": True,
                "outcomes": outcomes,
            }
    except Exception as error:
        if _sqlstate(error) != UNDEFINED_TABLE:
            logger.warning("usage summary failed for %s: %s", identifier, error)
        return empty
