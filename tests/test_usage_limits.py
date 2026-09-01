"""Plan usage limits: the accounting, the gate, and the ways past it.

The fake connection below models `ghic_usage_events` including its partial
unique index, because that index -- not application care -- is what makes an
issue billable at most once per period. A fake that accepted every insert
would let a double-billing bug pass every test in this file.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from ghic.service import usage


class FakeUsageDatabase:
    """Enough Postgres to exercise usage.py: two tables and one index."""

    def __init__(self, plans=None, workspaces=None):
        self.plans = plans if plans is not None else {
            "starter": (1, 500, "month"),
            "pro": (10, 10000, "month"),
            "enterprise": (None, None, "month"),
        }
        self.workspaces = workspaces if workspaces is not None else {"ws-1": "starter"}
        self.events: list[dict] = []
        self.statements: list[str] = []
        self.fail_with: Exception | None = None
        self.committed = 0
        self.rolled_back = 0

    # -- the connection interface pg8000.native exposes ---------------------
    def run(self, sql, **params):
        self.statements.append(sql)
        if self.fail_with is not None:
            raise self.fail_with
        text = " ".join(sql.split())
        if text == "BEGIN":
            return []
        if text == "COMMIT":
            self.committed += 1
            return []
        if text == "ROLLBACK":
            self.rolled_back += 1
            return []
        if "pg_advisory_xact_lock" in text:
            return [[None]]
        if text.startswith("SELECT p.plan"):
            plan = self.workspaces.get(params["workspace_id"])
            if plan is None or plan not in self.plans:
                return []
            repos, issues, period = self.plans[plan]
            return [[plan, repos, issues, period]]
        if text.startswith("SELECT 1 FROM ghic_usage_events"):
            return [[1]] if self._find(params) else []
        if text.startswith("SELECT count(*) FROM ghic_usage_events"):
            return [[sum(1 for e in self._scope(params) if e["outcome"] == params["outcome"])]]
        if text.startswith("SELECT outcome, count(*)"):
            counts: dict[str, int] = {}
            for event in self._scope(params):
                counts[event["outcome"]] = counts.get(event["outcome"], 0) + 1
            return [[outcome, n] for outcome, n in sorted(counts.items())]
        if text.startswith("INSERT INTO ghic_usage_events"):
            # ghic_usage_counted_once_idx: one counted row per issue per period.
            if params["outcome"] == usage.COUNTED and self._find(params):
                return []
            self.events.append(dict(params))
            return []
        if text.startswith("DELETE FROM ghic_usage_events"):
            before = len(self.events)
            self.events = [e for e in self.events if not self._matches(e, params)]
            return [[before - len(self.events)]]
        raise AssertionError(f"unexpected SQL: {text}")

    def close(self):
        pass

    # -- helpers ------------------------------------------------------------
    def _scope(self, params):
        return [
            e for e in self.events
            if e["workspace_id"] == params["workspace_id"]
            and e["period"] == params["period"]
        ]

    @staticmethod
    def _matches(event, params):
        return (
            event["workspace_id"] == params["workspace_id"]
            and event["period"] == params["period"]
            and event["repo"] == params["repo"]
            and event["issue_number"] == params["issue_number"]
            and event["outcome"] == params["outcome"]
        )

    def _find(self, params):
        target = dict(params)
        target["outcome"] = usage.COUNTED
        return [e for e in self.events if self._matches(e, target)]

    def counted(self):
        return [e for e in self.events if e["outcome"] == usage.COUNTED]


@pytest.fixture
def db(monkeypatch):
    fake = FakeUsageDatabase()

    @contextmanager
    def fake_session(_url):
        yield fake

    monkeypatch.setattr(usage, "session", fake_session)
    return fake


def undefined_table():
    return Exception({"S": "ERROR", "C": "42P01", "M": "relation does not exist"})


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------
def test_period_is_the_utc_calendar_month():
    moment = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
    assert usage.period_key("month", moment) == "2026-08"


def test_a_local_late_evening_still_belongs_to_the_utc_month():
    # 2026-08-31 23:00 in UTC+5 is 2026-08-31 18:00 UTC -- still August. The
    # point is that the boundary is decided by one clock, not the customer's.
    from datetime import timedelta

    local = datetime(2026, 9, 1, 3, 0, tzinfo=timezone(timedelta(hours=5)))
    assert usage.period_key("month", local) == "2026-08"


def test_a_naive_datetime_is_read_as_utc_rather_than_guessed():
    assert usage.period_key("month", datetime(2026, 8, 27, 12, 0)) == "2026-08"


# ---------------------------------------------------------------------------
# Reading the plan
# ---------------------------------------------------------------------------
def test_a_plan_is_read_as_written(db):
    plan = usage.plan_for_workspace("postgres://test", "ws-1")
    assert (plan.plan, plan.max_issues_per_period, plan.enforced) == ("starter", 500, True)


def test_unlimited_is_none_and_not_zero(db):
    db.workspaces["ws-1"] = "enterprise"
    plan = usage.plan_for_workspace("postgres://test", "ws-1")
    # int(None) raises and Number(null) is 0. Either coercion turns the most
    # expensive plan into the most restrictive one, so None has to survive.
    assert plan.max_issues_per_period is None
    assert plan.unlimited_issues is True
    assert plan.enforced is True


def test_an_unknown_workspace_is_unmetered(db):
    plan = usage.plan_for_workspace("postgres://test", "ws-missing")
    assert plan.enforced is False


def test_absent_plan_tables_fail_open(db):
    db.fail_with = undefined_table()
    plan = usage.plan_for_workspace("postgres://test", "ws-1")
    assert plan.enforced is False
    assert plan.max_issues_per_period is None


def test_a_blank_workspace_is_not_looked_up(monkeypatch):
    monkeypatch.setattr(usage, "session", pytest.fail)
    assert usage.plan_for_workspace("postgres://test", "").enforced is False
    assert usage.plan_for_workspace("", "ws-1").enforced is False


# ---------------------------------------------------------------------------
# Reserving
# ---------------------------------------------------------------------------
def test_a_reservation_counts_once(db):
    decision = usage.reserve("postgres://test", "ws-1", "acme/widgets", 7)
    assert decision.allowed is True
    assert decision.reserved is True
    assert decision.used == 1
    assert len(db.counted()) == 1


def test_the_same_issue_twice_is_charged_once(db):
    first = usage.reserve("postgres://test", "ws-1", "acme/widgets", 7)
    second = usage.reserve("postgres://test", "ws-1", "acme/widgets", 7)
    assert first.reserved is True
    # A redelivered webhook or a QStash retry runs the work again but must
    # not bill again -- and must not own the reservation it did not take.
    assert second.allowed is True
    assert second.reserved is False
    assert second.reason == "already_counted"
    assert len(db.counted()) == 1


def test_the_reservation_is_serialized_per_workspace_and_period(db):
    usage.reserve("postgres://test", "ws-1", "acme/widgets", 7)
    locks = [s for s in db.statements if "pg_advisory_xact_lock" in s]
    assert len(locks) == 1


def test_the_limit_refuses_the_next_issue(db):
    db.plans["starter"] = (1, 2, "month")
    assert usage.reserve("postgres://test", "ws-1", "a/b", 1).allowed is True
    assert usage.reserve("postgres://test", "ws-1", "a/b", 2).allowed is True
    third = usage.reserve("postgres://test", "ws-1", "a/b", 3)
    assert third.allowed is False
    assert third.reason == "issue_quota_exhausted"
    assert third.used == 2 and third.limit == 2
    assert len(db.counted()) == 2


def test_only_the_first_refusal_of_a_period_is_announced(db):
    db.plans["starter"] = (1, 1, "month")
    usage.reserve("postgres://test", "ws-1", "a/b", 1)
    assert usage.reserve("postgres://test", "ws-1", "a/b", 2).first_refusal is True
    assert usage.reserve("postgres://test", "ws-1", "a/b", 3).first_refusal is False


def test_a_refusal_is_recorded_as_limited_not_counted(db):
    db.plans["starter"] = (1, 0, "month")
    usage.reserve("postgres://test", "ws-1", "a/b", 1)
    assert [e["outcome"] for e in db.events] == [usage.LIMITED]
    assert db.counted() == []


def test_a_zero_limit_blocks_everything(db):
    # Zero is a real limit, distinct from NULL. A plan priced at nothing
    # analyses nothing.
    db.plans["starter"] = (1, 0, "month")
    assert usage.reserve("postgres://test", "ws-1", "a/b", 1).allowed is False


def test_an_unlimited_plan_never_writes_a_counted_row(db):
    db.workspaces["ws-1"] = "enterprise"
    decision = usage.reserve("postgres://test", "ws-1", "a/b", 1)
    assert decision.allowed is True
    assert decision.reserved is False
    assert db.events == []


def test_reservation_falls_open_when_the_tables_are_absent(db):
    db.fail_with = undefined_table()
    decision = usage.reserve("postgres://test", "ws-1", "a/b", 1)
    assert decision.allowed is True
    assert decision.reserved is False
    assert decision.plan.enforced is False


def test_a_failed_transaction_is_rolled_back(db):
    class Exploding(FakeUsageDatabase):
        def run(self, sql, **params):
            if sql.lstrip().startswith("INSERT"):
                raise RuntimeError("connection lost")
            return super().run(sql, **params)

    exploding = Exploding()

    @contextmanager
    def fake_session(_url):
        yield exploding

    import ghic.service.usage as module

    original = module.session
    module.session = fake_session
    try:
        decision = usage.reserve("postgres://test", "ws-1", "a/b", 1)
    finally:
        module.session = original
    assert decision.allowed is True  # fails open
    assert exploding.rolled_back == 1
    assert exploding.committed == 0


# ---------------------------------------------------------------------------
# Releasing
# ---------------------------------------------------------------------------
def test_releasing_gives_the_slot_back_and_leaves_a_trace(db):
    usage.reserve("postgres://test", "ws-1", "acme/widgets", 7)
    # The period comes from the same clock reserve() used. A literal here
    # passes for a month and then deletes nothing on the 1st.
    assert usage.release(
        "postgres://test", "ws-1", usage.period_key(), "acme/widgets", 7,
        reason="TimeoutError",
    ) is True
    assert db.counted() == []
    # Deleting without trace would answer neither "why was I charged" nor
    # "why was I not analysed".
    failures = [e for e in db.events if e["outcome"] == usage.FAILED]
    assert len(failures) == 1
    assert failures[0]["reason"] == "TimeoutError"


def test_a_released_slot_can_be_taken_again(db):
    db.plans["starter"] = (1, 1, "month")
    usage.reserve("postgres://test", "ws-1", "a/b", 1)
    period = usage.period_key()
    usage.release("postgres://test", "ws-1", period, "a/b", 1)
    assert usage.reserve("postgres://test", "ws-1", "a/b", 2).allowed is True


def test_release_never_raises(db):
    db.fail_with = RuntimeError("gone")
    assert usage.release("postgres://test", "ws-1", "2026-08", "a/b", 1) is False


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def test_record_refuses_to_write_a_counted_row(db):
    # `counted` is written only by reserve(), under its lock. Any other
    # writer would put the billing row outside the one place that checks
    # the limit before writing it.
    assert usage.record(
        "postgres://test", "ws-1", "2026-08", "a/b", 1, usage.COUNTED
    ) is False
    assert db.events == []


def test_record_writes_an_audit_row(db):
    assert usage.record(
        "postgres://test", "ws-1", "2026-08", "a/b", 1, usage.SKIPPED, "bot_author"
    ) is True
    assert db.events[0]["outcome"] == usage.SKIPPED


# ---------------------------------------------------------------------------
# Summarizing
# ---------------------------------------------------------------------------
def test_the_summary_separates_the_four_outcomes(db):
    usage.reserve("postgres://test", "ws-1", "a/b", 1)
    period = usage.period_key()
    usage.record("postgres://test", "ws-1", period, "a/b", 2, usage.FAILED, "boom")
    usage.record("postgres://test", "ws-1", period, "a/b", 3, usage.SKIPPED, "bot")
    summary = usage.usage_summary("postgres://test", "ws-1")
    assert summary["used"] == 1
    assert summary["limit"] == 500
    assert summary["remaining"] == 499
    assert summary["outcomes"] == {"counted": 1, "failed": 1, "skipped": 1}


def test_remaining_never_goes_negative(db):
    db.plans["starter"] = (1, 1, "month")
    usage.reserve("postgres://test", "ws-1", "a/b", 1)
    db.plans["starter"] = (1, 0, "month")
    assert usage.usage_summary("postgres://test", "ws-1")["remaining"] == 0


def test_an_unlimited_plan_has_no_remaining(db):
    db.workspaces["ws-1"] = "enterprise"
    summary = usage.usage_summary("postgres://test", "ws-1")
    assert summary["limit"] is None
    assert summary["remaining"] is None


def test_has_capacity_is_permissive_when_unreadable(db):
    db.fail_with = undefined_table()
    assert usage.has_capacity("postgres://test", "ws-1") is True
