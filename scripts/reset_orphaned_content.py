"""One-off cleanup of orphaned test residue, so the install flow can be
exercised from a clean slate.

This deliberately does NOT use the new purge_installation path. That function
keys off ghic_github_installations, and production holds zero installation
rows -- the installations these records belonged to are already gone. What is
left is orphaned content: chunks and state for a repository that has no
repository row, and ledger rows with no workspace.

Those are exactly the rows the uninstall purge refuses to touch, by design:
it will not guess an owner for unattributable data. So clearing them is a
deliberate manual act, itemised here, and not something any webhook can do.

Nothing here deletes a user, a workspace, a membership, workspace settings,
org settings, schema migrations, or ghic_idempotency.

Usage: python reset_test_environment.py [--confirm] [--include-ledger]
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import closing
from urllib.parse import unquote, urlparse

import pg8000.native

TARGET_REPO = "Faisalhakimi22/github-issue-classifier"
TARGET_WORKSPACE = "ghic-default-workspace"

# Tables this script will never write to, asserted rather than assumed.
PROTECTED = (
    "ghic_users", "ghic_workspaces", "ghic_workspace_members",
    "ghic_workspace_settings", "ghic_org_settings", "ghic_schema_migrations",
    "ghic_idempotency", "ghic_github_installation_intents",
)


def connect(read_only: bool):
    p = urlparse(os.environ["DATABASE_URL_UNPOOLED"])
    kwargs = dict(
        user=unquote(p.username), password=unquote(p.password or ""),
        host=p.hostname, port=p.port or 5432,
        database=unquote(p.path.lstrip("/")), ssl_context=True,
        application_name="ghic-test-environment-reset",
    )
    if read_only:
        kwargs["startup_params"] = {
            "options": "-c default_transaction_read_only=on"}
    return pg8000.native.Connection(**kwargs)


def snapshot(conn) -> dict:
    protected = {}
    for table in PROTECTED:
        present = conn.run(
            "SELECT to_regclass(:q) IS NOT NULL", q="public." + table)[0][0]
        protected[table] = (
            conn.run(f'SELECT count(*) FROM "{table}"')[0][0] if present else None
        )
    return {
        "installations": conn.run(
            "SELECT count(*) FROM ghic_github_installations")[0][0],
        "repositories": conn.run(
            "SELECT count(*) FROM ghic_github_repositories")[0][0],
        "chunks_target": conn.run(
            "SELECT count(*) FROM ghic_repo_chunks "
            "WHERE workspace_id = :ws AND repo = :repo",
            ws=TARGET_WORKSPACE, repo=TARGET_REPO)[0][0],
        "chunks_other": conn.run(
            "SELECT count(*) FROM ghic_repo_chunks "
            "WHERE NOT (workspace_id = :ws AND repo = :repo)",
            ws=TARGET_WORKSPACE, repo=TARGET_REPO)[0][0],
        "state_target": conn.run(
            "SELECT count(*) FROM ghic_repository_state "
            "WHERE workspace_id = :ws AND repo = :repo",
            ws=TARGET_WORKSPACE, repo=TARGET_REPO)[0][0],
        "state_other": conn.run(
            "SELECT count(*) FROM ghic_repository_state "
            "WHERE NOT (workspace_id = :ws AND repo = :repo)",
            ws=TARGET_WORKSPACE, repo=TARGET_REPO)[0][0],
        "ledger_unattributed": conn.run(
            "SELECT count(*) FROM ghic_ledger WHERE workspace_id IS NULL")[0][0],
        "ledger_attributed": conn.run(
            "SELECT count(*) FROM ghic_ledger "
            "WHERE workspace_id IS NOT NULL")[0][0],
        "protected": protected,
    }


def main() -> int:
    confirm = "--confirm" in sys.argv
    include_ledger = "--include-ledger" in sys.argv

    with closing(connect(read_only=True)) as conn:
        before = snapshot(conn)

    print("BEFORE")
    print(json.dumps(before, indent=2))

    # Refuse to run if anything is connected: the orphan assumption would no
    # longer hold and the uninstall path should be used instead.
    if before["installations"] or before["repositories"]:
        print("\nREFUSING: installations/repositories exist. Use the uninstall "
              "purge path instead of this orphan cleanup.")
        return 2

    plan = [
        ("ghic_repo_chunks", before["chunks_target"],
         f"workspace_id = {TARGET_WORKSPACE!r} AND repo = {TARGET_REPO!r}"),
        ("ghic_repository_state", before["state_target"],
         f"workspace_id = {TARGET_WORKSPACE!r} AND repo = {TARGET_REPO!r}"),
    ]
    if include_ledger:
        plan.append(("ghic_ledger", before["ledger_unattributed"],
                     "workspace_id IS NULL"))

    print("\nPLAN")
    for table, count, predicate in plan:
        print(f"  DELETE {count:>5} FROM {table}  WHERE {predicate}")
    print("\nRETAINED")
    for table, count in before["protected"].items():
        print(f"  {table}: {count if count is not None else 'absent'} (untouched)")
    if not include_ledger:
        print(f"  ghic_ledger: {before['ledger_unattributed']} unattributed rows "
              "(untouched; pass --include-ledger to clear)")

    if not confirm:
        print("\nDRY RUN -- nothing written. Re-run with --confirm to apply.")
        return 0

    deleted = {}
    conn = connect(read_only=False)
    try:
        conn.run("BEGIN")
        for table, _, _ in plan:
            if table == "ghic_ledger":
                rows = conn.run(
                    "DELETE FROM ghic_ledger WHERE workspace_id IS NULL "
                    "RETURNING 1")
            else:
                rows = conn.run(
                    f'DELETE FROM "{table}" '
                    "WHERE workspace_id = :ws AND repo = :repo RETURNING 1",
                    ws=TARGET_WORKSPACE, repo=TARGET_REPO)
            deleted[table] = len(rows)
        conn.run("COMMIT")
    except Exception:
        conn.run("ROLLBACK")
        raise
    finally:
        conn.close()

    with closing(connect(read_only=True)) as conn:
        after = snapshot(conn)

    print("\nDELETED")
    print(json.dumps(deleted, indent=2))
    print("\nAFTER")
    print(json.dumps(after, indent=2))

    ok = all(after["protected"][t] == before["protected"][t] for t in PROTECTED)
    print("\nprotected tables unchanged:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
