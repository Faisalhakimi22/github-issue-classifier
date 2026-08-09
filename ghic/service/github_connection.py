"""Keep the dashboard's GitHub connection tables honest.

The hub writes `ghic_github_installations` and `ghic_github_repositories`
when a signed-in user completes an installation. Those rows then go stale
the moment someone changes anything on GitHub -- removes a repository from
the installation, uninstalls the App entirely -- and the dashboard would
keep showing repositories GHIC can no longer read.

GitHub already tells us: it delivers `installation` and
`installation_repositories` events to the same webhook that receives issue
events. This is the handler for them.

Two deliberate limits:

*Nothing here creates an installation row.* `connected_by_firebase_uid` is
NOT NULL because a GitHub installation only becomes a GHIC workspace
connection when an authenticated user claims it, having proved they own or
administer the account. A webhook payload proves neither, so an
installation GHIC has never seen is recorded as nothing at all -- it stays
invisible until someone signs in and connects it. Additions are only
applied to installations a user already claimed.

*Nothing here raises.* A failure to reconcile must not turn into a non-2xx
webhook response, because GitHub would retry the delivery and the retry
would be rejected by the idempotency check anyway. Failures are logged and
the event is acknowledged.
"""
from __future__ import annotations

import logging
from typing import Any

from .._pg import session

logger = logging.getLogger("ghic.service.github_connection")

# The hub's schema is the owner of these tables. This module only ever
# writes to rows it can prove already exist, so it never needs the DDL.
_TABLE_CHECK = (
    "SELECT to_regclass('public.ghic_github_installations') IS NOT NULL "
    "AND to_regclass('public.ghic_github_repositories') IS NOT NULL AS ok"
)


def _tables_exist(conn: Any) -> bool:
    rows = conn.run(_TABLE_CHECK)
    return bool(rows and rows[0][0])


def _installation_is_connected(conn: Any, installation_id: int) -> bool:
    rows = conn.run(
        "SELECT 1 FROM ghic_github_installations "
        "WHERE installation_id = :iid AND revoked_at IS NULL",
        iid=installation_id,
    )
    return bool(rows)


def _mark_revoked(conn: Any, installation_id: int) -> None:
    conn.run(
        "UPDATE ghic_github_installations "
        "SET revoked_at = now(), updated_at = now() "
        "WHERE installation_id = :iid AND revoked_at IS NULL",
        iid=installation_id,
    )
    conn.run(
        "UPDATE ghic_github_repositories "
        "SET active = false, removed_at = now(), updated_at = now() "
        "WHERE installation_id = :iid AND active = true",
        iid=installation_id,
    )


def _restore(conn: Any, installation_id: int) -> None:
    """Un-suspend: the connection was never withdrawn by the user."""
    conn.run(
        "UPDATE ghic_github_installations "
        "SET revoked_at = NULL, updated_at = now() "
        "WHERE installation_id = :iid",
        iid=installation_id,
    )


def _deactivate_repos(conn: Any, installation_id: int, full_names: list[str]) -> None:
    for name in full_names:
        conn.run(
            "UPDATE ghic_github_repositories "
            "SET active = false, removed_at = now(), updated_at = now() "
            "WHERE installation_id = :iid AND repo_full_name = :repo",
            iid=installation_id,
            repo=name,
        )


def _activate_repos(conn: Any, installation_id: int, repos: list[dict]) -> None:
    for repo in repos:
        full_name = str(repo.get("full_name") or "")
        if not full_name:
            continue
        repo_id = repo.get("id")
        conn.run(
            "INSERT INTO ghic_github_repositories ("
            "  repo_full_name, installation_id, github_repo_id, private,"
            "  active, updated_at, removed_at"
            ") VALUES (:repo, :iid, :gid, :priv, true, now(), NULL) "
            "ON CONFLICT (repo_full_name) DO UPDATE SET "
            "  installation_id = EXCLUDED.installation_id, "
            "  github_repo_id = COALESCE(EXCLUDED.github_repo_id,"
            "                            ghic_github_repositories.github_repo_id), "
            "  private = EXCLUDED.private, "
            "  active = true, updated_at = now(), removed_at = NULL",
            repo=full_name,
            iid=installation_id,
            gid=int(repo_id) if repo_id is not None else None,
            priv=bool(repo.get("private")),
        )


def _repo_names(payload: dict, key: str) -> list[str]:
    entries = payload.get(key) or []
    names = []
    for entry in entries:
        name = str((entry or {}).get("full_name") or "")
        if name:
            names.append(name)
    return names


def handle_installation_event(
    database_url: str, event: str, payload: dict
) -> dict[str, Any]:
    """Reconcile one installation lifecycle delivery. Never raises."""
    if not database_url:
        return {"ok": True, "connection_sync": "no_database"}

    installation = payload.get("installation") or {}
    installation_id = installation.get("id")
    if installation_id is None:
        return {"ok": True, "connection_sync": "no_installation_id"}

    action = str(payload.get("action") or "")

    try:
        installation_id = int(installation_id)
    except (TypeError, ValueError):
        return {"ok": True, "connection_sync": "bad_installation_id"}

    try:
        with session(database_url) as conn:
            if not _tables_exist(conn):
                return {"ok": True, "connection_sync": "tables_absent"}

            # `deleted` and `suspend` both mean GHIC has lost access now.
            # They are distinct on GitHub, but neither leaves GHIC able to
            # read the repositories, and showing them as connected would be
            # wrong in exactly the same way.
            if event == "installation" and action in ("deleted", "suspend"):
                _mark_revoked(conn, installation_id)
                return {
                    "ok": True,
                    "connection_sync": f"revoked:{action}",
                    "installation_id": installation_id,
                }

            if event == "installation" and action == "unsuspend":
                if not _installation_is_connected(conn, installation_id):
                    _restore(conn, installation_id)
                _activate_repos(
                    conn, installation_id, list(payload.get("repositories") or [])
                )
                return {
                    "ok": True,
                    "connection_sync": "unsuspended",
                    "installation_id": installation_id,
                }

            # Everything below only makes sense for a connection a user
            # already claimed. An unclaimed installation is not an error --
            # it is simply not this workspace's yet.
            if not _installation_is_connected(conn, installation_id):
                return {"ok": True, "connection_sync": "not_connected"}

            if event == "installation_repositories":
                removed = _repo_names(payload, "repositories_removed")
                added = list(payload.get("repositories_added") or [])
                if removed:
                    _deactivate_repos(conn, installation_id, removed)
                if added:
                    _activate_repos(conn, installation_id, added)
                return {
                    "ok": True,
                    "connection_sync": "repositories",
                    "added": len(added),
                    "removed": len(removed),
                }

            if event == "repository" and action in ("deleted", "archived"):
                repo = payload.get("repository") or {}
                name = str(repo.get("full_name") or "")
                if name:
                    _deactivate_repos(conn, installation_id, [name])
                return {"ok": True, "connection_sync": f"repository:{action}"}

        return {"ok": True, "connection_sync": f"ignored:{event}/{action}"}
    except Exception as exc:  # never fail a webhook over bookkeeping
        logger.warning(
            "installation sync failed for %s (%s/%s): %s",
            installation_id,
            event,
            action,
            exc,
        )
        return {"ok": True, "connection_sync": "error"}
