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
from contextlib import contextmanager
from typing import Any

from .._pg import session

logger = logging.getLogger("ghic.service.github_connection")


def authorize_issue(
    database_url: str,
    installation_id: Any,
    repo_full_name: str,
    github_client: Any = None,
) -> dict[str, Any]:
    """Authorize an issue against the current GHIC connection state.

    The webhook payload supplies only lookup candidates. PostgreSQL remains the
    source of truth for whether GHIC claimed the installation and repository.
    When the App client is available, its app-level installation lookup adds a
    current check for suspension and accidental cross-App installation ids.
    """
    try:
        installation_id = int(installation_id)
    except (TypeError, ValueError):
        return {"authorized": False, "reason": "unknown_installation"}
    repo = str(repo_full_name or "")
    if installation_id <= 0 or not repo:
        return {"authorized": False, "reason": "unknown_installation"}
    if not database_url:
        return {"authorized": False, "reason": "connection_state_unavailable"}

    try:
        with session(database_url) as conn:
            if not _tables_exist(conn):
                return {"authorized": False, "reason": "connection_state_unavailable"}
            installations = conn.run(
                "SELECT installation_id, workspace_id, revoked_at, connection_status "
                "FROM ghic_github_installations "
                "WHERE installation_id = :iid",
                iid=installation_id,
            )
            if not installations:
                return {"authorized": False, "reason": "unclaimed_installation"}
            if len(installations[0]) < 4:
                return {"authorized": False, "reason": "connection_state_unavailable"}
            workspace_id = installations[0][1]
            if not workspace_id:
                return {"authorized": False, "reason": "connection_state_unavailable"}

            def rejected(reason: str) -> dict[str, Any]:
                result = {"authorized": False, "reason": reason}
                result["workspace_id"] = workspace_id
                return result

            revoked_at = installations[0][2]
            connection_status = str(installations[0][3] or "")
            if revoked_at is not None or connection_status == "revoked":
                return rejected("revoked_installation")
            if connection_status == "suspended":
                return rejected("suspended_installation")
            if connection_status != "connected":
                return rejected("connection_state_unavailable")

            repositories = conn.run(
                "SELECT installation_id, workspace_id, active "
                "FROM ghic_github_repositories "
                "WHERE repo_full_name = :repo",
                repo=repo,
            )
            if not repositories:
                return rejected("repository_not_connected")
            if len(repositories[0]) < 3:
                return rejected("connection_state_unavailable")
            connected_installation, repository_workspace, active = repositories[0]
            if repository_workspace != workspace_id:
                return rejected("workspace_repository_mismatch")
            if int(connected_installation) != installation_id:
                return rejected("installation_repository_mismatch")
            if not active:
                return rejected("repository_removed")
    except Exception:
        logger.exception(
            "issue authorization lookup failed for installation %s and %s",
            installation_id,
            repo,
        )
        return {"authorized": False, "reason": "connection_state_unavailable"}

    # The dashboard creates these rows only after verifying the installation
    # with the same App credentials. Re-check the live installation when the
    # webhook service has its App client, so suspension and App mismatches are
    # rejected even if their lifecycle webhook has not reached us yet.
    verify = getattr(github_client, "verify_installation", None)
    if not callable(verify):
        return {"authorized": False, "reason": "github_installation_unavailable"}
    try:
        current = verify(installation_id)
    except Exception:
        logger.exception("current GitHub installation verification failed")
        return {"authorized": False, "reason": "github_installation_unavailable"}
    if current.get("status") == "suspended":
        return rejected("suspended_installation")
    if current.get("status") == "app_mismatch":
        return rejected("installation_app_mismatch")
    if current.get("status") != "active":
        return rejected("unknown_installation")

    result = {
        "authorized": True,
        "installation_id": installation_id,
        "repo": repo,
    }
    result["workspace_id"] = workspace_id
    return result


@contextmanager
def authorization_lease(
    database_url: str,
    installation_id: Any,
    repo_full_name: str,
):
    """Hold row locks while a GitHub mutation is performed.

    Lifecycle updates use the same PostgreSQL rows, so disconnect/revoke is
    blocked until the mutation finishes. This closes the authorization
    check-to-write gap without changing GitHub or vector storage.
    """
    from .._pg import connect

    conn = None
    begun = False
    try:
        installation_id = int(installation_id)
        repo = str(repo_full_name or "")
        if installation_id <= 0 or not database_url or not repo:
            yield {"authorized": False, "reason": "connection_state_unavailable"}
            return
        conn = connect(database_url)
        conn.run("BEGIN")
        begun = True
        rows = conn.run(
            "SELECT i.connection_status, i.revoked_at, i.workspace_id, r.active, r.installation_id, r.workspace_id "
            "FROM ghic_github_installations i "
            "JOIN ghic_github_repositories r "
            "  ON r.installation_id = i.installation_id "
            "WHERE i.installation_id = :iid AND r.repo_full_name = :repo "
            "FOR UPDATE OF i, r",
            iid=installation_id,
            repo=repo,
        )
        if not rows:
            decision = {"authorized": False, "reason": "repository_not_connected"}
        else:
            status, revoked_at, installation_workspace, active, connected_installation, repository_workspace = rows[0]
            if not installation_workspace or installation_workspace != repository_workspace:
                decision = {"authorized": False, "reason": "workspace_repository_mismatch"}
            elif revoked_at is not None or status == "revoked":
                decision = {"authorized": False, "reason": "revoked_installation"}
            elif status == "suspended":
                decision = {"authorized": False, "reason": "suspended_installation"}
            elif status != "connected":
                decision = {"authorized": False, "reason": "connection_state_unavailable"}
            elif int(connected_installation) != installation_id:
                decision = {
                    "authorized": False,
                    "reason": "installation_repository_mismatch",
                }
            elif not active:
                decision = {"authorized": False, "reason": "repository_removed"}
            else:
                decision = {
                    "authorized": True,
                    "installation_id": installation_id,
                    "repo": repo,
                    "workspace_id": installation_workspace,
                }
    except Exception:
        if conn is not None and begun:
            try:
                conn.run("ROLLBACK")
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        yield {"authorized": False, "reason": "connection_state_unavailable"}
        return

    try:
        yield decision
    except Exception:
        if conn is not None and begun:
            conn.run("ROLLBACK")
        raise
    else:
        if conn is not None and begun:
            conn.run("COMMIT")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

# The hub's schema is the owner of these tables. This module only ever
# writes to rows it can prove already exist, so it never needs the DDL.
_TABLE_CHECK = (
    "SELECT to_regclass('public.ghic_github_installations') IS NOT NULL "
    "AND to_regclass('public.ghic_github_repositories') IS NOT NULL AS ok"
)


def _tables_exist(conn: Any) -> bool:
    rows = conn.run(_TABLE_CHECK)
    return bool(rows and rows[0][0])


@contextmanager
def _transaction(conn: Any):
    conn.run("BEGIN")
    try:
        yield
    except Exception:
        conn.run("ROLLBACK")
        raise
    else:
        conn.run("COMMIT")


def _installation_is_connected(conn: Any, installation_id: int) -> bool:
    rows = conn.run(
        "SELECT 1 FROM ghic_github_installations "
        "WHERE installation_id = :iid AND revoked_at IS NULL "
        "AND connection_status = 'connected'",
        iid=installation_id,
    )
    return bool(rows)


def _mark_revoked(conn: Any, installation_id: int) -> None:
    with _transaction(conn):
        conn.run(
            "UPDATE ghic_github_installations "
            "SET revoked_at = now(), connection_status = 'revoked', "
            "status_reason = 'github_installation_revoked', "
            "status_changed_at = now(), updated_at = now() "
            "WHERE installation_id = :iid AND revoked_at IS NULL",
            iid=installation_id,
        )
        conn.run(
            "UPDATE ghic_github_repositories "
            "SET active = false, removed_at = now(), updated_at = now() "
            "WHERE installation_id = :iid AND active = true",
            iid=installation_id,
        )


def _mark_suspended(conn: Any, installation_id: int) -> None:
    with _transaction(conn):
        conn.run(
            "UPDATE ghic_github_installations "
            "SET connection_status = 'suspended', "
            "status_reason = 'github_installation_suspended', "
            "status_changed_at = now(), updated_at = now() "
            "WHERE installation_id = :iid "
            "AND connection_status = 'connected' AND revoked_at IS NULL",
            iid=installation_id,
        )
        conn.run(
            "UPDATE ghic_github_repositories "
            "SET active = false, removed_at = now(), updated_at = now() "
            "WHERE installation_id = :iid AND active = true "
            "AND EXISTS (SELECT 1 FROM ghic_github_installations "
            "            WHERE installation_id = :iid "
            "              AND connection_status = 'suspended' "
            "              AND revoked_at IS NULL)",
            iid=installation_id,
        )


def _restore(conn: Any, installation_id: int) -> None:
    """Un-suspend: the connection was never withdrawn by the user."""
    conn.run(
        "UPDATE ghic_github_installations "
        "SET revoked_at = NULL, connection_status = 'connected', "
        "status_reason = NULL, status_changed_at = now(), updated_at = now() "
        "WHERE installation_id = :iid AND connection_status = 'suspended'",
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


def _deactivate_all_repos(conn: Any, installation_id: int) -> None:
    conn.run(
        "UPDATE ghic_github_repositories "
        "SET active = false, removed_at = now(), updated_at = now() "
        "WHERE installation_id = :iid AND active = true",
        iid=installation_id,
    )


def _activate_repos(conn: Any, installation_id: int, repos: list[dict]) -> None:
    for repo in repos:
        full_name = str(repo.get("full_name") or "")
        if not full_name:
            continue
        repo_id = repo.get("id")
        conn.run(
            "INSERT INTO ghic_github_repositories ("
            "  repo_full_name, installation_id, workspace_id, github_repo_id, private,"
            "  active, updated_at, removed_at"
            ") VALUES (:repo, :iid, "
            "  (SELECT workspace_id FROM ghic_github_installations WHERE installation_id = :iid),"
            "  :gid, :priv, true, now(), NULL) "
            "ON CONFLICT (repo_full_name) DO UPDATE SET "
            "  workspace_id = EXCLUDED.workspace_id, "
            "  github_repo_id = COALESCE(EXCLUDED.github_repo_id,"
            "                            ghic_github_repositories.github_repo_id), "
            "  private = EXCLUDED.private, "
            "  active = true, updated_at = now(), removed_at = NULL "
            "WHERE ghic_github_repositories.installation_id = EXCLUDED.installation_id",
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
    database_url: str, event: str, payload: dict, github_client: Any = None
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
            if event == "installation" and action == "deleted":
                _mark_revoked(conn, installation_id)
                return {
                    "ok": True,
                    "connection_sync": "revoked:deleted",
                    "installation_id": installation_id,
                }

            if event == "installation" and action == "suspend":
                _mark_suspended(conn, installation_id)
                return {
                    "ok": True,
                    "connection_sync": "suspended",
                    "installation_id": installation_id,
                }

            if event == "installation" and action == "unsuspend":
                suspended = conn.run(
                    "SELECT 1 FROM ghic_github_installations "
                    "WHERE installation_id = :iid AND revoked_at IS NULL "
                    "AND connection_status = 'suspended'",
                    iid=installation_id,
                )
                authoritative_repos = None
                if suspended and github_client is not None:
                    list_repos = getattr(
                        github_client, "list_installation_repositories", None
                    )
                    if callable(list_repos):
                        try:
                            authoritative_repos = list_repos(installation_id)
                        except Exception:
                            logger.exception(
                                "could not reconcile repositories after unsuspend for %s",
                                installation_id,
                            )
                if suspended and authoritative_repos is not None:
                    with _transaction(conn):
                        _restore(conn, installation_id)
                        if _installation_is_connected(conn, installation_id):
                            _deactivate_all_repos(conn, installation_id)
                            _activate_repos(conn, installation_id, authoritative_repos)
                return {
                    "ok": True,
                    "connection_sync": "unsuspended" if suspended else "ignored_unsuspend",
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
                if added:
                    authoritative_repos = None
                    list_repos = getattr(
                        github_client, "list_installation_repositories", None
                    ) if github_client is not None else None
                    if callable(list_repos):
                        try:
                            authoritative_repos = list_repos(installation_id)
                        except Exception:
                            logger.exception(
                                "could not reconcile added repositories for %s",
                                installation_id,
                            )
                    if authoritative_repos is None:
                        if removed:
                            with _transaction(conn):
                                _deactivate_repos(conn, installation_id, removed)
                        return {
                            "ok": True,
                            "connection_sync": "reconciliation_unavailable",
                            "added": 0,
                            "removed": len(removed),
                        }
                    with _transaction(conn):
                        _deactivate_all_repos(conn, installation_id)
                        _activate_repos(conn, installation_id, authoritative_repos)
                elif removed:
                    with _transaction(conn):
                        _deactivate_repos(conn, installation_id, removed)
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
                    with _transaction(conn):
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
