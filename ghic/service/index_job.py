"""One repository-index job, shared by every place that runs one.

Two callers: the QStash callback (`POST /internal/index-repository`) and
the GitHub Actions worker (`python -m ghic.index_worker`). They must make
the same authorization decision, so there is one implementation, and
neither caller trusts the payload it was handed: the installation and
repository are re-checked against PostgreSQL.

Locks are held only for an instant. An earlier version kept the
installation and repository rows locked (`FOR UPDATE`) for the whole clone
and index, which in production meant minutes: every Hub request queued
behind it (its cold-start migrations need those tables), and the idle
locking transaction was eventually dropped by the database, failing a job
whose indexing had actually succeeded. Instead:

  1. authorize, without locks;
  2. under a brief lease, re-check and mint the installation token, then
     release it;
  3. clone and index with no locks held;
  4. authorize again. If access was revoked or the repository removed while
     the index was being built, delete what was just written.

That keeps the guarantee the long lease existed for -- nothing indexed for
a revoked installation survives -- without blocking the product. An uninstall
that lands during step 3 runs its own purge; step 4 removes anything this job
wrote after that purge.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .. import utils

logger = utils.get_logger(__name__)


def run_index_job(
    payload: dict[str, Any],
    *,
    database_url: str,
    repo_intelligence: Any,
    gh: Any,
    authorization_gate: Callable[..., dict[str, Any]],
    authorization_lease: Callable[..., Any],
) -> dict[str, Any]:
    """Authorize and run one index job. Returns a result dict; raises only
    for an unexpected failure.

    A failed index returns indexed=False rather than raising: retrying a
    clone that failed because the App lacks contents:read on a private repo
    would fail identically every time.
    """
    repo = str(payload.get("repo") or "")
    if not repo:
        raise ValueError("missing repo")

    from ..repository_intelligence.metrics import set_correlation_id

    set_correlation_id(str(payload.get("correlation_id") or ""))

    installation_id = payload.get("installation_id")
    if installation_id in (None, ""):
        return _refused(repo, "unknown_installation")
    decision = authorization_gate(database_url, installation_id, repo, gh)
    if not decision.get("authorized"):
        return _refused(repo, str(decision.get("reason") or "authorization_rejected"))

    # Brief: re-check under lock and mint the token, then let go.
    token = ""
    with authorization_lease(database_url, installation_id, repo) as lease:
        if not lease.get("authorized"):
            return _refused(repo, str(lease.get("reason") or "authorization_rejected"))
        workspace_id = lease.get("workspace_id")
        if gh is not None:
            try:
                token = gh.installation_token(int(installation_id))
            except Exception as e:
                # A public repo still clones without a token, so this is a
                # degradation (private repos fail later), not a hard stop.
                logger.warning("could not mint installation token for %s: %s", repo, e)

    indexed = repo_intelligence.index_repository(
        repo, token=token, force=bool(payload.get("force")),
        workspace_id=workspace_id,
    )

    # Access may have been revoked while the index was built. Whatever this
    # job wrote must not outlive that.
    after = authorization_gate(database_url, installation_id, repo, gh)
    if not after.get("authorized"):
        reason = str(after.get("reason") or "authorization_rejected")
        logger.warning(
            "access to %s ended during indexing (%s); removing the new index", repo, reason,
        )
        try:
            repo_intelligence.forget(repo, workspace_id=workspace_id)
        except Exception as e:
            # Surface it: an index that should have been deleted and was not
            # is exactly the failure the uninstall purge exists to prevent.
            logger.error("could not remove the index for %s after revocation: %s", repo, e)
            raise
        return _refused(repo, f"revoked_during_index:{reason}")

    return {"ok": True, "repo": repo, "indexed": bool(indexed)}


def _refused(repo: str, reason: str) -> dict[str, Any]:
    return {
        "ok": True,
        "repo": repo,
        "indexed": False,
        "skipped": True,
        "authorization": "rejected",
        "reason": reason,
    }
