"""One repository-index job, shared by every place that runs one.

Two callers: the QStash callback (`POST /internal/index-repository`) and
the GitHub Actions worker (`python -m ghic.index_worker`). They must make
the same authorization decision, so there is one implementation, and
neither caller trusts the payload it was handed: the installation and
repository are re-checked against PostgreSQL, and the rows stay locked for
the whole token/clone/index operation so a disconnect or revoke cannot race
the job between the check and the read of repository contents.
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
    """Authorize and run one index job. Never raises for an expected
    refusal; returns a result dict either way.

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

    with authorization_lease(database_url, installation_id, repo) as lease:
        if not lease.get("authorized"):
            return _refused(repo, str(lease.get("reason") or "authorization_rejected"))
        token = ""
        if gh is not None:
            try:
                token = gh.installation_token(int(installation_id))
            except Exception as e:
                # A public repo still clones without a token, so this is a
                # degradation (private repos fail later), not a hard stop.
                logger.warning("could not mint installation token for %s: %s", repo, e)

        indexed = repo_intelligence.index_repository(
            repo, token=token, force=bool(payload.get("force")),
            workspace_id=lease.get("workspace_id"),
        )
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
