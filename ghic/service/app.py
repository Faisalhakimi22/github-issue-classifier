"""FastAPI application: the GitHub App webhook endpoint.

Endpoints:
  GET  /healthz      liveness + model identity (for load balancers)
  POST /webhook      GitHub webhook receiver (HMAC-verified)
  POST /api/predict  direct scoring API (same auth caveat as /webhook)

Event handling: `issues`/opened is scored; `issues`/closed feeds the online
evaluation loop (the bot grades its own earlier prediction against the final
labels/state_reason). Everything else is acknowledged and ignored — GitHub
retries on non-2xx, so unknown events must still return 200. Issues opened by
bots are ignored (the model was trained with bot authors excluded).

Run locally:
  uvicorn --factory ghic.service.app:create_app --reload   # uses GHIC_* env vars
  python -m ghic.service.app                                # same, without reload
"""
from __future__ import annotations

import hashlib
import hmac
import os
from collections import deque
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from .. import utils
from ..config import get_config
from ..label import label_issue
from .github_app import GitHubAppClient
from .inference import IssuePredictor, format_comment
from .settings import ServiceSettings, load_settings
from .tracking import PredictionTracker

logger = utils.get_logger(__name__)

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Webhook signature verification
# ---------------------------------------------------------------------------
def verify_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    """Validate GitHub's X-Hub-Signature-256 header (constant-time compare)."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature_header[len("sha256="):], expected)


def _is_bot(login: str) -> bool:
    return login.endswith("[bot]") or login.endswith("-bot")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app(
    settings: ServiceSettings | None = None,
    predictor: IssuePredictor | None = None,
    gh_client: GitHubAppClient | None = None,
) -> FastAPI:
    """Build the app. `predictor` / `gh_client` are injectable for tests."""
    settings = settings or load_settings()
    if predictor is None:
        settings.validate()
        predictor = IssuePredictor(settings.model_path, settings.threshold)
    if gh_client is None and settings.can_call_github:
        gh_client = GitHubAppClient(
            settings.app_id,
            settings.private_key_pem,
            base_url=settings.api_base_url,
            timeout=settings.request_timeout,
        )

    app = FastAPI(title="GitHub Issue Triage Bot", version=__version__)
    app.state.settings = settings
    app.state.predictor = predictor
    app.state.gh = gh_client
    # Rolling in-memory observability: totals survive for the process
    # lifetime, `recent` keeps the last 500 scored issues for /stats.
    app.state.totals = {"scored": 0, "positive": 0, "proba_sum": 0.0}
    app.state.recent = deque(maxlen=500)
    # Online evaluation ledger: predictions at open time, graded at close time.
    app.state.tracker = PredictionTracker(ledger_path=settings.ledger_path)

    def _require_token(request: Request) -> None:
        s: ServiceSettings = app.state.settings
        if s.webhook_secret:
            token = request.headers.get("X-GHIC-Token", "")
            if not hmac.compare_digest(token, s.webhook_secret):
                raise HTTPException(status_code=401, detail="invalid token")
        elif not s.allow_unsigned:
            raise HTTPException(status_code=503, detail="webhook secret not configured")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "model": app.state.predictor.model_name,
            "threshold": app.state.settings.threshold,
            "repo_thresholds": app.state.settings.repo_thresholds,
            "dry_run": app.state.settings.dry_run,
        }

    @app.get("/stats")
    def stats(request: Request) -> dict[str, Any]:
        """What has this deploy actually predicted? Token-gated (repo names)."""
        _require_token(request)
        t = app.state.totals
        return {
            "scored": t["scored"],
            "predicted_actionable": t["positive"],
            "positive_rate": round(t["positive"] / t["scored"], 4) if t["scored"] else None,
            "mean_proba": round(t["proba_sum"] / t["scored"], 4) if t["scored"] else None,
            "dry_run": app.state.settings.dry_run,
            "online_evaluation": app.state.tracker.summary(),
            "recent": list(app.state.recent)[-20:],
        }

    @app.post("/webhook")
    async def webhook(request: Request) -> dict[str, Any]:
        body = await request.body()
        s: ServiceSettings = app.state.settings
        if s.webhook_secret:
            if not verify_signature(
                s.webhook_secret, body, request.headers.get("X-Hub-Signature-256")
            ):
                raise HTTPException(status_code=401, detail="invalid webhook signature")
        elif not s.allow_unsigned:
            raise HTTPException(status_code=503, detail="webhook secret not configured")

        event = request.headers.get("X-GitHub-Event", "")
        payload = await request.json()

        if event == "ping":
            return {"ok": True, "pong": payload.get("zen", "")}
        if event == "issues" and payload.get("action") == "opened":
            return _handle_issue_opened(app, payload)
        if event == "issues" and payload.get("action") == "closed":
            return _handle_issue_closed(app, payload)
        return {"ok": True, "ignored": f"{event}/{payload.get('action')}"}

    @app.post("/api/predict")
    async def api_predict(request: Request) -> dict[str, Any]:
        """Score an arbitrary issue body without GitHub side effects.

        Guarded by the same webhook secret (send it as X-GHIC-Token) so the
        model is not an open scoring oracle when deployed publicly.
        """
        _require_token(request)
        data = await request.json()
        if not data.get("title") and not data.get("body"):
            raise HTTPException(status_code=422, detail="title or body is required")
        pred = app.state.predictor.predict(
            repo_full_name=data.get("repo", "api/adhoc"),
            issue_number=int(data.get("number", 0)),
            title=data.get("title", ""),
            body=data.get("body", ""),
            created_at=data.get("created_at") or _utcnow_iso(),
        )
        return pred.as_dict()

    return app


def _utcnow_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# The issues.opened flow
# ---------------------------------------------------------------------------
def _handle_issue_opened(app: FastAPI, payload: dict[str, Any]) -> dict[str, Any]:
    s: ServiceSettings = app.state.settings
    predictor: IssuePredictor = app.state.predictor
    gh: GitHubAppClient | None = app.state.gh

    issue = payload.get("issue") or {}
    repo = (payload.get("repository") or {}).get("full_name", "")
    installation_id = (payload.get("installation") or {}).get("id")
    number = int(issue.get("number", 0))
    author = ((issue.get("user") or {}).get("login")) or ""

    if not repo or not number:
        raise HTTPException(status_code=422, detail="malformed issues payload")
    if _is_bot(author):
        return {"ok": True, "ignored": f"bot author {author}"}

    # Enrichment: author profile + latest release. Best-effort — any failure
    # degrades to NaN features, which the pipeline imputes.
    author_created_at = author_public_repos = author_followers = None
    latest_release = None
    if s.enrich and gh is not None and installation_id:
        user = gh.get_user(author, installation_id)
        if user:
            author_created_at = user.get("created_at")
            author_public_repos = user.get("public_repos")
            author_followers = user.get("followers")
        latest_release = gh.get_latest_release_date(repo, installation_id)

    pred = predictor.predict(
        repo_full_name=repo,
        issue_number=number,
        title=issue.get("title", ""),
        body=issue.get("body") or "",
        created_at=issue.get("created_at") or _utcnow_iso(),
        author_login=author,
        author_created_at=author_created_at,
        author_public_repos=author_public_repos,
        author_followers=author_followers,
        latest_release_iso=latest_release,
        threshold=s.threshold_for(repo),
        explain=s.post_comment and not s.dry_run,  # only pay for it if it ships
    )
    totals = app.state.totals
    totals["scored"] += 1
    totals["positive"] += pred.predicted_label
    totals["proba_sum"] += pred.proba
    app.state.tracker.record_prediction(repo, number, pred.proba, pred.predicted_label)
    app.state.recent.append({
        "repo": repo,
        "issue": number,
        "proba": round(pred.proba, 4),
        "predicted": pred.as_dict()["predicted_class"],
        "at": _utcnow_iso(),
    })
    logger.info(
        "scored %s#%d: P(bug)=%.3f -> %s%s",
        repo, number, pred.proba, pred.as_dict()["predicted_class"],
        " [dry-run]" if s.dry_run else "",
    )

    actions: list[str] = []
    if not s.dry_run and gh is not None and installation_id:
        if s.post_comment:
            gh.post_comment(repo, number, format_comment(pred), installation_id)
            actions.append("comment")
        if s.apply_label and pred.predicted_label == 1:
            gh.add_labels(repo, number, [s.label_name], installation_id)
            actions.append("label")

    return {"ok": True, "prediction": pred.as_dict(), "actions": actions,
            "dry_run": s.dry_run}


# ---------------------------------------------------------------------------
# The issues.closed flow: grade our earlier prediction against the outcome
# ---------------------------------------------------------------------------
def _handle_issue_closed(app: FastAPI, payload: dict[str, Any]) -> dict[str, Any]:
    issue = payload.get("issue") or {}
    repo = (payload.get("repository") or {}).get("full_name", "")
    number = int(issue.get("number", 0))
    if not repo or not number:
        raise HTTPException(status_code=422, detail="malformed issues payload")

    # Rebuild the training-time issue dict from the close payload and apply
    # the SAME labeling rules used to build the dataset. REST/webhook payloads
    # use lowercase state reasons; the rules expect the GraphQL uppercase form.
    state_reason = (issue.get("state_reason") or "")
    issue_dict = {
        "number": number,
        "author_login": ((issue.get("user") or {}).get("login")) or None,
        "locked": bool(issue.get("locked")),
        "state_reason": state_reason.upper() or None,
        "labels_at_close": [
            lab.get("name", "") for lab in (issue.get("labels") or [])
        ],
        "closed_by_merged_pr": False,   # timeline data is invisible to webhooks
        "cross_referenced_prs": [],
    }
    cfg = get_config(require_token=False)
    result = label_issue(issue_dict, cfg.labeling)
    if result.label is None:
        return {"ok": True, "ignored": f"outcome dropped by rule {result.rule}"}

    matched = app.state.tracker.record_outcome(repo, number, result.label)
    logger.info(
        "outcome %s#%d: truth=%d (rule %s)%s",
        repo, number, result.label, result.rule,
        "" if matched else " — no tracked prediction",
    )
    return {"ok": True, "outcome": result.label, "rule": result.rule,
            "matched_prediction": matched}


def main() -> int:
    """`python -m ghic.service.app` / `ghic-serve` — run a local server."""
    import uvicorn

    host = os.environ.get("GHIC_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", os.environ.get("GHIC_PORT", "8000")))
    uvicorn.run(create_app(), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
