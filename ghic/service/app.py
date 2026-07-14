"""FastAPI application: the GitHub App webhook endpoint.

Endpoints:
  GET  /healthz      liveness + model identity (for load balancers)
  POST /webhook      GitHub webhook receiver (HMAC-verified)
  POST /api/predict  direct scoring API (same auth caveat as /webhook)

Event handling: `issues`/opened is scored (+ optional actions); `edited`
re-scores with the improved text (never posts anything); `closed` feeds the
online evaluation loop (the bot grades its own earlier prediction against
the final labels/state_reason); `labeled`/`unlabeled` are recorded to the
ledger as future ground truth. Everything else is acknowledged and ignored —
GitHub retries on non-2xx, so unknown events must still return 200. Issues
opened by bots are ignored (the model was trained with bot authors excluded).

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

__version__ = "1.1.0"


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
    dup_index: Any = None,
    category_predictor: Any = None,
    effort_predictor: Any = None,
    assignment_recommender: Any = None,
) -> FastAPI:
    """Build the app. `predictor` / `gh_client` / `dup_index` /
    `category_predictor` / `effort_predictor` / `assignment_recommender`
    are injectable for tests."""
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
    # Per-endpoint latency samples (ms) for /stats percentiles, plus a
    # structured one-line log per request: method, path, status, duration.
    app.state.latencies = {}
    app.state.errors = {"count": 0}

    @app.middleware("http")
    async def observe(request: Request, call_next: Any) -> Any:
        import time as _time

        started = _time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            app.state.errors["count"] += 1
            raise
        elapsed_ms = (_time.perf_counter() - started) * 1000
        path = request.url.path
        app.state.latencies.setdefault(path, deque(maxlen=1000)).append(elapsed_ms)
        if response.status_code >= 500:
            app.state.errors["count"] += 1
        logger.info(
            '{"method": "%s", "path": "%s", "status": %d, "ms": %.1f}',
            request.method, path, response.status_code, elapsed_ms,
        )
        return response
    # Rolling in-memory observability: totals survive for the process
    # lifetime, `recent` keeps the last 500 scored issues for /stats.
    app.state.totals = {"scored": 0, "positive": 0, "proba_sum": 0.0, "rescored": 0}
    app.state.recent = deque(maxlen=500)
    # Online evaluation ledger: predictions at open time, graded at close time.
    app.state.tracker = PredictionTracker(ledger_path=settings.ledger_path)
    # Duplicate-candidate index (optional; assistive only).
    if dup_index is None and settings.suggest_related:
        from ..dupdetect import load_index

        dup_index = load_index()
        if dup_index is not None:
            logger.info("duplicate index loaded (%d issues)", len(dup_index.meta))
    app.state.dup_index = dup_index
    # Category head (optional; assistive suggestion, never auto-labeled).
    if category_predictor is None and settings.suggest_category:
        from ..category import load_category_predictor

        category_predictor = load_category_predictor()
        if category_predictor is not None:
            logger.info("category model loaded (%s)", ", ".join(category_predictor.classes))
    app.state.category_predictor = category_predictor
    # Effort head: the artifact exists only if a run met the declared ship bar.
    if effort_predictor is None and settings.estimate_effort:
        from ..effort import load_effort_predictor

        effort_predictor = load_effort_predictor()
        if effort_predictor is not None:
            logger.info("effort model loaded")
    app.state.effort_predictor = effort_predictor
    # Assignment suggestions (similarity mechanism — won its evaluation;
    # response-level only, never an assignment action).
    if assignment_recommender is None and settings.suggest_assignees and dup_index is not None:
        from ..assign import load_assignment_recommender

        assignment_recommender = load_assignment_recommender(dup_index)
        if assignment_recommender is not None:
            logger.info("assignment recommender loaded")
    app.state.assignment_recommender = assignment_recommender

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
            "rescored_after_edit": t["rescored"],
            "predicted_actionable": t["positive"],
            "positive_rate": round(t["positive"] / t["scored"], 4) if t["scored"] else None,
            "mean_proba": round(t["proba_sum"] / t["scored"], 4) if t["scored"] else None,
            "dry_run": app.state.settings.dry_run,
            "online_evaluation": app.state.tracker.summary(),
            "latency_ms": {
                path: _percentiles(samples)
                for path, samples in app.state.latencies.items()
            },
            "errors_5xx": app.state.errors["count"],
            "analytics": app.state.tracker.analytics(),
            "recent": list(app.state.recent)[-20:],
        }

    @app.get("/dashboard")
    def dashboard() -> Any:
        """Read-only operator view over /stats (data loads with the token,
        client-side; the page itself carries no repo data)."""
        from fastapi.responses import HTMLResponse

        return HTMLResponse(_DASHBOARD_HTML)

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
        if event == "issues" and payload.get("action") == "edited":
            return _handle_issue_edited(app, payload)
        if event == "issues" and payload.get("action") == "closed":
            return _handle_issue_closed(app, payload)
        if event == "issues" and payload.get("action") in ("labeled", "unlabeled"):
            return _handle_label_event(app, payload)
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


def _percentiles(samples: Any) -> dict[str, float]:
    values = sorted(samples)
    if not values:
        return {}

    def pct(p: float) -> float:
        return round(values[min(len(values) - 1, int(len(values) * p))], 1)

    return {"n": len(values), "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)}


def _enrich(
    s: ServiceSettings, gh: GitHubAppClient | None, author: str, repo: str,
    installation_id: Any,
) -> tuple[Any, Any, Any, Any]:
    """Author profile + latest release. Best-effort — any failure degrades to
    NaN features, which the pipeline imputes."""
    author_created_at = author_public_repos = author_followers = None
    latest_release = None
    if s.enrich and gh is not None and installation_id:
        user = gh.get_user(author, installation_id)
        if user:
            author_created_at = user.get("created_at")
            author_public_repos = user.get("public_repos")
            author_followers = user.get("followers")
        latest_release = gh.get_latest_release_date(repo, installation_id)
    return author_created_at, author_public_repos, author_followers, latest_release


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

    author_created_at, author_public_repos, author_followers, latest_release = _enrich(
        s, gh, author, repo, installation_id
    )

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
    # Assistive duplicate candidates — surfaced for a maintainer to confirm,
    # never acted on automatically (pairwise ground truth doesn't exist).
    # Computed before the ledger write so the duplicate-rate facet is real.
    related: list[dict[str, Any]] = []
    if s.suggest_related and app.state.dup_index is not None:
        try:
            related = app.state.dup_index.query(
                repo, issue.get("title", ""), issue.get("body") or "",
                min_sim=s.related_min_similarity,
            )
        except Exception as e:  # index problems must never block a prediction
            logger.warning("duplicate lookup failed: %s", e)

    totals = app.state.totals
    totals["scored"] += 1
    totals["positive"] += pred.predicted_label
    totals["proba_sum"] += pred.proba
    app.state.tracker.record_prediction(repo, number, pred.proba, pred.predicted_label,
                                        related_count=len(related))
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

    # Assistive auxiliary heads (category suggestion, coarse resolution-time
    # bucket). One shared feature frame; any failure degrades to None and
    # never blocks the main prediction.
    category: dict[str, Any] | None = None
    effort: dict[str, Any] | None = None
    if app.state.category_predictor is not None or app.state.effort_predictor is not None:
        from .inference import build_feature_frame

        frame = None
        try:
            frame = build_feature_frame(
                predictor.cfg,
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
            )
        except Exception as e:
            logger.warning("aux feature frame failed: %s", e)
        if frame is not None and app.state.category_predictor is not None:
            try:
                category = app.state.category_predictor.predict_frame(frame)
            except Exception as e:
                logger.warning("category prediction failed: %s", e)
        if frame is not None and app.state.effort_predictor is not None:
            try:
                effort = app.state.effort_predictor.predict_frame(frame)
            except Exception as e:
                logger.warning("effort estimate failed: %s", e)

    # Assistive maintainer suggestions — response only, never an assignment
    # (a wrong automatic assignment costs a real person's time).
    suggested_assignees: list[dict[str, Any]] = []
    if app.state.assignment_recommender is not None:
        try:
            suggested_assignees = app.state.assignment_recommender.recommend(
                repo, issue.get("title", ""), issue.get("body") or ""
            )
        except Exception as e:  # recommender problems never block a prediction
            logger.warning("assignment suggestion failed: %s", e)

    # Optional LLM-drafted "missing information" request. Triggered only by
    # the deterministic under-specified check; the classifier's decision is
    # never delegated to the generator.
    info_request: dict[str, Any] | None = None
    if s.draft_missing_info:
        from .drafting import draft_missing_info

        try:
            info_request = draft_missing_info(
                issue.get("title", ""), issue.get("body") or "", related=related
            )
        except Exception as e:  # drafting must never block a prediction
            logger.warning("missing-info draft failed: %s", e)

    actions: list[str] = []
    if not s.dry_run and gh is not None and installation_id:
        if s.post_comment:
            comment = format_comment(pred, related, category)
            if info_request:
                comment += "\n\n---\n\n" + info_request["draft"]
            gh.post_comment(repo, number, comment, installation_id)
            actions.append("comment")
        if s.apply_label and pred.predicted_label == 1:
            gh.add_labels(repo, number, [s.label_name], installation_id)
            actions.append("label")
        if s.project_id and pred.predicted_label == 1 and issue.get("node_id"):
            gh.add_issue_to_project(s.project_id, issue["node_id"], installation_id)
            actions.append("project")
    for action in actions:
        app.state.tracker.record_action(repo, number, action)

    return {"ok": True, "prediction": pred.as_dict(), "category": category,
            "estimated_resolution": effort,   # API-only by design; see EFFORT_CARD.md
            "related_issues": related,
            "suggested_assignees": suggested_assignees,  # API-only; never assigned
            "missing_info": info_request,
            "actions": actions, "dry_run": s.dry_run}


# ---------------------------------------------------------------------------
# The issues.edited flow: re-score with the improved text, no write actions
# ---------------------------------------------------------------------------
def _handle_issue_edited(app: FastAPI, payload: dict[str, Any]) -> dict[str, Any]:
    """Reporters frequently add repro steps after opening (often because the
    bot asked). Re-scoring updates the pending ledger entry so the prediction
    graded at close reflects the text maintainers actually triaged. No
    comment or label is ever posted on edit — one issue, at most one comment."""
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
    if issue.get("state") == "closed":
        return {"ok": True, "ignored": "edit on closed issue"}

    author_created_at, author_public_repos, author_followers, latest_release = _enrich(
        s, gh, author, repo, installation_id
    )
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
        explain=False,
    )
    app.state.totals["rescored"] += 1
    app.state.tracker.record_prediction(repo, number, pred.proba, pred.predicted_label)
    logger.info("rescored %s#%d after edit: P(bug)=%.3f", repo, number, pred.proba)
    return {"ok": True, "rescored": True, "prediction": pred.as_dict()}


# ---------------------------------------------------------------------------
# Label events: maintainer labeling is future ground truth — record it
# ---------------------------------------------------------------------------
def _handle_label_event(app: FastAPI, payload: dict[str, Any]) -> dict[str, Any]:
    issue = payload.get("issue") or {}
    repo = (payload.get("repository") or {}).get("full_name", "")
    number = int(issue.get("number", 0))
    label = ((payload.get("label") or {}).get("name")) or ""
    if not repo or not number or not label:
        return {"ok": True, "ignored": "label event without label/repo/number"}
    added = payload.get("action") == "labeled"
    app.state.tracker.record_label_event(repo, number, label, added)
    return {"ok": True, "recorded": ("+" if added else "-") + label}


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


# ---------------------------------------------------------------------------
# Dashboard page (self-contained; token entered client-side, sent as header)
# ---------------------------------------------------------------------------
_DASHBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>GHIC dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
 body { margin: 2rem auto; max-width: 60rem; padding: 0 1rem; line-height: 1.45; }
 h1 { font-size: 1.3rem; } h2 { font-size: 1.05rem; margin-top: 1.6rem; }
 .tiles { display: flex; gap: 1rem; flex-wrap: wrap; }
 .tile { border: 1px solid #8884; border-radius: 8px; padding: .8rem 1.1rem; min-width: 9rem; }
 .tile b { display: block; font-size: 1.5rem; }
 table { border-collapse: collapse; width: 100%; font-size: .9rem; }
 td, th { border-bottom: 1px solid #8883; padding: .35rem .5rem; text-align: left; }
 input { padding: .4rem; min-width: 18rem; } button { padding: .4rem .9rem; }
 .muted { opacity: .65; font-size: .85rem; }
 .err { color: #c33; }
</style></head><body>
<h1>Issue Triage Bot — dashboard</h1>
<p><input id="token" type="password" placeholder="webhook secret (X-GHIC-Token)">
<button onclick="load()">Load</button> <span id="msg" class="muted"></span></p>
<div id="content" hidden>
 <div class="tiles">
  <div class="tile"><b id="scored">–</b>issues scored</div>
  <div class="tile"><b id="posrate">–</b>flagged actionable</div>
  <div class="tile"><b id="liveprec">–</b>live precision</div>
  <div class="tile"><b id="liverec">–</b>live recall (lower bound)</div>
  <div class="tile"><b id="p95">–</b>webhook p95 (ms)</div>
 </div>
 <h2>Model</h2><p id="model" class="muted"></p>
 <h2>Online evaluation</h2><p id="online" class="muted"></p>
 <h2>Issue trends <span class="muted">(predictions/day, last 30d)</span></h2>
 <table><thead><tr><th>date</th><th>predictions</th></tr></thead><tbody id="trends"></tbody></table>
 <h2>Duplicate rate</h2><p id="duprate" class="muted"></p>
 <h2>Resolution analytics</h2><p id="resolutions" class="muted"></p>
 <h2>Confidence distribution <span class="muted">(P(actionable) deciles)</span></h2>
 <p id="confhist" style="font-family: ui-monospace, monospace; white-space: pre;"></p>
 <h2>Label stats <span class="muted">(applied by maintainers, observed live)</span></h2>
 <table><thead><tr><th>label</th><th>added</th></tr></thead><tbody id="labelstats"></tbody></table>
 <h2>Component analytics <span class="muted">(per repo)</span></h2>
 <table><thead><tr><th>repo</th><th>scored</th><th>positive rate</th><th>mean P</th></tr></thead>
 <tbody id="components"></tbody></table>
 <h2>Recent predictions</h2>
 <table><thead><tr><th>repo</th><th>issue</th><th>P(bug)</th><th>predicted</th><th>at</th></tr></thead>
 <tbody id="recent"></tbody></table>
</div>
<script>
async function load() {
  const t = document.getElementById('token').value;
  const msg = document.getElementById('msg');
  msg.textContent = 'loading…'; msg.className = 'muted';
  try {
    const h = { 'X-GHIC-Token': t };
    const [stats, health] = await Promise.all([
      fetch('/stats', { headers: h }).then(r => { if (!r.ok) throw new Error('stats HTTP ' + r.status); return r.json(); }),
      fetch('/healthz').then(r => r.json()),
    ]);
    document.getElementById('content').hidden = false;
    msg.textContent = 'updated ' + new Date().toLocaleTimeString();
    const fmt = v => v == null ? 'n/a' : (typeof v === 'number' && v <= 1 ? (v * 100).toFixed(1) + '%' : v);
    document.getElementById('scored').textContent = stats.scored;
    document.getElementById('posrate').textContent = fmt(stats.positive_rate);
    const oe = stats.online_evaluation || {};
    document.getElementById('liveprec').textContent = fmt(oe.live_precision);
    document.getElementById('liverec').textContent = fmt(oe.live_recall_lower_bound);
    const wh = (stats.latency_ms || {})['/webhook'] || {};
    document.getElementById('p95').textContent = wh.p95 ?? 'n/a';
    document.getElementById('model').textContent =
      health.model + ' · threshold ' + health.threshold + ' · dry_run ' + health.dry_run +
      ' · per-repo thresholds ' + JSON.stringify(health.repo_thresholds);
    document.getElementById('online').textContent =
      'resolved ' + (oe.resolved ?? 0) + ' · awaiting outcome ' + (oe.awaiting_outcome ?? 0) +
      ' · confusion ' + JSON.stringify(oe.confusion) + ' · audited GitHub writes ' +
      (oe.github_writes_audited ?? 0) + ' — ' + (oe.note || '');
    const a = stats.analytics || {};
    const trends = (a.issue_trends || {}).predictions_per_day || {};
    document.getElementById('trends').innerHTML = Object.keys(trends).map(d =>
      '<tr><td>' + d + '</td><td>' + trends[d] + '</td></tr>').join('') ||
      '<tr><td colspan="2" class="muted">no predictions yet</td></tr>';
    const dr = a.duplicate_rate || {};
    document.getElementById('duprate').textContent =
      (dr.predictions_with_related_candidates ?? 0) + ' predictions had similar-prior candidates (rate ' +
      fmt(dr.rate) + ') · duplicate labels observed live: ' + (dr.duplicate_labels_observed_live ?? 0);
    const rs = a.resolution_analytics || {};
    document.getElementById('resolutions').textContent =
      'resolved actionable ' + (rs.resolved_actionable ?? 0) + ' · resolved non-actionable ' +
      (rs.resolved_non_actionable ?? 0) + ' · awaiting outcome ' + (rs.awaiting_outcome ?? 0);
    const histo = ((a.confidence_metrics || {}).proba_histogram_deciles) || [];
    const hmax = Math.max(1, ...histo);
    document.getElementById('confhist').textContent = histo.map((n, i) =>
      (i / 10).toFixed(1) + '–' + ((i + 1) / 10).toFixed(1) + ' ' +
      '█'.repeat(Math.round(20 * n / hmax)).padEnd(20, '·') + ' ' + n).join('\\n') || 'no data';
    const ls = ((a.label_stats || {}).top_labels_added) || {};
    document.getElementById('labelstats').innerHTML = Object.keys(ls).map(k =>
      '<tr><td>' + k + '</td><td>' + ls[k] + '</td></tr>').join('') ||
      '<tr><td colspan="2" class="muted">no label events yet</td></tr>';
    const comp = a.component_analytics || {};
    document.getElementById('components').innerHTML = Object.keys(comp).map(k =>
      '<tr><td>' + k + '</td><td>' + comp[k].scored + '</td><td>' + fmt(comp[k].positive_rate) +
      '</td><td>' + (comp[k].mean_proba ?? 'n/a') + '</td></tr>').join('') ||
      '<tr><td colspan="4" class="muted">no predictions yet</td></tr>';
    document.getElementById('recent').innerHTML = (stats.recent || []).slice().reverse().map(r =>
      '<tr><td>' + r.repo + '</td><td>#' + r.issue + '</td><td>' + r.proba.toFixed(3) +
      '</td><td>' + r.predicted + '</td><td>' + r.at.replace('T', ' ').slice(0, 19) + '</td></tr>'
    ).join('');
  } catch (e) { msg.textContent = e.message; msg.className = 'err'; }
}
</script></body></html>"""


def export_openapi(path: Any = None) -> Any:
    """Write the OpenAPI spec FastAPI generates to docs/openapi.json.

    Builds the app with stubs so no model artifact is needed — the spec
    describes the API surface, not the model.
    """
    import json
    from pathlib import Path

    from .settings import ServiceSettings

    spec_app = create_app(
        ServiceSettings(model_path=Path("unused.joblib"), webhook_secret="spec",
                        suggest_related=False, suggest_category=False,
                        estimate_effort=False),
        predictor=object(),  # never called during spec generation
    )
    spec = spec_app.openapi()
    path = Path(path) if path else utils.PROJECT_ROOT / "docs" / "openapi.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d paths)", path, len(spec.get("paths", {})))
    return path


def main() -> int:
    """`python -m ghic.service.app` / `ghic-serve` — run a local server."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the webhook service.")
    parser.add_argument("--openapi", nargs="?", const="", metavar="PATH",
                        help="export the OpenAPI spec (default docs/openapi.json) and exit")
    args = parser.parse_args()
    if args.openapi is not None:
        export_openapi(args.openapi or None)
        return 0

    import uvicorn

    host = os.environ.get("GHIC_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", os.environ.get("GHIC_PORT", "8000")))
    uvicorn.run(create_app(), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
