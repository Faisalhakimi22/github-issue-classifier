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
from ..llm import IssueContext, LLMService
from .github_app import GitHubAppClient
from .inference import IssuePredictor, format_comment, format_llm_comment
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
    llm_service: Any = None,
    idempotency: Any = None,
    repo_intelligence: Any = None,
    automation_service: Any = None,
    authorization_gate: Any = None,
    authorization_lease: Any = None,
) -> FastAPI:
    """Build the app. `predictor` / `gh_client` / `dup_index` /
    `category_predictor` / `effort_predictor` / `assignment_recommender` /
    `llm_service` / `idempotency` / `repo_intelligence` are injectable for
    tests."""
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
    app.state.tracker = PredictionTracker(
        ledger_path=settings.ledger_path, database_url=settings.database_url
    )
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
    # LLM-assisted analysis (optional; off unless GHIC_USE_LLM_ANALYSIS=true
    # AND an OPENROUTER_API_KEY is set — see settings.can_use_llm).
    if llm_service is None and settings.can_use_llm:
        llm_service = LLMService(_build_llm_provider(settings))
    app.state.llm_service = llm_service
    # Idempotency: dedup GitHub webhook deliveries by X-GitHub-Delivery.
    # Shares database_url (same Postgres) with the ledger when set; a
    # file-backed fallback otherwise; in-memory (this process's lifetime
    # only) if neither is configured -- see idempotency.py.
    if idempotency is None:
        from .idempotency import build_idempotency_store

        idempotency = build_idempotency_store(
            database_url=settings.database_url, file_path=settings.idempotency_path,
        )
    app.state.idempotency = idempotency
    # Repository Intelligence: semantic code retrieval over the issue's own
    # repository (off unless GHIC_USE_REPO_INTELLIGENCE=true). Constructed
    # with an index_scheduler bound to this app so a repo that has never
    # been indexed gets queued rather than indexed inline -- indexing takes
    # far longer than a webhook may.
    if repo_intelligence is None and settings.use_repo_intelligence:
        from ..repository_intelligence import build_service

        # build_service resolves every provider from the environment and
        # returns None when the engine shouldn't run here at all -- most
        # importantly on an ephemeral filesystem with no persistent vector
        # store, where indexing would be rebuilt and discarded on every
        # cold start. None is handled everywhere exactly like "feature off".
        repo_intelligence = build_service(
            qstash_token=settings.qstash_token,
            callback_url=(
                f"{settings.public_base_url}/internal/index-repository"
                if settings.public_base_url else ""
            ),
            qstash_region=settings.qstash_region,
            database_url=settings.database_url,
        )
    app.state.repo_intelligence = repo_intelligence
    # repo full name -> installation id, populated from webhook payloads so
    # a queued indexing job can authenticate a private clone.
    app.state.repo_installations = {}
    if authorization_gate is None:
        from .github_connection import authorize_issue

        authorization_gate = authorize_issue
    if authorization_lease is None:
        from .github_connection import authorization_lease as default_authorization_lease

        authorization_lease = default_authorization_lease
    app.state.authorization_gate = authorization_gate
    app.state.authorization_lease = authorization_lease
    # Phase 4 automation. Constructed only when at least one capability
    # flag is set; holds no GitHub client by design (see ghic/automation/).
    if automation_service is None:
        from ..automation import AutomationFlags, AutomationService

        flags = AutomationFlags.from_env()
        automation_service = (
            AutomationService(flags, engine_version=__version__) if flags.any_enabled else None
        )
        if automation_service is not None:
            logger.info("automation enabled: %s", flags)
    app.state.automation = automation_service

    def _require_token(request: Request) -> None:
        s: ServiceSettings = app.state.settings
        if s.webhook_secret:
            token = request.headers.get("X-GHIC-Token", "")
            if not hmac.compare_digest(token, s.webhook_secret):
                raise HTTPException(status_code=401, detail="invalid token")
        elif not s.allow_unsigned:
            raise HTTPException(status_code=503, detail="webhook secret not configured")

    def _legacy_repository_route_disabled() -> None:
        """Disable pre-Hub shared-token routes that lack tenant auth."""
        raise HTTPException(
            status_code=503,
            detail="legacy repository endpoint disabled; use the authenticated Hub API",
        )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        payload = {
            "status": "ok",
            "version": __version__,
            "model": app.state.predictor.model_name,
            "threshold": app.state.settings.threshold,
            # Count, not the mapping. /healthz is unauthenticated -- it exists
            # for load balancers -- and the keys of this dict are repository
            # names. Empty until now, so nothing leaked; the moment a per-repo
            # threshold is configured the repository it belongs to would be
            # readable by anyone, including for a private repo. Operators set
            # these through GHIC_REPO_THRESHOLDS and already know the values.
            "repo_thresholds_configured": len(app.state.settings.repo_thresholds),
            "dry_run": app.state.settings.dry_run,
        }
        if app.state.settings.use_repo_intelligence:
            # Reports durability explicitly: "enabled but nothing persists"
            # looks identical to a healthy deploy until someone wonders why
            # no repository is ever ready.
            from ..repository_intelligence import health_snapshot

            payload["repository_intelligence"] = health_snapshot(app.state.repo_intelligence)
        return payload

    @app.get("/repositories")
    def repositories(request: Request, limit: int = 50) -> dict[str, Any]:
        """Indexing status per repository, for the dashboard.

        Token-gated like /stats: index state discloses which repositories
        this installation can see, plus their languages and README
        summaries.
        """
        _require_token(request)
        _legacy_repository_route_disabled()
        service = app.state.repo_intelligence
        if service is None:
            return {"enabled": False, "repositories": []}
        return {
            "enabled": True,
            "repositories": [r.as_dict() for r in service.list_repositories(limit=limit)],
        }

    @app.get("/repositories/metrics")
    def repository_metrics(request: Request) -> dict[str, Any]:
        """Engine counters and timings. Process-scoped -- see metrics.py."""
        _require_token(request)
        from ..repository_intelligence import METRICS

        service = app.state.repo_intelligence
        snapshot = METRICS.snapshot()
        if service is not None and service.index_queue is not None:
            depth = service.index_queue.depth()
            if depth is not None:
                snapshot["gauges"]["index_queue_depth"] = depth
        return snapshot

    @app.get("/stats")
    def stats(request: Request) -> dict[str, Any]:
        """Process-level observability; never expose tenant or repository data."""
        _require_token(request)
        return {
            "scope": "process",
            "dry_run": app.state.settings.dry_run,
            "latency_ms": {
                path: _percentiles(samples)
                for path, samples in app.state.latencies.items()
            },
            "errors_5xx": app.state.errors["count"],
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

        # GitHub reuses the same X-GitHub-Delivery ID when it retries a
        # delivery that didn't get a timely 2xx -- the natural idempotency
        # key, checked before any processing so a retried delivery (or one
        # replayed by mistake) never produces a second comment.
        delivery_id = request.headers.get("X-GitHub-Delivery")
        if delivery_id and not app.state.idempotency.mark_if_new(delivery_id):
            logger.info("duplicate delivery %s ignored", delivery_id)
            return {"ok": True, "duplicate": True}

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
        if event in ("installation", "installation_repositories", "repository"):
            # Keeps the dashboard's connected-repository list in step with
            # what the installation actually grants. Bookkeeping only: it
            # never raises, so a sync problem cannot make GitHub retry a
            # delivery that was otherwise handled.
            from .github_connection import handle_installation_event

            result = handle_installation_event(
                s.database_url, event, payload, app.state.gh
            )
            if not result.get("ok", True) and result.get("retryable"):
                # This delivery was marked consumed before the handler ran, so
                # GitHub's retry would be discarded as a duplicate unless the
                # key is released first -- which would make the 500 below
                # meaningless. Safe only because the purge is idempotent.
                release = getattr(app.state.idempotency, "release", None)
                if delivery_id and callable(release):
                    release(delivery_id)
                raise HTTPException(
                    status_code=500,
                    detail=str(result.get("connection_sync") or "cleanup failed"),
                )
            return result
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

    @app.post("/internal/process-issue")
    async def process_issue_callback(request: Request) -> dict[str, Any]:
        """QStash's callback target -- see settings.can_use_async_queue and
        ghic/service/qstash.py. Verifies the Upstash-Signature JWT (not the
        GitHub HMAC -- this request comes from QStash, not GitHub) before
        doing any work, since this endpoint can trigger a real GitHub
        comment. A non-2xx response makes QStash retry per its own policy.
        """
        s: ServiceSettings = app.state.settings
        if not s.can_use_async_queue:
            raise HTTPException(status_code=503, detail="async processing not configured")

        body = await request.body()
        from .qstash import verify_signature

        signing_keys = [s.qstash_current_signing_key, s.qstash_next_signing_key]
        if not verify_signature(
            request.headers.get("Upstash-Signature"), body, signing_keys,
            f"{s.public_base_url}/internal/process-issue",
        ):
            raise HTTPException(status_code=401, detail="invalid qstash signature")

        payload = await request.json()
        return _process_issue_job(app, payload)

    @app.post("/automation/analyze")
    def automation_analyze(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        """Advisory automation for an arbitrary issue (Feature 10).

        One endpoint rather than five (`/fix-plan`, `/pr-draft`,
        `/test-plan`, `/checklist`): they share every input and the same
        retrieval + analysis pipeline, so splitting them would mean
        re-retrieving the same evidence up to four times for one issue.
        Callers select what they want with `include`; the response shape
        is the same bundle either way, and each section is separately
        flagged server-side regardless.

        Read-only and advisory: this endpoint cannot label, assign, close,
        comment, or open a pull request.
        """
        _require_token(request)
        _legacy_repository_route_disabled()
        service = app.state.repo_intelligence
        automation = app.state.automation
        if service is None or automation is None:
            raise HTTPException(
                status_code=503,
                detail="repository intelligence and automation must both be enabled",
            )

        repo = str(payload.get("repo") or "")
        title = str(payload.get("title") or "")
        body = str(payload.get("body") or "")
        number = int(payload.get("issue_number") or 0)
        if not repo or not title:
            raise HTTPException(status_code=422, detail="repo and title are required")

        context = service.get_context(repo, title, body)
        analysis = None
        if app.state.settings.use_engineering_intelligence:
            from ..engineering_intelligence import EngineeringAnalyzer

            analysis = EngineeringAnalyzer().analyze(
                repo, title, body, context, issue_number=number,
            )

        bundle = automation.build(
            repo, number, title, body, analysis=analysis, repository_context=context,
        )
        return {
            "ok": True,
            "advisory_only": True,
            "repository_context": context.as_dict() if context else None,
            "engineering_analysis": analysis.as_dict() if analysis else None,
            "automation": bundle.as_dict() if bundle else None,
        }

    @app.get("/automation/weekly-digest")
    def automation_weekly_digest(
        request: Request, repo: str, fmt: str = "markdown"
    ) -> dict[str, Any]:
        """Repository digest from indexed history (Feature 9).

        `fmt=markdown` for GitHub/email, `fmt=text` for Slack/Teams.
        """
        _require_token(request)
        _legacy_repository_route_disabled()
        service = app.state.repo_intelligence
        if service is None:
            raise HTTPException(status_code=503, detail="repository intelligence not enabled")

        from ..automation import build_weekly_digest
        from ..engineering_intelligence import ComponentAnalyzer, cluster_history

        chunks = _all_indexed_chunks(service, repo)
        if not chunks:
            return {"ok": True, "repo": repo, "digest": "", "note": "repository not indexed"}

        stats = ComponentAnalyzer().analyze(chunks)
        clusters = cluster_history(chunks)
        return {
            "ok": True,
            "repo": repo,
            "format": fmt,
            "digest": build_weekly_digest(repo, stats, clusters, fmt=fmt),
        }

    @app.get("/automation/repository-analytics")
    def automation_repository_analytics(request: Request, repo: str) -> dict[str, Any]:
        """Structured analytics behind the dashboard widgets (Feature 11)."""
        _require_token(request)
        _legacy_repository_route_disabled()
        service = app.state.repo_intelligence
        if service is None:
            raise HTTPException(status_code=503, detail="repository intelligence not enabled")

        from ..engineering_intelligence import (
            ComponentAnalyzer,
            cluster_history,
            summarize_repository,
        )

        chunks = _all_indexed_chunks(service, repo)
        analyzer = ComponentAnalyzer()
        stats = analyzer.analyze(chunks)
        return {
            "ok": True,
            "repo": repo,
            "summary": summarize_repository(stats),
            "components": [s.as_dict() for s in stats.values()],
            "hotspots": [s.as_dict() for s in analyzer.hotspots(stats)],
            "clusters": [c.as_dict() for c in cluster_history(chunks)],
        }

    @app.post("/internal/index-repository")
    async def index_repository_callback(request: Request) -> dict[str, Any]:
        """QStash's callback target for repository indexing.

        Same signature verification as process-issue. Indexing is slow by
        nature (clone + walk + embed), which is exactly why it lives behind
        the queue instead of in the webhook: this endpoint may take minutes,
        and nothing is waiting on it. A failed index returns 200 with
        indexed=false rather than a 5xx -- QStash retrying a clone that
        failed because the repo is private and the App lacks contents:read
        would just fail identically several more times.
        """
        s: ServiceSettings = app.state.settings
        if app.state.repo_intelligence is None:
            raise HTTPException(status_code=503, detail="repository intelligence not enabled")
        if not s.can_use_async_queue:
            raise HTTPException(status_code=503, detail="async processing not configured")

        body = await request.body()
        from .qstash import verify_signature

        signing_keys = [s.qstash_current_signing_key, s.qstash_next_signing_key]
        if not verify_signature(
            request.headers.get("Upstash-Signature"), body, signing_keys,
            f"{s.public_base_url}/internal/index-repository",
        ):
            raise HTTPException(status_code=401, detail="invalid qstash signature")

        payload = await request.json()
        repo = str(payload.get("repo") or "")
        if not repo:
            raise HTTPException(status_code=422, detail="missing repo")

        # Carried from the webhook that queued this job, minutes and a
        # process boundary ago -- it's what joins the two in the logs.
        from ..repository_intelligence.metrics import set_correlation_id

        set_correlation_id(str(payload.get("correlation_id") or ""))

        # A durable job must carry the installation identity that authorized
        # it. Process-local observations are intentionally not a fallback.
        installation_id = payload.get("installation_id")
        decision = _authorize_index_job(app, installation_id, repo)
        if not decision["authorized"]:
            return {
                "ok": True,
                "repo": repo,
                "indexed": False,
                "skipped": True,
                "authorization": "rejected",
                "reason": decision["reason"],
            }
        # Keep the authoritative rows locked for the whole token/clone/index
        # operation. A disconnect or revoke therefore cannot race this job
        # after the check and before repository contents are read.
        with app.state.authorization_lease(
            s.database_url, installation_id, repo
        ) as lease:
            if not lease["authorized"]:
                return {
                    "ok": True,
                    "repo": repo,
                    "indexed": False,
                    "skipped": True,
                    "authorization": "rejected",
                    "reason": lease["reason"],
                }
            token = ""
            if app.state.gh is not None and installation_id:
                try:
                    token = app.state.gh.installation_token(int(installation_id))
                except Exception as e:
                    # A public repo still clones without a token, so this is a
                    # degradation (private repos fail later), not a hard stop.
                    logger.warning("could not mint installation token for %s: %s", repo, e)

            indexed = app.state.repo_intelligence.index_repository(
                repo, token=token, workspace_id=lease.get("workspace_id")
            )
        return {"ok": True, "repo": repo, "indexed": indexed}

    return app


def _all_indexed_chunks(service: Any, repo: str) -> list[Any]:
    """Every chunk indexed for `repo`, for repository-wide analytics.

    Analytics need the whole corpus, not a query's top-k -- component
    health computed over ten retrieved chunks would describe the query,
    not the repository. Returns [] when the repo isn't indexed, which the
    callers report honestly rather than rendering empty widgets.
    """
    pointer = service.index_cache.get_latest(repo)
    if pointer is None:
        return []
    loaded = service.index_cache.load(repo, *pointer)
    if loaded is None:
        return []
    store, _ = loaded
    return list(getattr(store, "_chunks", []))


def _epoch(iso: Any) -> float | None:
    """GitHub's ISO-8601 timestamp -> epoch seconds. None if unparseable,
    which makes the analyzer fall back to "now" rather than mis-dating an
    issue and inventing a bogus regression window."""
    if not iso:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


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


def _build_llm_provider(settings: ServiceSettings) -> Any:
    """Groq (fast, ~1.9s measured) is the primary; OpenRouter (nemotron,
    free tier, ~17s measured) is the fallback -- see
    models/LLM_ANALYSIS_CARD.md for the comparison. Which of the two keys
    are actually set decides what gets built: both -> a priority chain,
    either alone -> that one provider. Only called when settings.can_use_llm
    is already True, so at least one key is guaranteed present.
    """
    from ..llm.fallback import FallbackLLMProvider
    from ..llm.groq import GroqProvider
    from ..llm.openrouter import OpenRouterProvider

    providers: list[Any] = []
    if settings.groq_api_key:
        providers.append(GroqProvider(api_key=settings.groq_api_key, model=settings.llm_model))
    if settings.openrouter_api_key:
        providers.append(OpenRouterProvider(
            api_key=settings.openrouter_api_key, model=settings.openrouter_model,
        ))

    provider = providers[0] if len(providers) == 1 else FallbackLLMProvider(providers)
    logger.info(
        "llm analysis enabled (%s)",
        " -> ".join(type(p).__name__ for p in providers),
    )
    return provider


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
    """Thin ingest: cheap validation, then either hand off to QStash (async
    configured) or process inline (default, current behavior unchanged).

    The bot/malformed-payload checks stay here rather than moving into the
    deferred job -- no reason to enqueue (or spend a synchronous call on) an
    issue that's getting ignored either way.
    """
    issue = payload.get("issue") or {}
    repo = (payload.get("repository") or {}).get("full_name", "")
    number = int(issue.get("number", 0))
    author = ((issue.get("user") or {}).get("login")) or ""

    if not repo or not number:
        raise HTTPException(status_code=422, detail="malformed issues payload")

    decision = _authorize_issue(app, payload, repo, number)
    if not decision["authorized"]:
        return _authorization_skip_response(decision)
    if _is_bot(author):
        return {"ok": True, "ignored": f"bot author {author}"}

    if app.state.settings.can_use_async_queue:
        from .qstash import QStashError, publish

        s: ServiceSettings = app.state.settings
        try:
            message_id = publish(
                f"{s.public_base_url}/internal/process-issue", payload, s.qstash_token,
                region=s.qstash_region,
            )
            return {"ok": True, "queued": True, "message_id": message_id}
        except QStashError as e:
            # Queueing failed -- fall through to synchronous processing
            # rather than silently dropping the issue.
            logger.warning("qstash publish failed (%s); processing synchronously instead", e)

    return _process_issue_job(app, payload, authorization=decision)


def _process_issue_job(
    app: FastAPI, payload: dict[str, Any], authorization: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run one issue job and persist its complete, sanitized lifecycle."""
    import time as _time

    started = _time.perf_counter()
    issue = payload.get("issue") or {}
    repo = (payload.get("repository") or {}).get("full_name", "")
    number = int(issue.get("number", 0))
    decision = authorization or _authorize_issue(app, payload, repo, number)
    if not decision["authorized"]:
        return _authorization_skip_response(decision)
    payload["_workspace_id"] = decision.get("workspace_id")
    try:
        result = _process_issue_job_impl(app, payload)
    except Exception as error:
        try:
            app.state.tracker.record_processing_failure(
                repo, number, error, workspace_id=payload.get("_workspace_id")
            )
        except Exception as ledger_error:
            logger.warning("could not persist processing failure: %s", ledger_error)
        raise

    duration_ms = (_time.perf_counter() - started) * 1000
    if result.get("authorization") == "rejected":
        return result
    app.state.tracker.record_analysis(
        repo,
        number,
        _dashboard_analysis_record(payload, result, duration_ms),
        workspace_id=payload.get("_workspace_id"),
    )
    return result


def _authorize_issue(
    app: FastAPI, payload: dict[str, Any], repo: str, number: int
) -> dict[str, Any]:
    installation_id = (payload.get("installation") or {}).get("id")
    decision = app.state.authorization_gate(
        app.state.settings.database_url,
        installation_id,
        repo,
        app.state.gh,
    )
    workspace_id = decision.get("workspace_id")
    if workspace_id:
        payload["_workspace_id"] = workspace_id
    if decision.get("authorized"):
        return decision
    try:
        app.state.tracker.record_authorization_skip(
            repo,
            number,
            installation_id,
            str(decision.get("reason") or "authorization_rejected"),
            workspace_id=workspace_id,
        )
    except Exception as error:
        logger.warning("could not persist authorization skip: %s", error)
    return decision


def _authorize_index_job(app: FastAPI, installation_id: Any, repo: str) -> dict[str, Any]:
    if installation_id is None:
        return {"authorized": False, "reason": "unknown_installation"}
    return app.state.authorization_gate(
        app.state.settings.database_url,
        installation_id,
        repo,
        app.state.gh,
    )


def _authorization_skip_response(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "skipped": True,
        "authorization": "rejected",
        "reason": str(decision.get("reason") or "authorization_rejected"),
    }


def _process_issue_job_impl(app: FastAPI, payload: dict[str, Any]) -> dict[str, Any]:
    """The actual work: ML scoring, assistive heads, LLM analysis, and any
    GitHub writes. Runs inline when async processing is off (default) or
    QStash publish failed; runs from POST /internal/process-issue when
    QStash calls back. Same function either way -- the caller decides
    when it runs, this doesn't know or care.
    """
    s: ServiceSettings = app.state.settings
    predictor: IssuePredictor = app.state.predictor
    gh: GitHubAppClient | None = app.state.gh

    issue = payload.get("issue") or {}
    repo = (payload.get("repository") or {}).get("full_name", "")
    installation_id = (payload.get("installation") or {}).get("id")
    number = int(issue.get("number", 0))
    author = ((issue.get("user") or {}).get("login")) or ""

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
    app.state.tracker.record_prediction(
        repo, number, pred.proba, pred.predicted_label,
        related_count=len(related), workspace_id=payload.get("_workspace_id"),
    )
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

    # LLM-assisted analysis (category/priority/severity/summary/missing
    # info/labels) layered on top of the ML prediction -- never recomputes
    # or overrides pred.proba. None when disabled or on any failure;
    # LLMService itself never raises. See ghic/llm/ and
    # models/LLM_ANALYSIS_CARD.md.
    # Repository Intelligence: semantic code retrieval over this repo, fed
    # to the LLM as evidence and rendered as the comment's Repository
    # Evidence section. Never raises and never blocks on indexing -- an
    # unindexed repo queues a background job and returns empty, so the
    # first issue on a new repo is analyzed from text alone. See
    # ghic/repository_intelligence/repository_service.py.
    repo_context = None
    if app.state.repo_intelligence is not None:
        if installation_id:
            app.state.repo_installations[repo] = installation_id
        # get_context is contractually non-raising, but this is wrapped
        # anyway -- same as every other assistive head above. The contract
        # is one implementation's promise; this is the webhook's guarantee,
        # and the webhook shouldn't be able to break because a component
        # someone swapped in violated its interface.
        try:
            repo_context = app.state.repo_intelligence.get_context(
                repo,
                issue.get("title", ""),
                issue.get("body") or "",
                category=(category or {}).get("predicted", "") if category else "",
                predicted_label=pred.predicted_label,
                installation_id=installation_id,
                workspace_id=payload.get("_workspace_id"),
            )
            logger.info(
                "repository context for %s#%d: indexed=%s files=%d",
                repo, number, repo_context.indexed, len(repo_context.relevant_files),
            )
        except Exception as e:
            logger.warning("repository intelligence failed for %s#%d: %s", repo, number, e)
            from ..repository_intelligence.models import (
                RepositoryContext,
                UNAVAILABLE_CONTEXT_NOTE,
            )

            repo_context = RepositoryContext(
                repo=repo, indexed=False, note=UNAVAILABLE_CONTEXT_NOTE,
            )

    # Phase 3: engineering analysis over the evidence just retrieved.
    # Deterministic and local -- no network, no LLM call -- so it adds
    # milliseconds, not seconds, to the webhook. Never raises (the
    # analyzer catches internally; this wraps it anyway, same as every
    # other assistive head).
    engineering_analysis = None
    if s.use_engineering_intelligence and repo_context is not None:
        try:
            from ..engineering_intelligence import EngineeringAnalyzer

            engineering_analysis = EngineeringAnalyzer().analyze(
                repo, issue.get("title", ""), issue.get("body") or "", repo_context,
                issue_number=number,
                issue_created_at=_epoch(issue.get("created_at")),
            )
        except Exception as e:
            logger.warning("engineering analysis failed for %s#%d: %s", repo, number, e)
            engineering_analysis = None

    # Phase 4 automation: advisory artifacts only. This service holds no
    # GitHub client and cannot act on anything it suggests -- see
    # ghic/automation/. Each capability is separately flagged.
    automation = None
    if app.state.automation is not None and repo_context is not None:
        try:
            automation = app.state.automation.build(
                repo, number, issue.get("title", ""), issue.get("body") or "",
                analysis=engineering_analysis, repository_context=repo_context,
            )
        except Exception as e:
            logger.warning("automation failed for %s#%d: %s", repo, number, e)
            automation = None

    llm_analysis = None
    if app.state.llm_service is not None:
        context = IssueContext(
            repo=repo,
            title=issue.get("title", ""),
            body=issue.get("body") or "",
            labels=[lab.get("name", "") for lab in (issue.get("labels") or [])],
            author=author,
            metadata={"created_at": issue.get("created_at")},
            ml_probability=pred.proba,
            ml_predicted_label=pred.predicted_label,
            repository_context=repo_context,
        )
        llm_analysis = app.state.llm_service.analyze_issue(context)

    # Consistency check: does the LLM's own priority/severity/reasoning
    # strongly contradict the ML verdict it was given as fixed evidence?
    # Deterministic, not a second LLM call -- see ghic/llm/consistency.py
    # for why. Never silently picks a side; the comment shows the tension.
    disagreement = False
    if llm_analysis is not None:
        from ..llm import detect_disagreement

        disagreement = detect_disagreement(
            pred.predicted_label, llm_analysis.priority, llm_analysis.severity,
            llm_analysis.reasoning,
        )
        if disagreement:
            logger.warning(
                "llm/ml disagreement on %s#%d: ML=non-actionable but LLM priority=%s severity=%s",
                repo, number, llm_analysis.priority, llm_analysis.severity,
            )

    # Optional LLM-drafted "missing information" request. Skipped when the
    # analysis above already produced one (avoids a redundant second LLM
    # call and a duplicate section in the comment). Triggered only by the
    # deterministic under-specified check either way; the classifier's
    # decision is never delegated to the generator.
    info_request: dict[str, Any] | None = None
    if s.draft_missing_info and llm_analysis is None:
        from .drafting import draft_missing_info

        try:
            info_request = draft_missing_info(
                issue.get("title", ""), issue.get("body") or "", related=related
            )
        except Exception as e:  # drafting must never block a prediction
            logger.warning("missing-info draft failed: %s", e)

    actions: list[str] = []
    if not s.dry_run and gh is not None and installation_id:
        with app.state.authorization_lease(
            s.database_url, installation_id, repo
        ) as final_decision:
            if not final_decision["authorized"]:
                app.state.tracker.record_authorization_skip(
                    repo,
                    number,
                    installation_id,
                    str(final_decision.get("reason") or "authorization_rejected"),
                    workspace_id=final_decision.get("workspace_id"),
                )
                return _authorization_skip_response(final_decision)
            if s.post_comment:
                if llm_analysis is not None:
                    comment = format_llm_comment(
                        pred, llm_analysis, related, disagreement=disagreement,
                        repository_context=repo_context,
                        engineering_analysis=engineering_analysis,
                        automation=automation,
                    )
                else:
                    comment = format_comment(
                        pred, related, category, repository_context=repo_context,
                    )
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
        app.state.tracker.record_action(
            repo, number, action, workspace_id=payload.get("_workspace_id")
        )

    return {"ok": True, "prediction": pred.as_dict(), "category": category,
            "estimated_resolution": effort,   # API-only by design; see EFFORT_CARD.md
            "related_issues": related,
            "suggested_assignees": suggested_assignees,  # API-only; never assigned
            "missing_info": info_request,
            "llm_analysis": llm_analysis.as_dict() if llm_analysis else None,
            "llm_ml_disagreement": disagreement,
            "repository_context": repo_context.as_dict() if repo_context else None,
            "engineering_analysis": (
                engineering_analysis.as_dict() if engineering_analysis else None
            ),
            "automation": automation.as_dict() if automation else None,
            "actions": actions, "dry_run": s.dry_run}


def _dashboard_analysis_record(
    payload: dict[str, Any], result: dict[str, Any], duration_ms: float
) -> dict[str, Any]:
    """Reduce a processing result to durable, dashboard-safe evidence.

    Issue bodies, retrieved source text, generated comments, credentials,
    and provider errors are intentionally excluded.  Paths and symbols are
    retained because they are the evidence the dashboard must be able to
    audit after the webhook request has finished.
    """
    issue = payload.get("issue") or {}
    repository = payload.get("repository") or {}
    installation = payload.get("installation") or {}
    sender = payload.get("sender") or {}
    prediction = result.get("prediction") or {}
    llm = result.get("llm_analysis") or None
    context = result.get("repository_context") or None

    evidence_status = "disabled"
    persisted_context = None
    if context is not None:
        chunks = []
        for chunk in (context.get("chunks") or [])[:10]:
            chunks.append({
                key: chunk.get(key)
                for key in (
                    "path", "language", "kind", "symbol", "parent_symbol",
                    "start_line", "end_line", "source", "reference", "url", "score",
                )
                if chunk.get(key) not in (None, "")
            })
        note = str(context.get("note") or "")
        if chunks:
            evidence_status = "available"
        elif note.startswith("Repository evidence unavailable:"):
            evidence_status = "unavailable"
        elif context.get("indexed"):
            evidence_status = "insufficient"
        else:
            evidence_status = "not_indexed"
        persisted_context = {
            "indexed": bool(context.get("indexed")),
            "indexing_queued": bool(context.get("indexing_queued")),
            "note": note,
            "relevant_files": list(context.get("relevant_files") or [])[:10],
            "relevant_symbols": list(context.get("relevant_symbols") or [])[:10],
            "chunks": chunks,
        }

    related = [
        {
            key: candidate.get(key)
            for key in ("number", "title", "similarity")
            if candidate.get(key) not in (None, "")
        }
        for candidate in (result.get("related_issues") or [])[:10]
    ]
    missing_information = list((llm or {}).get("missing_information") or [])
    review_reasons = []
    if result.get("llm_ml_disagreement"):
        review_reasons.append("classifier and AI analysis disagree")
    if missing_information:
        review_reasons.append("reported issue is missing requested information")
    if llm is None:
        review_reasons.append("AI reasoning was unavailable")
    if evidence_status == "unavailable":
        review_reasons.append("repository retrieval was unavailable")

    labels = [
        str(label.get("name") if isinstance(label, dict) else label)
        for label in (issue.get("labels") or [])
        if (label.get("name") if isinstance(label, dict) else label)
    ]
    actions = [str(action) for action in (result.get("actions") or [])]

    return {
        "title": str(issue.get("title") or "")[:500],
        "url": str(issue.get("html_url") or ""),
        "issue_state": str(issue.get("state") or "open"),
        "issue_created_at": issue.get("created_at"),
        "issue_updated_at": issue.get("updated_at"),
        "labels": labels[:30],
        "repository_url": str(repository.get("html_url") or ""),
        "repository_private": repository.get("private"),
        "default_branch": str(repository.get("default_branch") or ""),
        "installation_id": installation.get("id"),
        "sender_github_id": sender.get("id"),
        "prediction": {
            key: prediction.get(key)
            for key in (
                "proba_actionable_bug", "threshold", "predicted_label",
                "predicted_class", "model",
            )
            if prediction.get(key) is not None
        },
        "category": result.get("category"),
        "llm_analysis": llm,
        "llm_ml_disagreement": bool(result.get("llm_ml_disagreement")),
        "repository_context": persisted_context,
        "repository_evidence_status": evidence_status,
        "related_issues": related,
        "actions": actions,
        "comment_posted": "comment" in actions,
        "analysis_status": "complete" if llm is not None else "scored_only",
        "needs_maintainer_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "duration_ms": round(duration_ms, 1),
        "dry_run": bool(result.get("dry_run")),
    }


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
    decision = _authorize_issue(app, payload, repo, number)
    if not decision["authorized"]:
        return _authorization_skip_response(decision)
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
    app.state.tracker.record_prediction(
        repo, number, pred.proba, pred.predicted_label,
        workspace_id=decision.get("workspace_id"),
    )
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
    decision = _authorize_issue(app, payload, repo, number)
    if not decision["authorized"]:
        return _authorization_skip_response(decision)
    added = payload.get("action") == "labeled"
    app.state.tracker.record_label_event(
        repo, number, label, added, workspace_id=decision.get("workspace_id")
    )
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
    decision = _authorize_issue(app, payload, repo, number)
    if not decision["authorized"]:
        return _authorization_skip_response(decision)

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

    matched = app.state.tracker.record_outcome(
        repo, number, result.label, workspace_id=decision.get("workspace_id")
    )
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
      ' · per-repo thresholds ' + (health.repo_thresholds_configured ?? 0);
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
