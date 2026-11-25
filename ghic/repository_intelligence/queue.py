"""Index-job queue abstraction.

Phase 1 called QStash directly from a lambda wired into the service. That
works but hardcodes one provider into the webhook module, which is exactly
what this phase is meant to undo.

`IndexQueue` is the seam. Two implementations ship:

  `QStashIndexQueue` -- production. Wraps the QStash publisher the service
  already uses for async issue processing, so there is one queue to operate
  rather than two.
  `GitHubActionsIndexQueue` -- production on Vercel, which has no git:
  starts a workflow run on a runner that does.
  `InlineIndexQueue`  -- development and single-box Docker. Runs the job on
  a background *thread* rather than pretending to enqueue it.

`InlineIndexQueue` deserves its warning label, which is why it isn't the
default anywhere a real deploy would land: a thread dies with its process,
so a restart mid-index loses the job silently, and nothing retries it. It
exists so `docker run` and a laptop work without standing up a queue, not
as a production option. The state store's STALE_IN_FLIGHT_SECONDS is what
eventually rescues a repository wedged in INDEXING by exactly this.

Redis/Celery/RabbitMQ are not implemented here, deliberately. Each is a
broker this test suite cannot exercise, and an unverified adapter that
looks plausible is worse than a documented interface -- implementing
`IndexQueue` is three methods, and REPOSITORY_INTELLIGENCE_CARD.md states
the contract a new one must satisfy.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from inspect import Parameter, signature
from typing import Any

from .. import utils
from .metrics import METRICS, RepositoryIntelligenceMetrics, get_correlation_id

logger = utils.get_logger(__name__)


class IndexQueue(ABC):
    """Somewhere to put "index this repository" so a webhook doesn't wait.

    Implementations must not raise: `enqueue` returns False when the job
    could not be queued, and the caller continues without repository
    context. A queue outage degrades evidence, never a webhook.
    """

    @abstractmethod
    def enqueue(
        self, repo: str, *, installation_id: Any = None, reason: str = "",
        workspace_id: Any = None,
    ) -> bool:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def durable(self) -> bool:
        """False when a process restart loses queued work. Surfaced in
        /healthz so this is visible before it matters, not after."""
        return True

    def depth(self) -> int | None:
        """Pending jobs, or None when the backend can't report it. QStash
        can't cheaply, which is why the return is optional rather than a
        lie."""
        return None


class QStashIndexQueue(IndexQueue):
    """Durable delivery through Upstash QStash, with retries and signature
    verification handled by the existing endpoint."""

    def __init__(
        self,
        token: str,
        callback_url: str,
        *,
        region: str = "us-east-1",
        metrics: RepositoryIntelligenceMetrics | None = None,
    ) -> None:
        if not token or not callback_url:
            raise ValueError("token and callback_url are required")
        self.token = token
        self.callback_url = callback_url
        self.region = region
        self.metrics = metrics or METRICS

    def enqueue(
        self, repo: str, *, installation_id: Any = None, reason: str = "",
        workspace_id: Any = None,
    ) -> bool:
        from ..service.qstash import publish

        try:
            publish(
                destination_url=self.callback_url,
                payload={
                    "repo": repo,
                    "installation_id": installation_id,
                    # Diagnostic only. The worker derives the authoritative
                    # workspace from installation/repository PostgreSQL state.
                    "workspace_id": workspace_id,
                    "reason": reason,
                    # Carried so the worker's logs join up with the webhook
                    # that triggered it -- the two run minutes and a process
                    # boundary apart.
                    "correlation_id": get_correlation_id(),
                },
                token=self.token,
                region=self.region,
            )
            self.metrics.increment("index_jobs_enqueued")
            return True
        except Exception as e:
            self.metrics.increment("index_jobs_enqueue_failed")
            logger.warning("could not enqueue index job for %s: %s", repo, e)
            return False


class GitHubActionsIndexQueue(IndexQueue):
    """Durable delivery by starting a GitHub Actions workflow run.

    Exists because the Vercel runtime has no `git`, so a QStash callback
    into the same deploy can never clone anything. An Actions runner has
    git, writes the index to the same Postgres the service reads, and GitHub
    keeps the run (and its retry button) whether or not anything here
    survives.

    The workflow (`.github/workflows/index-on-demand.yml`) receives only the
    repository and installation id. It re-derives authorization from the
    database itself: a dispatch is a request, never a grant.
    """

    API = "https://api.github.com"

    def __init__(
        self,
        token: str,
        dispatch_repo: str,
        *,
        workflow: str = "index-on-demand.yml",
        ref: str = "main",
        timeout: float = 10.0,
        metrics: RepositoryIntelligenceMetrics | None = None,
        session: Any = None,
    ) -> None:
        if not token or not dispatch_repo:
            raise ValueError("token and dispatch_repo are required")
        self.token = token
        self.dispatch_repo = dispatch_repo
        self.workflow = workflow
        self.ref = ref
        self.timeout = timeout
        self.metrics = metrics or METRICS
        if session is None:
            import requests

            session = requests.Session()
        self._session = session

    def enqueue(
        self, repo: str, *, installation_id: Any = None, reason: str = "",
        workspace_id: Any = None,
    ) -> bool:
        if installation_id in (None, ""):
            # The worker refuses a job it cannot authorize, so dispatching
            # one would only burn runner minutes.
            logger.info("not dispatching index for %s: no installation id", repo)
            return False
        url = (
            f"{self.API}/repos/{self.dispatch_repo}/actions/workflows/"
            f"{self.workflow}/dispatches"
        )
        try:
            resp = self._session.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={
                    "ref": self.ref,
                    "inputs": {
                        "repo": repo,
                        "installation_id": str(installation_id),
                        "reason": (reason or "")[:100],
                        "correlation_id": get_correlation_id() or "",
                    },
                },
                timeout=self.timeout,
            )
        except Exception as e:
            self.metrics.increment("index_jobs_enqueue_failed")
            logger.warning("could not dispatch index workflow for %s: %s", repo, e)
            return False
        if resp.status_code != 204:
            self.metrics.increment("index_jobs_enqueue_failed")
            logger.warning(
                "index workflow dispatch for %s returned HTTP %s: %s",
                repo, resp.status_code, (resp.text or "")[:200],
            )
            return False
        self.metrics.increment("index_jobs_enqueued")
        return True


class InlineIndexQueue(IndexQueue):
    """Background thread. Development only -- see the module docstring."""

    def __init__(
        self,
        worker: Callable[..., Any],
        *,
        max_concurrent: int = 1,
        metrics: RepositoryIntelligenceMetrics | None = None,
    ) -> None:
        self.worker = worker
        self.metrics = metrics or METRICS
        try:
            parameters = signature(worker).parameters.values()
            self._worker_accepts_workspace = any(
                parameter.kind == Parameter.VAR_POSITIONAL
                or parameter.kind == Parameter.VAR_KEYWORD
                for parameter in parameters
            ) or len(signature(worker).parameters) >= 3
        except (TypeError, ValueError):
            self._worker_accepts_workspace = True
        # A semaphore, not a pool: indexing is IO- and CPU-heavy, and two
        # concurrent clones of a large repository on a laptop is already
        # more than enough.
        self._slots = threading.Semaphore(max_concurrent)
        self._active = 0
        self._lock = threading.Lock()

    @property
    def durable(self) -> bool:
        return False

    def depth(self) -> int | None:
        with self._lock:
            return self._active

    def enqueue(
        self, repo: str, *, installation_id: Any = None, reason: str = "",
        workspace_id: Any = None,
    ) -> bool:
        def run() -> None:
            try:
                if self._worker_accepts_workspace:
                    self.worker(repo, installation_id, workspace_id)
                else:
                    # Preserve the small two-argument development adapter
                    # contract used by existing local callers.
                    self.worker(repo, installation_id)
            except Exception as e:
                self.metrics.increment("index_jobs_failed")
                logger.warning("inline index job failed for %s: %s", repo, e)
            finally:
                with self._lock:
                    self._active -= 1
                self._slots.release()

        if not self._slots.acquire(blocking=False):
            logger.info("inline index queue is busy; skipping %s (it will retry)", repo)
            return False
        with self._lock:
            self._active += 1
        threading.Thread(target=run, name=f"ghic-index-{repo}", daemon=True).start()
        self.metrics.increment("index_jobs_enqueued")
        return True


class NullIndexQueue(IndexQueue):
    """Explicitly no background indexing.

    The correct configuration for a deploy that indexes out of band (a CI
    step, `python -m ghic.repo_index`, a cron job) -- and honest about it,
    where silently doing nothing would leave an operator wondering why
    repositories never become READY.
    """

    def enqueue(
        self, repo: str, *, installation_id: Any = None, reason: str = "",
        workspace_id: Any = None,
    ) -> bool:
        logger.info(
            "background indexing is disabled; %s will not be indexed automatically "
            "(run: python -m ghic.repo_index --repo %s)", repo, repo,
        )
        return False
