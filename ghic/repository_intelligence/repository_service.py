"""The facade: one object the webhook talks to, one contract it guarantees.

`RepositoryIntelligenceService.get_context()` never raises and never blocks
on indexing. Those two properties are the whole design:

  **Never raises.** Every failure mode -- git missing, clone denied, index
  corrupt, embedding API down, database unreachable, repository empty -- is
  caught and turned into an empty `RepositoryContext`. Issue analysis
  continues with issue text alone, exactly as it did before this module
  existed. A repository intelligence outage degrades the comment; it never
  costs a webhook.

  **Never blocks.** Cloning and indexing takes seconds to minutes; GitHub's
  webhook delivery window is ~10s. So the read path only ever *reads* an
  existing index, and an unindexed repository gets its indexing enqueued
  while the current issue is analyzed without repository context. The first
  issue on a new repository is the one that pays.

Phase 1.5 added durable state, a swappable queue, incremental re-indexing,
and metrics -- all as optional constructor arguments. A service built the
Phase 1 way (`RepositoryIntelligenceService(cfg)`) still works and still
behaves the same, which is what makes "favour extension over replacement"
true here rather than just claimed: the new collaborators default to None
and every code path checks for that.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .. import utils
from .cache import IndexCache, RepositoryCache, RepositoryCacheError
from .config import RepositoryIntelligenceConfig
from .embeddings import EmbeddingProvider, build_embedding_provider
from .incremental import diff_change_set, should_use_incremental
from .indexer import RepositoryIndexer
from .metrics import METRICS, RepositoryIntelligenceMetrics, set_repo_context
from .models import EMPTY_CONTEXT_NOTE, RepositoryContext, RepositoryMetadata
from .queue import IndexQueue
from .retriever import SemanticRetriever
from .state import (
    STALE_IN_FLIGHT_SECONDS,
    RepositoryRecord,
    RepositoryState,
    RepositoryStateStore,
)

logger = utils.get_logger(__name__)

# Phase 1 signature, kept working: a bare callable scheduler instead of an
# IndexQueue. Nothing in the tree still passes one, but a caller outside it
# might, and honouring it costs one branch.
IndexScheduler = Callable[[str], bool]


class RepositoryIntelligenceService:
    def __init__(
        self,
        cfg: RepositoryIntelligenceConfig | None = None,
        *,
        embedder: EmbeddingProvider | None = None,
        repo_cache: RepositoryCache | None = None,
        index_cache: IndexCache | None = None,
        indexer: RepositoryIndexer | None = None,
        retriever: SemanticRetriever | None = None,
        index_scheduler: IndexScheduler | None = None,
        state_store: RepositoryStateStore | None = None,
        index_queue: IndexQueue | None = None,
        metrics: RepositoryIntelligenceMetrics | None = None,
    ) -> None:
        self.cfg = cfg or RepositoryIntelligenceConfig()
        self.embedder = embedder or build_embedding_provider(self.cfg)
        self.repo_cache = repo_cache or RepositoryCache(self.cfg)
        self.index_cache = index_cache or IndexCache(self.cfg)
        self.indexer = indexer or RepositoryIndexer(self.embedder, self.cfg)
        self.retriever = retriever or SemanticRetriever(self.embedder, self.cfg)
        self.index_scheduler = index_scheduler
        self.state_store = state_store
        self.index_queue = index_queue
        self.metrics = metrics or METRICS

    # ------------------------------------------------------------------
    # Read path -- runs inside the webhook. Fast, or it returns nothing.
    # ------------------------------------------------------------------
    def get_context(
        self,
        repo: str,
        title: str,
        body: str,
        *,
        category: str = "",
        predicted_label: int | None = None,
    ) -> RepositoryContext:
        """Retrieve repository evidence for one issue. Never raises."""
        set_repo_context(repo)
        self.metrics.increment("retrievals_total")
        try:
            with self.metrics.timed("retrieval"):
                context = self._get_context(repo, title, body, category, predicted_label)
        except Exception as e:  # the contract: analysis continues regardless
            self.metrics.increment("retrieval_errors")
            logger.warning("repository intelligence unavailable for %s (%s: %s)",
                           repo, type(e).__name__, e)
            return RepositoryContext(repo=repo, note=EMPTY_CONTEXT_NOTE)

        if not context.is_empty:
            self.metrics.increment("retrievals_with_results")
        if self.state_store is not None and context.indexed:
            self.state_store.record_retrieval(repo, hit=not context.is_empty)
        return context

    def _get_context(
        self, repo: str, title: str, body: str, category: str, predicted_label: int | None,
    ) -> RepositoryContext:
        if not self.cfg.vector_search_enabled:
            return RepositoryContext(repo=repo, note=EMPTY_CONTEXT_NOTE)

        record = self.state_store.get(repo) if self.state_store is not None else None

        # With a state store, lifecycle decides whether to search at all --
        # state is authoritative, never inferred from the filesystem.
        if record is not None and not record.state.is_searchable:
            return self._schedule_and_return_empty(repo, f"state={record.state.value}", record)

        store, metadata = self._load_index(repo, record)
        if store is None:
            return self._schedule_and_return_empty(repo, "no usable index", record)

        chunks = self.retriever.retrieve(
            store, title, body, category=category, predicted_label=predicted_label,
        )
        return RepositoryContext(
            repo=repo, metadata=metadata, chunks=chunks, indexed=True,
            note="" if chunks else EMPTY_CONTEXT_NOTE,
        )

    def _load_index(
        self, repo: str, record: RepositoryRecord | None
    ) -> tuple[Any, RepositoryMetadata | None]:
        """Open this repo's index, whichever backend holds it."""
        self.metrics.increment("index_cache_lookups")

        if self.cfg.vector_provider == "postgres":
            from .vector_pg import PostgresVectorStore

            store = PostgresVectorStore(
                self.cfg.database_url, repo, self.embedder.dimensions
            )
            if not store.size:
                return None, None
            self.metrics.increment("index_cache_hits")
            metadata = (
                RepositoryMetadata.from_dict(record.metadata_json)
                if record and record.metadata_json else None
            )
            return store, metadata

        pointer = self.index_cache.get_latest(repo)
        if pointer is None:
            return None, None
        commit_sha, embedder_name = pointer
        if embedder_name != self.embedder.name:
            # The embedding provider changed under an existing index; its
            # vectors aren't comparable to a query from the current one.
            return None, None
        loaded = self.index_cache.load(repo, commit_sha, embedder_name)
        if loaded is None:
            return None, None
        self.metrics.increment("index_cache_hits")
        return loaded

    def _schedule_and_return_empty(
        self, repo: str, reason: str, record: RepositoryRecord | None = None,
    ) -> RepositoryContext:
        queued = self._request_indexing(repo, record, reason)
        logger.info("no repository index for %s (%s); queued=%s", repo, reason, queued)
        return RepositoryContext(
            repo=repo, indexed=False, indexing_queued=queued, note=EMPTY_CONTEXT_NOTE,
        )

    def _request_indexing(
        self, repo: str, record: RepositoryRecord | None, reason: str,
    ) -> bool:
        """Queue an index job unless one is already in flight.

        The in-flight check is what stops a busy repository from queueing a
        job per issue. It trusts the state store's timestamp only up to
        STALE_IN_FLIGHT_SECONDS, so a worker that died mid-index doesn't
        wedge a repository in INDEXING forever with nothing ever retrying.
        """
        if not self.cfg.auto_index:
            return False
        if record is not None and record.state.is_in_flight:
            age = time.time() - record.updated_at
            if age < STALE_IN_FLIGHT_SECONDS:
                return False
            logger.warning(
                "%s has been %s for %.0f minutes; re-queueing (previous worker likely died)",
                repo, record.state.value, age / 60,
            )

        queued = False
        try:
            if self.index_queue is not None:
                queued = self.index_queue.enqueue(repo, reason=reason)
            elif self.index_scheduler is not None:      # Phase 1 compatibility
                queued = bool(self.index_scheduler(repo))
        except Exception as e:  # a queue outage must not surface here
            logger.warning("could not queue indexing for %s: %s", repo, e)
            return False

        if queued and self.state_store is not None:
            self.state_store.mark(repo, RepositoryState.QUEUED)
        return queued

    # ------------------------------------------------------------------
    # Write path -- runs in the background, never inside a webhook.
    # ------------------------------------------------------------------
    def index_repository(self, repo: str, *, token: str = "", force: bool = False) -> bool:
        """Clone/update, index, persist. Returns True when an index exists
        afterwards. Never raises -- the background job logs and gives up."""
        set_repo_context(repo)
        started = time.time()
        record = self.state_store.get(repo) if self.state_store is not None else None
        # UPDATING, not INDEXING, when a usable index already exists: it
        # stays searchable while the update runs.
        in_progress = (
            RepositoryState.UPDATING
            if record and record.state == RepositoryState.READY
            else RepositoryState.INDEXING
        )
        self._mark(repo, in_progress)

        try:
            checkout = self.repo_cache.ensure(repo, token=token)
        except (RepositoryCacheError, Exception) as e:
            self.metrics.increment("index_jobs_failed")
            self._mark(repo, RepositoryState.FAILED, error=f"clone failed: {e}")
            logger.warning("clone/fetch failed for %s: %s", repo, e)
            return False

        previous_sha = record.indexed_commit_sha if record else ""
        embedder_changed = bool(
            record and record.embedding_provider and
            record.embedding_provider != self.embedder.name
        )

        # Nothing changed: the cheapest possible outcome, and the common one
        # on a repo that gets many issues and few pushes.
        if (
            not force and not embedder_changed
            and previous_sha == checkout.commit_sha
            and self._index_exists(repo, checkout.commit_sha)
        ):
            self.metrics.increment("index_jobs_skipped_unchanged")
            self._mark(repo, RepositoryState.READY, indexed_commit_sha=checkout.commit_sha)
            logger.info("index for %s already current at %s", repo, checkout.commit_sha[:8])
            return True

        try:
            with self.metrics.timed("indexing"):
                ok = self._run_index(
                    repo, checkout, previous_sha,
                    force=force or embedder_changed,
                )
        except Exception as e:
            self.metrics.increment("index_jobs_failed")
            self._mark(repo, RepositoryState.FAILED, error=f"indexing failed: {e}")
            logger.warning("indexing failed for %s (%s: %s)", repo, type(e).__name__, e)
            return False

        if ok:
            self.metrics.increment("index_jobs_completed")
            self.metrics.observe("index_job_wallclock", time.time() - started)
        return ok

    def _run_index(
        self, repo: str, checkout: Any, previous_sha: str, *, force: bool
    ) -> bool:
        """Full or incremental index, depending on what changed."""
        change_set = None
        if self.cfg.incremental_indexing and not force and previous_sha:
            change_set = diff_change_set(
                checkout.path, previous_sha, checkout.commit_sha,
                self.cfg.git_timeout_seconds,
            )
            if change_set is not None and change_set.is_empty:
                self._mark(repo, RepositoryState.READY,
                           indexed_commit_sha=checkout.commit_sha)
                return True

        incremental = (
            self.cfg.vector_provider == "postgres"      # needs per-path deletes
            and should_use_incremental(change_set, self.cfg.max_incremental_files)
        )
        if incremental:
            return self._index_incremental(repo, checkout, change_set)
        return self._index_full(repo, checkout)

    def _index_full(self, repo: str, checkout: Any) -> bool:
        store, metadata = self.indexer.build(
            repo, checkout.path,
            default_branch=checkout.default_branch, commit_sha=checkout.commit_sha,
        )
        if self.cfg.vector_provider == "postgres":
            from .vector_pg import PostgresVectorStore

            target = PostgresVectorStore(
                self.cfg.database_url, repo, self.embedder.dimensions
            )
            target.clear()          # replace wholesale; no partial old state
            chunks = list(getattr(store, "_chunks", []))
            vectors = getattr(store, "_vectors", None)
            if chunks and vectors is not None:
                target.add(vectors, chunks)
        else:
            self.index_cache.save(
                repo, checkout.commit_sha, self.embedder.name, store, metadata
            )
            self.index_cache.set_latest(repo, checkout.commit_sha, self.embedder.name)

        self.metrics.increment("index_full_rebuilds")
        self._mark_ready(repo, checkout, metadata)
        return True

    def _index_incremental(self, repo: str, checkout: Any, change_set: Any) -> bool:
        """Re-embed only what changed, and drop what disappeared.

        Postgres-only: it needs per-path deletes, which a monolithic local
        index file can't do without rewriting itself entirely -- at which
        point it isn't incremental.
        """
        from . import parser
        from .vector_pg import PostgresVectorStore

        store = PostgresVectorStore(self.cfg.database_url, repo, self.embedder.dimensions)
        removed = store.delete_paths(change_set.stale_paths)

        chunks = []
        for relative in change_set.changed:
            path = checkout.path / relative
            if not path.is_file() or not parser.is_indexable(path, checkout.path, self.cfg):
                continue
            text = parser.read_text(path)
            if text:
                chunks.extend(parser.chunk_file(repo, relative, text, self.cfg))

        if chunks:
            vectors = self.embedder.embed_documents([c.embedding_text() for c in chunks])
            store.add(vectors, chunks)

        # Metadata (language mix, frameworks, README) is cheap next to
        # embedding and can shift with any change, so it's always refreshed.
        files = parser.walk_repository(checkout.path, self.cfg)
        metadata = replace(
            parser.detect_metadata(
                checkout.path, repo, files, self.cfg,
                default_branch=checkout.default_branch, commit_sha=checkout.commit_sha,
            ),
            chunk_count=store.size,
        )

        self.metrics.increment("index_incremental_updates")
        self.metrics.increment("index_chunks_removed", removed)
        self.metrics.increment("index_chunks_added", len(chunks))
        logger.info(
            "incremental index for %s: %d files changed, %d deleted, "
            "%d chunks removed, %d added",
            repo, len(change_set.changed), len(change_set.deleted), removed, len(chunks),
        )
        self._mark_ready(repo, checkout, metadata)
        return True

    def _index_exists(self, repo: str, commit_sha: str) -> bool:
        if self.cfg.vector_provider == "postgres":
            from .vector_pg import PostgresVectorStore

            return PostgresVectorStore(
                self.cfg.database_url, repo, self.embedder.dimensions
            ).size > 0
        return self.index_cache.has(repo, commit_sha, self.embedder.name)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _mark(self, repo: str, state: RepositoryState, **fields: Any) -> None:
        if self.state_store is None:
            return
        try:
            self.state_store.mark(repo, state, **fields)
        except Exception as e:  # state is observability, never the critical path
            logger.debug("could not record state %s for %s: %s", state.value, repo, e)

    def _mark_ready(self, repo: str, checkout: Any, metadata: RepositoryMetadata) -> None:
        self._mark(
            repo, RepositoryState.READY,
            indexed_commit_sha=checkout.commit_sha,
            default_branch=checkout.default_branch,
            embedding_provider=self.embedder.name,
            vector_backend=self.cfg.vector_provider,
            chunk_count=metadata.chunk_count,
            file_count=metadata.file_count,
            primary_language=metadata.primary_language,
            frameworks=metadata.frameworks,
            readme_summary=metadata.readme_summary,
            indexed_at=time.time(),
            metadata_json=metadata.as_dict(),
        )

    # ------------------------------------------------------------------
    # Local / CLI entry points
    # ------------------------------------------------------------------
    def index_local_path(
        self, repo: str, root: Path, *, commit_sha: str = "local", default_branch: str = "",
    ) -> RepositoryMetadata | None:
        """Index a directory already on disk (CLI, tests, monorepo
        subdirectory). Bypasses git entirely."""
        try:
            store, metadata = self.indexer.build(
                repo, root, default_branch=default_branch, commit_sha=commit_sha,
            )
        except Exception as e:
            logger.warning("local indexing failed for %s (%s: %s)", repo, type(e).__name__, e)
            return None

        if self.cfg.vector_provider == "postgres":
            from .vector_pg import PostgresVectorStore

            target = PostgresVectorStore(
                self.cfg.database_url, repo, self.embedder.dimensions
            )
            target.clear()
            chunks = list(getattr(store, "_chunks", []))
            vectors = getattr(store, "_vectors", None)
            if chunks and vectors is not None:
                target.add(vectors, chunks)
        else:
            self.index_cache.save(repo, commit_sha, self.embedder.name, store, metadata)
            self.index_cache.set_latest(repo, commit_sha, self.embedder.name)

        self._mark(
            repo, RepositoryState.READY,
            indexed_commit_sha=commit_sha, default_branch=default_branch,
            embedding_provider=self.embedder.name, vector_backend=self.cfg.vector_provider,
            chunk_count=metadata.chunk_count, file_count=metadata.file_count,
            primary_language=metadata.primary_language, frameworks=metadata.frameworks,
            readme_summary=metadata.readme_summary, indexed_at=time.time(),
            metadata_json=metadata.as_dict(),
        )
        return metadata

    def is_indexed(self, repo: str) -> bool:
        if self.state_store is not None:
            record = self.state_store.get(repo)
            if record is not None:
                return record.state.is_searchable
        return self.index_cache.get_latest(repo) is not None

    def status(self, repo: str) -> RepositoryRecord | None:
        return self.state_store.get(repo) if self.state_store is not None else None

    def list_repositories(self, limit: int = 100) -> list[RepositoryRecord]:
        return self.state_store.list(limit=limit) if self.state_store is not None else []

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def forget(self, repo: str) -> None:
        """Remove everything for one repository -- used when an App
        installation is deleted or a repo is removed."""
        try:
            self.index_cache.purge(repo)
            self.repo_cache.remove(repo)
            if self.cfg.vector_provider == "postgres" and self.cfg.database_url:
                from .vector_pg import PostgresVectorStore

                PostgresVectorStore(
                    self.cfg.database_url, repo, self.embedder.dimensions
                ).clear()
        except Exception as e:
            logger.warning("cleanup failed for %s: %s", repo, e)
        if self.state_store is not None:
            self.state_store.delete(repo)

    def cleanup(self, *, stale_days: int | None = None, dry_run: bool = False) -> dict[str, Any]:
        """Drop indexes for repositories nothing has touched in a while.

        Storage is the cost that grows silently: a repository indexed once
        after an App install and never queried again keeps its chunks
        forever. Uses `last_accessed_at` rather than `indexed_at` on
        purpose -- a stable repository that is still being searched should
        not be evicted just because its code hasn't changed.
        """
        stale_days = stale_days if stale_days is not None else self.cfg.stale_index_days
        cutoff = time.time() - stale_days * 86400
        removed: list[str] = []
        if self.state_store is None:
            return {"removed": [], "reason": "no state store; cleanup needs durable state"}

        for record in self.state_store.list(limit=10_000):
            touched = max(record.last_accessed_at, record.indexed_at, record.updated_at)
            if record.state == RepositoryState.ARCHIVED or (touched and touched < cutoff):
                removed.append(record.repo)
                if not dry_run:
                    self.forget(record.repo)
        if removed:
            logger.info("cleanup removed %d stale repository index(es)", len(removed))
        return {"removed": removed, "stale_days": stale_days, "dry_run": dry_run}
