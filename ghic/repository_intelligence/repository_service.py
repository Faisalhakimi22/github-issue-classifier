"""The facade: one object the webhook talks to, one contract it guarantees.

`RepositoryIntelligenceService.get_context()` never raises and never blocks
on indexing. Those two properties are the whole design:

  **Never raises.** Every failure mode -- git missing, clone denied, index
  corrupt, embedding API down, repository empty -- is caught and turned into
  an empty `RepositoryContext`. Issue analysis continues with issue text
  alone, exactly as it did before this module existed. A repository
  intelligence outage degrades the comment; it never costs a webhook.

  **Never blocks.** Cloning and indexing a repository takes seconds to
  minutes; GitHub's webhook delivery window is ~10s. So the read path
  (`get_context`) only ever *reads* an existing index, and a repository that
  has never been indexed gets its indexing enqueued through the injected
  `index_scheduler` while the current issue is analyzed without repository
  context. The first issue on a new repository is the one that pays: it gets
  no code evidence, and every issue after it does.

Dependency injection throughout -- embedder, repo cache, index cache,
indexer, retriever, and scheduler are all constructor arguments with working
defaults. Tests inject fakes; a future Commit/PR intelligence module reuses
the same pieces without inheriting the webhook's assumptions.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .. import utils
from .cache import IndexCache, RepositoryCache, RepositoryCacheError
from .config import RepositoryIntelligenceConfig
from .embeddings import EmbeddingProvider, build_embedding_provider
from .indexer import RepositoryIndexer
from .models import EMPTY_CONTEXT_NOTE, RepositoryContext, RepositoryMetadata
from .retriever import SemanticRetriever

logger = utils.get_logger(__name__)

# Called with the repo full name when an index is missing. Returns True if
# the caller successfully queued an indexing job. app.py wires this to
# QStash; tests pass a lambda; None disables background indexing entirely.
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
    ) -> None:
        self.cfg = cfg or RepositoryIntelligenceConfig()
        self.embedder = embedder or build_embedding_provider(self.cfg)
        self.repo_cache = repo_cache or RepositoryCache(self.cfg)
        self.index_cache = index_cache or IndexCache(self.cfg)
        self.indexer = indexer or RepositoryIndexer(self.embedder, self.cfg)
        self.retriever = retriever or SemanticRetriever(self.embedder, self.cfg)
        self.index_scheduler = index_scheduler

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
        try:
            return self._get_context(repo, title, body, category, predicted_label)
        except Exception as e:  # the contract: analysis continues regardless
            logger.warning("repository intelligence unavailable for %s (%s: %s)",
                           repo, type(e).__name__, e)
            return RepositoryContext(repo=repo, note=EMPTY_CONTEXT_NOTE)

    def _get_context(
        self, repo: str, title: str, body: str, category: str, predicted_label: int | None,
    ) -> RepositoryContext:
        pointer = self.index_cache.get_latest(repo)
        if pointer is None:
            return self._schedule_and_return_empty(repo, "never indexed")

        commit_sha, embedder_name = pointer
        if embedder_name != self.embedder.name:
            # The embedding provider changed under an existing index; its
            # vectors are not comparable to a query from the current one.
            return self._schedule_and_return_empty(repo, "embedder changed")

        loaded = self.index_cache.load(repo, commit_sha, embedder_name)
        if loaded is None:
            return self._schedule_and_return_empty(repo, "index missing or stale")

        store, metadata = loaded
        chunks = self.retriever.retrieve(
            store, title, body, category=category, predicted_label=predicted_label,
        )
        return RepositoryContext(
            repo=repo,
            metadata=metadata,
            chunks=chunks,
            indexed=True,
            note="" if chunks else EMPTY_CONTEXT_NOTE,
        )

    def _schedule_and_return_empty(self, repo: str, reason: str) -> RepositoryContext:
        queued = False
        if self.index_scheduler is not None:
            try:
                queued = bool(self.index_scheduler(repo))
            except Exception as e:  # a queue outage must not surface here
                logger.warning("could not queue indexing for %s: %s", repo, e)
        logger.info("no repository index for %s (%s); queued=%s", repo, reason, queued)
        return RepositoryContext(
            repo=repo, indexed=False, indexing_queued=queued, note=EMPTY_CONTEXT_NOTE,
        )

    # ------------------------------------------------------------------
    # Write path -- runs in the background, never inside a webhook.
    # ------------------------------------------------------------------
    def index_repository(self, repo: str, *, token: str = "", force: bool = False) -> bool:
        """Clone/update, index, persist. Returns True when an index exists
        afterwards. Never raises -- the background job logs and gives up."""
        try:
            checkout = self.repo_cache.ensure(repo, token=token)
        except RepositoryCacheError as e:
            logger.warning("clone/fetch failed for %s: %s", repo, e)
            return False
        except Exception as e:
            logger.warning("unexpected clone failure for %s (%s: %s)", repo, type(e).__name__, e)
            return False

        if not force and self.index_cache.has(repo, checkout.commit_sha, self.embedder.name):
            # Already indexed at this exact commit: refresh the pointer (the
            # index may have been built before the pointer existed) and stop.
            self.index_cache.set_latest(repo, checkout.commit_sha, self.embedder.name)
            logger.info("index for %s already current at %s", repo, checkout.commit_sha[:8])
            return True

        try:
            store, metadata = self.indexer.build(
                repo, checkout.path,
                default_branch=checkout.default_branch, commit_sha=checkout.commit_sha,
            )
        except Exception as e:
            logger.warning("indexing failed for %s (%s: %s)", repo, type(e).__name__, e)
            return False

        self.index_cache.save(repo, checkout.commit_sha, self.embedder.name, store, metadata)
        self.index_cache.set_latest(repo, checkout.commit_sha, self.embedder.name)
        return True

    def index_local_path(
        self, repo: str, root: Path, *, commit_sha: str = "local", default_branch: str = "",
    ) -> RepositoryMetadata | None:
        """Index a directory that's already on disk (CLI, tests, monorepo
        subdirectory). Bypasses git entirely."""
        try:
            store, metadata = self.indexer.build(
                repo, root, default_branch=default_branch, commit_sha=commit_sha,
            )
        except Exception as e:
            logger.warning("local indexing failed for %s (%s: %s)", repo, type(e).__name__, e)
            return None
        self.index_cache.save(repo, commit_sha, self.embedder.name, store, metadata)
        self.index_cache.set_latest(repo, commit_sha, self.embedder.name)
        return metadata

    def is_indexed(self, repo: str) -> bool:
        return self.index_cache.get_latest(repo) is not None
