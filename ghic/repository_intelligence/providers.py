"""Provider resolution: config strings -> concrete infrastructure.

The one place in the engine that maps a provider *name* to a class. Every
other module holds an interface (`VectorStore`, `EmbeddingProvider`,
`RepositoryStateStore`, `IndexQueue`) and has no idea what's behind it,
which is the property that makes "swap the vector database" a config change
rather than a refactor.

`build_service()` also owns the auto-disable rule, which is the most
important behaviour in this module: on an ephemeral filesystem with no
persistent vector backend configured, repository intelligence turns itself
**off** and says why. The alternative -- running anyway -- means every cold
start re-clones and re-embeds a repository whose index is discarded minutes
later. That is slow, costs real money against a paid embedding provider,
and produces no working feature, all while appearing to be enabled. Failing
visibly at startup beats degrading invisibly forever.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from .. import utils
from .config import (
    RepositoryIntelligenceConfig,
    ephemeral_filesystem,
    platform_name,
)
from .embeddings import EmbeddingProvider, build_embedding_provider
from .indexer import VectorStore, build_vector_store
from .queue import IndexQueue, InlineIndexQueue, NullIndexQueue, QStashIndexQueue
from .state import (
    FileStateStore,
    MemoryStateStore,
    PostgresStateStore,
    RepositoryStateStore,
    build_state_store,
)

logger = utils.get_logger(__name__)


class ProviderUnavailable(RuntimeError):
    """A provider was requested by name but can't be constructed."""


# ---------------------------------------------------------------------------
# Vector stores
# ---------------------------------------------------------------------------
def build_vector_store_for(
    repo: str, cfg: RepositoryIntelligenceConfig, dimensions: int, *, expected_chunks: int = 0
) -> VectorStore:
    """A vector store for one repository.

    Per-repo rather than global because the two backends disagree about
    what a "store" is: a local index is a directory per repo, while
    Postgres is one table scoped by a repo column. Constructing per repo is
    the shape that fits both.
    """
    if cfg.vector_provider == "postgres":
        if not cfg.database_url:
            raise ProviderUnavailable(
                "GHIC_VECTOR_PROVIDER=postgres requires a database URL "
                "(GHIC_DATABASE_URL / DATABASE_URL / POSTGRES_URL)"
            )
        from .vector_pg import PostgresVectorStore

        return PostgresVectorStore(cfg.database_url, repo, dimensions)

    if cfg.vector_provider not in ("local", "faiss", ""):
        # Named but unimplemented (qdrant, pinecone, milvus, chroma):
        # refuse rather than silently falling back to a local index that
        # won't survive, which is the failure the operator was trying to
        # avoid by naming a real database.
        raise ProviderUnavailable(
            f"vector provider {cfg.vector_provider!r} is not implemented. "
            "Available: 'local' (filesystem, dev/single-box) or 'postgres' "
            "(pgvector, production). Implement the VectorStore interface in "
            "indexer.py to add another -- see models/REPOSITORY_INTELLIGENCE_CARD.md."
        )
    return build_vector_store(dimensions, expected_chunks=expected_chunks)


# ---------------------------------------------------------------------------
# State stores
# ---------------------------------------------------------------------------
def build_state_store_for(cfg: RepositoryIntelligenceConfig) -> RepositoryStateStore:
    if cfg.state_provider == "postgres":
        if not cfg.database_url:
            raise ProviderUnavailable("GHIC_STATE_PROVIDER=postgres requires a database URL")
        return PostgresStateStore(cfg.database_url)
    if cfg.state_provider == "file":
        return FileStateStore(cfg.cache_dir / "state.json")
    if cfg.state_provider == "memory":
        return MemoryStateStore()
    # "auto": best available, same precedence as the ledger.
    return build_state_store(
        database_url=cfg.database_url, file_path=cfg.cache_dir / "state.json"
    )


# ---------------------------------------------------------------------------
# Queues
# ---------------------------------------------------------------------------
def build_index_queue(
    cfg: RepositoryIntelligenceConfig,
    *,
    qstash_token: str = "",
    callback_url: str = "",
    qstash_region: str = "us-east-1",
    inline_worker: Callable[[str, Any], Any] | None = None,
) -> IndexQueue:
    """Resolve the queue. "auto" prefers QStash, then inline, then none."""
    provider = cfg.queue_provider

    def qstash() -> IndexQueue:
        if not (qstash_token and callback_url):
            raise ProviderUnavailable(
                "queue provider 'qstash' requires QSTASH_TOKEN and GHIC_PUBLIC_BASE_URL"
            )
        return QStashIndexQueue(qstash_token, callback_url, region=qstash_region)

    if provider == "qstash":
        return qstash()
    if provider == "inline":
        if inline_worker is None:
            raise ProviderUnavailable("queue provider 'inline' requires a worker callable")
        return InlineIndexQueue(inline_worker)
    if provider == "none":
        return NullIndexQueue()
    if provider not in ("auto", ""):
        raise ProviderUnavailable(
            f"queue provider {provider!r} is not implemented. Available: "
            "'qstash', 'inline' (dev only), 'none'. Implement IndexQueue in "
            "queue.py to add Redis/Celery/RabbitMQ."
        )

    if qstash_token and callback_url:
        return QStashIndexQueue(qstash_token, callback_url, region=qstash_region)
    if inline_worker is not None:
        logger.info("no queue configured; using an in-process worker (development only)")
        return InlineIndexQueue(inline_worker)
    return NullIndexQueue()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def build_embeddings(cfg: RepositoryIntelligenceConfig) -> EmbeddingProvider:
    """Delegates to embeddings.build_embedding_provider.

    Re-exported here so callers have one import for "resolve my providers"
    and so a future provider registry has a single place to live.
    """
    return build_embedding_provider(cfg)


# ---------------------------------------------------------------------------
# The whole service
# ---------------------------------------------------------------------------
def build_service(
    cfg: RepositoryIntelligenceConfig | None = None,
    *,
    qstash_token: str = "",
    callback_url: str = "",
    qstash_region: str = "us-east-1",
    database_url: str = "",
) -> Any | None:
    """Fully-wired `RepositoryIntelligenceService`, or None if it shouldn't run.

    None is a supported, documented outcome, not a failure: the caller
    treats it exactly like the feature being switched off, which the
    webhook already handles (repo_context stays None, the comment renders
    no evidence section, analysis continues on issue text alone).
    """
    from .repository_service import RepositoryIntelligenceService

    cfg = cfg or RepositoryIntelligenceConfig.from_env()
    if database_url and not cfg.database_url:
        cfg = replace(cfg, database_url=database_url)

    platform = platform_name()
    if ephemeral_filesystem() and not cfg.uses_persistent_vectors:
        logger.warning(
            "repository intelligence is DISABLED on %s: this platform has an "
            "ephemeral filesystem and no persistent vector store is configured, "
            "so every cold start would re-index from scratch and throw the "
            "result away. Set GHIC_VECTOR_PROVIDER=postgres with a database URL "
            "to enable it here, or run it on a host with a volume. Issue "
            "analysis continues normally without repository evidence.",
            platform,
        )
        return None

    try:
        embedder = build_embeddings(cfg)
        state_store = build_state_store_for(cfg)
    except ProviderUnavailable as e:
        logger.warning("repository intelligence disabled: %s", e)
        return None

    service = RepositoryIntelligenceService(
        cfg, embedder=embedder, state_store=state_store,
    )

    # The inline worker closes over the service, so the queue can only be
    # built after it exists.
    try:
        service.index_queue = build_index_queue(
            cfg,
            qstash_token=qstash_token,
            callback_url=callback_url,
            qstash_region=qstash_region,
            inline_worker=lambda repo, installation_id: service.index_repository(repo),
        )
    except ProviderUnavailable as e:
        logger.warning("index queue unavailable (%s); auto-indexing is off", e)
        service.index_queue = NullIndexQueue()

    logger.info(
        "repository intelligence ready on %s: vectors=%s state=%s(durable=%s) "
        "queue=%s(durable=%s) embedder=%s",
        platform, cfg.vector_provider, type(state_store).__name__, state_store.durable,
        service.index_queue.name, service.index_queue.durable, embedder.name,
    )
    return service


def health_snapshot(service: Any | None, cfg: RepositoryIntelligenceConfig | None = None) -> dict:
    """What /healthz reports about this subsystem.

    Surfaces durability explicitly, because "enabled but nothing persists"
    is the failure mode most likely to go unnoticed -- it looks identical
    to a healthy deploy until someone asks why no repository is ever READY.
    """
    if service is None:
        cfg = cfg or RepositoryIntelligenceConfig.from_env()
        return {
            "enabled": False,
            "platform": platform_name(),
            "reason": (
                "ephemeral filesystem without a persistent vector store"
                if ephemeral_filesystem() and not cfg.uses_persistent_vectors
                else "not configured"
            ),
        }
    # Whether this runtime can index at all. Cloning shells out to git, and
    # serverless Python runtimes generally don't ship it -- so a deploy can
    # be fully "enabled" (persistent vectors, durable queue, healthy state)
    # and still never produce an index, because every queued job dies at
    # the clone. That is a confusing failure to diagnose from the outside,
    # so it is reported rather than inferred: can_index=false means index
    # out of band (python -m ghic.repo_index) and let this deploy serve
    # reads, which needs no git.
    import shutil

    git_available = shutil.which("git") is not None

    embedding = service.embedder.metadata()
    return {
        "enabled": True,
        "platform": platform_name(),
        "vector_provider": service.cfg.vector_provider,
        "embedding_provider": embedding.provider,
        "embedding_model": embedding.model,
        "embedding_dimensions": embedding.dimensions,
        "embedding_signature": embedding.signature,
        "min_similarity": service.cfg.min_similarity,
        "auto_index": service.cfg.auto_index,
        "state_durable": service.state_store.durable if service.state_store else False,
        "queue": service.index_queue.name if service.index_queue else None,
        "queue_durable": service.index_queue.durable if service.index_queue else False,
        "incremental_indexing": service.cfg.incremental_indexing,
        "can_index": git_available,
        "indexing_note": (
            ""
            if git_available
            else "git is not available in this runtime; index out of band with "
                 "`python -m ghic.repo_index` against the same database. Retrieval "
                 "works without it."
        ),
    }
