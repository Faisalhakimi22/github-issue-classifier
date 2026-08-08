"""Tests for Phase 1.5 production infrastructure: lifecycle state, queues,
incremental indexing, secret exclusion, metrics, and provider resolution.

Postgres-backed paths (state store, pgvector) are exercised through their
interfaces with the in-memory/file backends; the Postgres classes
themselves need a live database and are covered by the deployment
checklist, not by mocks that would only assert that pg8000 was called the
way the test expected.
"""
from __future__ import annotations

import time

import pytest

from ghic.repository_intelligence import (
    NullIndexQueue,
    RepositoryIntelligenceConfig,
    RepositoryIntelligenceService,
    RepositoryState,
    UNAVAILABLE_CONTEXT_NOTE,
    build_embedding_provider,
)
from ghic.repository_intelligence.config import (
    ephemeral_filesystem,
    is_sensitive_path,
    platform_name,
    redact_secrets,
)
from ghic.repository_intelligence.incremental import ChangeSet, should_use_incremental
from ghic.repository_intelligence.metrics import RepositoryIntelligenceMetrics
from ghic.repository_intelligence.providers import (
    ProviderUnavailable,
    build_index_queue,
    build_service,
    build_state_store_for,
    build_vector_store_for,
    health_snapshot,
)
from ghic.repository_intelligence.queue import InlineIndexQueue
from ghic.repository_intelligence.state import (
    STALE_IN_FLIGHT_SECONDS,
    FileStateStore,
    MemoryStateStore,
    RepositoryRecord,
)


def test_openrouter_embedding_config_reuses_existing_openrouter_key(monkeypatch):
    monkeypatch.setenv("GHIC_REPO_INTEL_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv(
        "GHIC_REPO_INTEL_EMBEDDING_MODEL", "mistralai/codestral-embed-2505"
    )
    monkeypatch.setenv("GHIC_REPO_INTEL_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv(
        "GHIC_REPO_INTEL_EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1"
    )
    monkeypatch.setenv("GHIC_REPO_AUTO_INDEX", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "existing-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "key-for-a-different-endpoint")
    monkeypatch.delenv("GHIC_REPO_INTEL_EMBEDDING_API_KEY", raising=False)

    cfg = RepositoryIntelligenceConfig.from_env()

    assert cfg.embedding_api_key == "existing-openrouter-key"
    assert cfg.embedding_provider == "openai"
    assert cfg.embedding_model == "mistralai/codestral-embed-2505"
    assert cfg.embedding_dimensions == 1536
    assert (
        build_embedding_provider(cfg).metadata().signature
        == "openai:mistralai/codestral-embed-2505:1536"
    )
    assert cfg.auto_index is False


def test_openrouter_key_is_not_used_for_other_embedding_endpoints(monkeypatch):
    monkeypatch.setenv("GHIC_REPO_INTEL_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv(
        "GHIC_REPO_INTEL_EMBEDDING_BASE_URL", "https://api.openai.com/v1"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "llm-only-key")
    monkeypatch.delenv("GHIC_REPO_INTEL_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert RepositoryIntelligenceConfig.from_env().embedding_api_key == ""


@pytest.fixture
def cfg(tmp_path):
    return RepositoryIntelligenceConfig(cache_dir=tmp_path / "_cache")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text(
        "def handler(request):\n    return process(request)\n", encoding="utf-8"
    )
    (root / "README.md").write_text("# Demo\n\nA demo project for tests.\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Lifecycle state
# ---------------------------------------------------------------------------
class TestRepositoryState:
    def test_only_ready_and_updating_are_searchable(self):
        assert RepositoryState.READY.is_searchable
        # Searchable on purpose: an incremental update leaves the previous
        # index usable, so a mid-update issue gets stale evidence not none.
        assert RepositoryState.UPDATING.is_searchable
        for state in (RepositoryState.NOT_INDEXED, RepositoryState.QUEUED,
                      RepositoryState.INDEXING, RepositoryState.FAILED,
                      RepositoryState.ARCHIVED):
            assert not state.is_searchable, state

    def test_in_flight_states(self):
        assert RepositoryState.QUEUED.is_in_flight
        assert RepositoryState.INDEXING.is_in_flight
        assert not RepositoryState.READY.is_in_flight
        assert not RepositoryState.FAILED.is_in_flight

    def test_record_round_trips_through_dict(self):
        record = RepositoryRecord(
            repo="acme/demo", owner="acme", name="demo", state=RepositoryState.READY,
            chunk_count=42, frameworks=["FastAPI"], indexed_commit_sha="abc123",
            embedding_provider="openai", embedding_model="compatible", embedding_dimensions=256,
        )
        assert RepositoryRecord.from_dict(record.as_dict()) == record

    def test_legacy_embedding_provider_is_read_as_metadata(self):
        restored = RepositoryRecord.from_dict({
            "repo": "acme/demo",
            "state": "ready",
            "embedding_provider": "hashing-512",
        })
        assert restored.embedding_provider == "hashing"
        assert restored.embedding_model == "hashing"
        assert restored.embedding_dimensions == 512
        assert restored.embedding_signature == "hashing:hashing:512"

    def test_round_trip_backfills_owner_and_name_when_absent(self):
        """from_dict derives owner/name from repo, so a record written by an
        older version (or constructed without them) reads back complete."""
        sparse = RepositoryRecord(repo="acme/demo", state=RepositoryState.READY)
        restored = RepositoryRecord.from_dict(sparse.as_dict())
        assert (restored.owner, restored.name) == ("acme", "demo")
        assert restored.state == sparse.state
        assert restored.chunk_count == sparse.chunk_count

    def test_unknown_state_from_a_newer_version_degrades_safely(self):
        restored = RepositoryRecord.from_dict({"repo": "acme/demo", "state": "teleporting"})
        assert restored.state == RepositoryState.NOT_INDEXED

    def test_owner_and_name_are_derived_from_the_repo(self):
        assert RepositoryRecord.from_dict({"repo": "acme/demo"}).owner == "acme"
        assert RepositoryRecord.from_dict({"repo": "acme/demo"}).name == "demo"

    def test_hit_rate(self):
        record = RepositoryRecord(repo="a/b", retrieval_count=4, retrieval_hit_count=3)
        assert record.hit_rate == 0.75
        assert RepositoryRecord(repo="a/b").hit_rate == 0.0


class TestStateStores:
    @pytest.fixture(params=["memory", "file"])
    def store(self, request, tmp_path):
        if request.param == "memory":
            return MemoryStateStore()
        return FileStateStore(tmp_path / "state.json")

    def test_put_and_get(self, store):
        store.put(RepositoryRecord(repo="acme/demo", state=RepositoryState.READY))
        assert store.get("acme/demo").state == RepositoryState.READY

    def test_missing_repo_is_none(self, store):
        assert store.get("acme/nope") is None

    def test_mark_creates_then_transitions(self, store):
        store.mark("acme/demo", RepositoryState.QUEUED)
        assert store.get("acme/demo").state == RepositoryState.QUEUED
        store.mark("acme/demo", RepositoryState.READY, chunk_count=10)
        record = store.get("acme/demo")
        assert record.state == RepositoryState.READY
        assert record.chunk_count == 10

    def test_mark_failed_keeps_the_error(self, store):
        store.mark("acme/demo", RepositoryState.FAILED, error="clone denied")
        assert "clone denied" in store.get("acme/demo").last_error

    def test_list_is_newest_first(self, store):
        store.mark("acme/one", RepositoryState.READY)
        time.sleep(0.01)
        store.mark("acme/two", RepositoryState.READY)
        assert [r.repo for r in store.list()][0] == "acme/two"

    def test_delete(self, store):
        store.mark("acme/demo", RepositoryState.READY)
        store.delete("acme/demo")
        assert store.get("acme/demo") is None

    def test_record_retrieval_updates_counters(self, store):
        store.mark("acme/demo", RepositoryState.READY)
        store.record_retrieval("acme/demo", hit=True)
        store.record_retrieval("acme/demo", hit=False)
        record = store.get("acme/demo")
        assert record.retrieval_count == 2
        assert record.retrieval_hit_count == 1
        assert record.last_accessed_at > 0

    def test_record_retrieval_on_unknown_repo_is_a_noop(self, store):
        store.record_retrieval("acme/unknown", hit=True)   # must not raise

    def test_memory_store_declares_itself_non_durable(self):
        assert MemoryStateStore().durable is False

    def test_file_store_survives_a_new_instance(self, tmp_path):
        path = tmp_path / "state.json"
        FileStateStore(path).mark("acme/demo", RepositoryState.READY, chunk_count=7)
        assert FileStateStore(path).get("acme/demo").chunk_count == 7

    def test_file_store_tolerates_corruption(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{not json", encoding="utf-8")
        store = FileStateStore(path)
        assert store.get("acme/demo") is None
        store.mark("acme/demo", RepositoryState.READY)      # recovers by rewriting
        assert store.get("acme/demo").state == RepositoryState.READY


# ---------------------------------------------------------------------------
# Queues
# ---------------------------------------------------------------------------
class TestQueues:
    def test_null_queue_never_queues(self):
        assert NullIndexQueue().enqueue("acme/demo") is False

    def test_inline_queue_runs_the_worker(self):
        done = []
        queue = InlineIndexQueue(lambda repo, installation_id: done.append(repo))
        assert queue.enqueue("acme/demo") is True
        for _ in range(200):
            if done:
                break
            time.sleep(0.01)
        assert done == ["acme/demo"]

    def test_inline_queue_declares_itself_non_durable(self):
        assert InlineIndexQueue(lambda r, i: None).durable is False

    def test_inline_queue_swallows_worker_failure(self):
        def boom(repo, installation_id):
            raise RuntimeError("indexing exploded")

        queue = InlineIndexQueue(boom)
        assert queue.enqueue("acme/demo") is True       # the thread absorbs it
        time.sleep(0.1)

    def test_qstash_queue_requires_token_and_url(self):
        from ghic.repository_intelligence.queue import QStashIndexQueue

        with pytest.raises(ValueError):
            QStashIndexQueue("", "https://example.com/cb")
        with pytest.raises(ValueError):
            QStashIndexQueue("token", "")


# ---------------------------------------------------------------------------
# Incremental indexing
# ---------------------------------------------------------------------------
class TestIncremental:
    def test_change_set_separates_changed_from_deleted(self):
        cs = ChangeSet(changed=["a.py"], deleted=["b.py"])
        assert not cs.is_empty
        assert cs.total == 2
        assert set(cs.stale_paths) == {"a.py", "b.py"}

    def test_empty_change_set_is_not_worth_an_incremental_pass(self):
        assert should_use_incremental(ChangeSet(), 100) is False

    def test_none_change_set_forces_a_full_rebuild(self):
        assert should_use_incremental(None, 100) is False

    def test_small_change_set_is_incremental(self):
        assert should_use_incremental(ChangeSet(changed=["a.py"]), 100) is True

    def test_large_change_set_falls_back_to_a_full_rebuild(self):
        big = ChangeSet(changed=[f"f{i}.py" for i in range(500)])
        assert should_use_incremental(big, 100) is False

    def test_diff_against_a_non_repo_returns_none(self, tmp_path):
        from ghic.repository_intelligence.incremental import diff_change_set

        assert diff_change_set(tmp_path, "aaa", "bbb") is None

    def test_diff_with_missing_shas_returns_none(self, repo):
        from ghic.repository_intelligence.incremental import diff_change_set

        assert diff_change_set(repo, "", "abc") is None

    def test_identical_shas_are_an_empty_change_set(self, repo):
        from ghic.repository_intelligence.incremental import diff_change_set

        result = diff_change_set(repo, "abc123", "abc123")
        assert result is not None and result.is_empty


# ---------------------------------------------------------------------------
# Security -- Requirement 16
# ---------------------------------------------------------------------------
class TestSecretHandling:
    @pytest.mark.parametrize("path", [
        ".env", ".env.production", "config/.env.local",
        "deploy/server.pem", "certs/private.key", "id_rsa",
        "app/credentials.json", "secrets.yaml", ".npmrc",
        "infra/terraform.tfstate", "gcp/service-account.json",
    ])
    def test_sensitive_files_are_excluded(self, path):
        assert is_sensitive_path(path) is True

    @pytest.mark.parametrize("path", [
        "ghic/service/app.py", "README.md", "src/environment.ts",
        "docs/keyboard.md", "tests/test_secrets_are_excluded.py",
    ])
    def test_ordinary_files_are_not_excluded(self, path):
        assert is_sensitive_path(path) is False

    @pytest.mark.parametrize("secret", [
        'token = "ghp_abcdefghijklmnopqrstuvwxyz1234"',
        'key = "sk-or-v1-abcdefghijklmnopqrstuvwxyz"',
        'groq = "gsk_abcdefghijklmnopqrstuvwxyz1234"',
        'aws = "AKIAIOSFODNN7EXAMPLE"',
        'url = "postgresql://user:hunter2@db.example.com/app"',
        'password = "correcthorsebattery"',
        'api_key: "abcdefghijklmnop"',
    ])
    def test_secret_values_are_redacted(self, secret):
        assert "REDACTED" in redact_secrets(secret)

    def test_private_key_blocks_are_redacted(self):
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF\n"
            "-----END RSA PRIVATE KEY-----"
        )
        assert "MIIEow" not in redact_secrets(pem)

    @pytest.mark.parametrize("ordinary", [
        "def parse_csv(path):\n    return open(path).read()",
        'name = "widget"',
        "MAX_RETRIES = 3",
        "# See the API key documentation for setup",
    ])
    def test_ordinary_code_is_untouched(self, ordinary):
        assert redact_secrets(ordinary) == ordinary

    def test_secrets_never_reach_chunks(self, tmp_path, cfg):
        """End to end: a committed token must not survive into the index."""
        root = tmp_path / "leaky"
        root.mkdir()
        (root / "settings.py").write_text(
            'API_TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz1234"\n'
            "def load():\n    return API_TOKEN\n",
            encoding="utf-8",
        )
        (root / ".env").write_text("SECRET=hunter2\n", encoding="utf-8")

        service = RepositoryIntelligenceService(cfg)
        service.index_local_path("acme/leaky", root)
        context = service.get_context("acme/leaky", "token handling", "API_TOKEN")

        blob = " ".join(rc.chunk.text for rc in context.chunks)
        assert "ghp_abcdefghijklmnopqrstuvwxyz1234" not in blob
        assert "hunter2" not in blob
        assert all(".env" not in rc.chunk.path for rc in context.chunks)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_counters_and_derived_rates(self):
        metrics = RepositoryIntelligenceMetrics()
        metrics.increment("retrievals_total", 4)
        metrics.increment("retrievals_with_results", 3)
        snapshot = metrics.snapshot()
        assert snapshot["counters"]["retrievals_total"] == 4
        assert snapshot["derived"]["retrieval_hit_rate"] == 0.75

    def test_zero_denominator_does_not_divide_by_zero(self):
        assert RepositoryIntelligenceMetrics().snapshot()["derived"]["retrieval_hit_rate"] == 0.0

    def test_timings_are_recorded(self):
        metrics = RepositoryIntelligenceMetrics()
        with metrics.timed("indexing"):
            time.sleep(0.01)
        assert metrics.snapshot()["timings"]["indexing"]["n"] == 1

    def test_a_raising_block_is_still_timed(self):
        metrics = RepositoryIntelligenceMetrics()
        with pytest.raises(ValueError), metrics.timed("indexing"):
            raise ValueError("boom")
        assert metrics.snapshot()["timings"]["indexing"]["n"] == 1

    def test_correlation_id_round_trips(self):
        from ghic.repository_intelligence.metrics import (
            get_correlation_id,
            log_context,
            set_correlation_id,
            set_repo_context,
        )

        set_correlation_id("delivery-123")
        set_repo_context("acme/demo")
        assert get_correlation_id() == "delivery-123"
        assert log_context() == {"correlation_id": "delivery-123", "repo": "acme/demo"}

    def test_blank_correlation_id_gets_a_generated_one(self):
        from ghic.repository_intelligence.metrics import set_correlation_id

        assert set_correlation_id("") != ""

    def test_service_records_retrieval_metrics(self, repo, cfg):
        metrics = RepositoryIntelligenceMetrics()
        service = RepositoryIntelligenceService(
            cfg, state_store=MemoryStateStore(), metrics=metrics
        )
        service.index_local_path("acme/demo", repo)
        service.get_context("acme/demo", "handler request", "")
        snapshot = metrics.snapshot()
        assert snapshot["counters"]["retrievals_total"] == 1
        assert "retrieval" in snapshot["timings"]


# ---------------------------------------------------------------------------
# Provider resolution + auto-disable
# ---------------------------------------------------------------------------
class TestProviders:
    def test_local_vector_provider_is_the_default(self, cfg):
        from ghic.repository_intelligence.indexer import NumpyVectorStore

        assert isinstance(build_vector_store_for("a/b", cfg, 64), NumpyVectorStore)

    def test_postgres_vector_provider_requires_a_url(self, tmp_path):
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path, vector_provider="postgres")
        with pytest.raises(ProviderUnavailable, match="database URL"):
            build_vector_store_for("a/b", cfg, 64)

    def test_unimplemented_vector_provider_refuses_rather_than_silently_falling_back(
        self, tmp_path
    ):
        """Naming a real database and silently getting a local index is the
        exact failure the operator was trying to avoid."""
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path, vector_provider="pinecone")
        with pytest.raises(ProviderUnavailable, match="not implemented"):
            build_vector_store_for("a/b", cfg, 64)

    def test_state_provider_auto_falls_back_to_file(self, cfg):
        assert isinstance(build_state_store_for(cfg), FileStateStore)

    def test_state_provider_memory_is_explicit(self, tmp_path):
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path, state_provider="memory")
        assert isinstance(build_state_store_for(cfg), MemoryStateStore)

    def test_queue_auto_prefers_qstash_when_configured(self, cfg):
        from ghic.repository_intelligence.queue import QStashIndexQueue

        queue = build_index_queue(
            cfg, qstash_token="t", callback_url="https://example.com/cb"
        )
        assert isinstance(queue, QStashIndexQueue)

    def test_queue_auto_falls_back_to_inline_then_null(self, cfg):
        assert isinstance(build_index_queue(cfg, inline_worker=lambda r, i: None),
                          InlineIndexQueue)
        assert isinstance(build_index_queue(cfg), NullIndexQueue)

    def test_unimplemented_queue_provider_refuses(self, tmp_path):
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path, queue_provider="rabbitmq")
        with pytest.raises(ProviderUnavailable, match="not implemented"):
            build_index_queue(cfg)

    def test_auto_disable_on_ephemeral_filesystem_without_persistence(
        self, tmp_path, monkeypatch
    ):
        """The most important behaviour in providers.py: running anyway
        would re-index on every cold start and throw it away."""
        monkeypatch.setenv("VERCEL", "1")
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path, vector_provider="local")
        assert build_service(cfg) is None

    def test_not_disabled_when_persistent_vectors_are_configured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERCEL", "1")
        cfg = RepositoryIntelligenceConfig(
            cache_dir=tmp_path, vector_provider="postgres",
            database_url="postgresql://u:p@h/db", state_provider="memory",
        )
        assert cfg.uses_persistent_vectors is True
        assert ephemeral_filesystem() is True

    def test_not_disabled_on_a_normal_host(self, tmp_path, monkeypatch):
        for var in ("VERCEL", "AWS_LAMBDA_FUNCTION_NAME", "FUNCTIONS_WORKER_RUNTIME",
                    "K_SERVICE"):
            monkeypatch.delenv(var, raising=False)
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path, state_provider="memory")
        service = build_service(cfg)
        assert service is not None
        assert service.index_queue is not None

    def test_platform_detection(self, monkeypatch):
        monkeypatch.setenv("VERCEL", "1")
        assert platform_name() == "vercel"
        monkeypatch.delenv("VERCEL")
        monkeypatch.setenv("FLY_APP_NAME", "ghic")
        assert platform_name() == "fly.io"

    def test_health_snapshot_reports_disabled_with_a_reason(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VERCEL", "1")
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path)
        snapshot = health_snapshot(None, cfg)
        assert snapshot["enabled"] is False
        assert "ephemeral" in snapshot["reason"]

    def test_health_snapshot_reports_durability(self, tmp_path):
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path, state_provider="memory")
        service = RepositoryIntelligenceService(cfg, state_store=MemoryStateStore())
        service.index_queue = NullIndexQueue()
        snapshot = health_snapshot(service)
        assert snapshot["enabled"] is True
        assert snapshot["state_durable"] is False


# ---------------------------------------------------------------------------
# Service behaviour with the new collaborators
# ---------------------------------------------------------------------------
class TestServiceLifecycle:
    def test_indexing_marks_ready_with_metadata(self, repo, cfg):
        store = MemoryStateStore()
        service = RepositoryIntelligenceService(cfg, state_store=store)
        service.index_local_path("acme/demo", repo)

        record = store.get("acme/demo")
        assert record.state == RepositoryState.READY
        assert record.chunk_count > 0
        assert record.primary_language == "Python"
        assert record.embedding_provider == "hashing"
        assert record.embedding_model == "hashing"
        assert record.embedding_dimensions == service.embedder.dimensions
        assert record.embedding_signature == service.embedder.metadata().signature

    def test_embedding_metadata_persists_in_file_state_store(self, repo, tmp_path):
        state_path = tmp_path / "state.json"
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path / "cache")
        service = RepositoryIntelligenceService(cfg, state_store=FileStateStore(state_path))
        service.index_local_path("acme/demo", repo, commit_sha="sha1")

        record = FileStateStore(state_path).get("acme/demo")
        assert record.embedding_provider == "hashing"
        assert record.embedding_model == "hashing"
        assert record.embedding_dimensions == 512
        assert record.indexed_commit_sha == "sha1"

    def test_incompatible_embedding_state_is_not_retrieved(self, repo, tmp_path):
        state_store = MemoryStateStore()
        old_cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path / "old", auto_index=False)
        old_service = RepositoryIntelligenceService(old_cfg, state_store=state_store)
        old_service.index_local_path("acme/demo", repo, commit_sha="sha1")

        new_cfg = RepositoryIntelligenceConfig(
            cache_dir=tmp_path / "old", embedding_dimensions=128, auto_index=False
        )
        queued: list[str] = []
        new_service = RepositoryIntelligenceService(
            new_cfg,
            state_store=state_store,
            index_scheduler=lambda name: queued.append(name) or True,
        )
        context = new_service.get_context("acme/demo", "handler request", "")

        assert context.is_empty
        assert not context.indexed
        assert context.note == UNAVAILABLE_CONTEXT_NOTE
        assert state_store.get("acme/demo").state == RepositoryState.READY
        assert queued == []

    def test_index_repository_forces_full_rebuild_when_embedding_changes(self, repo, cfg):
        from ghic.repository_intelligence.cache import Checkout

        state_store = MemoryStateStore()
        state_store.put(RepositoryRecord(
            repo="acme/demo",
            state=RepositoryState.READY,
            indexed_commit_sha="sha1",
            embedding_provider="hashing",
            embedding_model="hashing",
            embedding_dimensions=128,
        ))
        service = RepositoryIntelligenceService(cfg, state_store=state_store)
        service.repo_cache.ensure = lambda *a, **kw: Checkout(
            path=repo, commit_sha="sha1", default_branch="main"
        )
        service._index_exists = lambda *a, **kw: True
        calls: list[bool] = []

        def capture_run(*args, force: bool):
            calls.append(force)
            return True

        service._run_index = capture_run
        assert service.index_repository("acme/demo")
        assert calls == [True]

    def test_forced_full_rebuild_enables_guarded_dimension_migration(self, repo, cfg):
        from ghic.repository_intelligence.cache import Checkout

        service = RepositoryIntelligenceService(cfg, state_store=MemoryStateStore())
        checkout = Checkout(path=repo, commit_sha="sha2", default_branch="main")
        calls: list[bool] = []

        def capture_full(*args, allow_dimension_migration: bool = False):
            calls.append(allow_dimension_migration)
            return True

        service._index_full = capture_full
        assert service._run_index("acme/demo", checkout, "sha1", force=True)
        assert calls == [True]

    def test_unindexed_repo_is_queued_once_not_per_issue(self, cfg):
        calls: list[str] = []

        class CountingQueue(NullIndexQueue):
            def enqueue(self, repo, *, installation_id=None, reason=""):
                calls.append(repo)
                return True

        service = RepositoryIntelligenceService(
            cfg, state_store=MemoryStateStore(), index_queue=CountingQueue()
        )
        for _ in range(3):
            service.get_context("acme/new", "csv import fails", "")
        assert calls == ["acme/new"]        # queued once; then QUEUED suppresses

    def test_a_wedged_in_flight_job_is_eventually_requeued(self, cfg):
        """A worker that died mid-index must not wedge a repo forever."""
        calls: list[str] = []

        class CountingQueue(NullIndexQueue):
            def enqueue(self, repo, *, installation_id=None, reason=""):
                calls.append(repo)
                return True

        store = MemoryStateStore()
        service = RepositoryIntelligenceService(
            cfg, state_store=store, index_queue=CountingQueue()
        )
        store.put(RepositoryRecord(
            repo="acme/stuck", state=RepositoryState.INDEXING,
            updated_at=time.time() - STALE_IN_FLIGHT_SECONDS - 60,
        ))
        service.get_context("acme/stuck", "anything", "")
        assert calls == ["acme/stuck"]

    def test_failed_state_is_not_searched(self, repo, cfg):
        store = MemoryStateStore()
        service = RepositoryIntelligenceService(
            cfg, state_store=store, index_queue=NullIndexQueue()
        )
        service.index_local_path("acme/demo", repo)
        store.mark("acme/demo", RepositoryState.FAILED, error="clone denied")
        assert service.get_context("acme/demo", "handler", "").is_empty

    def test_auto_index_flag_disables_queueing(self, tmp_path):
        calls: list[str] = []

        class CountingQueue(NullIndexQueue):
            def enqueue(self, repo, *, installation_id=None, reason=""):
                calls.append(repo)
                return True

        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path, auto_index=False)
        service = RepositoryIntelligenceService(
            cfg, state_store=MemoryStateStore(), index_queue=CountingQueue()
        )
        service.get_context("acme/new", "anything", "")
        assert calls == []

    def test_vector_search_flag_disables_retrieval_entirely(self, repo, tmp_path):
        cfg = RepositoryIntelligenceConfig(
            cache_dir=tmp_path / "c", vector_search_enabled=False
        )
        service = RepositoryIntelligenceService(cfg, state_store=MemoryStateStore())
        service.index_local_path("acme/demo", repo)
        assert service.get_context("acme/demo", "handler", "").is_empty

    def test_phase_one_construction_still_works(self, repo, cfg):
        """Backward compatibility: no state store, no queue, a bare
        scheduler callable -- exactly how Phase 1 built it."""
        queued: list[str] = []
        service = RepositoryIntelligenceService(
            cfg, index_scheduler=lambda r: queued.append(r) or True
        )
        service.index_local_path("acme/demo", repo)
        assert not service.get_context("acme/demo", "handler request", "").is_empty
        assert service.get_context("acme/other", "anything", "").indexing_queued
        assert queued == ["acme/other"]

    def test_status_and_listing(self, repo, cfg):
        service = RepositoryIntelligenceService(cfg, state_store=MemoryStateStore())
        service.index_local_path("acme/demo", repo)
        assert service.status("acme/demo").state == RepositoryState.READY
        assert [r.repo for r in service.list_repositories()] == ["acme/demo"]

    def test_status_without_a_state_store_is_none(self, cfg):
        assert RepositoryIntelligenceService(cfg).status("acme/demo") is None

    def test_forget_removes_state_and_index(self, repo, cfg):
        store = MemoryStateStore()
        service = RepositoryIntelligenceService(cfg, state_store=store)
        service.index_local_path("acme/demo", repo)
        service.forget("acme/demo")
        assert store.get("acme/demo") is None
        assert service.get_context("acme/demo", "handler", "").is_empty

    def test_cleanup_removes_stale_repositories(self, repo, cfg):
        store = MemoryStateStore()
        service = RepositoryIntelligenceService(
            cfg, state_store=store, index_queue=NullIndexQueue()
        )
        service.index_local_path("acme/stale", repo)
        # Backdate every timestamp so the record reads as long-untouched.
        store.put(RepositoryRecord(
            repo="acme/stale", state=RepositoryState.READY,
            last_accessed_at=time.time() - 90 * 86400,
            indexed_at=time.time() - 90 * 86400,
            updated_at=time.time() - 90 * 86400,
        ))
        result = service.cleanup(stale_days=30)
        assert "acme/stale" in result["removed"]

    def test_cleanup_dry_run_changes_nothing(self, repo, cfg):
        store = MemoryStateStore()
        service = RepositoryIntelligenceService(cfg, state_store=store)
        store.put(RepositoryRecord(
            repo="acme/stale", state=RepositoryState.READY,
            last_accessed_at=time.time() - 90 * 86400,
            updated_at=time.time() - 90 * 86400,
        ))
        result = service.cleanup(stale_days=30, dry_run=True)
        assert result["removed"] == ["acme/stale"]
        assert store.get("acme/stale") is not None

    def test_cleanup_without_a_state_store_reports_why(self, cfg):
        result = RepositoryIntelligenceService(cfg).cleanup()
        assert result["removed"] == []
        assert "state" in result["reason"]

    def test_active_repository_is_not_cleaned_up(self, cfg):
        store = MemoryStateStore()
        service = RepositoryIntelligenceService(cfg, state_store=store)
        store.put(RepositoryRecord(
            repo="acme/active", state=RepositoryState.READY,
            last_accessed_at=time.time(), updated_at=time.time(),
        ))
        assert service.cleanup(stale_days=30)["removed"] == []
