from __future__ import annotations

from pathlib import Path

import pytest

from ghic import repo_index
from ghic.repository_intelligence import (
    RepositoryIntelligenceConfig,
    RepositoryIntelligenceService,
    RepositoryState,
    build_service,
)
from ghic.repository_intelligence.cache import Checkout, RepositoryCacheError
from ghic.repository_intelligence.state import FileStateStore, MemoryStateStore


@pytest.fixture
def local_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text(
        "class App:\n"
        "    def handle(self, request):\n"
        "        return request\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Demo\n\nCLI indexing fixture.\n", encoding="utf-8")
    return root


def _configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state_provider: str) -> Path:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("GHIC_REPO_INTEL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("GHIC_VECTOR_PROVIDER", "local")
    monkeypatch.setenv("GHIC_STATE_PROVIDER", state_provider)
    monkeypatch.setenv("GHIC_INDEX_QUEUE_PROVIDER", "none")
    for name in ("GHIC_DATABASE_URL", "DATABASE_URL", "POSTGRES_URL", "VERCEL"):
        monkeypatch.delenv(name, raising=False)
    return cache_dir


@pytest.mark.parametrize("state_provider", ["memory", "file"])
def test_cli_path_index_uses_configured_state_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    local_repo: Path,
    state_provider: str,
):
    cache_dir = _configure(monkeypatch, tmp_path, state_provider)
    services: list[RepositoryIntelligenceService] = []

    def capture_service(cfg: RepositoryIntelligenceConfig):
        service = build_service(cfg)
        services.append(service)
        return service

    monkeypatch.setattr(repo_index, "build_service", capture_service)

    assert repo_index.main(["--repo", "acme/demo", "--path", str(local_repo)]) == 0

    record = services[0].status("acme/demo")
    assert record is not None
    assert record.state == RepositoryState.READY
    assert record.indexed_commit_sha == "local"
    assert record.chunk_count > 0

    if state_provider == "file":
        persisted = FileStateStore(cache_dir / "state.json").get("acme/demo")
        assert persisted is not None
        assert persisted.state == RepositoryState.READY


def test_cli_repository_index_persists_indexed_commit_sha(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    local_repo: Path,
):
    _configure(monkeypatch, tmp_path, "memory")
    state_store = MemoryStateStore()
    services: list[RepositoryIntelligenceService] = []

    def fake_build_service(cfg: RepositoryIntelligenceConfig):
        service = RepositoryIntelligenceService(cfg, state_store=state_store)
        service.repo_cache.ensure = lambda *a, **kw: Checkout(
            path=local_repo, commit_sha="abc123def456", default_branch="main"
        )
        services.append(service)
        return service

    monkeypatch.setattr(repo_index, "build_service", fake_build_service)

    assert repo_index.main(["--repo", "acme/demo"]) == 0

    record = services[0].status("acme/demo")
    assert record is not None
    assert record.state == RepositoryState.READY
    assert record.indexed_commit_sha == "abc123def456"
    assert record.chunk_count > 0


def test_cli_failed_repository_index_does_not_become_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _configure(monkeypatch, tmp_path, "memory")
    state_store = MemoryStateStore()
    services: list[RepositoryIntelligenceService] = []

    def fake_build_service(cfg: RepositoryIntelligenceConfig):
        service = RepositoryIntelligenceService(cfg, state_store=state_store)

        def fail_clone(*args, **kwargs):
            raise RepositoryCacheError("clone denied")

        service.repo_cache.ensure = fail_clone
        services.append(service)
        return service

    monkeypatch.setattr(repo_index, "build_service", fake_build_service)

    assert repo_index.main(["--repo", "acme/private"]) == 1

    record = services[0].status("acme/private")
    assert record is not None
    assert record.state == RepositoryState.FAILED
    assert record.state != RepositoryState.READY
    assert "clone denied" in record.last_error


def test_cli_repairs_missing_state_without_rebuilding_an_existing_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    local_repo: Path,
):
    _configure(monkeypatch, tmp_path, "memory")
    cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path / "cache")
    old_service = RepositoryIntelligenceService(cfg)
    assert old_service.index_local_path(
        "acme/demo", local_repo, commit_sha="abc123def456", default_branch="main"
    )

    state_store = MemoryStateStore()
    services: list[RepositoryIntelligenceService] = []

    def fake_build_service(config: RepositoryIntelligenceConfig):
        service = RepositoryIntelligenceService(config, state_store=state_store)
        service.repo_cache.ensure = lambda *a, **kw: Checkout(
            path=local_repo, commit_sha="abc123def456", default_branch="main"
        )

        def fail_if_rebuilt(*args, **kwargs):
            raise AssertionError("existing index should not be rebuilt")

        service._run_index = fail_if_rebuilt
        services.append(service)
        return service

    monkeypatch.setattr(repo_index, "build_service", fake_build_service)

    assert repo_index.main(["--repo", "acme/demo"]) == 0

    record = services[0].status("acme/demo")
    assert record is not None
    assert record.state == RepositoryState.READY
    assert record.indexed_commit_sha == "abc123def456"
