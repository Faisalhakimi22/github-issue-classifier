"""Tests for the Repository Intelligence Engine.

Built against a real fixture repository on disk (tmp_path) rather than
mocks, because the things most likely to break here -- chunk boundaries,
exclusion rules, cache invalidation -- are exactly the things a mock would
paper over. The only mocked pieces are the ones with an external dependency
(the embeddings HTTP API, git).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ghic.repository_intelligence import (
    CodeChunk,
    HashingEmbeddingProvider,
    NumpyVectorStore,
    RepositoryIntelligenceConfig,
    RepositoryIntelligenceService,
    RetrievedChunk,
    SemanticRetriever,
    build_query,
)
from ghic.repository_intelligence.cache import IndexCache
from ghic.repository_intelligence.embeddings import EmbeddingError, tokenize
from ghic.repository_intelligence.indexer import RepositoryIndexer
from ghic.repository_intelligence.models import EMPTY_CONTEXT_NOTE, UNAVAILABLE_CONTEXT_NOTE
from ghic.repository_intelligence.parser import (
    chunk_file,
    detect_metadata,
    is_indexable,
    walk_repository,
)

PYTHON_SOURCE = '''\
"""Module docstring."""
import os

MAX_RETRIES = 3


def parse_csv(path):
    """Parse a CSV file."""
    with open(path) as handle:
        return handle.read()


class CsvImporter:
    """Imports CSV data."""

    kind = "csv"

    def __init__(self, encoding="utf-8"):
        self.encoding = encoding

    def decode_utf8(self, raw):
        return raw.decode(self.encoding)
'''

JS_SOURCE = """\
import fs from 'fs';

export function parseCsv(path) {
  const text = fs.readFileSync(path);
  return text.toString();
}

export class Importer {
  constructor(encoding) {
    this.encoding = encoding;
  }
}
"""

GO_SOURCE = """\
package importer

import "os"

func ParseCSV(path string) ([]byte, error) {
	return os.ReadFile(path)
}

type Importer struct {
	Encoding string
}
"""

MARKDOWN_SOURCE = """\
# CSV Importer

Imports CSV data and validates file encoding.

## Installation

Run the installer.

## Usage

```bash
# this is not a heading
importer run
```

## Troubleshooting

Check the encoding.
"""


@pytest.fixture
def repo(tmp_path):
    """A small but structurally realistic repository."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "csv_parser.py").write_text(PYTHON_SOURCE, encoding="utf-8")
    (tmp_path / "src" / "importer.js").write_text(JS_SOURCE, encoding="utf-8")
    (tmp_path / "src" / "importer.go").write_text(GO_SOURCE, encoding="utf-8")
    (tmp_path / "README.md").write_text(MARKDOWN_SOURCE, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\ndependencies = ["fastapi", "scikit-learn"]\n',
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")

    # Things that must never be indexed.
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("module.exports = 1;", encoding="utf-8")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.js").write_text("var a=1;", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    (tmp_path / "app.min.js").write_text("var b=2;", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    return tmp_path


@pytest.fixture
def cfg(tmp_path):
    return RepositoryIntelligenceConfig(cache_dir=tmp_path / "_cache")


@pytest.fixture
def service(cfg):
    return RepositoryIntelligenceService(cfg)


# ---------------------------------------------------------------------------
# Walking / exclusion
# ---------------------------------------------------------------------------
class TestWalking:
    def test_indexes_source_and_docs_only(self, repo, cfg):
        names = {p.name for p in walk_repository(repo, cfg)}
        assert names == {"csv_parser.py", "importer.js", "importer.go", "README.md", "main.py"}

    def test_excludes_dependencies_build_output_and_binaries(self, repo, cfg):
        paths = {p.as_posix() for p in walk_repository(repo, cfg)}
        for excluded in ("node_modules", "dist", "package-lock.json", "app.min.js", "logo.png"):
            assert not any(excluded in p for p in paths), excluded

    def test_oversized_files_skipped(self, repo, cfg):
        big = repo / "src" / "generated.py"
        big.write_text("x = 1\n" * 200_000, encoding="utf-8")
        assert not is_indexable(big, repo, cfg)

    def test_nested_excluded_directory_is_excluded(self, repo, cfg):
        nested = repo / "src" / "node_modules"
        nested.mkdir()
        target = nested / "inner.js"
        target.write_text("var x = 1;", encoding="utf-8")
        assert not is_indexable(target, repo, cfg)

    def test_empty_repository_yields_nothing(self, tmp_path, cfg):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert walk_repository(empty, cfg) == []

    def test_max_files_prefers_source_over_archive_and_tests(self, tmp_path):
        """The cap must degrade by relevance, not alphabetically -- an
        `_archive/` directory sorting before `src/` must not evict the
        source it precedes."""
        root = tmp_path / "big"
        (root / "_archive").mkdir(parents=True)
        (root / "src").mkdir()
        (root / "tests").mkdir()
        for i in range(5):
            (root / "_archive" / f"old{i}.py").write_text("x = 1\n", encoding="utf-8")
            (root / "tests" / f"test_{i}.py").write_text("x = 1\n", encoding="utf-8")
            (root / "src" / f"mod{i}.py").write_text("x = 1\n", encoding="utf-8")

        small = RepositoryIntelligenceConfig(cache_dir=tmp_path / "_c", max_files=5)
        kept = {p.parent.name for p in walk_repository(root, small)}
        assert kept == {"src"}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
class TestChunking:
    def test_python_splits_on_declarations(self, cfg):
        chunks = chunk_file("acme/demo", "src/csv_parser.py", PYTHON_SOURCE, cfg)
        by_symbol = {c.symbol: c for c in chunks}
        assert "parse_csv" in by_symbol
        assert by_symbol["parse_csv"].kind == "function"
        assert "decode_utf8" in by_symbol
        assert by_symbol["decode_utf8"].kind == "method"
        assert by_symbol["decode_utf8"].parent_symbol == "CsvImporter"
        assert by_symbol["decode_utf8"].qualified_symbol == "CsvImporter.decode_utf8"

    def test_python_chunks_never_split_mid_declaration(self, cfg):
        for chunk in chunk_file("acme/demo", "src/csv_parser.py", PYTHON_SOURCE, cfg):
            if chunk.symbol == "parse_csv":
                assert chunk.text.lstrip().startswith("def parse_csv")
                assert "handle.read()" in chunk.text

    def test_python_module_level_code_is_kept(self, cfg):
        chunks = chunk_file("acme/demo", "src/csv_parser.py", PYTHON_SOURCE, cfg)
        module_text = " ".join(c.text for c in chunks if c.kind == "module")
        assert "MAX_RETRIES" in module_text

    def test_line_numbers_point_at_real_source(self, cfg):
        lines = PYTHON_SOURCE.split("\n")
        for chunk in chunk_file("acme/demo", "src/csv_parser.py", PYTHON_SOURCE, cfg):
            assert chunk.start_line >= 1
            assert chunk.end_line <= len(lines)
            assert chunk.text.split("\n")[0] == lines[chunk.start_line - 1]

    def test_syntax_error_falls_back_to_whole_file(self, cfg):
        broken = "def oops(:\n    value = 1\n    other = 2\n    return value + other\n"
        chunks = chunk_file("acme/demo", "broken.py", broken, cfg)
        assert len(chunks) == 1
        assert chunks[0].kind == "module"

    def test_trivial_unnamed_fragment_is_dropped_as_noise(self, cfg):
        assert chunk_file("acme/demo", "tiny.py", "x = 1\n", cfg) == []

    def test_short_named_declaration_is_kept(self, cfg):
        """A three-line method is short but highly retrievable -- its name
        is what an issue reporter types."""
        source = "class A:\n    def decode_utf8(self, raw):\n        return raw.decode()\n"
        symbols = {c.symbol for c in chunk_file("acme/demo", "small.py", source, cfg)}
        assert "decode_utf8" in symbols

    def test_javascript_declarations(self, cfg):
        symbols = {c.symbol for c in chunk_file("acme/demo", "src/importer.js", JS_SOURCE, cfg)}
        assert "parseCsv" in symbols
        assert "Importer" in symbols

    def test_go_declarations(self, cfg):
        symbols = {c.symbol for c in chunk_file("acme/demo", "src/importer.go", GO_SOURCE, cfg)}
        assert "ParseCSV" in symbols
        assert "Importer" in symbols

    def test_markdown_splits_on_headings_not_inside_fences(self, cfg):
        chunks = chunk_file("acme/demo", "README.md", MARKDOWN_SOURCE, cfg)
        headings = [c.symbol for c in chunks]
        assert "Installation" in headings
        assert "Troubleshooting" in headings
        # "# this is not a heading" lives inside a bash fence.
        assert not any("not a heading" in h for h in headings)

    def test_oversized_declaration_is_split(self, tmp_path):
        narrow = RepositoryIntelligenceConfig(cache_dir=tmp_path, max_chunk_chars=200)
        source = "def big():\n" + "".join(f"    value_{i} = {i}\n" for i in range(200))
        chunks = chunk_file("acme/demo", "big.py", source, narrow)
        assert len(chunks) > 1
        assert all(len(c.text) <= 400 for c in chunks)

    def test_empty_file_produces_no_chunks(self, cfg):
        assert chunk_file("acme/demo", "empty.py", "\n\n", cfg) == []

    def test_unknown_language_produces_no_chunks(self, cfg):
        assert chunk_file("acme/demo", "notes.xyz", "hello", cfg) == []


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
class TestMetadata:
    def test_detects_language_frameworks_and_toolchain(self, repo, cfg):
        files = walk_repository(repo, cfg)
        metadata = detect_metadata(repo, "acme/demo", files, cfg)
        assert metadata.primary_language == "Python"
        assert "FastAPI" in metadata.frameworks
        assert "scikit-learn" in metadata.frameworks
        assert "pip/pyproject" in metadata.dependency_managers
        assert "main.py" in metadata.entry_points

    def test_readme_summary_skips_headings_and_badges(self, repo, cfg):
        metadata = detect_metadata(repo, "acme/demo", walk_repository(repo, cfg), cfg)
        assert metadata.readme_summary.startswith("Imports CSV data")

    def test_package_json_frameworks_come_from_dependencies_not_prose(self, tmp_path, cfg):
        root = tmp_path / "js"
        root.mkdir()
        (root / "package.json").write_text(
            json.dumps({"dependencies": {"react": "^18"}}), encoding="utf-8"
        )
        # A README merely mentioning Vue must not produce a Vue label.
        (root / "README.md").write_text("We migrated away from vue.", encoding="utf-8")
        metadata = detect_metadata(root, "acme/js", walk_repository(root, cfg), cfg)
        assert "React" in metadata.frameworks
        assert "Vue" not in metadata.frameworks

    def test_empty_repository_metadata_is_blank_not_guessed(self, tmp_path, cfg):
        empty = tmp_path / "empty"
        empty.mkdir()
        metadata = detect_metadata(empty, "acme/empty", [], cfg)
        assert metadata.primary_language == ""
        assert metadata.frameworks == []
        assert metadata.file_count == 0


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class TestEmbeddings:
    def test_tokenizer_splits_identifiers(self):
        tokens = set(tokenize("parse_csv_file"))
        assert {"parse_csv_file", "parse", "csv", "file"} <= tokens
        assert {"parse", "csv", "file"} <= set(tokenize("parseCSVFile"))

    def test_vectors_are_unit_length(self):
        provider = HashingEmbeddingProvider(128)
        vectors = provider.embed_documents(["def parse_csv(path):", "class Importer:"])
        assert vectors.shape == (2, 128)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)

    def test_deterministic(self):
        provider = HashingEmbeddingProvider(64)
        first = provider.embed_documents(["decode_utf8"])
        second = provider.embed_documents(["decode_utf8"])
        np.testing.assert_array_equal(first, second)

    def test_similar_text_scores_higher_than_unrelated(self):
        provider = HashingEmbeddingProvider(512)
        query = provider.embed_query("csv parsing fails on encoding")
        docs = provider.embed_documents([
            "def parse_csv(path): decode the csv encoding",
            "def render_button(props): return jsx markup",
        ])
        assert float(docs[0] @ query) > float(docs[1] @ query)

    def test_empty_input_returns_empty_matrix(self):
        assert HashingEmbeddingProvider(32).embed_documents([]).shape == (0, 32)

    def test_all_stopword_text_yields_zero_vector_not_a_crash(self):
        vector = HashingEmbeddingProvider(32).embed_query("the and for")
        assert np.isfinite(vector).all()

    def test_openai_provider_requires_a_key(self):
        from ghic.repository_intelligence.embeddings import OpenAICompatibleEmbeddingProvider

        with pytest.raises(ValueError):
            OpenAICompatibleEmbeddingProvider(api_key="")

    def test_openai_provider_parses_a_successful_response(self):
        import httpx

        from ghic.repository_intelligence.embeddings import OpenAICompatibleEmbeddingProvider

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert payload["dimensions"] == 4
            assert "output_dimension" not in payload
            return httpx.Response(200, json={
                "data": [{"embedding": [0.0, 1.0, 0.0, 0.0]} for _ in payload["input"]]
            })

        provider = OpenAICompatibleEmbeddingProvider(
            api_key="k", dimensions=4,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        vectors = provider.embed_documents(["a", "b"])
        assert vectors.shape == (2, 4)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)

    def test_codestral_provider_uses_output_dimension(self):
        import httpx

        from ghic.repository_intelligence.embeddings import OpenAICompatibleEmbeddingProvider

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert payload["model"] == "mistralai/codestral-embed-2505"
            assert payload["output_dimension"] == 1536
            assert "dimensions" not in payload
            return httpx.Response(200, json={
                "data": [{"embedding": [1.0] + [0.0] * 1535}]
            })

        provider = OpenAICompatibleEmbeddingProvider(
            api_key="k",
            model="mistralai/codestral-embed-2505",
            dimensions=1536,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        vector = provider.embed_query("find webhook verification code")
        assert vector.shape == (1536,)

    def test_openai_provider_detects_response_dimensions(self):
        import httpx

        from ghic.repository_intelligence.embeddings import OpenAICompatibleEmbeddingProvider

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0, 0.0]}]})

        provider = OpenAICompatibleEmbeddingProvider(
            api_key="k", dimensions=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        vectors = provider.embed_documents(["semantic text"])
        assert vectors.shape == (1, 3)
        assert provider.dimensions == 3
        assert provider.metadata().as_dict() == {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 3,
            "signature": "openai:text-embedding-3-small:3",
        }

    def test_openai_provider_raises_embedding_error_on_4xx(self):
        import httpx

        from ghic.repository_intelligence.embeddings import OpenAICompatibleEmbeddingProvider

        provider = OpenAICompatibleEmbeddingProvider(
            api_key="bad", dimensions=4,
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(401, json={}))
            ),
        )
        with pytest.raises(EmbeddingError):
            provider.embed_documents(["a"])

    def test_openai_provider_raises_embedding_error_on_transient_failure(self):
        import httpx

        from ghic.repository_intelligence.embeddings import OpenAICompatibleEmbeddingProvider

        provider = OpenAICompatibleEmbeddingProvider(
            api_key="bad", dimensions=4,
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(500, text="down"))
            ),
        )
        with pytest.raises(EmbeddingError):
            provider.embed_documents(["a"])

    def test_build_provider_falls_back_when_openai_key_missing(self, tmp_path):
        from ghic.repository_intelligence.embeddings import build_embedding_provider

        cfg = RepositoryIntelligenceConfig(
            cache_dir=tmp_path, embedding_provider="openai", embedding_api_key="",
        )
        provider = build_embedding_provider(cfg)
        assert isinstance(provider, HashingEmbeddingProvider)
        assert provider.metadata().as_dict() == {
            "provider": "hashing",
            "model": "hashing",
            "dimensions": 512,
            "signature": "hashing:hashing:512",
        }

    def test_build_provider_selects_openai_compatible_when_key_is_present(self, tmp_path):
        from ghic.repository_intelligence.embeddings import (
            OpenAICompatibleEmbeddingProvider,
            build_embedding_provider,
        )

        cfg = RepositoryIntelligenceConfig(
            cache_dir=tmp_path,
            embedding_provider="openai",
            embedding_api_key="test-key",
            embedding_model="compatible-embedder",
            embedding_dimensions=256,
        )
        provider = build_embedding_provider(cfg)
        assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
        assert provider.metadata().as_dict() == {
            "provider": "openai",
            "model": "compatible-embedder",
            "dimensions": 256,
            "signature": "openai:compatible-embedder:256",
        }


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
class TestVectorStore:
    def _chunk(self, path: str) -> CodeChunk:
        return CodeChunk(repo="a/b", path=path, language="Python", text="x",
                         start_line=1, end_line=1)

    def test_search_ranks_by_similarity(self):
        store = NumpyVectorStore(2)
        store.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                  [self._chunk("near.py"), self._chunk("far.py")])
        results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
        assert [r.chunk.path for r in results] == ["near.py", "far.py"]
        assert results[0].score > results[1].score

    def test_empty_store_returns_nothing(self):
        assert NumpyVectorStore(4).search(np.zeros(4, dtype=np.float32), top_k=5) == []

    def test_dimension_mismatch_is_rejected(self):
        store = NumpyVectorStore(3)
        with pytest.raises(ValueError):
            store.add(np.zeros((1, 5), dtype=np.float32), [self._chunk("a.py")])

    def test_roundtrip_through_disk(self, tmp_path):
        store = NumpyVectorStore(2)
        store.add(np.array([[1.0, 0.0]], dtype=np.float32), [self._chunk("a.py")])
        store.save(tmp_path / "idx")
        reloaded = NumpyVectorStore.load(tmp_path / "idx")
        assert reloaded.size == 1
        assert reloaded.search(np.array([1.0, 0.0], dtype=np.float32), 1)[0].chunk.path == "a.py"

    def test_indexer_uses_detected_embedding_dimensions(self, repo, cfg):
        import httpx

        from ghic.repository_intelligence.embeddings import OpenAICompatibleEmbeddingProvider

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            return httpx.Response(200, json={
                "data": [{"embedding": [1.0, 0.0, 0.0]} for _ in payload["input"]]
            })

        provider = OpenAICompatibleEmbeddingProvider(
            api_key="k",
            dimensions=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        store, _ = RepositoryIndexer(provider, cfg).build("acme/demo", repo)
        assert store.dimensions == 3


# ---------------------------------------------------------------------------
# Query construction + retrieval
# ---------------------------------------------------------------------------
class TestQueryBuilding:
    def test_keeps_identifiers_from_the_body(self):
        query = build_query(
            "Import fails",
            "Calling parse_csv() raises UnicodeDecodeError in src/importer.py.",
        )
        assert "parse_csv" in query
        assert "UnicodeDecodeError" in query
        assert "src/importer.py" in query

    def test_title_is_weighted(self):
        assert build_query("encoding bug", "").count("encoding bug") > 1

    def test_drops_stack_trace_noise_but_keeps_prose(self):
        body = "The import breaks.\n\n```\n" + "\n".join(f"  at frame {i}" for i in range(80)) + "\n```"
        query = build_query("Import breaks", body)
        assert "The import breaks." in query
        assert "at frame 40" not in query

    def test_empty_issue_yields_empty_query(self):
        assert build_query("", "") == ""


class TestRetrieval:
    def test_finds_the_relevant_function(self, repo, service):
        service.index_local_path("acme/demo", repo)
        context = service.get_context(
            "acme/demo", "CSV parsing crashes", "parse_csv fails on this file",
        )
        assert context.indexed
        assert "src/csv_parser.py" in context.relevant_files

    def test_unrelated_query_returns_no_confident_match(self, repo, cfg):
        strict = RepositoryIntelligenceConfig(cache_dir=cfg.cache_dir, min_similarity=0.95)
        service = RepositoryIntelligenceService(strict)
        service.index_local_path("acme/demo", repo)
        context = service.get_context("acme/demo", "kubernetes ingress tls renewal", "")
        assert context.indexed
        assert context.is_empty
        assert context.note == EMPTY_CONTEXT_NOTE

    def test_results_are_capped_at_top_k(self, repo, tmp_path):
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path / "c", top_k=3, min_similarity=0.0)
        service = RepositoryIntelligenceService(cfg)
        service.index_local_path("acme/demo", repo)
        context = service.get_context("acme/demo", "csv", "importer encoding")
        assert len(context.chunks) <= 3

    def test_no_single_file_dominates_results(self, tmp_path):
        root = tmp_path / "mono"
        root.mkdir()
        body = "".join(
            f"def handler_{i}(request):\n    return process_csv_import(request)\n\n"
            for i in range(40)
        )
        (root / "huge.py").write_text(body, encoding="utf-8")
        (root / "other.py").write_text(
            "def process_csv_import(request):\n    return True\n", encoding="utf-8"
        )
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path / "c", min_similarity=0.0)
        service = RepositoryIntelligenceService(cfg)
        service.index_local_path("acme/mono", root)
        context = service.get_context("acme/mono", "csv import handler", "")
        assert sum(1 for c in context.chunks if c.chunk.path == "huge.py") <= 2

    def test_scores_are_returned_and_ordered_within_the_floor(self, repo, service):
        service.index_local_path("acme/demo", repo)
        context = service.get_context("acme/demo", "csv parser encoding", "")
        assert all(rc.score >= service.cfg.min_similarity for rc in context.chunks)

    def test_prefers_implementation_without_discarding_test_evidence(self):
        candidates = [
            RetrievedChunk(
                CodeChunk(
                    repo="acme/demo",
                    path="tests/test_import.py",
                    language="Python",
                    text="def test_csv_import_unicode_error(): pass",
                    start_line=1,
                    end_line=1,
                    kind="function",
                    symbol="test_csv_import_unicode_error",
                ),
                0.62,
            ),
            RetrievedChunk(
                CodeChunk(
                    repo="acme/demo",
                    path="docs/importing.md",
                    language="Markdown",
                    text="# CSV imports\nUnicode errors can occur during import.",
                    start_line=1,
                    end_line=2,
                    kind="section",
                    symbol="CSV imports",
                ),
                0.56,
            ),
            RetrievedChunk(
                CodeChunk(
                    repo="acme/demo",
                    path="ghic/collect.py",
                    language="Python",
                    text="def read_csv(path): return path.read_text(encoding='utf-8')",
                    start_line=10,
                    end_line=11,
                    kind="function",
                    symbol="read_csv",
                ),
                0.54,
            ),
        ]

        class CandidateStore:
            size = len(candidates)

            def search(self, query, top_k):
                return candidates[:top_k]

        retriever = SemanticRetriever(
            HashingEmbeddingProvider(8),
            RepositoryIntelligenceConfig(min_similarity=0.15, top_k=3),
        )
        results = retriever.retrieve(CandidateStore(), "CSV import UnicodeDecodeError", "")

        assert [item.chunk.path for item in results] == [
            "ghic/collect.py",
            "docs/importing.md",
            "tests/test_import.py",
        ]


# ---------------------------------------------------------------------------
# Caching + lifecycle
# ---------------------------------------------------------------------------
class TestCaching:
    def test_index_is_reused_for_the_same_commit(self, repo, cfg):
        cache = IndexCache(cfg)
        embedder = HashingEmbeddingProvider(cfg.embedding_dimensions)
        store, metadata = RepositoryIndexer(embedder, cfg).build("acme/demo", repo)
        cache.save("acme/demo", "sha1", embedder.name, store, metadata)

        loaded = cache.load("acme/demo", "sha1", embedder.name)
        assert loaded is not None
        assert loaded[0].size == store.size
        assert loaded[1].primary_language == metadata.primary_language

    def test_different_commit_is_a_miss(self, repo, cfg):
        cache = IndexCache(cfg)
        embedder = HashingEmbeddingProvider(cfg.embedding_dimensions)
        store, metadata = RepositoryIndexer(embedder, cfg).build("acme/demo", repo)
        cache.save("acme/demo", "sha1", embedder.name, store, metadata)
        assert cache.load("acme/demo", "sha2", embedder.name) is None

    def test_different_embedder_is_a_miss(self, repo, cfg):
        cache = IndexCache(cfg)
        embedder = HashingEmbeddingProvider(cfg.embedding_dimensions)
        store, metadata = RepositoryIndexer(embedder, cfg).build("acme/demo", repo)
        cache.save("acme/demo", "sha1", embedder.name, store, metadata)
        assert cache.load("acme/demo", "sha1", "openai:other-1536") is None

    def test_corrupt_index_degrades_to_a_miss(self, repo, cfg):
        cache = IndexCache(cfg)
        embedder = HashingEmbeddingProvider(cfg.embedding_dimensions)
        store, metadata = RepositoryIndexer(embedder, cfg).build("acme/demo", repo)
        cache.save("acme/demo", "sha1", embedder.name, store, metadata)
        (cache.directory("acme/demo", "sha1", embedder.name) / "manifest.json").write_text(
            "{not json", encoding="utf-8"
        )
        assert cache.load("acme/demo", "sha1", embedder.name) is None

    def test_latest_pointer_round_trips_and_purges(self, cfg):
        cache = IndexCache(cfg)
        cache.set_latest("acme/demo", "sha1", "hashing-512")
        assert cache.get_latest("acme/demo") == ("sha1", "hashing-512")
        cache.purge("acme/demo")
        assert cache.get_latest("acme/demo") is None

    def test_reindexing_the_same_repo_replaces_the_pointer(self, repo, service):
        service.index_local_path("acme/demo", repo, commit_sha="sha1")
        service.index_local_path("acme/demo", repo, commit_sha="sha2")
        assert service.index_cache.get_latest("acme/demo")[0] == "sha2"


# ---------------------------------------------------------------------------
# Fallback behaviour -- the "never fail the webhook" contract
# ---------------------------------------------------------------------------
class TestFallback:
    def test_unindexed_repo_returns_empty_context_not_an_error(self, service):
        context = service.get_context("acme/never-indexed", "anything", "")
        assert context.is_empty
        assert not context.indexed
        assert context.note == UNAVAILABLE_CONTEXT_NOTE

    def test_unindexed_repo_queues_indexing_when_a_scheduler_exists(self, cfg):
        queued: list[str] = []
        service = RepositoryIntelligenceService(
            cfg, index_scheduler=lambda repo: queued.append(repo) or True
        )
        context = service.get_context("acme/new", "csv", "")
        assert queued == ["acme/new"]
        assert context.indexing_queued

    def test_scheduler_failure_is_swallowed(self, cfg):
        def broken(repo: str) -> bool:
            raise RuntimeError("queue is down")

        service = RepositoryIntelligenceService(cfg, index_scheduler=broken)
        context = service.get_context("acme/new", "csv", "")
        assert context.is_empty
        assert not context.indexing_queued

    def test_embedding_outage_during_retrieval_degrades_to_empty(self, repo, service):
        service.index_local_path("acme/demo", repo)

        class BrokenEmbedder(HashingEmbeddingProvider):
            def embed_documents(self, texts):
                raise EmbeddingError("provider down")

        service.retriever.embedder = BrokenEmbedder(service.cfg.embedding_dimensions)
        context = service.get_context("acme/demo", "csv parsing", "")
        assert context.is_empty

    def test_indexing_a_missing_directory_returns_none(self, service, tmp_path):
        assert service.index_local_path("acme/gone", tmp_path / "does-not-exist") is None

    def test_index_repository_returns_false_when_clone_fails(self, service, monkeypatch):
        from ghic.repository_intelligence import cache as cache_module

        def boom(*args, **kwargs):
            raise cache_module.RepositoryCacheError("no such repo")

        monkeypatch.setattr(service.repo_cache, "ensure", boom)
        assert service.index_repository("acme/private") is False

    def test_empty_repository_indexes_without_error(self, tmp_path, service):
        empty = tmp_path / "empty"
        empty.mkdir()
        metadata = service.index_local_path("acme/empty", empty)
        assert metadata is not None
        assert metadata.chunk_count == 0
        assert service.get_context("acme/empty", "anything", "").is_empty


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
class TestPromptConstruction:
    def test_repository_section_contains_only_retrieved_files(self, repo, service):
        from ghic.llm.prompts import build_repository_section

        service.index_local_path("acme/demo", repo)
        context = service.get_context("acme/demo", "csv parsing crashes", "parse_csv fails")
        section = build_repository_section(context)

        assert "src/csv_parser.py" in section
        for path in ("node_modules", "dist/bundle.js", "package-lock.json"):
            assert path not in section

    def test_repository_section_states_its_limits(self, repo, service):
        from ghic.llm.prompts import build_repository_section

        service.index_local_path("acme/demo", repo)
        section = build_repository_section(
            service.get_context("acme/demo", "csv parsing", "parse_csv")
        )
        assert "partial view" in section
        assert "do not" in section.lower()

    def test_empty_context_produces_no_section(self):
        from ghic.llm.prompts import build_repository_section

        assert build_repository_section(None) == ""

    def test_user_prompt_says_so_when_no_code_was_retrieved(self):
        from ghic.llm.models import IssueContext
        from ghic.llm.prompts import build_user_prompt

        prompt = build_user_prompt(IssueContext(
            repo="acme/demo", title="t", body="b", labels=[], author="a",
            metadata={}, ml_probability=0.5, ml_predicted_label=1,
        ))
        assert "No repository code was retrieved" in prompt
        assert "do not speculate" in prompt.lower()

    def test_user_prompt_embeds_retrieved_code(self, repo, service):
        from ghic.llm.models import IssueContext
        from ghic.llm.prompts import build_user_prompt

        service.index_local_path("acme/demo", repo)
        context = service.get_context("acme/demo", "csv parsing crashes", "parse_csv fails")
        prompt = build_user_prompt(IssueContext(
            repo="acme/demo", title="csv parsing crashes", body="parse_csv fails",
            labels=[], author="a", metadata={}, ml_probability=0.9,
            ml_predicted_label=1, repository_context=context,
        ))
        assert "src/csv_parser.py" in prompt
        assert "def parse_csv" in prompt


# ---------------------------------------------------------------------------
# Large-repository behaviour
# ---------------------------------------------------------------------------
class TestLargeRepository:
    def test_chunk_ceiling_is_enforced(self, tmp_path):
        root = tmp_path / "large"
        root.mkdir()
        for f in range(20):
            (root / f"mod{f}.py").write_text(
                "".join(f"def fn_{f}_{i}():\n    return {i}\n\n" for i in range(30)),
                encoding="utf-8",
            )
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path / "c", max_chunks=50)
        store, metadata = RepositoryIndexer(HashingEmbeddingProvider(64), cfg).build(
            "acme/large", root
        )
        assert store.size <= 50
        assert metadata.chunk_count <= 50

    def test_indexing_many_files_stays_within_the_file_cap(self, tmp_path):
        root = tmp_path / "many"
        root.mkdir()
        for i in range(60):
            (root / f"file{i:03d}.py").write_text(f"def fn{i}():\n    return {i}\n", encoding="utf-8")
        cfg = RepositoryIntelligenceConfig(cache_dir=tmp_path / "c", max_files=25)
        _, metadata = RepositoryIndexer(HashingEmbeddingProvider(64), cfg).build("acme/many", root)
        assert metadata.file_count == 25
