"""Vector store abstraction + the indexing pipeline.

`VectorStore` is the seam the spec asks for: nothing outside this module
knows which backend is in use. Two implementations ship, and the default is
the one that is actually correct for this workload:

  `NumpyVectorStore` (default) -- exact cosine similarity via one dense
  matrix-vector product. For a repository-scale corpus (a few thousand to a
  few tens of thousands of chunks) this is not a compromise, it is the
  better algorithm: FAISS's value is *approximate* search over millions of
  vectors, and at 20k x 512 float32 (~40MB, one BLAS call, low single-digit
  milliseconds) approximation buys nothing and costs recall. It also has no
  dependency beyond numpy, which already ships.

  `FaissVectorStore` -- used automatically when `faiss` is importable and
  the corpus is large enough to justify it. Same interface, same results
  for exact index types.

Adding Chroma/Pinecone/Qdrant/Milvus later means writing one subclass and
returning it from `build_vector_store()`; no caller changes.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .. import utils
from .config import RepositoryIntelligenceConfig
from .embeddings import EmbeddingProvider
from .models import CodeChunk, RetrievedChunk

logger = utils.get_logger(__name__)

INDEX_FORMAT_VERSION = 1

# Below this many chunks, FAISS's index-build cost exceeds the search time
# it saves; numpy stays faster end to end.
_FAISS_MIN_CHUNKS = 50_000


class VectorStore(ABC):
    """Add vectors with their chunks, search by vector, persist, reload."""

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def add(self, vectors: np.ndarray, chunks: list[CodeChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> VectorStore:
        raise NotImplementedError


class NumpyVectorStore(VectorStore):
    """Exact cosine similarity over L2-normalized vectors.

    Vectors arrive normalized from `EmbeddingProvider`, so cosine similarity
    is a plain dot product and search is one `matrix @ query`.
    """

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self._vectors: np.ndarray = np.zeros((0, dimensions), dtype=np.float32)
        self._chunks: list[CodeChunk] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: list[CodeChunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        if not len(chunks):
            return
        if vectors.shape[1] != self.dimensions:
            raise ValueError(
                f"expected {self.dimensions}-dim vectors, got {vectors.shape[1]}"
            )
        self._vectors = (
            vectors.astype(np.float32) if not self.size
            else np.vstack([self._vectors, vectors.astype(np.float32)])
        )
        self._chunks.extend(chunks)

    def search(self, query: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if not self.size or top_k <= 0:
            return []
        scores = self._vectors @ query.astype(np.float32)
        k = min(top_k, len(scores))
        # argpartition first: O(n) to isolate the top k, then sort only those.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [RetrievedChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._vectors)
        # JSONL, not one JSON array: a partially-written index is detectable
        # and skippable line by line instead of poisoning the whole file.
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path) -> NumpyVectorStore:
        vectors = np.load(directory / "vectors.npy")
        store = cls(int(vectors.shape[1]) if vectors.size else 0)
        chunks: list[CodeChunk] = []
        with (directory / "chunks.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    chunks.append(CodeChunk.from_dict(json.loads(line)))
        if len(chunks) != len(vectors):
            raise ValueError("index is inconsistent: vector/chunk count mismatch")
        store._vectors = vectors.astype(np.float32)
        store._chunks = chunks
        return store


class FaissVectorStore(VectorStore):
    """FAISS-backed exact inner-product search.

    Kept behind the same interface and only selected for genuinely large
    corpora (see `build_vector_store`). Uses IndexFlatIP -- exact, not
    approximate -- because at this scale correctness is free and a recall
    regression in a maintainer-facing evidence section is not worth the
    milliseconds.
    """

    def __init__(self, dimensions: int) -> None:
        import faiss

        self.dimensions = dimensions
        self._index = faiss.IndexFlatIP(dimensions)
        self._chunks: list[CodeChunk] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: list[CodeChunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        if not len(chunks):
            return
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._chunks.extend(chunks)

    def search(self, query: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if not self.size or top_k <= 0:
            return []
        scores, indices = self._index.search(
            np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32),
            min(top_k, self.size),
        )
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(s))
            for s, i in zip(scores[0], indices[0]) if i >= 0
        ]

    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path) -> FaissVectorStore:
        import faiss

        index = faiss.read_index(str(directory / "index.faiss"))
        store = cls.__new__(cls)
        store.dimensions = index.d
        store._index = index
        store._chunks = []
        with (directory / "chunks.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    store._chunks.append(CodeChunk.from_dict(json.loads(line)))
        return store


def build_vector_store(dimensions: int, expected_chunks: int = 0) -> VectorStore:
    """Pick a backend. Callers never name one."""
    if expected_chunks >= _FAISS_MIN_CHUNKS:
        try:
            return FaissVectorStore(dimensions)
        except ImportError:
            logger.info("faiss not installed; using exact numpy search for %d chunks",
                        expected_chunks)
    return NumpyVectorStore(dimensions)


def load_vector_store(directory: Path) -> VectorStore:
    if (directory / "index.faiss").exists():
        return FaissVectorStore.load(directory)
    return NumpyVectorStore.load(directory)


# ---------------------------------------------------------------------------
# Indexing pipeline
# ---------------------------------------------------------------------------
class RepositoryIndexer:
    """Checkout directory -> chunks -> embeddings -> a persisted index.

    Deliberately does no cloning and no HTTP of its own: it is handed a
    directory that already exists, which makes it trivially testable against
    a fixture tree and reusable by a future commit/PR indexer that gets its
    files from somewhere else entirely.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        cfg: RepositoryIntelligenceConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.cfg = cfg or RepositoryIntelligenceConfig()

    def build(
        self, repo: str, root: Path, *, default_branch: str = "", commit_sha: str = "",
    ) -> tuple[VectorStore, object]:
        """Index `root`. Returns the store and the detected metadata.

        Raises nothing the caller has to catch beyond genuine I/O failure --
        an empty repository yields an empty store and metadata with
        chunk_count=0, which downstream renders as "no evidence found"
        rather than an error.
        """
        from . import parser  # local: keeps module import cheap for consumers

        started = time.time()
        files = parser.walk_repository(root, self.cfg)
        metadata = parser.detect_metadata(
            root, repo, files, self.cfg,
            default_branch=default_branch, commit_sha=commit_sha,
        )

        chunks: list[CodeChunk] = []
        for path in files:
            if len(chunks) >= self.cfg.max_chunks:
                logger.warning("max_chunks=%d reached for %s; index is partial",
                               self.cfg.max_chunks, repo)
                break
            relative = path.relative_to(root).as_posix()
            text = parser.read_text(path)
            if text:
                chunks.extend(parser.chunk_file(repo, relative, text, self.cfg))
        chunks = chunks[:self.cfg.max_chunks]

        store = build_vector_store(self.embedder.dimensions, expected_chunks=len(chunks))
        if chunks:
            vectors = self.embedder.embed_documents([c.embedding_text() for c in chunks])
            store.add(vectors, chunks)

        from dataclasses import replace

        metadata = replace(metadata, chunk_count=len(chunks))
        logger.info(
            "indexed %s: %d files -> %d chunks in %.1fs (%s)",
            repo, len(files), len(chunks), time.time() - started, self.embedder.name,
        )
        return store, metadata
