"""Embedding provider abstraction.

Two implementations ship:

  `OpenAICompatibleEmbeddingProvider` -- any `/v1/embeddings` API (OpenAI,
  Jina, Voyage's compatible endpoint, a local llama.cpp or Ollama server).
  Set GHIC_REPO_INTEL_EMBEDDING_PROVIDER=openai plus a key.

  `HashingEmbeddingProvider` -- the default. Signed feature hashing over
  identifier-aware code tokens, L2-normalized. Deterministic, offline, no
  API key, no extra dependency, and no per-repo embedding cost.

The honest framing of the default, because it decides what this feature can
claim: hashing embeddings are **lexical, not semantic**. They match an issue
mentioning `parse_csv` to the function named `parse_csv`, and an issue about
"UTF-8 decoding" to code containing those tokens. They will not match
"the importer chokes on foreign characters" to `decode()` with no shared
vocabulary -- a real embedding model would. That tradeoff is deliberate:
lexical retrieval that always works, costs nothing, and ships inside a
serverless bundle beats semantic retrieval that requires an API key most
deployments won't set. Operators who want semantic matching set the
provider to `openai`, and nothing else in the engine changes.

Identifier splitting is what makes the lexical default work as well as it
does: `parse_csv_file` indexes as `parse`, `csv`, `file` *and* the whole
token, so an issue saying "CSV parsing" hits it without an exact match.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .. import utils
from .config import RepositoryIntelligenceConfig

logger = utils.get_logger(__name__)


class EmbeddingError(RuntimeError):
    """Any failure to produce embeddings. Callers degrade, never crash."""


@dataclass(frozen=True)
class EmbeddingMetadata:
    """Stable identity for vectors written by one embedding configuration."""

    provider: str
    model: str
    dimensions: int

    @property
    def signature(self) -> str:
        return f"{self.provider}:{self.model}:{self.dimensions}"

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "signature": self.signature,
        }


class EmbeddingProvider(ABC):
    """One provider = one way to turn texts into unit vectors.

    `embed_documents` and `embed_query` are separate because asymmetric
    models (and future rerankers) prefix the two differently; for symmetric
    models the second just delegates to the first.
    """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier baked into the index cache key, so switching
        providers or dimensions invalidates a stale index automatically
        instead of silently comparing incompatible vectors."""
        raise NotImplementedError

    @property
    def provider_id(self) -> str:
        """Provider family, separate from model and dimensions."""
        return self.name.split(":", 1)[0].split("-", 1)[0]

    @property
    def model_id(self) -> str:
        """Model identity within the provider family."""
        return self.name

    def metadata(self) -> EmbeddingMetadata:
        return EmbeddingMetadata(
            provider=self.provider_id,
            model=self.model_id,
            dimensions=self.dimensions,
        )

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """(len(texts), dimensions) float32, L2-normalized row-wise."""
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# Default: dependency-free lexical hashing
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Tokens present in nearly every source file carry no discriminative signal
# and would otherwise dominate short chunks.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "import", "return",
    "def", "class", "self", "none", "true", "false", "null", "if", "else",
    "int", "str", "bool", "var", "let", "const", "function", "new", "public",
    "private", "static", "void", "func", "fn", "pub", "use", "package",
})


def tokenize(text: str) -> list[str]:
    """Identifier-aware tokens: whole identifier plus its sub-words.

    `parse_csv_file` -> ["parse_csv_file", "parse", "csv", "file"]
    `parseCSVFile`   -> ["parsecsvfile", "parse", "csv", "file"]
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        lowered = raw.lower()
        if len(lowered) > 1 and lowered not in _STOPWORDS:
            tokens.append(lowered)
        parts = [p for chunk in raw.split("_") for p in _CAMEL_BOUNDARY.split(chunk)]
        if len(parts) > 1:
            tokens.extend(
                p.lower() for p in parts
                if len(p) > 1 and p.lower() not in _STOPWORDS
            )
    return tokens


class HashingEmbeddingProvider(EmbeddingProvider):
    """Signed feature hashing with sublinear term-frequency scaling.

    Signed hashing (each token contributes +1 or -1 by an independent bit of
    its digest) keeps hash collisions unbiased in expectation instead of
    letting them inflate similarity, which is the standard fix and the
    reason this behaves like a real vector space rather than a bag of
    accidents.
    """

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"hashing-{self._dimensions}"

    @property
    def provider_id(self) -> str:
        return "hashing"

    @property
    def model_id(self) -> str:
        return "hashing"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            signs: dict[int, float] = {}
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                index = value % self._dimensions
                counts[index] = counts.get(index, 0.0) + 1.0
                signs.setdefault(index, 1.0 if (value >> 63) & 1 else -1.0)
            for index, count in counts.items():
                # Sublinear tf: a token repeated 50 times is more relevant
                # than one appearing once, but not 50x more.
                matrix[row, index] = signs[index] * (1.0 + np.log(count))
        return _l2_normalize(matrix)


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP provider
# ---------------------------------------------------------------------------
class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Any service exposing POST {base_url}/embeddings in OpenAI's shape.

    Batched, with the project's existing retry policy applied only to
    transient failures -- a 401 is never retried, matching
    llm/_chat_completions.py. Indexing runs in the background (see
    repository_service.py), so unlike the LLM path this can afford to
    retry a timeout.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dimensions: int = 512,
        batch_size: int = 64,
        timeout: float = 30.0,
        http_client: object | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for the OpenAI-compatible provider")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dimensions = dimensions
        self.batch_size = batch_size
        self.timeout = timeout
        self._client = http_client

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"openai:{self.model}-{self._dimensions}"

    @property
    def provider_id(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return self.model

    def _post(self, texts: list[str]) -> list[list[float]]:
        import requests

        client = self._client
        payload = {"model": self.model, "input": texts}
        # `dimensions` is an OpenAI v3 extension; older/compatible servers
        # reject unknown fields, so it is only sent when it would change
        # anything.
        if self._dimensions:
            payload["dimensions"] = self._dimensions

        def _call() -> list[list[float]]:
            owned = client is None
            active = requests.Session() if owned else client
            try:
                response = active.post(  # type: ignore[union-attr]
                    f"{self.base_url}/embeddings",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )
                if response.status_code >= 400:
                    detail = response.text[:200]
                    if response.status_code == 429 or response.status_code >= 500:
                        raise _TransientEmbeddingError(
                            f"{response.status_code} from embeddings API: {detail}"
                        )
                    raise EmbeddingError(f"{response.status_code} from embeddings API: {detail}")
                data = response.json().get("data")
                if not isinstance(data, list) or len(data) != len(texts):
                    raise EmbeddingError("unexpected embeddings response shape")
                return [item["embedding"] for item in data]
            finally:
                if owned:
                    active.close()  # type: ignore[union-attr]

        retrying = utils.retry_with_backoff(
            max_attempts=3, base_delay=1.0, max_delay=8.0,
            exceptions=(_TransientEmbeddingError, OSError),
            logger=logger,
        )(_call)
        try:
            return retrying()
        except _TransientEmbeddingError as e:
            raise EmbeddingError(str(e)) from e
        except EmbeddingError:
            raise
        except Exception as e:  # transport errors, malformed JSON, ...
            raise EmbeddingError(f"embeddings request failed: {e}") from e

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimensions), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._post(texts[start:start + self.batch_size]))
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape[1] != self._dimensions:
            # Trust the server over the config rather than truncating and
            # silently degrading every similarity in the index.
            logger.info("embeddings API returned %d dims (configured %d); using the API's",
                        matrix.shape[1], self._dimensions)
            self._dimensions = int(matrix.shape[1])
        return _l2_normalize(matrix)


class _TransientEmbeddingError(RuntimeError):
    """Internal retry marker; never escapes this module."""


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # An all-zero row (a chunk of pure stopwords) would divide by zero; it
    # stays zero, which scores 0 against every query -- correct, since it
    # carries no signal.
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32)


def build_embedding_provider(cfg: RepositoryIntelligenceConfig) -> EmbeddingProvider:
    """Factory: config -> provider. The only place a provider name string is
    branched on; everything downstream holds an `EmbeddingProvider`."""
    if cfg.embedding_provider == "openai":
        if not cfg.embedding_api_key:
            logger.warning(
                "embedding_provider=openai but no API key is set; "
                "falling back to the offline hashing provider"
            )
            return HashingEmbeddingProvider(cfg.embedding_dimensions or 512)
        return OpenAICompatibleEmbeddingProvider(
            api_key=cfg.embedding_api_key,
            model=cfg.embedding_model,
            base_url=cfg.embedding_base_url,
            dimensions=cfg.embedding_dimensions,
            batch_size=cfg.embedding_batch_size,
            timeout=cfg.embedding_timeout,
        )
    return HashingEmbeddingProvider(cfg.embedding_dimensions or 512)
