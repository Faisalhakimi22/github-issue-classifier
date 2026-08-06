"""Repository Intelligence Engine: retrieval-augmented reasoning over the
code an issue is actually about.

The pipeline this adds, between ML scoring and LLM reasoning:

    issue -> query construction -> semantic search over an indexed
    repository -> retrieved code chunks -> engineering context -> prompt

The rule the whole module is built around: **every statement about a
repository must come from a retrieved chunk.** That isn't enforced by asking
the LLM nicely. The prompt only ever contains chunks that were actually
retrieved (`prompts.build_repository_section`), and the comment's evidence
section is rendered in Python from `RetrievedChunk` metadata rather than
from model output (`service/inference.py`) -- so a file path in a GHIC
comment is a path that exists in the index, structurally, not aspirationally.
When retrieval finds nothing above the confidence floor, every layer says
so: `RepositoryContext.is_empty`, the prompt's explicit "no repository
context available" line, and the comment's "No directly related source files
were confidently identified."

Public surface is `RepositoryIntelligenceService` plus the dataclasses.
Everything else (parsers, stores, providers) is swappable behind an
interface -- see `models/REPOSITORY_INTELLIGENCE_CARD.md`.
"""
from .config import RepositoryIntelligenceConfig
from .embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    HashingEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    build_embedding_provider,
)
from .indexer import NumpyVectorStore, RepositoryIndexer, VectorStore
from .models import (
    EMPTY_CONTEXT_NOTE,
    CodeChunk,
    RepositoryContext,
    RepositoryMetadata,
    RetrievedChunk,
)
from .repository_service import RepositoryIntelligenceService
from .retriever import SemanticRetriever, build_query

__all__ = [
    "RepositoryIntelligenceConfig",
    "RepositoryIntelligenceService",
    "RepositoryIndexer",
    "SemanticRetriever",
    "EmbeddingProvider",
    "EmbeddingError",
    "HashingEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "build_embedding_provider",
    "VectorStore",
    "NumpyVectorStore",
    "CodeChunk",
    "RetrievedChunk",
    "RepositoryContext",
    "RepositoryMetadata",
    "EMPTY_CONTEXT_NOTE",
    "build_query",
]
