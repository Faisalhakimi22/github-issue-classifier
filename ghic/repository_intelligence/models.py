"""Domain types for the Repository Intelligence Engine.

Frozen dataclasses + manual validation, matching the rest of the codebase
(service/inference.py's `Prediction`, llm/models.py's `IssueAnalysis`)
rather than introducing pydantic for one subsystem.

The type that matters most here is `RetrievedChunk`: it carries the file
path, line span, and symbol name of a chunk that was *actually retrieved
from an indexed repository*. Everything the GitHub comment says about a
repository is rendered from these objects in Python -- never from LLM
output -- which is what makes the "never hallucinate a file" rule
mechanical rather than a request the model can ignore. See
repository_service.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# What produced a chunk. `code` is Phase 1; the rest are Phase 2 corpora
# (see sources.py). This is the field that let commit/PR/issue intelligence
# reuse the entire Phase 1 pipeline -- embedder, vector store, retriever,
# prompt builder -- rather than growing a parallel one: they differ in
# where the text comes from, not in what happens to it afterwards.
SOURCE_CODE = "code"
SOURCE_COMMIT = "commit"
SOURCE_ISSUE = "issue"
SOURCE_PULL_REQUEST = "pull_request"

ALL_SOURCES = (SOURCE_CODE, SOURCE_COMMIT, SOURCE_ISSUE, SOURCE_PULL_REQUEST)


@dataclass(frozen=True)
class CodeChunk:
    """One semantically-bounded, retrievable unit.

    For source files (`source="code"`) the boundaries come from the
    language's own structure -- a function, method, class, or markdown
    section -- never a fixed character window, which would cut mid-statement
    and produce chunks that retrieve well but read as nonsense when handed
    to an LLM. See parser.py.

    Phase 2 reuses this type for commits, pull requests, and resolved
    issues, where the natural unit is the whole record rather than a slice
    of one. `path` carries a stable synthetic identifier in those cases
    (`commit:abc1234`, `issue:412`) and `reference` carries the
    human-facing one, so nothing downstream has to special-case them --
    including the anti-hallucination guarantee, which still holds because
    every rendered reference comes from a retrieved chunk.
    """
    repo: str
    path: str
    language: str
    text: str
    start_line: int
    end_line: int
    kind: str = "module"                 # function | method | class | section | module
    symbol: str = ""                     # function/class/heading name, "" for whole-module
    parent_symbol: str = ""              # enclosing class for methods
    source: str = SOURCE_CODE
    reference: str = ""                  # "#412", "abc1234" -- what a human cites
    url: str = ""
    timestamp: float = 0.0               # authored/closed time, for recency ranking

    @property
    def chunk_id(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def qualified_symbol(self) -> str:
        if self.parent_symbol and self.symbol:
            return f"{self.parent_symbol}.{self.symbol}"
        return self.symbol

    def embedding_text(self) -> str:
        """What actually gets embedded.

        Prefixed with the path and symbol so a query mentioning a filename
        or function name can match lexically as well as semantically --
        issue reporters name files and functions constantly, and dropping
        that signal to embed the bare body measurably hurts retrieval on
        exactly the queries that should be easiest.
        """
        header = f"{self.path}"
        if self.qualified_symbol:
            header += f" {self.qualified_symbol}"
        return f"{header}\n{self.text}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "path": self.path,
            "language": self.language,
            "text": self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "kind": self.kind,
            "symbol": self.symbol,
            "parent_symbol": self.parent_symbol,
            "source": self.source,
            "reference": self.reference,
            "url": self.url,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CodeChunk:
        # The Phase 2 fields default, so an index written by Phase 1 loads
        # unchanged and reads as `source="code"` -- which is what it is.
        return cls(
            repo=raw["repo"],
            path=raw["path"],
            language=raw["language"],
            text=raw["text"],
            start_line=int(raw["start_line"]),
            end_line=int(raw["end_line"]),
            kind=raw.get("kind", "module"),
            symbol=raw.get("symbol", ""),
            parent_symbol=raw.get("parent_symbol", ""),
            source=raw.get("source", SOURCE_CODE),
            reference=raw.get("reference", ""),
            url=raw.get("url", ""),
            timestamp=float(raw.get("timestamp", 0) or 0),
        )


@dataclass(frozen=True)
class RepositoryMetadata:
    """What kind of project this is, inferred from files on disk.

    Every field is evidence-backed: `primary_language` comes from counting
    indexed source files, `frameworks`/`dependency_manager` from manifest
    files that were actually read. Nothing here is guessed from the repo
    name.
    """
    repo: str
    default_branch: str = ""
    commit_sha: str = ""
    primary_language: str = ""
    languages: dict[str, int] = field(default_factory=dict)   # language -> file count
    frameworks: list[str] = field(default_factory=list)
    dependency_managers: list[str] = field(default_factory=list)
    project_type: str = ""                                    # library | application | ...
    entry_points: list[str] = field(default_factory=list)
    top_level_dirs: list[str] = field(default_factory=list)
    readme_summary: str = ""
    file_count: int = 0
    chunk_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "default_branch": self.default_branch,
            "commit_sha": self.commit_sha,
            "primary_language": self.primary_language,
            "languages": self.languages,
            "frameworks": self.frameworks,
            "dependency_managers": self.dependency_managers,
            "project_type": self.project_type,
            "entry_points": self.entry_points,
            "top_level_dirs": self.top_level_dirs,
            "readme_summary": self.readme_summary,
            "file_count": self.file_count,
            "chunk_count": self.chunk_count,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RepositoryMetadata:
        return cls(
            repo=raw["repo"],
            default_branch=raw.get("default_branch", ""),
            commit_sha=raw.get("commit_sha", ""),
            primary_language=raw.get("primary_language", ""),
            languages=dict(raw.get("languages") or {}),
            frameworks=list(raw.get("frameworks") or []),
            dependency_managers=list(raw.get("dependency_managers") or []),
            project_type=raw.get("project_type", ""),
            entry_points=list(raw.get("entry_points") or []),
            top_level_dirs=list(raw.get("top_level_dirs") or []),
            readme_summary=raw.get("readme_summary", ""),
            file_count=int(raw.get("file_count", 0)),
            chunk_count=int(raw.get("chunk_count", 0)),
        )


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk the retriever actually returned, with its similarity score.

    `score` is cosine similarity in [-1, 1] (in practice [0, 1] for the
    non-negative embeddings both shipped providers produce). It is a
    *retrieval* score, not a calibrated probability that this file is
    relevant -- the comment never presents it as one.
    """
    chunk: CodeChunk
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {**self.chunk.as_dict(), "score": round(self.score, 4)}


@dataclass(frozen=True)
class RepositoryContext:
    """The engineering context handed to the LLM and rendered as evidence.

    `is_empty` is the single question every caller asks: when it's True the
    prompt gets no repository section at all and the comment says so
    plainly, rather than either side inventing filler. An unindexed repo, a
    failed index, and a repo where nothing scored above the similarity
    floor all produce an explicit, non-speculative answer.
    """
    repo: str
    metadata: RepositoryMetadata | None = None
    chunks: list[RetrievedChunk] = field(default_factory=list)
    indexed: bool = False
    indexing_queued: bool = False
    note: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def code_chunks(self) -> list[RetrievedChunk]:
        return [rc for rc in self.chunks if rc.chunk.source == SOURCE_CODE]

    @property
    def history_chunks(self) -> list[RetrievedChunk]:
        """Commits, PRs, and resolved issues -- the Phase 2 corpora."""
        return [rc for rc in self.chunks if rc.chunk.source != SOURCE_CODE]

    @property
    def relevant_files(self) -> list[str]:
        """Unique source-file paths, most-relevant first.

        Code only: a commit or issue chunk has a synthetic `path`
        (`commit:abc1234`) that is not a file and must never be rendered
        as one.
        """
        return list(dict.fromkeys(rc.chunk.path for rc in self.code_chunks))

    @property
    def relevant_symbols(self) -> list[str]:
        seen = {rc.chunk.qualified_symbol for rc in self.chunks}
        ordered = [rc.chunk.qualified_symbol for rc in self.chunks if rc.chunk.qualified_symbol]
        return list(dict.fromkeys(s for s in ordered if s in seen))

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "indexed": self.indexed,
            "indexing_queued": self.indexing_queued,
            "note": self.note,
            "metadata": self.metadata.as_dict() if self.metadata else None,
            "relevant_files": self.relevant_files,
            "relevant_symbols": self.relevant_symbols,
            "chunks": [rc.as_dict() for rc in self.chunks],
        }


EMPTY_CONTEXT_NOTE = "No directly related source files were confidently identified."
UNAVAILABLE_CONTEXT_NOTE = (
    "Repository evidence unavailable: repository intelligence could not retrieve "
    "indexed code. This analysis is based on the issue description only."
)
