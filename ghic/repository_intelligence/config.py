"""Configuration for the Repository Intelligence Engine.

Separate from ServiceSettings on purpose: the engine is usable outside the
webhook service (a CLI reindex, a notebook, a future Commit/PR intelligence
module) and shouldn't need a fully-populated web-service settings object to
construct. `ServiceSettings` holds the operator-facing on/off toggle; this
holds the engine's own tuning, and `from_env()` is the one place that reads
the environment.

Every limit here exists because a repository is untrusted input: a repo can
contain a 400MB vendored blob, a single 200k-line generated file, or half a
million files. The defaults are chosen to keep one indexing run bounded in
time and memory rather than to maximize recall on a pathological repo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .. import utils

# Source files worth indexing, extension -> language name. Markdown is in
# here deliberately: READMEs and architecture docs answer "what is this
# project" better than any source file, and they retrieve well.
LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "Python",
    ".pyi": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".cs": "C#",
    ".md": "Markdown",
    ".mdx": "Markdown",
    ".rst": "reStructuredText",
}

# Directories never worth indexing: dependencies, build output, VCS
# internals, and caches. Matched on any path segment, so `a/node_modules/b`
# is excluded as surely as a top-level `node_modules`.
EXCLUDED_DIRECTORIES: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "bower_components", "vendor", "third_party",
    "dist", "build", "out", "target", "bin", "obj",
    ".next", ".nuxt", ".svelte-kit", ".output",
    "coverage", "htmlcov", ".nyc_output",
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox",
    ".venv", "venv", "env", "site-packages",
    ".idea", ".vscode", ".gradle", ".terraform",
    "migrations",            # usually generated, always noisy
    "__snapshots__", "fixtures", "testdata",
})

# Filename suffixes that are generated or vendored even inside an otherwise
# indexable directory. Lock files carry no engineering meaning and are huge.
EXCLUDED_FILE_SUFFIXES: tuple[str, ...] = (
    ".min.js", ".min.css", ".bundle.js", ".map",
    ".lock", "-lock.json", ".sum",
    ".pb.go", ".pb.cc", ".pb.h", "_pb2.py", "_pb2_grpc.py",
    ".generated.ts", ".g.dart", ".designer.cs",
    ".snap", ".pyc", ".pyo", ".so", ".dll", ".dylib", ".class", ".jar",
)

EXCLUDED_FILENAMES: frozenset[str] = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Cargo.lock", "Gemfile.lock", "composer.lock", "go.sum",
})

# Manifest -> (dependency manager, language hint). Read to detect the
# project's toolchain, and (for frameworks) scanned for known dependency
# names -- see parser.detect_metadata().
DEPENDENCY_MANIFESTS: dict[str, str] = {
    "pyproject.toml": "pip/pyproject",
    "requirements.txt": "pip",
    "Pipfile": "pipenv",
    "setup.py": "setuptools",
    "package.json": "npm/yarn/pnpm",
    "go.mod": "go modules",
    "Cargo.toml": "cargo",
    "pom.xml": "maven",
    "build.gradle": "gradle",
    "build.gradle.kts": "gradle",
    "Gemfile": "bundler",
    "composer.json": "composer",
}

# Dependency name -> framework label. Deliberately small and high-precision:
# a wrong framework label in a maintainer-facing comment is worse than none.
FRAMEWORK_MARKERS: dict[str, str] = {
    "fastapi": "FastAPI",
    "django": "Django",
    "flask": "Flask",
    "starlette": "Starlette",
    "scikit-learn": "scikit-learn",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "react": "React",
    "next": "Next.js",
    "vue": "Vue",
    "svelte": "Svelte",
    "angular": "Angular",
    "express": "Express",
    "nestjs": "NestJS",
    "@nestjs/core": "NestJS",
    "gin-gonic/gin": "Gin",
    "labstack/echo": "Echo",
    "actix-web": "Actix Web",
    "rocket": "Rocket",
    "axum": "Axum",
    "spring-boot-starter": "Spring Boot",
    "aspnetcore": "ASP.NET Core",
}

# Files that, when present at the repo root, name the project's entry point.
ENTRY_POINT_CANDIDATES: tuple[str, ...] = (
    "main.py", "app.py", "__main__.py", "manage.py", "wsgi.py", "asgi.py",
    "index.js", "index.ts", "main.js", "main.ts", "server.js", "server.ts",
    "main.go", "main.rs", "Program.cs", "Main.java",
)


@dataclass(frozen=True)
class RepositoryIntelligenceConfig:
    # --- Clone cache -------------------------------------------------
    cache_dir: Path = field(default_factory=lambda: utils.DATA_RAW / "repo_intel")
    clone_depth: int = 1                  # shallow: history is a later phase's problem
    git_timeout_seconds: float = 120.0

    # --- Indexing limits ---------------------------------------------
    max_file_bytes: int = 400_000         # a single source file bigger than this is generated
    max_files: int = 4_000                # hard ceiling on one indexing run
    max_chunks: int = 20_000
    max_chunk_chars: int = 6_000          # oversized functions get split at this width
    min_chunk_chars: int = 40             # one-line stubs add noise, not signal

    # --- Retrieval ----------------------------------------------------
    top_k: int = 10
    min_similarity: float = 0.15          # below this, "no confident match" is the honest answer
    max_files_in_prompt: int = 6
    max_chunk_chars_in_prompt: int = 1_200
    readme_summary_chars: int = 800

    # --- Embeddings ---------------------------------------------------
    embedding_provider: str = "hashing"   # hashing | openai
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_dimensions: int = 512
    embedding_batch_size: int = 64
    embedding_timeout: float = 30.0

    # --- Behaviour ----------------------------------------------------
    index_ttl_seconds: int = 7 * 24 * 60 * 60   # re-index weekly even if the SHA is unknown

    @property
    def clones_dir(self) -> Path:
        return self.cache_dir / "clones"

    @property
    def index_dir(self) -> Path:
        return self.cache_dir / "index"

    @classmethod
    def from_env(cls) -> RepositoryIntelligenceConfig:
        """Build from GHIC_REPO_INTEL_* environment variables.

        The embedding key follows the existing convention of matching the
        provider's own variable name (OPENAI_API_KEY, like GROQ_API_KEY in
        settings.py) so a key already set for another tool works here.
        """
        cache_dir = os.environ.get("GHIC_REPO_INTEL_CACHE_DIR", "")
        return cls(
            cache_dir=Path(cache_dir) if cache_dir else utils.DATA_RAW / "repo_intel",
            max_files=_env_int("GHIC_REPO_INTEL_MAX_FILES", cls.max_files),
            max_chunks=_env_int("GHIC_REPO_INTEL_MAX_CHUNKS", cls.max_chunks),
            top_k=_env_int("GHIC_REPO_INTEL_TOP_K", cls.top_k),
            min_similarity=_env_float("GHIC_REPO_INTEL_MIN_SIMILARITY", cls.min_similarity),
            embedding_provider=os.environ.get(
                "GHIC_REPO_INTEL_EMBEDDING_PROVIDER", cls.embedding_provider
            ).strip().lower(),
            embedding_model=os.environ.get(
                "GHIC_REPO_INTEL_EMBEDDING_MODEL", cls.embedding_model
            ),
            embedding_api_key=(
                os.environ.get("GHIC_REPO_INTEL_EMBEDDING_API_KEY")
                or os.environ.get("OPENAI_API_KEY", "")
            ),
            embedding_base_url=os.environ.get(
                "GHIC_REPO_INTEL_EMBEDDING_BASE_URL", cls.embedding_base_url
            ),
            embedding_dimensions=_env_int(
                "GHIC_REPO_INTEL_EMBEDDING_DIMENSIONS", cls.embedding_dimensions
            ),
        )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default
