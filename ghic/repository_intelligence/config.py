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
import re
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

# Files that may hold credentials. Excluded by *name*, independently of the
# extension allowlist, because the allowlist is the wrong place to rely on
# for this: it happens to exclude `.env` and `.pem` today, and one added
# extension later would silently start indexing them. A secret that reaches
# the index reaches the embedding provider, the vector database, and
# potentially a public GitHub comment -- so this is defence in depth on
# purpose, not redundancy. See `is_sensitive_path`.
SENSITIVE_FILENAME_PATTERNS: tuple[str, ...] = (
    r"(^|/)\.env($|\.)",                       # .env, .env.local, .env.production
    r"(^|/)\.npmrc$", r"(^|/)\.pypirc$", r"(^|/)\.netrc$",
    r"(^|/)\.htpasswd$",
    r"(^|/)id_(rsa|dsa|ecdsa|ed25519)($|\.)",
    r"(^|/)\.ssh/",
    r"(^|/)credentials?($|\.)", r"(^|/)secrets?($|\.)",
    r"\.(pem|key|p12|pfx|jks|keystore|asc|gpg|ppk)$",
    r"(^|/)service[-_]?account.*\.json$",
    r"(^|/)(kubeconfig|\.kube/config)$",
    r"\.tfstate(\.backup)?$",                  # terraform state embeds secrets
)

_SENSITIVE_RE = re.compile("|".join(SENSITIVE_FILENAME_PATTERNS), re.IGNORECASE)

# Value-shaped secrets that appear *inside* otherwise-indexable files: a
# real key pasted into a README, a token in a config module, a private key
# block in a test fixture. Redacted before chunking, so no secret is ever
# embedded, stored, or shown as evidence.
_SECRET_VALUE_PATTERNS: tuple[str, ...] = (
    r"-----BEGIN[ A-Z]*PRIVATE KEY-----[\s\S]*?-----END[ A-Z]*PRIVATE KEY-----",
    r"\bgh[pousr]_[A-Za-z0-9]{20,}",                  # GitHub tokens
    r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    r"\bsk-(?:or-|ant-|proj-)?[A-Za-z0-9_-]{20,}",     # OpenAI/Anthropic/OpenRouter
    r"\bgsk_[A-Za-z0-9]{20,}",                         # Groq
    r"\bxox[abprs]-[A-Za-z0-9-]{10,}",                 # Slack
    r"\bAKIA[0-9A-Z]{16}\b",                           # AWS access key id
    r"\bAIza[0-9A-Za-z_-]{35}\b",                      # Google API key
    r"\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",   # JWT
    # Scoped (?i:...) groups, not a bare (?i): Python treats an inline (?i)
    # as a *global* flag and rejects it anywhere but the very start of a
    # pattern, which these become the moment they're joined with "|".
    r"(?i:\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://"
    r"[^\s'\"]*:[^\s'\"@]+@[^\s'\"]+)",
    # key = "value" / KEY: value, where the name looks credential-ish
    r"(?i:\b\w*(?:secret|password|passwd|api[_-]?key|access[_-]?token|"
    r"auth[_-]?token|private[_-]?key|client[_-]?secret)\w*\s*[=:]\s*"
    r"['\"][^'\"\n]{8,}['\"])",
)

_SECRET_VALUE_RE = re.compile("|".join(_SECRET_VALUE_PATTERNS))

REDACTION_PLACEHOLDER = "[REDACTED-SECRET]"


def is_sensitive_path(relative_path: str) -> bool:
    """True for files that must never be indexed, whatever their extension."""
    return bool(_SENSITIVE_RE.search(relative_path.replace("\\", "/")))


def redact_secrets(text: str) -> str:
    """Replace credential-shaped values with a placeholder.

    Applied to every file before chunking. Pattern-based, so it catches
    known token shapes and `password = "..."` assignments, not arbitrary
    high-entropy strings -- an entropy heuristic on source code flags
    hashes, UUIDs, and base64 test fixtures constantly, and a redactor that
    shreds ordinary code is one an operator turns off. The file-name
    exclusions above are the primary defence; this is the second layer for
    secrets committed somewhere they don't belong.
    """
    return _SECRET_VALUE_RE.sub(REDACTION_PLACEHOLDER, text)

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

    # --- Infrastructure providers (Phase 1.5) --------------------------
    # Names, not classes: the factories in providers.py resolve them, and
    # nothing else in the engine branches on these strings.
    vector_provider: str = "local"        # local | postgres
    state_provider: str = "auto"          # auto | postgres | file | memory
    queue_provider: str = "auto"          # auto | qstash | inline | none
    database_url: str = ""                # shared with the ledger/idempotency store

    # --- Feature flags --------------------------------------------------
    # Each gates one capability so a rollout can be narrowed without a
    # redeploy (they are read from the environment, and the service reads
    # config at construction).
    incremental_indexing: bool = True
    vector_search_enabled: bool = True
    repo_summarization: bool = True
    auto_index: bool = True               # queue an index for unknown repos

    # --- Phase 2 corpora (commits / PRs / resolved issues) -------------
    # Each is separately gated: they have different costs (commit history
    # needs a deeper clone, issue history needs API calls or collected
    # data) and different value per repository, so an operator should be
    # able to run one without the others.
    index_commits: bool = False
    index_issue_history: bool = False
    max_commits: int = 2_000
    max_history_items: int = 2_000
    # Slots reserved for history in a retrieval result. Code and history
    # answer different questions ("where is this" vs "how was this handled
    # before"), so they get separate budgets rather than competing --
    # the same lesson the docs-vs-code cap taught in Phase 1.
    history_result_slots: int = 3

    # --- Behaviour ----------------------------------------------------
    index_ttl_seconds: int = 7 * 24 * 60 * 60   # re-index weekly even if the SHA is unknown
    max_incremental_files: int = 200      # above this, a full rebuild is cheaper
    stale_index_days: int = 30            # cleanup: drop indexes untouched this long
    retrieval_cache_ttl_seconds: int = 300

    @property
    def clones_dir(self) -> Path:
        return self.cache_dir / "clones"

    @property
    def index_dir(self) -> Path:
        return self.cache_dir / "index"

    @property
    def uses_persistent_vectors(self) -> bool:
        """True when the index lives somewhere that survives a restart.

        This is the question that decides whether the feature is worth
        running at all on a given platform -- see `ephemeral_filesystem()`
        and the auto-disable in providers.build_service().
        """
        return self.vector_provider == "postgres" and bool(self.database_url)

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
            vector_provider=os.environ.get(
                "GHIC_VECTOR_PROVIDER", cls.vector_provider
            ).strip().lower(),
            state_provider=os.environ.get(
                "GHIC_STATE_PROVIDER", cls.state_provider
            ).strip().lower(),
            queue_provider=os.environ.get(
                "GHIC_INDEX_QUEUE_PROVIDER", cls.queue_provider
            ).strip().lower(),
            # Same precedence the ledger uses, so one Postgres serves every
            # subsystem without a second connection string to configure.
            database_url=(
                os.environ.get("GHIC_DATABASE_URL")
                or os.environ.get("DATABASE_URL")
                or os.environ.get("POSTGRES_URL", "")
            ),
            incremental_indexing=_env_bool(
                "GHIC_INCREMENTAL_INDEXING", cls.incremental_indexing
            ),
            vector_search_enabled=_env_bool("GHIC_VECTOR_SEARCH", cls.vector_search_enabled),
            repo_summarization=_env_bool("GHIC_REPO_SUMMARIZATION", cls.repo_summarization),
            auto_index=_env_bool("GHIC_REPO_AUTO_INDEX", cls.auto_index),
            max_incremental_files=_env_int(
                "GHIC_INDEX_BATCH_SIZE", cls.max_incremental_files
            ),
            index_ttl_seconds=_env_int("GHIC_REPO_CACHE_TTL", cls.index_ttl_seconds),
            stale_index_days=_env_int("GHIC_REPO_STALE_DAYS", cls.stale_index_days),
            index_commits=_env_bool("GHIC_COMMIT_INTELLIGENCE", cls.index_commits),
            index_issue_history=_env_bool(
                "GHIC_HISTORY_INTELLIGENCE", cls.index_issue_history
            ),
            max_commits=_env_int("GHIC_MAX_COMMITS", cls.max_commits),
            max_history_items=_env_int("GHIC_MAX_HISTORY_ITEMS", cls.max_history_items),
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def ephemeral_filesystem() -> bool:
    """Whether this process is running somewhere its disk won't survive.

    Detected from the platform's own environment markers rather than
    guessed. This is what powers the auto-disable in providers.py: on a
    serverless platform a local-filesystem index is rebuilt on every cold
    start, which is slow, expensive (with a paid embedding provider), and
    silently useless -- so the engine refuses to run that way unless a
    persistent vector backend is configured.
    """
    return bool(
        os.environ.get("VERCEL")
        or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        or os.environ.get("FUNCTIONS_WORKER_RUNTIME")      # Azure Functions
        or os.environ.get("K_SERVICE")                     # Cloud Run / Knative
    )


def platform_name() -> str:
    """Best-effort platform label, for logs and /healthz."""
    for env_var, label in (
        ("VERCEL", "vercel"),
        ("AWS_LAMBDA_FUNCTION_NAME", "aws-lambda"),
        ("FUNCTIONS_WORKER_RUNTIME", "azure-functions"),
        ("K_SERVICE", "cloud-run"),
        ("FLY_APP_NAME", "fly.io"),
        ("RAILWAY_ENVIRONMENT", "railway"),
        ("RENDER", "render"),
        ("KUBERNETES_SERVICE_HOST", "kubernetes"),
    ):
        if os.environ.get(env_var):
            return label
    return "local"


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
