"""Clone cache and index cache.

Two independent caches with different invalidation rules:

  `RepositoryCache` -- a shallow git checkout per repository, refreshed with
  `git fetch` rather than re-cloned. Keyed by repo full name.

  `IndexCache` -- a persisted vector index per (repo, commit SHA, embedder).
  Including the commit SHA in the key is the whole invalidation strategy:
  a repository that hasn't changed has the same SHA and reuses its index; a
  repository that has changed produces a different key and gets rebuilt. No
  mtime heuristics, no manual invalidation, no way to serve an index that
  doesn't match the code it claims to describe. The embedder name is in the
  key too, so switching embedding providers or dimensions can't silently
  compare incompatible vectors.

Security note: an installation token is injected into the clone URL for
private repositories, and is deliberately never written to disk or logged.
`git remote set-url` rewrites the stored remote back to a tokenless URL
immediately after cloning, so the token doesn't persist in
`.git/config` inside the cache directory.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .. import utils
from .config import RepositoryIntelligenceConfig
from .indexer import INDEX_FORMAT_VERSION, VectorStore, load_vector_store
from .models import RepositoryMetadata

logger = utils.get_logger(__name__)


class RepositoryCacheError(RuntimeError):
    """Clone/fetch failed. Callers degrade to "not indexed", never crash."""


@dataclass(frozen=True)
class Checkout:
    path: Path
    commit_sha: str
    default_branch: str


def run_git(args: list[str], cwd: Path | None, timeout: float) -> str:
    """Run git, raising RepositoryCacheError with a redacted message.

    Public because incremental.py drives its own git plumbing (diff,
    cat-file) against a checkout this module produced.

    stderr from a failed authenticated clone can echo the remote URL, which
    contains the installation token; `_redact` strips it before the message
    reaches a log or an exception.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        raise RepositoryCacheError("git is not installed or not on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise RepositoryCacheError(f"git {args[0]} timed out after {timeout:.0f}s") from e
    if completed.returncode != 0:
        raise RepositoryCacheError(
            f"git {args[0]} failed ({completed.returncode}): "
            f"{_redact(completed.stderr.strip())[:300]}"
        )
    return completed.stdout.strip()


def _redact(text: str) -> str:
    """Strip credentials out of any URL in `text`."""
    import re

    return re.sub(r"(https://)[^@/\s]+@", r"\1***@", text)


class RepositoryCache:
    """Shallow checkouts of repositories, reused across webhook deliveries."""

    def __init__(self, cfg: RepositoryIntelligenceConfig | None = None) -> None:
        self.cfg = cfg or RepositoryIntelligenceConfig()

    def path_for(self, repo: str) -> Path:
        # Flatten owner/name so one directory level holds every repo and a
        # crafted repo name can't escape the cache root.
        safe = repo.replace("/", "__").replace("..", "__")
        return self.cfg.clones_dir / safe

    def is_cloned(self, repo: str) -> bool:
        return (self.path_for(repo) / ".git").is_dir()

    def ensure(self, repo: str, *, token: str = "", base_url: str = "https://github.com") -> Checkout:
        """Clone or update `repo`, returning the checkout and its commit SHA.

        Never re-clones an existing checkout: a fetch of a shallow clone
        transfers only what changed, which is the difference between a few
        hundred KB and a full repository download on every webhook.
        """
        target = self.path_for(repo)
        url = f"{base_url}/{repo}.git"
        authed = (
            f"https://x-access-token:{token}@{base_url.split('://', 1)[-1]}/{repo}.git"
            if token else url
        )

        if self.is_cloned(repo):
            self._update(target, authed, url)
        else:
            self._clone(repo, target, authed, url)

        commit_sha = run_git(["rev-parse", "HEAD"], target, self.cfg.git_timeout_seconds)
        branch = run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"], target, self.cfg.git_timeout_seconds
        )
        return Checkout(path=target, commit_sha=commit_sha, default_branch=branch)

    def _clone(self, repo: str, target: Path, authed_url: str, clean_url: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        logger.info("cloning %s (depth=%d)", repo, self.cfg.clone_depth)
        run_git(
            ["clone", "--depth", str(self.cfg.clone_depth), "--single-branch",
             "--no-tags", authed_url, str(target)],
            None, self.cfg.git_timeout_seconds,
        )
        # Drop the token from .git/config immediately.
        run_git(["remote", "set-url", "origin", clean_url], target,
                 self.cfg.git_timeout_seconds)

    def _update(self, target: Path, authed_url: str, clean_url: str) -> None:
        try:
            run_git(["remote", "set-url", "origin", authed_url], target,
                     self.cfg.git_timeout_seconds)
            run_git(["fetch", "--depth", str(self.cfg.clone_depth), "origin"], target,
                     self.cfg.git_timeout_seconds)
            run_git(["reset", "--hard", "FETCH_HEAD"], target, self.cfg.git_timeout_seconds)
            run_git(["clean", "-fdx"], target, self.cfg.git_timeout_seconds)
        finally:
            # Restore the tokenless remote even if the fetch failed.
            try:
                run_git(["remote", "set-url", "origin", clean_url], target,
                         self.cfg.git_timeout_seconds)
            except RepositoryCacheError:
                pass

    def remove(self, repo: str) -> None:
        shutil.rmtree(self.path_for(repo), ignore_errors=True)


class IndexCache:
    """Persisted vector indexes, keyed by (repo, commit SHA, embedder)."""

    def __init__(self, cfg: RepositoryIntelligenceConfig | None = None) -> None:
        self.cfg = cfg or RepositoryIntelligenceConfig()

    def key(self, repo: str, commit_sha: str, embedder_name: str) -> str:
        return utils.cache_key("repo-index", repo, commit_sha, embedder_name)

    def directory(self, repo: str, commit_sha: str, embedder_name: str) -> Path:
        return self.cfg.index_dir / self.key(repo, commit_sha, embedder_name)

    def has(self, repo: str, commit_sha: str, embedder_name: str) -> bool:
        return (self.directory(repo, commit_sha, embedder_name) / "manifest.json").is_file()

    def load(
        self, repo: str, commit_sha: str, embedder_name: str
    ) -> tuple[VectorStore, RepositoryMetadata] | None:
        """None on any miss, corruption, staleness, or version mismatch.

        A cache is an optimization; a broken one degrades to a rebuild, and
        must never propagate an exception into a webhook.
        """
        directory = self.directory(repo, commit_sha, embedder_name)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format_version") != INDEX_FORMAT_VERSION:
                return None
            age = time.time() - float(manifest.get("built_at", 0))
            if age > self.cfg.index_ttl_seconds:
                logger.info("index for %s is %.1f days old; rebuilding", repo, age / 86400)
                return None
            store = load_vector_store(directory)
            metadata = RepositoryMetadata.from_dict(manifest["metadata"])
        except (OSError, ValueError, KeyError) as e:
            logger.warning("index cache for %s unreadable (%s); rebuilding", repo, e)
            return None
        return store, metadata

    def save(
        self, repo: str, commit_sha: str, embedder_name: str,
        store: VectorStore, metadata: RepositoryMetadata,
    ) -> None:
        """Best-effort. A read-only or ephemeral filesystem (Vercel) makes
        this a no-op rather than a failure -- the same tradeoff LLMService
        makes for its own cache."""
        directory = self.directory(repo, commit_sha, embedder_name)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            store.save(directory)
            # manifest.json is written last: `has()` keys off it, so a run
            # interrupted mid-save leaves an incomplete directory that is
            # correctly treated as a miss.
            (directory / "manifest.json").write_text(
                json.dumps({
                    "format_version": INDEX_FORMAT_VERSION,
                    "repo": repo,
                    "commit_sha": commit_sha,
                    "embedder": embedder_name,
                    "built_at": time.time(),
                    "metadata": metadata.as_dict(),
                }, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("could not persist index for %s (%s); continuing uncached", repo, e)

    # --- "latest index for this repo" pointer ---------------------------
    # Retrieval needs to find a repo's index without knowing its current
    # commit SHA -- and resolving the SHA means running git, which is far
    # too slow for the webhook path (and impossible on a deploy with no
    # checkout). Indexing writes this pointer; retrieval reads it. The
    # pointer is what makes the read path pure filesystem.
    def _pointer_path(self, repo: str) -> Path:
        safe = repo.replace("/", "__").replace("..", "__")
        return self.cfg.index_dir / "_latest" / f"{safe}.json"

    def set_latest(self, repo: str, commit_sha: str, embedder_name: str) -> None:
        path = self._pointer_path(repo)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({
                    "repo": repo, "commit_sha": commit_sha,
                    "embedder": embedder_name, "updated_at": time.time(),
                }),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("could not write index pointer for %s (%s)", repo, e)

    def get_latest(self, repo: str) -> tuple[str, str] | None:
        """(commit_sha, embedder_name) for this repo's newest index, or None."""
        path = self._pointer_path(repo)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data["commit_sha"]), str(data["embedder"])
        except (OSError, ValueError, KeyError):
            return None

    def purge(self, repo: str) -> int:
        """Drop every index for `repo` regardless of SHA. Returns the count."""
        removed = 0
        if not self.cfg.index_dir.is_dir():
            return 0
        self._pointer_path(repo).unlink(missing_ok=True)
        for directory in self.cfg.index_dir.iterdir():
            manifest = directory / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                if json.loads(manifest.read_text(encoding="utf-8")).get("repo") == repo:
                    shutil.rmtree(directory, ignore_errors=True)
                    removed += 1
            except (OSError, ValueError):
                continue
        return removed
