"""Phase 2 corpora: commits, pull requests, and resolved issues.

Phase 1 indexed source files. These three index a repository's *history*,
and they answer questions the code alone can't:

  **Commit Intelligence** -- "when did this break, and what changed?"
  A regression report matched against commit subjects and touched paths
  points at the change that likely introduced it.

  **Pull Request Intelligence** -- "is someone already fixing this?"
  The most useful triage answer for a duplicate-effort issue is a link to
  the open PR that already addresses it.

  **Historical Issue Intelligence** -- "how did we resolve this last time?"
  A closed issue with its resolution is the single highest-value piece of
  triage context that exists, and it's the one a new maintainer never has.

All three reuse the entire Phase 1 pipeline unchanged -- the same embedder,
vector store, retriever, similarity floor, and diversification -- because
they produce the same `CodeChunk` type with a different `source`. That was
the point of the Phase 1 seam: `RepositoryIndexer` takes content, not a
repository, so a corpus that arrives from `git log` or the GitHub API
reuses everything downstream of chunking.

**What is not chunked here.** A commit, PR, or issue is indexed as one
record, not split. Splitting a 40-line issue body into three pieces
retrieves fragments that read as decontextualized noise, and the whole
value of this corpus is the *pairing* of a problem statement with its
resolution -- which is destroyed by cutting between them.

**Diffs are summarized, not embedded.** A commit's patch can be tens of
thousands of lines and is mostly noise for retrieval (whitespace, imports,
lockfiles). What retrieves well is the subject, the body, and the list of
touched paths, so that is what gets embedded. The alternative -- embedding
raw patches -- measurably dilutes the signal and costs far more tokens.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import utils
from .cache import RepositoryCacheError, run_git
from .config import RepositoryIntelligenceConfig, redact_secrets
from .models import (
    SOURCE_COMMIT,
    SOURCE_ISSUE,
    SOURCE_PULL_REQUEST,
    CodeChunk,
)

logger = utils.get_logger(__name__)

# Bounds, for the same reason every Phase 1 limit exists: a repository is
# untrusted input. A monorepo can have half a million commits.
DEFAULT_MAX_COMMITS = 2_000
DEFAULT_MAX_HISTORY_ITEMS = 2_000
_MAX_RECORD_CHARS = 4_000
_MAX_PATHS_PER_COMMIT = 25

# Commits that carry no triage signal. Merge commits restate their branch,
# and release/bump commits match everything and explain nothing.
_NOISE_SUBJECT_PREFIXES = (
    "merge branch", "merge pull request", "merge remote-tracking",
    "bump version", "release v", "chore(release)", "version bump",
    "update changelog", "regenerate", "[skip ci]",
)


class DocumentSource(ABC):
    """Anything that can produce indexable chunks for a repository.

    Three methods' worth of contract, deliberately small so a future
    corpus (discussions, releases, CI failures) is a small class rather
    than a change to the engine.
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def fetch(self, repo: str, **kwargs: Any) -> list[CodeChunk]:
        raise NotImplementedError


def _clip(text: str, limit: int = _MAX_RECORD_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def _parse_timestamp(value: Any) -> float:
    """ISO-8601 or epoch -> epoch seconds. 0.0 when unparseable."""
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, TypeError):
        return 0.0


def is_noise_commit(subject: str) -> bool:
    """Whether a commit subject carries no triage signal."""
    lowered = (subject or "").strip().lower()
    if not lowered:
        return True
    return any(lowered.startswith(prefix) for prefix in _NOISE_SUBJECT_PREFIXES)


# ---------------------------------------------------------------------------
# Commit Intelligence
# ---------------------------------------------------------------------------
class CommitSource(DocumentSource):
    """Commits from a local checkout, via `git log`.

    Reads the clone Phase 1 already maintains, so this costs no additional
    network I/O -- but it does need history, and Phase 1 clones with
    `--depth 1`. `fetch` deepens the clone on demand (`git fetch
    --deepen`) rather than silently indexing a single commit, and reports
    honestly when it can't.
    """

    @property
    def source_type(self) -> str:
        return SOURCE_COMMIT

    def fetch(
        self,
        repo: str,
        *,
        root: Path | None = None,
        max_commits: int = DEFAULT_MAX_COMMITS,
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> list[CodeChunk]:
        if root is None:
            raise ValueError("CommitSource.fetch requires root=<checkout path>")
        try:
            raw = self._git_log(root, max_commits, timeout)
        except RepositoryCacheError as e:
            logger.warning("could not read commit history for %s: %s", repo, e)
            return []
        return self._to_chunks(repo, raw)

    def _git_log(self, root: Path, max_commits: int, timeout: float) -> str:
        # A unit separator between fields and a record separator between
        # commits: commit messages contain newlines, tabs, and every
        # punctuation character, so anything printable is ambiguous.
        #
        # The record separator goes at the *start* of the format, not the
        # end. `--name-only` appends each commit's file list after the
        # formatted line, so a trailing separator puts those paths at the
        # beginning of the *next* record -- where they get parsed as part
        # of its SHA. (Caught by an end-to-end run: a commit came back
        # referenced as "src/csv_" instead of its hash.) Leading the record
        # keeps a commit's paths inside its own record.
        fmt = "%x1e" + "%x1f".join(["%H", "%an", "%aI", "%s", "%b"])
        return run_git(
            ["log", f"--max-count={max_commits}", f"--pretty=format:{fmt}",
             "--name-only", "--no-merges"],
            root, timeout,
        )

    def _to_chunks(self, repo: str, raw: str) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for record in raw.split("\x1e"):
            record = record.strip("\n")
            if not record.strip():
                continue
            fields = record.split("\x1f")
            if len(fields) < 5:
                continue
            sha, author, authored_at, subject, rest = fields[:5]
            sha = sha.strip("\n ")
            if not sha or is_noise_commit(subject):
                continue

            # `--name-only` appends touched paths after the body, separated
            # by a blank line.
            body_lines: list[str] = []
            paths: list[str] = []
            for line in rest.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                # A path has no spaces and contains a separator or an
                # extension; anything else is still message body.
                if ("/" in stripped or "." in stripped) and " " not in stripped:
                    paths.append(stripped)
                else:
                    body_lines.append(stripped)

            text = self._render(sha, author, subject, body_lines, paths)
            chunks.append(CodeChunk(
                repo=repo,
                path=f"commit:{sha[:12]}",
                language="",
                text=redact_secrets(text),
                start_line=0,
                end_line=0,
                kind="commit",
                symbol=subject.strip()[:120],
                source=SOURCE_COMMIT,
                reference=sha[:8],
                timestamp=_parse_timestamp(authored_at),
            ))
        return chunks

    def _render(
        self, sha: str, author: str, subject: str, body: list[str], paths: list[str]
    ) -> str:
        parts = [f"Commit {sha[:8]} by {author}", subject.strip()]
        if body:
            parts.append(" ".join(body))
        if paths:
            shown = paths[:_MAX_PATHS_PER_COMMIT]
            parts.append("Files changed: " + ", ".join(shown))
            if len(paths) > len(shown):
                parts.append(f"(and {len(paths) - len(shown)} more)")
        return _clip("\n".join(parts))


def deepen_clone(root: Path, commits: int = DEFAULT_MAX_COMMITS, timeout: float = 180.0) -> bool:
    """Fetch more history into a shallow clone. False if it couldn't.

    Phase 1 clones at `--depth 1` because file content is all it needs.
    Commit intelligence needs history, so it is fetched on demand rather
    than making every Phase 1 clone pay for depth it doesn't use.
    """
    try:
        run_git(["fetch", f"--deepen={commits}"], root, timeout)
        return True
    except RepositoryCacheError as e:
        # Already complete, or the remote refuses -- either way, index
        # whatever history is present rather than failing.
        logger.info("could not deepen clone at %s (%s); indexing available history", root, e)
        return False


# ---------------------------------------------------------------------------
# Issue / Pull Request Intelligence
# ---------------------------------------------------------------------------
class IssueHistorySource(DocumentSource):
    """Resolved issues and pull requests.

    Deliberately takes already-fetched records rather than calling the
    GitHub API itself: the project already collects issues
    (`ghic/collect.py`), a webhook handler already has an authenticated
    client, and a source that owns its own HTTP is a source that can't be
    tested without mocking a network. `fetch` accepts `items=[...]`;
    `from_github` is the thin adapter for the live case.

    GitHub returns pull requests through the issues API, distinguished by
    a `pull_request` key -- which is why one class covers both corpora.
    They're separated by `source` at index time so retrieval can weight
    them differently: an open PR that fixes the issue is a *different*
    kind of answer from a closed issue that resolved it before.
    """

    @property
    def source_type(self) -> str:
        return SOURCE_ISSUE

    def fetch(
        self,
        repo: str,
        *,
        items: list[dict[str, Any]] | None = None,
        max_items: int = DEFAULT_MAX_HISTORY_ITEMS,
        include_open_prs: bool = True,
        **kwargs: Any,
    ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for item in (items or [])[:max_items]:
            chunk = self._to_chunk(repo, item, include_open_prs=include_open_prs)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def _to_chunk(
        self, repo: str, item: dict[str, Any], *, include_open_prs: bool
    ) -> CodeChunk | None:
        number = item.get("number")
        title = (item.get("title") or "").strip()
        if not number or not title:
            return None

        is_pr = bool(item.get("pull_request") or item.get("is_pull_request"))
        state = (item.get("state") or "").lower()
        state_reason = (item.get("state_reason") or item.get("stateReason") or "").lower()

        # A closed-as-not-planned issue records that nobody fixed it, which
        # is genuinely useful ("we've seen this and declined it") -- but it
        # must never be retrieved as though it were a resolution.
        resolved = state == "closed" and state_reason not in ("not_planned", "duplicate")

        if is_pr:
            if state != "closed" and not include_open_prs:
                return None
            source = SOURCE_PULL_REQUEST
            kind = "pull_request"
        else:
            # Open issues are the *current* backlog, not history. Surfacing
            # them here would duplicate the existing duplicate-detector.
            if state != "closed":
                return None
            source = SOURCE_ISSUE
            kind = "issue"

        text = self._render(item, is_pr=is_pr, resolved=resolved, state=state,
                            state_reason=state_reason)
        return CodeChunk(
            repo=repo,
            path=f"{'pr' if is_pr else 'issue'}:{number}",
            language="",
            text=redact_secrets(text),
            start_line=0,
            end_line=0,
            kind=kind,
            symbol=title[:160],
            source=source,
            reference=f"#{number}",
            url=item.get("html_url") or item.get("url") or "",
            timestamp=_parse_timestamp(
                item.get("closed_at") or item.get("closedAt") or item.get("updated_at")
            ),
        )

    def _render(
        self, item: dict[str, Any], *, is_pr: bool, resolved: bool,
        state: str, state_reason: str,
    ) -> str:
        number = item.get("number")
        label = "Pull request" if is_pr else "Issue"
        parts = [f"{label} #{number}: {(item.get('title') or '').strip()}"]

        labels = item.get("labels") or []
        names = [
            lab.get("name") if isinstance(lab, dict) else str(lab)
            for lab in (labels if isinstance(labels, list) else [])
        ]
        names = [n for n in names if n]
        if names:
            parts.append("Labels: " + ", ".join(names[:8]))

        # Outcome first, before the body: it is the part that makes this
        # record worth retrieving, and a long body would otherwise bury it
        # past the truncation limit.
        if state == "closed":
            if is_pr:
                parts.append("Outcome: merged or closed.")
            elif resolved:
                parts.append("Outcome: closed as completed (resolved).")
            else:
                parts.append(f"Outcome: closed as {state_reason or 'not planned'} "
                             "(not a resolution).")

        body = (item.get("body") or "").strip()
        if body:
            parts.append(body)

        resolution = (item.get("resolution") or item.get("resolution_comment") or "").strip()
        if resolution:
            parts.append("Resolution: " + resolution)

        return _clip("\n".join(parts))

    @staticmethod
    def from_github(
        gh_client: Any, repo: str, installation_id: Any, *, limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Fetch recent closed issues/PRs through an authenticated client.

        Thin by design -- the adapter, not the logic. Returns [] on any
        failure so a history index degrades to "no history" rather than
        failing an indexing run.
        """
        try:
            return gh_client.list_recent_closed_issues(
                repo, installation_id=installation_id, limit=limit
            )
        except Exception as e:
            logger.warning("could not fetch issue history for %s: %s", repo, e)
            return []


def load_collected_issues(repo: str, path: Path | None = None) -> list[dict[str, Any]]:
    """Read this project's own collected issue corpus, if present.

    `ghic/collect.py` already gathers issues for training, including
    `closed_at` and `state_reason`. Reusing that file means a deploy can
    build history intelligence with no API calls at all -- and it's how the
    feature is exercised offline.
    """
    from .. import utils as _utils

    path = path or _utils.PROJECT_ROOT / "data" / "raw" / "issues.jsonl"
    if not path.is_file():
        return []
    items: list[dict[str, Any]] = []
    try:
        import json

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("repo_name") in (repo, None) or not repo:
                    items.append(record)
    except (OSError, ValueError) as e:
        logger.warning("could not read collected issues at %s: %s", path, e)
        return []
    return items


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_history_chunks(
    repo: str,
    cfg: RepositoryIntelligenceConfig,
    *,
    checkout_path: Path | None = None,
    issue_items: list[dict[str, Any]] | None = None,
    max_commits: int = DEFAULT_MAX_COMMITS,
) -> list[CodeChunk]:
    """Every Phase 2 chunk for one repository, from whichever sources are
    available. Missing inputs yield fewer chunks, never an error."""
    chunks: list[CodeChunk] = []

    if checkout_path is not None and cfg.index_commits:
        deepen_clone(checkout_path, max_commits)
        commit_chunks = CommitSource().fetch(
            repo, root=checkout_path, max_commits=max_commits,
            timeout=cfg.git_timeout_seconds,
        )
        chunks.extend(commit_chunks)
        logger.info("commit intelligence: %d commits indexed for %s", len(commit_chunks), repo)

    if issue_items and cfg.index_issue_history:
        history_chunks = IssueHistorySource().fetch(repo, items=issue_items)
        chunks.extend(history_chunks)
        logger.info(
            "history intelligence: %d issues/PRs indexed for %s",
            len(history_chunks), repo,
        )

    return chunks


def summarize_sources(chunks: list[CodeChunk]) -> dict[str, int]:
    """Chunk counts per source, for the state record and /repositories."""
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.source] = counts.get(chunk.source, 0) + 1
    return counts


def recency_weight(timestamp: float, *, half_life_days: float = 365.0) -> float:
    """Multiplier in (0, 1] that decays with age.

    History is the one corpus where age genuinely changes relevance: a
    resolution from last month is more likely to still apply than one from
    four years ago, across a codebase that has moved on. Applied as a
    gentle ranking prior, not a filter -- a five-year-old issue describing
    exactly this bug is still the right answer, and a half-life of a year
    keeps it competitive rather than burying it.
    """
    if timestamp <= 0:
        return 1.0          # unknown age is not evidence of staleness
    age_days = max(0.0, (time.time() - timestamp) / 86400.0)
    return float(0.5 ** (age_days / half_life_days))
