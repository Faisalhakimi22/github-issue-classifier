"""Preview the comment GHIC would post on any public GitHub issue.

    python -m ghic.preview https://github.com/owner/repo/issues/123
    ghic preview owner/repo#123 --out preview.md
    ghic preview owner/repo#123 --no-evidence --json

Built for outreach: before telling a maintainer what GHIC says about their
issue, run it on that issue and read the result. It runs the production
pipeline, in production order:

  - the issue and its author from GitHub's public API, enriched exactly as
    the webhook does (profile, latest release, one-year author history);
  - the champion classifier and the category head, with production's
    calibration rule deciding whether the score is shown;
  - repository evidence from a shallow clone, indexed locally with
    production's settings (hashing embeddings, 1536 dimensions, 2,000
    chunks, the live similarity threshold);
  - the AI analysis, if GROQ_API_KEY / OPENROUTER_API_KEY are set, through
    the same prompt, sanitizer and comment formatter.

It never writes anywhere. This module has no GitHub client that can write
(only GET requests exist here), the clone and index live in a temporary
directory deleted afterwards, and the repository-intelligence service is
built with local vectors and in-memory state, so the production database is
never configured even when DATABASE_URL is set.

Not included, because production doesn't show them on customer repositories
either: duplicate candidates (they need that repository's issue index) and
the automation drafts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import requests

from . import utils

logger = utils.get_logger(__name__)

# Production's indexing settings (see .github/workflows/index-on-demand.yml).
# A preview that indexed differently would show evidence GHIC wouldn't.
PRODUCTION_EMBEDDING_DIMENSIONS = 1536
PRODUCTION_MAX_CHUNKS = 2000
# The live value on Vercel (GHIC_REPO_INTEL_MIN_SIMILARITY). An explicit
# environment value wins, so a preview can follow production if it changes.
PRODUCTION_MIN_SIMILARITY = 0.12

_REF_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:github\.com/)?(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"(?:/issues/|#)(?P<number>\d+)/?(?:[?#].*)?$"
)


class PreviewError(Exception):
    """Something the person running the preview has to fix."""


def parse_issue_ref(ref: str) -> tuple[str, int]:
    """`https://github.com/o/r/issues/5`, `github.com/o/r/issues/5` or `o/r#5`."""
    text = (ref or "").strip()
    if "/pull/" in text:
        raise PreviewError("that is a pull request; GHIC analyses issues")
    match = _REF_RE.match(text)
    if not match:
        raise PreviewError(
            f"not an issue reference: {ref!r} (use https://github.com/owner/repo/issues/N "
            "or owner/repo#N)"
        )
    return match.group("repo"), int(match.group("number"))


class PublicGitHub:
    """Read-only access to GitHub's public API for previews.

    It has no method that writes, on purpose; the enrichment methods share
    their names with GitHubAppClient's reads so the webhook's own enrichment
    code can use it unchanged. GITHUB_TOKEN, if given, only raises the
    anonymous rate limits.
    """

    API = "https://api.github.com"

    def __init__(self, token: str = "", session: Any = None, timeout: float = 15.0) -> None:
        self.timeout = timeout
        self._session = session or requests.Session()
        self._headers = {"Accept": "application/vnd.github+json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._session.get(
            f"{self.API}{path}", headers=self._headers, params=params, timeout=self.timeout,
        )
        if resp.status_code == 401 and "Authorization" in self._headers:
            # Public data needs no token, and GitHub answers 401 to an expired
            # one even for public reads. Drop it, say so once, carry on.
            del self._headers["Authorization"]
            print("warning: GitHub rejected GITHUB_TOKEN/GH_TOKEN (expired or revoked); "
                  "continuing without it, at the lower anonymous rate limit", file=sys.stderr)
            resp = self._session.get(
                f"{self.API}{path}", headers=self._headers, params=params, timeout=self.timeout,
            )
        return resp

    def issue(self, repo: str, number: int) -> dict[str, Any]:
        resp = self._get(f"/repos/{repo}/issues/{number}")
        if resp.status_code == 404:
            raise PreviewError(
                f"{repo}#{number} was not found. Previews work on public repositories only."
            )
        if resp.status_code in (403, 429):
            raise PreviewError(
                "GitHub refused the request (usually the anonymous rate limit). "
                "Set GITHUB_TOKEN to any personal access token to raise it."
            )
        if resp.status_code != 200:
            raise PreviewError(f"GitHub returned HTTP {resp.status_code} for {repo}#{number}")
        data = resp.json()
        if "pull_request" in data:
            raise PreviewError(f"{repo}#{number} is a pull request; GHIC analyses issues")
        return data

    # -- the reads the webhook's _enrich() uses ------------------------------
    def get_user(self, login: str, installation_id: Any = None) -> dict[str, Any] | None:
        try:
            resp = self._get(f"/users/{login}")
            return resp.json() if resp.status_code == 200 else None
        except (requests.RequestException, ValueError):
            return None

    def get_latest_release_date(self, full_name: str, installation_id: Any = None) -> str | None:
        try:
            resp = self._get(f"/repos/{full_name}/releases/latest")
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("published_at") or data.get("created_at")
        except (requests.RequestException, ValueError):
            return None

    def author_has_prior_issues(
        self, full_name: str, login: str, before_iso: str | None,
        installation_id: Any = None, **_: Any,
    ) -> bool | None:
        from .service.github_app import prior_issue_query

        query = prior_issue_query(full_name, login, before_iso)
        if query is None:
            return None
        try:
            resp = self._get("/search/issues", {"q": query, "per_page": 1})
            return int(resp.json().get("total_count", 0)) > 0 if resp.status_code == 200 else None
        except (requests.RequestException, ValueError, TypeError):
            return None


@dataclass
class PreviewResult:
    repo: str
    number: int
    comment: str | None          # None: production would not comment at all
    prediction: dict[str, Any]
    score_shown: bool
    analysis: dict[str, Any] | None
    evidence_note: str
    relevant_files: list[str] = field(default_factory=list)


def _git_clone(repo: str, dest: Path) -> str:
    """Shallow clone of a public repository. Returns the commit SHA."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(
        ["git", "clone", "--depth", "1", "--single-branch", "--no-tags",
         f"https://github.com/{repo}.git", str(dest)],
        check=True, capture_output=True, text=True, timeout=300, env=env,
    )
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    return head.stdout.strip() or "local"


def _remove_tree(path: Path) -> None:
    """rmtree that also removes git's read-only object files on Windows."""
    def make_writable(func: Callable[..., Any], target: str, *_: Any) -> None:
        os.chmod(target, stat.S_IWRITE)
        func(target)

    try:
        shutil.rmtree(path, onexc=make_writable)  # Python 3.12+
    except TypeError:
        shutil.rmtree(path, onerror=make_writable)


def _evidence(
    repo: str, title: str, body: str, *, category: str, predicted_label: int,
    clone: Callable[[str, Path], str], workdir: Path,
) -> tuple[Any, str]:
    """Index the repository locally and retrieve evidence, as production would."""
    from .repository_intelligence import RepositoryIntelligenceConfig, build_service

    src = workdir / "src"
    try:
        commit = clone(repo, src)
    except subprocess.CalledProcessError as e:
        return None, f"could not clone {repo}: {(e.stderr or '').strip()[:200]}"
    except Exception as e:
        return None, f"could not clone {repo}: {e}"

    min_similarity = float(
        os.environ.get("GHIC_REPO_INTEL_MIN_SIMILARITY") or PRODUCTION_MIN_SIMILARITY
    )
    cfg = replace(
        RepositoryIntelligenceConfig.from_env(),
        cache_dir=workdir / "intel",
        vector_provider="local",
        state_provider="memory",
        queue_provider="none",
        database_url="",
        auto_index=False,
        embedding_provider="hashing",
        embedding_dimensions=PRODUCTION_EMBEDDING_DIMENSIONS,
        max_chunks=PRODUCTION_MAX_CHUNKS,
        min_similarity=min_similarity,
    )
    service = build_service(cfg)
    if service is None:
        return None, "repository intelligence is unavailable in this environment"
    metadata = service.index_local_path(repo, src, commit_sha=commit)
    if metadata is None:
        return None, "indexing failed (see the log)"
    context = service.get_context(
        repo, title, body, category=category, predicted_label=predicted_label,
    )
    return context, f"{metadata.file_count} files indexed at {commit[:8]}"


def run_preview(
    ref: str,
    *,
    github: Any,
    settings: Any,
    predictor: Any,
    category_predictor: Any = None,
    llm_service: Any = None,
    evidence: bool = True,
    clone: Callable[[str, Path], str] = _git_clone,
) -> PreviewResult:
    """Everything the webhook does for an opened issue, except writing."""
    from .llm import detect_disagreement
    from .llm.models import IssueContext
    from .service.app import _enrich
    from .service.inference import (
        build_feature_frame,
        format_comment,
        format_llm_comment,
        worth_commenting_without_ai,
    )

    repo, number = parse_issue_ref(ref)
    issue = github.issue(repo, number)
    title = issue.get("title") or ""
    body = issue.get("body") or ""
    author = ((issue.get("user") or {}).get("login")) or ""
    created_at = issue.get("created_at") or ""

    # The webhook's own enrichment. "preview" stands in for an installation
    # id; the public client ignores it.
    author_created_at, public_repos, followers, latest_release, prior = _enrich(
        settings, github, author, repo, "preview", created_at,
    )
    pred = predictor.predict(
        repo_full_name=repo, issue_number=number, title=title, body=body,
        created_at=created_at, author_login=author, author_created_at=author_created_at,
        author_public_repos=public_repos, author_followers=followers,
        latest_release_iso=latest_release, threshold=settings.threshold_for(repo),
        explain=True, author_has_prior_issues=prior,
    )

    category = None
    if category_predictor is not None:
        try:
            frame = build_feature_frame(
                predictor.cfg, repo_full_name=repo, issue_number=number, title=title,
                body=body, created_at=created_at, author_login=author,
                author_created_at=author_created_at, author_public_repos=public_repos,
                author_followers=followers, latest_release_iso=latest_release,
                author_has_prior_issues=prior,
            )
            category = category_predictor.predict_frame(frame)
        except Exception as e:
            logger.warning("category prediction failed: %s", e)

    repo_context, evidence_note = None, "skipped (--no-evidence)"
    workdir = Path(tempfile.mkdtemp(prefix="ghic-preview-")) if evidence else None
    try:
        if workdir is not None:
            repo_context, evidence_note = _evidence(
                repo, title, body,
                category=(category or {}).get("predicted", "") if category else "",
                predicted_label=pred.predicted_label, clone=clone, workdir=workdir,
            )

        engineering = None
        if getattr(settings, "use_engineering_intelligence", False) and repo_context is not None:
            try:
                from .engineering_intelligence import EngineeringAnalyzer

                engineering = EngineeringAnalyzer().analyze(
                    repo, title, body, repo_context, issue_number=number,
                )
            except Exception as e:
                logger.warning("engineering analysis failed: %s", e)
    finally:
        if workdir is not None:
            _remove_tree(workdir)

    calibrated = settings.score_is_calibrated_for(repo)
    analysis = None
    if llm_service is not None:
        analysis = llm_service.analyze_issue(IssueContext(
            repo=repo, title=title, body=body,
            labels=[lab.get("name", "") for lab in (issue.get("labels") or [])],
            author=author, metadata={"created_at": created_at},
            ml_probability=pred.proba, ml_predicted_label=pred.predicted_label,
            repository_context=repo_context, score_calibrated=calibrated,
        ))

    if analysis is not None:
        disagreement = detect_disagreement(
            pred.predicted_label, analysis.priority, analysis.severity, analysis.reasoning,
        )
        comment = format_llm_comment(
            pred, analysis, [], disagreement=disagreement,
            repository_context=repo_context, engineering_analysis=engineering,
            automation=None, score_is_calibrated=calibrated,
        )
    elif worth_commenting_without_ai(calibrated, repo_context, []):
        comment = format_comment(
            pred, [], category, repository_context=repo_context,
            score_is_calibrated=calibrated,
        )
    else:
        comment = None  # the webhook stays silent here too

    files = [getattr(f, "path", str(f)) for f in getattr(repo_context, "relevant_files", []) or []]
    return PreviewResult(
        repo=repo, number=number, comment=comment, prediction=pred.as_dict(),
        score_shown=calibrated, analysis=analysis.as_dict() if analysis else None,
        evidence_note=evidence_note, relevant_files=files,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ghic preview",
        description="Show the comment GHIC would post on a public GitHub issue. Never posts.",
    )
    parser.add_argument("issue", help="issue URL or owner/repo#number")
    parser.add_argument("--no-evidence", action="store_true",
                        help="skip cloning and indexing the repository")
    parser.add_argument("--out", help="also write the comment to this file")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--verbose", action="store_true", help="show pipeline logs")
    args = parser.parse_args(argv)

    # Validate before touching anything: a typo must not load .env or
    # change logging for the rest of the process.
    try:
        parse_issue_ref(args.issue)
    except PreviewError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # The Windows console's legacy code page can't encode "→" or "·", which
    # every GHIC comment uses. Never crash on printing the result.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    # Keys in .env (GROQ_API_KEY, OPENROUTER_API_KEY, GH_TOKEN) must be in the
    # environment before settings are read. Existing variables win.
    from dotenv import load_dotenv

    from .config import ENV_PATH

    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)

    import logging

    if not args.verbose:
        # Global, not per-logger: modules imported later in the run create
        # their loggers at INFO, and per-logger levels would miss them.
        # Restored before returning.
        logging.disable(logging.INFO)
    try:
        return _run_cli(args)
    finally:
        logging.disable(logging.NOTSET)


def _run_cli(args: argparse.Namespace) -> int:

    from .category import load_category_predictor
    from .llm.service import LLMService
    from .service.app import _build_llm_provider
    from .service.inference import IssuePredictor
    from .service.settings import load_settings

    settings = load_settings()
    llm_service = None
    if settings.groq_api_key or settings.openrouter_api_key:
        llm_service = LLMService(_build_llm_provider(settings))
    try:
        result = run_preview(
            args.issue,
            github=PublicGitHub(
                token=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "",
            ),
            settings=settings,
            predictor=IssuePredictor(settings.model_path, settings.threshold),
            category_predictor=load_category_predictor() if settings.suggest_category else None,
            llm_service=llm_service,
            evidence=not args.no_evidence,
        )
    except PreviewError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except requests.RequestException as e:
        print(f"error: could not reach GitHub ({type(e).__name__}): {e}", file=sys.stderr)
        return 2

    if args.out and result.comment is not None:
        Path(args.out).write_text(result.comment, encoding="utf-8")
    if args.json:
        print(json.dumps(result.__dict__, indent=2, default=str))
        return 0

    ai = "on" if result.analysis else (
        "off (set GROQ_API_KEY or OPENROUTER_API_KEY)" if llm_service is None
        else "unavailable (provider error or quota; classifier-only comment)"
    )
    print(
        f"GHIC preview of {result.repo}#{result.number}. Nothing was posted.\n"
        f"  score:    {'shown' if result.score_shown else 'hidden (repository not calibrated, as in production)'}\n"
        f"  evidence: {result.evidence_note}\n"
        f"  AI:       {ai}\n",
        file=sys.stderr,
    )
    if result.comment is None:
        print(
            "GHIC would not comment on this issue: no AI analysis, a score that isn't "
            "calibrated for this repository, and no repository evidence to show."
        )
    else:
        print(result.comment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
