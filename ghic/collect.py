"""GitHub data collection via GraphQL.

Idempotent: every page response is cached to data/raw/issues/<repo>/<sha256>.json
on receipt. Re-runs reuse cached pages and resume from the first uncached cursor.

Three phases:
  1. Issues — paged search per repo, per date-window slice. One query per page
     pulls the issue node plus labels(first:20) plus
     timelineItems(itemTypes:[CROSS_REFERENCED_EVENT, CLOSED_EVENT], first:50).
  2. Authors — deduplicated logins batched 50 per request via aliased
     user(login:...) queries. Each user is cached by login, not by batch, so
     partial cache hits skip already-fetched users on resume.
  3. Releases — per-repo release timestamps (paged 100, ascending), written to
     data/processed/releases.json. Feeds features.days_since_last_release.

CLI:
  python -m ghic.collect --measure-cost    # one probe page, log cost + projection
  python -m ghic.collect                   # full collection, writes processed CSVs
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import requests

from . import utils
from .config import Config, RepoConfig, get_config

logger = utils.get_logger(__name__)

GRAPHQL_URL = "https://api.github.com/graphql"
HOURLY_POINT_BUDGET = 5000
USER_BATCH_SIZE = 50
USER_NAMESPACE = "users"


# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------
ISSUE_QUERY = """
query IssueSearch($q: String!, $cursor: String, $pageSize: Int!) {
  rateLimit { remaining resetAt cost }
  search(type: ISSUE, first: $pageSize, after: $cursor, query: $q) {
    issueCount
    pageInfo { endCursor hasNextPage }
    nodes {
      ... on Issue {
        number
        title
        body
        createdAt
        closedAt
        stateReason
        locked
        authorAssociation
        author { login }
        labels(first: 20) { nodes { name } }
        timelineItems(first: 50, itemTypes: [CROSS_REFERENCED_EVENT, CLOSED_EVENT]) {
          nodes {
            __typename
            ... on ClosedEvent {
              createdAt
              closer {
                __typename
                ... on PullRequest { number merged mergedAt }
              }
            }
            ... on CrossReferencedEvent {
              createdAt
              source {
                __typename
                ... on PullRequest { number merged body }
              }
            }
          }
        }
      }
    }
  }
}
""".strip()

USER_FIELDS = """
  login
  createdAt
  repositories(privacy: PUBLIC) { totalCount }
  followers { totalCount }
""".strip()

# Releases power the temporal `days_since_last_release` feature. Paged 100 at a
# time, ascending, so the cached pages are stable across runs. publishedAt is
# the real ship date; createdAt is the fallback for drafts published later.
RELEASES_QUERY = """
query Releases($owner: String!, $name: String!, $cursor: String) {
  rateLimit { remaining resetAt cost }
  repository(owner: $owner, name: $name) {
    releases(first: 100, after: $cursor, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { endCursor hasNextPage }
      nodes { tagName publishedAt createdAt }
    }
  }
}
""".strip()

RELEASES_NAMESPACE = "releases"
RELEASES_PATH_NAME = "releases.json"


def build_user_query(n: int) -> str:
    """Build an aliased user batch query for `n` logins."""
    var_decls = ", ".join(f"$login{i}: String!" for i in range(n))
    aliases = "\n  ".join(
        f"u{i}: user(login: $login{i}) {{ {USER_FIELDS} }}" for i in range(n)
    )
    return (
        f"query Users({var_decls}) {{\n"
        f"  rateLimit {{ remaining resetAt cost }}\n"
        f"  {aliases}\n"
        f"}}"
    )


# ---------------------------------------------------------------------------
# HTTP + rate limit
# ---------------------------------------------------------------------------
class GraphQLError(Exception):
    pass


class CollectionDeadlineExceeded(Exception):
    """Raised when the runtime cap is hit, to trigger clean shutdown."""


def _post(token: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        GRAPHQL_URL,
        headers={"Authorization": f"bearer {token}"},
        json={"query": query, "variables": variables},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    errors = payload.get("errors")
    if errors:
        # GraphQL partial responses are valid: a deleted/renamed user alias in a
        # batch query returns its node as null PLUS a field-level NOT_FOUND error,
        # while every other alias resolves fine. Discarding that whole response
        # (and aborting the run) over one missing account is wrong. Only treat it
        # as fatal — and therefore retryable — when there is no data at all, which
        # is how GitHub signals genuine/transient query failures.
        if payload.get("data") is None:
            raise GraphQLError(json.dumps(errors))
        logger.warning(
            "GraphQL partial errors (continuing with returned data): %s",
            json.dumps(errors)[:500],
        )
    return payload


def _graphql(cfg: Config, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """POST with retry+backoff. Caller handles caching."""
    decorated = utils.retry_with_backoff(
        max_attempts=cfg.collection.backoff_max_attempts,
        base_delay=cfg.collection.backoff_base_seconds,
        exceptions=(requests.RequestException, GraphQLError),
        logger=logger,
    )(_post)
    return decorated(cfg.github_token, query, variables)


def _gate_rate_limit(rl: dict[str, Any], floor: int) -> None:
    """Log cost + remaining; sleep until reset if remaining < floor."""
    cost = rl.get("cost", "?")
    remaining = rl["remaining"]
    reset_at_s = rl["resetAt"]
    logger.info("rateLimit: cost=%s remaining=%s resetAt=%s", cost, remaining, reset_at_s)
    if remaining >= floor:
        return
    reset_at = datetime.fromisoformat(reset_at_s.replace("Z", "+00:00"))
    delay = max(1.0, (reset_at - datetime.now(tz=reset_at.tzinfo)).total_seconds())
    logger.warning(
        "Rate limit remaining=%d below floor=%d; sleeping %.0fs until %s",
        remaining, floor, delay, reset_at_s,
    )
    time.sleep(delay)


# ---------------------------------------------------------------------------
# Date windows
# ---------------------------------------------------------------------------
def date_windows(start: str, end: str, months: int) -> Iterator[tuple[date, date]]:
    """Yield consecutive inclusive [start, end] windows of `months * 30` days.

    Approximation is fine: what matters is no gaps and no overlaps, both of
    which this loop guarantees by construction.
    """
    cur = date.fromisoformat(start)
    final = date.fromisoformat(end)
    span = timedelta(days=months * 30)
    while cur <= final:
        nxt = min(cur + span - timedelta(days=1), final)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


# ---------------------------------------------------------------------------
# Issue page fetch with idempotent cache
# ---------------------------------------------------------------------------
def _issue_namespace(repo: RepoConfig) -> str:
    return f"issues/{repo.owner}__{repo.name}"


def _build_issue_vars(
    repo: RepoConfig, window_start: date, window_end: date, cursor: str | None, page_size: int,
) -> dict[str, Any]:
    q = (
        f"repo:{repo.slug} is:issue is:closed "
        f"created:{window_start.isoformat()}..{window_end.isoformat()}"
    )
    return {"q": q, "cursor": cursor, "pageSize": page_size}


def _fetch_issue_page(
    cfg: Config,
    repo: RepoConfig,
    window_start: date,
    window_end: date,
    cursor: str | None,
) -> dict[str, Any]:
    variables = _build_issue_vars(
        repo, window_start, window_end, cursor, cfg.collection.page_size
    )
    key = utils.cache_key(ISSUE_QUERY, variables)
    namespace = _issue_namespace(repo)

    cached = utils.cache_get(namespace, key)
    if cached is not None:
        logger.debug("cache hit %s/%s", namespace, key[:12])
        return cached

    logger.info(
        "fetch %s window=%s..%s cursor=%s",
        repo.slug, window_start, window_end, (cursor[:12] if cursor else "<start>"),
    )
    payload = _graphql(cfg, ISSUE_QUERY, variables)
    utils.cache_put(namespace, key, payload)
    _gate_rate_limit(payload["data"]["rateLimit"], cfg.collection.rate_limit_floor)
    return payload


def _iter_issue_pages(
    cfg: Config, repo: RepoConfig, deadline_epoch: float,
) -> Iterator[dict[str, Any]]:
    for ws, we in date_windows(
        repo.created_after, repo.created_before, cfg.collection.date_window_months
    ):
        cursor: str | None = None
        while True:
            if time.time() > deadline_epoch:
                logger.warning(
                    "Runtime ceiling reached at %s window=%s..%s; aborting cleanly.",
                    repo.slug, ws, we,
                )
                raise CollectionDeadlineExceeded()
            page = _fetch_issue_page(cfg, repo, ws, we, cursor)
            yield page
            pi = page["data"]["search"]["pageInfo"]
            if not pi["hasNextPage"]:
                break
            cursor = pi["endCursor"]


# ---------------------------------------------------------------------------
# Normalization — collect.py's output schema
# ---------------------------------------------------------------------------
def normalize_issue(node: dict[str, Any], repo: RepoConfig) -> dict[str, Any]:
    """Flatten a GraphQL issue node to a row dict. Every row carries repo_name."""
    timeline = (node.get("timelineItems") or {}).get("nodes") or []
    closing_pr_merged = False
    cross_refs: list[dict[str, Any]] = []
    for ev in timeline:
        tn = ev.get("__typename")
        if tn == "ClosedEvent":
            closer = ev.get("closer") or {}
            if closer.get("__typename") == "PullRequest" and closer.get("merged"):
                closing_pr_merged = True
        elif tn == "CrossReferencedEvent":
            src = ev.get("source") or {}
            if src.get("__typename") == "PullRequest":
                cross_refs.append({
                    "number": src.get("number"),
                    "merged": bool(src.get("merged")),
                    "body": src.get("body") or "",
                })
    return {
        "repo_name": repo.slug,
        "number": node["number"],
        "title": node.get("title") or "",
        "body": node.get("body") or "",
        "created_at": node.get("createdAt"),
        "closed_at": node.get("closedAt"),
        "state_reason": node.get("stateReason"),
        "locked": bool(node.get("locked")),
        "author_association": node.get("authorAssociation"),
        "author_login": (node.get("author") or {}).get("login"),
        "labels_at_close": [l["name"] for l in (node.get("labels") or {}).get("nodes", [])],
        "closed_by_merged_pr": closing_pr_merged,
        "cross_referenced_prs": cross_refs,
        # Author enrichment columns — None until enrich_authors() runs. Kept
        # in the schema up-front so partial-abort CSVs have stable columns.
        "author_created_at": None,
        "author_public_repos": None,
        "author_followers": None,
    }


# ---------------------------------------------------------------------------
# Author enrichment (phase 2)
# ---------------------------------------------------------------------------
def _fetch_user_batch(
    cfg: Config, logins: list[str],
) -> dict[str, dict[str, Any] | None]:
    """Return {login: user_data_or_None}, hitting cache first."""
    out: dict[str, dict[str, Any] | None] = {}
    missing: list[str] = []
    for login in logins:
        cached = utils.cache_get(USER_NAMESPACE, utils.cache_key(login))
        if cached is not None:
            # Cache stores {"data": value}; unwrap.
            out[login] = cached["data"]
        else:
            missing.append(login)
    if not missing:
        return out

    query = build_user_query(len(missing))
    variables = {f"login{i}": login for i, login in enumerate(missing)}
    payload = _graphql(cfg, query, variables)
    _gate_rate_limit(payload["data"]["rateLimit"], cfg.collection.rate_limit_floor)
    for i, login in enumerate(missing):
        user_node = payload["data"].get(f"u{i}")
        out[login] = user_node
        # Wrap in {"data": ...} so cache_get can distinguish "cached None"
        # (deleted account) from "not yet cached".
        utils.cache_put(USER_NAMESPACE, utils.cache_key(login), {"data": user_node})
    return out


def enrich_authors(
    cfg: Config, issues: list[dict[str, Any]], deadline_epoch: float,
) -> None:
    """Resolve unique authors in batches, attach metadata to each issue in place."""
    unique_logins = sorted({i["author_login"] for i in issues if i["author_login"]})
    logger.info(
        "Enriching %d unique authors in batches of %d",
        len(unique_logins), USER_BATCH_SIZE,
    )

    user_data: dict[str, dict[str, Any] | None] = {}
    for start in range(0, len(unique_logins), USER_BATCH_SIZE):
        if time.time() > deadline_epoch:
            logger.warning("Runtime ceiling reached during author enrichment; aborting cleanly.")
            raise CollectionDeadlineExceeded()
        batch = unique_logins[start:start + USER_BATCH_SIZE]
        user_data.update(_fetch_user_batch(cfg, batch))

    for issue in issues:
        u = user_data.get(issue["author_login"])
        if u is None:
            continue
        issue["author_created_at"] = u.get("createdAt")
        issue["author_public_repos"] = (u.get("repositories") or {}).get("totalCount")
        issue["author_followers"] = (u.get("followers") or {}).get("totalCount")


# ---------------------------------------------------------------------------
# Releases (phase 3) — feeds features.days_since_last_release
# ---------------------------------------------------------------------------
def _releases_namespace(repo: RepoConfig) -> str:
    return f"{RELEASES_NAMESPACE}/{repo.owner}__{repo.name}"


def _fetch_release_page(cfg: Config, repo: RepoConfig, cursor: str | None) -> dict[str, Any]:
    variables = {"owner": repo.owner, "name": repo.name, "cursor": cursor}
    key = utils.cache_key(RELEASES_QUERY, variables)
    namespace = _releases_namespace(repo)

    cached = utils.cache_get(namespace, key)
    if cached is not None:
        logger.debug("cache hit %s/%s", namespace, key[:12])
        return cached

    payload = _graphql(cfg, RELEASES_QUERY, variables)
    utils.cache_put(namespace, key, payload)
    _gate_rate_limit(payload["data"]["rateLimit"], cfg.collection.rate_limit_floor)
    return payload


def fetch_repo_releases(
    cfg: Config, repo: RepoConfig, deadline_epoch: float,
) -> list[str]:
    """Return this repo's release timestamps (ISO), ascending. Cached per page."""
    dates: list[str] = []
    cursor: str | None = None
    while True:
        if time.time() > deadline_epoch:
            logger.warning("Runtime ceiling reached fetching releases for %s; aborting.", repo.slug)
            raise CollectionDeadlineExceeded()
        page = _fetch_release_page(cfg, repo, cursor)
        repository = page["data"].get("repository")
        if not repository:  # repo renamed/removed, or no releases scope
            break
        releases = repository["releases"]
        for node in releases["nodes"] or []:
            ts = node.get("publishedAt") or node.get("createdAt")
            if ts:
                dates.append(ts)
        pi = releases["pageInfo"]
        if not pi["hasNextPage"]:
            break
        cursor = pi["endCursor"]
    return sorted(dates)


def collect_releases(
    cfg: Config, deadline_epoch: float,
) -> dict[str, list[str]]:
    """Fetch release dates for every repo. Maps repo slug -> sorted ISO list."""
    out: dict[str, list[str]] = {}
    for repo in cfg.repos:
        out[repo.slug] = fetch_repo_releases(cfg, repo, deadline_epoch)
        logger.info("releases %s: %d", repo.slug, len(out[repo.slug]))
    return out


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized: list[dict[str, Any]] = []
    for row in rows:
        s: dict[str, Any] = {}
        for k, v in row.items():
            s[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
        serialized.append(s)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(serialized[0].keys()))
        writer.writeheader()
        writer.writerows(serialized)
    logger.info("wrote %s (%d rows)", path, len(serialized))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class CollectionResult:
    repos: int
    issues: int
    authors: int
    releases: int
    pages: int
    elapsed_seconds: float
    aborted_at_deadline: bool


def run_collection(cfg: Config | None = None) -> CollectionResult:
    cfg = cfg or get_config()
    start_ts = time.time()
    deadline = start_ts + cfg.collection.max_runtime_minutes * 60

    all_issues: list[dict[str, Any]] = []
    releases: dict[str, list[str]] = {}
    pages = 0
    aborted = False

    try:
        for repo in cfg.repos:
            repo_issues: list[dict[str, Any]] = []
            for page in _iter_issue_pages(cfg, repo, deadline):
                pages += 1
                for node in page["data"]["search"]["nodes"] or []:
                    if node:  # search may return None for transferred/deleted issues
                        repo_issues.append(normalize_issue(node, repo))
            _write_csv(
                utils.DATA_PROCESSED / f"{repo.owner}_{repo.name}.csv",
                repo_issues,
            )
            all_issues.extend(repo_issues)
        enrich_authors(cfg, all_issues, deadline)
        releases = collect_releases(cfg, deadline)
    except CollectionDeadlineExceeded:
        aborted = True
    finally:
        _write_csv(utils.DATA_PROCESSED / "combined.csv", all_issues)
        if releases:
            utils.write_json(utils.DATA_PROCESSED / RELEASES_PATH_NAME, releases)

    unique_authors = len({i["author_login"] for i in all_issues if i["author_login"]})
    return CollectionResult(
        repos=len(cfg.repos),
        issues=len(all_issues),
        authors=unique_authors,
        releases=sum(len(v) for v in releases.values()),
        pages=pages,
        elapsed_seconds=time.time() - start_ts,
        aborted_at_deadline=aborted,
    )


# ---------------------------------------------------------------------------
# Cost measurement (per precommitment #2)
# ---------------------------------------------------------------------------
def measure_cost(cfg: Config | None = None) -> dict[str, Any]:
    """One probe page; log cost + project worst-case total budget."""
    cfg = cfg or get_config()
    repo = cfg.repos[0]
    ws = date.fromisoformat(repo.created_after)
    we = ws + timedelta(days=30)

    probe_page_size = 5
    variables = {
        "q": (
            f"repo:{repo.slug} is:issue is:closed "
            f"created:{ws.isoformat()}..{we.isoformat()}"
        ),
        "cursor": None,
        "pageSize": probe_page_size,
    }
    payload = _graphql(cfg, ISSUE_QUERY, variables)
    cost = payload["data"]["rateLimit"]["cost"]
    issue_count = payload["data"]["search"]["issueCount"]

    # Cost roughly scales with page size; extrapolate from probe to real page size.
    cost_at_real_page_size = cost * (cfg.collection.page_size / probe_page_size)
    pages_per_window = max(1, -(-issue_count // cfg.collection.page_size))  # ceil
    windows_per_repo = max(1, 12 // cfg.collection.date_window_months)
    projected_pages = pages_per_window * windows_per_repo * len(cfg.repos)
    projected_total = projected_pages * cost_at_real_page_size
    pct = projected_total / HOURLY_POINT_BUDGET * 100

    logger.info(
        "Probe: page_size=%d, cost=%s, issueCount-in-window=%d",
        probe_page_size, cost, issue_count,
    )
    logger.info(
        "Projection: ~%.1f cost/page at page_size=%d, ~%d pages total, "
        "~%.0f points (%.1f%% of %d/hr budget)",
        cost_at_real_page_size, cfg.collection.page_size,
        projected_pages, projected_total, pct, HOURLY_POINT_BUDGET,
    )
    if pct > 50:
        logger.warning(
            "Projected issue-collection budget exceeds 50%% of hourly limit. "
            "Reduce collection.page_size, or split the run across hours."
        )
    logger.info(
        "Note: author enrichment cost is NOT included in this probe — monitor "
        "rateLimit logs during the actual run."
    )
    return {
        "per_page_cost_probe": cost,
        "projected_cost_per_page_at_real_size": cost_at_real_page_size,
        "projected_pages": projected_pages,
        "projected_total_cost": projected_total,
        "pct_of_hourly_budget": pct,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect GitHub issues via GraphQL.")
    parser.add_argument(
        "--measure-cost",
        action="store_true",
        help="Issue one probe page, log per-page cost + projected budget, exit.",
    )
    args = parser.parse_args(argv)

    if args.measure_cost:
        measure_cost()
        return 0

    result = run_collection()
    logger.info(
        "Collection done: %d issues, %d authors, %d releases, %d pages, %.1fs%s",
        result.issues, result.authors, result.releases, result.pages, result.elapsed_seconds,
        " (aborted at deadline)" if result.aborted_at_deadline else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
