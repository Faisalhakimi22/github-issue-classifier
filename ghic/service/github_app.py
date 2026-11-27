"""GitHub App API client: JWT auth, installation tokens, REST helpers.

The webhook payload tells us which installation sent the event; this client
mints a short-lived installation token for it (cached until near expiry) and
performs the few REST calls the service needs:

  read  — GET /users/{login}            (author enrichment)
        — GET /repos/{full}/releases/latest
        — GET /search/issues               (has the author filed here before?)
  write — POST /repos/{full}/issues/{n}/comments
        — PATCH /repos/{full}/issues/comments/{id}  (retry-safe upsert)
        — POST /repos/{full}/issues/{n}/labels

All read failures degrade to None so a GitHub hiccup never blocks a
prediction — missing author fields are imputed by the model pipeline anyway.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .. import utils

logger = utils.get_logger(__name__)

_TOKEN_EXPIRY_MARGIN_S = 120       # refresh installation tokens 2 min early
_JWT_LIFETIME_S = 540              # GitHub max is 600; stay under clock skew
_COMMENT_SCAN_PAGES = 3            # the analysis lands early; 300 comments is plenty

# Hidden in the rendered comment. It is how a retried job recognises the
# comment an earlier attempt already posted. Never change it: comments
# posted under an old marker would stop being found, and retries would
# double-post on exactly those issues.
ANALYSIS_MARKER = "<!-- ghic:analysis -->"

# The classifier's `author_is_first_time_contributor` feature was learned
# from 2024 data as "no earlier issue by this author in this repository
# within the collected window"; for the held-out issues the reported metrics
# come from, that window reached back about 9-12 months. A one-year lookback
# reproduces that definition at serving time. Changing it changes what the
# feature means to the model.
PRIOR_ISSUE_LOOKBACK_DAYS = 365

_LOGIN_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def prior_issue_query(
    full_name: str, login: str, before_iso: str | None,
    lookback_days: int = PRIOR_ISSUE_LOOKBACK_DAYS,
) -> str | None:
    """The GitHub search query behind `author_has_prior_issues`, or None when
    the inputs can't be trusted into a query (so nothing can be injected).
    Shared by the App client and the read-only preview client, so a preview
    asks GitHub exactly what production asks."""
    if not (_LOGIN_RE.match(login or "") and _REPO_RE.match(full_name or "")):
        return None
    before = _parse_utc(before_iso)
    if before is None:
        return None
    lo = before - timedelta(days=lookback_days)
    hi = before - timedelta(seconds=1)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return f"repo:{full_name} is:issue author:{login} created:{lo.strftime(fmt)}..{hi.strftime(fmt)}"


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class GitHubAppClient:
    def __init__(
        self,
        app_id: str,
        private_key_pem: str,
        base_url: str = "https://api.github.com",
        timeout: float = 15.0,
    ) -> None:
        self.app_id = app_id
        self.private_key_pem = private_key_pem
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        # installation_id -> (token, expires_epoch)
        self._tokens: dict[int, tuple[str, float]] = {}

    # -- auth ---------------------------------------------------------------
    def _app_jwt(self) -> str:
        import jwt  # PyJWT

        now = int(time.time())
        payload = {"iat": now - 60, "exp": now + _JWT_LIFETIME_S, "iss": self.app_id}
        return jwt.encode(payload, self.private_key_pem, algorithm="RS256")

    def installation_token(self, installation_id: int) -> str:
        cached = self._tokens.get(installation_id)
        if cached and cached[1] - _TOKEN_EXPIRY_MARGIN_S > time.time():
            return cached[0]
        resp = self._session.post(
            f"{self.base_url}/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {self._app_jwt()}",
                "Accept": "application/vnd.github+json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # Tokens live ~1h; parsing expires_at exactly buys nothing over a
        # conservative fixed lifetime.
        self._tokens[installation_id] = (data["token"], time.time() + 55 * 60)
        return data["token"]

    def verify_installation(self, installation_id: int) -> dict[str, Any]:
        """Verify the current installation belongs to this App and is active."""
        resp = self._session.get(
            f"{self.base_url}/app/installations/{installation_id}",
            headers={
                "Authorization": f"Bearer {self._app_jwt()}",
                "Accept": "application/vnd.github+json",
            },
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            return {"status": "unknown"}
        resp.raise_for_status()
        data = resp.json()
        if str(data.get("app_id")) != str(self.app_id):
            return {"status": "app_mismatch"}
        if data.get("suspended_at"):
            return {"status": "suspended"}
        return {"status": "active"}

    def list_installation_repositories(self, installation_id: int) -> list[dict[str, Any]]:
        """Return authoritative repository membership for an installation."""
        resp = self._session.get(
            f"{self.base_url}/installation/repositories",
            headers={
                "Authorization": f"Bearer {self.installation_token(installation_id)}",
                "Accept": "application/vnd.github+json",
            },
            params={"per_page": 100},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return list(resp.json().get("repositories") or [])

    def _request(
        self,
        method: str,
        path: str,
        installation_id: int,
        json_body: Any = None,
    ) -> requests.Response:
        token = self.installation_token(installation_id)
        resp = self._session.request(
            method,
            f"{self.base_url}{path}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            json=json_body,
            timeout=self.timeout,
        )
        return resp

    # -- reads (degrade to None on failure) ----------------------------------
    def get_user(self, login: str, installation_id: int) -> dict[str, Any] | None:
        try:
            resp = self._request("GET", f"/users/{login}", installation_id)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("get_user(%s) -> HTTP %d", login, resp.status_code)
        except requests.RequestException as e:
            logger.warning("get_user(%s) failed: %s", login, e)
        return None

    def get_latest_release_date(self, full_name: str, installation_id: int) -> str | None:
        try:
            resp = self._request(
                "GET", f"/repos/{full_name}/releases/latest", installation_id
            )
            if resp.status_code == 200:
                return resp.json().get("published_at") or resp.json().get("created_at")
            if resp.status_code != 404:  # 404 = repo has no releases; expected
                logger.warning("latest release %s -> HTTP %d", full_name, resp.status_code)
        except requests.RequestException as e:
            logger.warning("latest release %s failed: %s", full_name, e)
        return None

    def author_has_prior_issues(
        self,
        full_name: str,
        login: str,
        before_iso: str | None,
        installation_id: int,
        *,
        lookback_days: int = PRIOR_ISSUE_LOOKBACK_DAYS,
    ) -> bool | None:
        """Did `login` open an issue in `full_name` in the `lookback_days`
        before `before_iso`? None when it can't be known (bad input, search
        refused, rate-limited, network): the caller then keeps the old
        default rather than guess.

        One search call. The window ends a second before this issue, so the
        issue being scored never counts as its own history.
        """
        query = prior_issue_query(full_name, login, before_iso, lookback_days)
        if query is None:
            return None
        try:
            resp = self._session.get(
                f"{self.base_url}/search/issues",
                headers={
                    "Authorization": f"Bearer {self.installation_token(installation_id)}",
                    "Accept": "application/vnd.github+json",
                },
                params={"q": query, "per_page": 1},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                logger.warning("prior-issue search for %s -> HTTP %d", full_name, resp.status_code)
                return None
            return int(resp.json().get("total_count", 0)) > 0
        except Exception as e:  # enrichment is best-effort, never a reason to fail
            logger.warning("prior-issue search for %s failed: %s", full_name, e)
            return None

    # -- writes (raise on failure so the caller can report it) ---------------
    def find_analysis_comment(self, full_name: str, issue_number: int,
                              installation_id: int) -> int | None:
        """Return the id of this App's earlier analysis comment, if any.

        Only bot-authored comments carrying ANALYSIS_MARKER count: a person
        who quotes the marker must never have their comment rewritten.
        Raises on a failed listing -- guessing "none" here is exactly how a
        retry ends up posting a second comment.
        """
        for page in range(1, _COMMENT_SCAN_PAGES + 1):
            resp = self._session.get(
                f"{self.base_url}/repos/{full_name}/issues/{issue_number}/comments",
                headers={
                    "Authorization": f"Bearer {self.installation_token(installation_id)}",
                    "Accept": "application/vnd.github+json",
                },
                params={"per_page": 100, "page": page},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            comments = resp.json() or []
            for comment in comments:
                user = comment.get("user") or {}
                if user.get("type") != "Bot":
                    continue
                if ANALYSIS_MARKER in (comment.get("body") or ""):
                    return int(comment["id"])
            if len(comments) < 100:
                break
        return None

    def post_comment(self, full_name: str, issue_number: int, body: str,
                     installation_id: int) -> None:
        """Create the analysis comment, or update it if one already exists.

        A job can run more than once for the same issue: QStash retries a
        callback that timed out or returned 5xx, and the function may have
        been killed after GitHub accepted the comment. Upserting on a hidden
        marker makes that retry rewrite the one comment instead of adding a
        second.
        """
        if ANALYSIS_MARKER not in body:
            body = f"{body}\n\n{ANALYSIS_MARKER}"
        existing = self.find_analysis_comment(full_name, issue_number, installation_id)
        if existing is not None:
            resp = self._request(
                "PATCH",
                f"/repos/{full_name}/issues/comments/{existing}",
                installation_id,
                json_body={"body": body},
            )
            resp.raise_for_status()
            logger.info("updated analysis comment on %s#%d", full_name, issue_number)
            return
        resp = self._request(
            "POST",
            f"/repos/{full_name}/issues/{issue_number}/comments",
            installation_id,
            json_body={"body": body},
        )
        resp.raise_for_status()
        logger.info("commented on %s#%d", full_name, issue_number)

    def add_labels(self, full_name: str, issue_number: int, labels: list[str],
                   installation_id: int) -> None:
        resp = self._request(
            "POST",
            f"/repos/{full_name}/issues/{issue_number}/labels",
            installation_id,
            json_body={"labels": labels},
        )
        resp.raise_for_status()
        logger.info("labeled %s#%d with %s", full_name, issue_number, labels)

    def add_issue_to_project(self, project_node_id: str, issue_node_id: str,
                             installation_id: int) -> None:
        """Add an issue to a Projects v2 board (GraphQL — v2 has no REST API).

        Requires the App to have the organization/repository Projects write
        permission; `project_node_id` is the board's node ID (`PVT_…`).
        """
        resp = self._request(
            "POST",
            "/graphql",
            installation_id,
            json_body={
                "query": (
                    "mutation($project: ID!, $content: ID!) {"
                    " addProjectV2ItemById(input:"
                    " {projectId: $project, contentId: $content})"
                    " { item { id } } }"
                ),
                "variables": {"project": project_node_id, "content": issue_node_id},
            },
        )
        resp.raise_for_status()
        errors = resp.json().get("errors")
        if errors:
            raise RuntimeError(f"addProjectV2ItemById failed: {errors}")
        logger.info("added issue %s to project %s", issue_node_id, project_node_id)
