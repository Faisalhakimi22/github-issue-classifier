"""GitHub App API client: JWT auth, installation tokens, REST helpers.

The webhook payload tells us which installation sent the event; this client
mints a short-lived installation token for it (cached until near expiry) and
performs the few REST calls the service needs:

  read  — GET /users/{login}            (author enrichment)
        — GET /repos/{full}/releases/latest
  write — POST /repos/{full}/issues/{n}/comments
        — POST /repos/{full}/issues/{n}/labels

All read failures degrade to None so a GitHub hiccup never blocks a
prediction — missing author fields are imputed by the model pipeline anyway.
"""
from __future__ import annotations

import time
from typing import Any

import requests

from .. import utils

logger = utils.get_logger(__name__)

_TOKEN_EXPIRY_MARGIN_S = 120       # refresh installation tokens 2 min early
_JWT_LIFETIME_S = 540              # GitHub max is 600; stay under clock skew


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

    # -- writes (raise on failure so the caller can report it) ---------------
    def post_comment(self, full_name: str, issue_number: int, body: str,
                     installation_id: int) -> None:
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
