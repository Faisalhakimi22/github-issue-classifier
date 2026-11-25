"""The analysis comment is upserted, so a retried job never double-posts.

QStash retries a callback that timed out, and Vercel can kill the function
after GitHub already accepted the comment. These tests pin the behaviour
that makes that retry harmless.
"""
from __future__ import annotations

from typing import Any

import pytest
import requests

from ghic.service.github_app import ANALYSIS_MARKER, GitHubAppClient


class _Resp:
    def __init__(self, status: int = 200, data: Any = None) -> None:
        self.status_code = status
        self._data = data

    def json(self) -> Any:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _Session:
    """Records every call; serves canned comment listings by page."""

    def __init__(self, pages: list[list[dict[str, Any]]], list_status: int = 200) -> None:
        self.pages = pages
        self.list_status = list_status
        self.calls: list[tuple[str, str, Any]] = []

    def get(self, url: str, headers: Any = None, params: Any = None, timeout: Any = None):
        page = (params or {}).get("page", 1)
        self.calls.append(("GET", url, page))
        if self.list_status != 200:
            return _Resp(self.list_status, {})
        return _Resp(200, self.pages[page - 1] if page <= len(self.pages) else [])

    def request(self, method: str, url: str, headers: Any = None, json: Any = None,
                timeout: Any = None):
        self.calls.append((method, url, json))
        return _Resp(201 if method == "POST" else 200, {})


def _client(session: _Session) -> GitHubAppClient:
    client = GitHubAppClient("1", "unused-key")
    client._session = session  # type: ignore[assignment]
    client._tokens[7] = ("tok", 10**12)
    return client


def _writes(session: _Session) -> list[tuple[str, str, Any]]:
    return [c for c in session.calls if c[0] in ("POST", "PATCH")]


def test_first_attempt_posts_with_marker() -> None:
    session = _Session(pages=[[]])
    _client(session).post_comment("o/r", 5, "analysis", 7)

    writes = _writes(session)
    assert len(writes) == 1
    method, url, body = writes[0]
    assert method == "POST"
    assert url.endswith("/repos/o/r/issues/5/comments")
    assert ANALYSIS_MARKER in body["body"]


def test_retry_updates_existing_comment_instead_of_posting() -> None:
    existing = {"id": 99, "user": {"type": "Bot"}, "body": f"old\n\n{ANALYSIS_MARKER}"}
    session = _Session(pages=[[{"id": 1, "user": {"type": "User"}, "body": "hi"}, existing]])
    _client(session).post_comment("o/r", 5, "new analysis", 7)

    writes = _writes(session)
    assert len(writes) == 1
    method, url, body = writes[0]
    assert method == "PATCH"
    assert url.endswith("/repos/o/r/issues/comments/99")
    assert body["body"].startswith("new analysis")


def test_human_quoting_the_marker_is_never_rewritten() -> None:
    quoted = {"id": 3, "user": {"type": "User"}, "body": f"look: {ANALYSIS_MARKER}"}
    session = _Session(pages=[[quoted]])
    _client(session).post_comment("o/r", 5, "analysis", 7)

    assert [w[0] for w in _writes(session)] == ["POST"]


def test_marker_found_on_a_later_page() -> None:
    filler = [{"id": i, "user": {"type": "User"}, "body": "x"} for i in range(100)]
    ours = {"id": 500, "user": {"type": "Bot"}, "body": ANALYSIS_MARKER}
    session = _Session(pages=[filler, [ours]])
    _client(session).post_comment("o/r", 5, "analysis", 7)

    assert _writes(session)[0][0] == "PATCH"


def test_failed_listing_raises_rather_than_guessing_and_double_posting() -> None:
    session = _Session(pages=[], list_status=502)
    with pytest.raises(requests.HTTPError):
        _client(session).post_comment("o/r", 5, "analysis", 7)
    assert _writes(session) == []


def test_marker_not_duplicated_when_body_already_has_it() -> None:
    session = _Session(pages=[[]])
    _client(session).post_comment("o/r", 5, f"analysis\n{ANALYSIS_MARKER}", 7)
    assert _writes(session)[0][2]["body"].count(ANALYSIS_MARKER) == 1
