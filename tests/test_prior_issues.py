"""author_is_first_time_contributor at serving time.

Training computed it from each author's earlier issues in the same repository
(within the collected 2024 window). A single webhook payload carries no
history, so production used to hard-code "first-time" for everyone, costing
about 4.5 points of recall on the held-out set. Production now asks GitHub
whether the author opened an issue there in the previous 365 days.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import requests

from ghic import features
from ghic.backtest import ReplayGitHub
from ghic.config import get_config
from ghic.service.github_app import GitHubAppClient
from ghic.service.inference import build_feature_frame

FLAG = "author_is_first_time_contributor"


# -- GitHubAppClient.author_has_prior_issues -------------------------------------------
class _Resp:
    def __init__(self, status: int, total: int = 0) -> None:
        self.status_code = status
        self._total = total

    def json(self) -> dict[str, Any]:
        return {"total_count": self._total, "incomplete_results": False, "items": []}


class _Session:
    def __init__(self, status: int = 200, total: int = 0, exc: Exception | None = None) -> None:
        self.status, self.total, self.exc = status, total, exc
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, headers: Any = None, params: Any = None, timeout: Any = None):
        self.calls.append({"url": url, "headers": headers, "params": params})
        if self.exc:
            raise self.exc
        return _Resp(self.status, self.total)


def _client(session: _Session) -> GitHubAppClient:
    client = GitHubAppClient("1", "unused-key")
    client._session = session  # type: ignore[assignment]
    client._tokens[7] = ("tok", 10**12)
    return client


def test_query_covers_the_year_before_the_issue_and_excludes_the_issue_itself() -> None:
    session = _Session(total=0)
    _client(session).author_has_prior_issues("o/r", "alice", "2026-11-27T15:48:12Z", 7)

    call = session.calls[0]
    assert call["url"].endswith("/search/issues")
    assert call["headers"]["Authorization"] == "Bearer tok"
    assert call["params"] == {
        "q": "repo:o/r is:issue author:alice "
             "created:2025-11-27T15:48:12Z..2026-11-27T15:48:11Z",
        "per_page": 1,
    }


def test_offset_timestamps_are_converted_to_utc() -> None:
    session = _Session()
    _client(session).author_has_prior_issues("o/r", "alice", "2026-11-27T20:48:12+05:00", 7)
    assert session.calls[0]["params"]["q"].endswith(
        "created:2025-11-27T15:48:12Z..2026-11-27T15:48:11Z"
    )


@pytest.mark.parametrize("total, expected", [(0, False), (1, True), (37, True)])
def test_any_earlier_issue_means_not_first_time(total: int, expected: bool) -> None:
    assert _client(_Session(total=total)).author_has_prior_issues(
        "o/r", "alice", "2026-11-27T15:48:12Z", 7
    ) is expected


@pytest.mark.parametrize("session", [
    _Session(status=403),            # search rate limit
    _Session(status=422),            # GitHub refused the query
    _Session(exc=requests.ConnectionError("down")),
])
def test_failures_mean_unknown_not_first_time(session: _Session) -> None:
    assert _client(session).author_has_prior_issues(
        "o/r", "alice", "2026-11-27T15:48:12Z", 7
    ) is None


@pytest.mark.parametrize("repo, login, before", [
    ("o/r", "alice repo:other/secret", "2026-11-27T15:48:12Z"),  # query injection
    ("o/r", "dependabot[bot]", "2026-11-27T15:48:12Z"),
    ("o/r", "", "2026-11-27T15:48:12Z"),
    ("o/r extra:thing", "alice", "2026-11-27T15:48:12Z"),
    ("o/r", "alice", None),
    ("o/r", "alice", "not a date"),
])
def test_bad_input_is_unknown_and_never_searched(repo: str, login: str, before: Any) -> None:
    session = _Session(total=5)
    assert _client(session).author_has_prior_issues(repo, login, before, 7) is None
    assert session.calls == []


# -- feature frame -------------------------------------------------------------------
@pytest.mark.parametrize("prior, flag", [(True, 0), (False, 1), (None, 1)])
def test_frame_uses_the_looked_up_history(prior: bool | None, flag: int) -> None:
    frame = build_feature_frame(
        get_config(require_token=False),
        repo_full_name="o/r", issue_number=1, title="Crash", body="Steps",
        created_at="2024-07-01T10:00:00Z", author_login="alice",
        author_has_prior_issues=prior,
    )
    assert int(frame[FLAG].iloc[0]) == flag


# -- backtest replay mirrors the training-time feature ----------------------------------
def _history() -> pd.DataFrame:
    return pd.DataFrame({
        "repo_name": ["o/r", "o/r", "o/r", "o/r", "x/y"],
        "number": [1, 2, 3, 4, 1],
        "title": ["a", "b", "c", "d", "e"],
        "body": ["", "", "", "", ""],
        "created_at": [
            "2024-01-10T00:00:00Z",   # alice, first
            "2024-06-01T00:00:00Z",   # alice, again
            "2024-03-01T00:00:00Z",   # bob, first
            "2024-11-30T00:00:00Z",   # carol, first
            "2024-02-01T00:00:00Z",   # alice, other repo
        ],
        "author_login": ["alice", "alice", "bob", "carol", "alice"],
        "author_created_at": [None] * 5,
    })


def test_replay_answers_the_same_question_as_production() -> None:
    gh = ReplayGitHub({}, history=_history())
    assert gh.author_has_prior_issues("o/r", "alice", "2024-06-01T00:00:00Z", 1) is True
    assert gh.author_has_prior_issues("o/r", "alice", "2024-01-10T00:00:00Z", 1) is False
    assert gh.author_has_prior_issues("o/r", "alice", "2025-06-02T00:00:00Z", 1) is False  # >365 days
    assert gh.author_has_prior_issues("x/y", "bob", "2024-06-01T00:00:00Z", 1) is False


def test_replay_reproduces_the_training_flag_for_every_row() -> None:
    history = _history()
    trained = features.engineer_features(history, get_config(require_token=False))
    gh = ReplayGitHub({}, history=history)
    for row, flag in zip(history.to_dict("records"), trained[FLAG]):
        prior = gh.author_has_prior_issues(
            row["repo_name"], row["author_login"], row["created_at"], 1,
        )
        assert int(flag) == (0 if prior else 1), row
