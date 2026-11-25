"""Indexing through GitHub Actions: dispatch, authorization, install hook.

Vercel has no git, so production indexing runs on an Actions runner. These
tests pin the three things that make that safe: the dispatch is well formed
and fails soft, the worker re-authorizes rather than trusting its inputs,
and installing the App queues indexing only for authorized repositories.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import pytest

from ghic.repository_intelligence import GitHubActionsIndexQueue, RepositoryIntelligenceConfig
from ghic.repository_intelligence.providers import build_index_queue
from ghic.repository_intelligence.queue import QStashIndexQueue
from ghic.service.index_job import run_index_job


class _Resp:
    def __init__(self, status: int, text: str = "") -> None:
        self.status_code = status
        self.text = text


class _Session:
    def __init__(self, status: int = 204, exc: Exception | None = None) -> None:
        self.status = status
        self.exc = exc
        self.posts: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def post(self, url: str, headers: Any = None, json: Any = None, timeout: Any = None):
        self.posts.append((url, headers, json))
        if self.exc:
            raise self.exc
        return _Resp(self.status, "nope")


# -- the queue ---------------------------------------------------------------
def test_dispatch_targets_the_workflow_with_string_inputs() -> None:
    session = _Session()
    queue = GitHubActionsIndexQueue("tok", "me/ghic-worker", session=session)

    assert queue.enqueue("cust/app", installation_id=42, reason="first_issue") is True
    url, headers, body = session.posts[0]
    assert url == (
        "https://api.github.com/repos/me/ghic-worker/actions/workflows/"
        "index-on-demand.yml/dispatches"
    )
    assert headers["Authorization"] == "Bearer tok"
    assert body["ref"] == "main"
    assert body["inputs"]["repo"] == "cust/app"
    # workflow_dispatch rejects non-string inputs
    assert body["inputs"]["installation_id"] == "42"
    assert all(isinstance(v, str) for v in body["inputs"].values())


@pytest.mark.parametrize("session", [_Session(status=404), _Session(exc=OSError("down"))])
def test_dispatch_failure_returns_false_never_raises(session: _Session) -> None:
    queue = GitHubActionsIndexQueue("tok", "me/w", session=session)
    assert queue.enqueue("cust/app", installation_id=1) is False


def test_no_installation_means_no_dispatch() -> None:
    session = _Session()
    queue = GitHubActionsIndexQueue("tok", "me/w", session=session)
    assert queue.enqueue("cust/app", installation_id=None) is False
    assert session.posts == []


def test_auto_prefers_actions_over_qstash_when_configured() -> None:
    cfg = replace(
        RepositoryIntelligenceConfig(),
        queue_provider="auto",
        actions_dispatch_repo="me/w",
        actions_dispatch_token="tok",
    )
    queue = build_index_queue(cfg, qstash_token="q", callback_url="https://x/cb")
    assert isinstance(queue, GitHubActionsIndexQueue)


def test_auto_keeps_qstash_without_actions_config() -> None:
    cfg = replace(RepositoryIntelligenceConfig(), queue_provider="auto")
    queue = build_index_queue(cfg, qstash_token="q", callback_url="https://x/cb")
    assert isinstance(queue, QStashIndexQueue)


# -- the shared job ------------------------------------------------------------
class _Service:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def index_repository(self, repo: str, **kwargs: Any) -> bool:
        self.calls.append({"repo": repo, **kwargs})
        return True


class _GH:
    def installation_token(self, installation_id: int) -> str:
        return f"token-{installation_id}"


def _lease(result: dict[str, Any]):
    @contextmanager
    def lease(database_url: str, installation_id: Any, repo: str):
        yield result
    return lease


def _job(payload: dict[str, Any], *, gate: dict[str, Any], lease: dict[str, Any]):
    service = _Service()
    result = run_index_job(
        payload,
        database_url="postgres://x",
        repo_intelligence=service,
        gh=_GH(),
        authorization_gate=lambda *a: gate,
        authorization_lease=_lease(lease),
    )
    return result, service


def test_authorized_job_indexes_with_installation_token() -> None:
    ok = {"authorized": True, "workspace_id": "ws1"}
    result, service = _job({"repo": "c/a", "installation_id": "7"}, gate=ok, lease=ok)

    assert result == {"ok": True, "repo": "c/a", "indexed": True}
    assert service.calls == [
        {"repo": "c/a", "token": "token-7", "force": False, "workspace_id": "ws1"}
    ]


def test_unclaimed_installation_is_refused_before_any_clone() -> None:
    result, service = _job(
        {"repo": "c/a", "installation_id": "7"},
        gate={"authorized": False, "reason": "unclaimed_installation"},
        lease={"authorized": True},
    )
    assert result["skipped"] and result["reason"] == "unclaimed_installation"
    assert service.calls == []


def test_revoked_between_check_and_lease_is_refused() -> None:
    result, service = _job(
        {"repo": "c/a", "installation_id": "7"},
        gate={"authorized": True},
        lease={"authorized": False, "reason": "revoked_installation"},
    )
    assert result["reason"] == "revoked_installation"
    assert service.calls == []


def test_missing_installation_is_refused() -> None:
    result, service = _job(
        {"repo": "c/a"}, gate={"authorized": True}, lease={"authorized": True}
    )
    assert result["reason"] == "unknown_installation"
    assert service.calls == []


# -- install hook ----------------------------------------------------------------
class _RequestingService:
    def __init__(self) -> None:
        self.requested: list[tuple[str, Any, Any]] = []

    def request_index(self, repo: str, *, installation_id: Any, reason: str,
                      workspace_id: Any = None) -> bool:
        self.requested.append((repo, installation_id, workspace_id))
        return True


class _State:
    pass


class _App:
    def __init__(self, service: Any, authorized: set[str]) -> None:
        self.state = _State()
        self.state.repo_intelligence = service
        self.state.gh = None
        self.state.settings = type("S", (), {"database_url": "postgres://x"})()
        self.state.authorization_gate = lambda db, inst, repo, gh: (
            {"authorized": True, "workspace_id": "ws"} if repo in authorized
            else {"authorized": False, "reason": "unclaimed_installation"}
        )


def test_install_queues_only_authorized_repositories() -> None:
    from ghic.service.app import _queue_indexing_for_new_repositories

    service = _RequestingService()
    app = _App(service, authorized={"c/one"})
    _queue_indexing_for_new_repositories(app, "installation", {
        "action": "created",
        "installation": {"id": 9},
        "repositories": [{"full_name": "c/one"}, {"full_name": "c/two"}],
    })
    assert service.requested == [("c/one", 9, "ws")]


def test_repositories_added_event_is_handled_and_others_ignored() -> None:
    from ghic.service.app import _queue_indexing_for_new_repositories

    service = _RequestingService()
    app = _App(service, authorized={"c/new"})
    _queue_indexing_for_new_repositories(app, "installation_repositories", {
        "action": "added", "installation": {"id": 9},
        "repositories_added": [{"full_name": "c/new"}],
    })
    _queue_indexing_for_new_repositories(app, "installation", {
        "action": "deleted", "installation": {"id": 9},
        "repositories": [{"full_name": "c/new"}],
    })
    assert service.requested == [("c/new", 9, "ws")]
