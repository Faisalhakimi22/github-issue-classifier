"""Tests for release collection (ghic/collect.py phase 3).

Network is mocked: we feed canned GraphQL pages and assert pagination,
publishedAt/createdAt fallback, ascending sort, and the deadline guard.
"""
from __future__ import annotations

import time

import pytest

from ghic import collect
from ghic.config import get_config


@pytest.fixture
def cfg():
    return get_config(require_token=False)


def _page(nodes, end_cursor=None, has_next=False):
    return {
        "data": {
            "rateLimit": {"remaining": 5000, "resetAt": "2030-01-01T00:00:00Z", "cost": 1},
            "repository": {
                "releases": {
                    "pageInfo": {"endCursor": end_cursor, "hasNextPage": has_next},
                    "nodes": nodes,
                }
            },
        }
    }


def _mock_graphql(monkeypatch, pages):
    it = iter(pages)
    monkeypatch.setattr(collect, "_graphql", lambda c, q, v: next(it))
    monkeypatch.setattr(collect, "_gate_rate_limit", lambda *a, **k: None)
    monkeypatch.setattr(collect.utils, "cache_get", lambda ns, k: None)
    monkeypatch.setattr(collect.utils, "cache_put", lambda ns, k, v: None)


def test_releases_paginate_and_sort(cfg, monkeypatch):
    _mock_graphql(monkeypatch, [
        _page(
            [{"publishedAt": "2024-02-01T00:00:00Z"}, {"publishedAt": "2024-01-01T00:00:00Z"}],
            end_cursor="c1", has_next=True,
        ),
        _page([{"publishedAt": "2024-03-01T00:00:00Z"}], has_next=False),
    ])
    dates = collect.fetch_repo_releases(cfg, cfg.repos[0], deadline_epoch=time.time() + 100)
    assert dates == [
        "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z", "2024-03-01T00:00:00Z",
    ]


def test_publishedat_falls_back_to_createdat(cfg, monkeypatch):
    _mock_graphql(monkeypatch, [
        _page([
            {"publishedAt": None, "createdAt": "2024-05-01T00:00:00Z"},
            {"publishedAt": "2024-04-01T00:00:00Z", "createdAt": "2024-03-01T00:00:00Z"},
        ]),
    ])
    dates = collect.fetch_repo_releases(cfg, cfg.repos[0], deadline_epoch=time.time() + 100)
    assert dates == ["2024-04-01T00:00:00Z", "2024-05-01T00:00:00Z"]


def test_missing_repository_returns_empty(cfg, monkeypatch):
    page = _page([])
    page["data"]["repository"] = None  # renamed/removed repo
    _mock_graphql(monkeypatch, [page])
    assert collect.fetch_repo_releases(cfg, cfg.repos[0], deadline_epoch=time.time() + 100) == []


def test_deadline_guard_aborts(cfg, monkeypatch):
    _mock_graphql(monkeypatch, [_page([{"publishedAt": "2024-01-01T00:00:00Z"}])])
    with pytest.raises(collect.CollectionDeadlineExceeded):
        collect.fetch_repo_releases(cfg, cfg.repos[0], deadline_epoch=time.time() - 1)
