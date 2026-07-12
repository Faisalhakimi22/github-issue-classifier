"""Regression tests for _post's GraphQL error handling.

Bug caught 2026-05-29: a deleted user login in a batch query returns a partial
response (data present + a field-level NOT_FOUND error). _post used to raise on
any `errors`, aborting the entire collection run over one missing account.
"""
from __future__ import annotations


import pytest

from ghic import collect


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _patch_requests(monkeypatch, payload):
    monkeypatch.setattr(collect.requests, "post", lambda *a, **k: _FakeResp(payload))


def test_partial_response_with_data_is_tolerated(monkeypatch):
    """data present + a NOT_FOUND field error -> return payload, do not raise."""
    payload = {
        "data": {"u0": {"login": "alice"}, "u1": None},
        "errors": [{"type": "NOT_FOUND", "path": ["u1"],
                    "message": "Could not resolve to a User with the login of 'ghost'."}],
    }
    _patch_requests(monkeypatch, payload)
    result = collect._post("tok", "query", {})
    assert result["data"]["u0"]["login"] == "alice"
    assert result["data"]["u1"] is None


def test_errors_without_data_still_raise(monkeypatch):
    """data: null signals a genuine/transient failure -> raise (so retry fires)."""
    payload = {"data": None, "errors": [{"message": "something went wrong"}]}
    _patch_requests(monkeypatch, payload)
    with pytest.raises(collect.GraphQLError):
        collect._post("tok", "query", {})


def test_clean_response_passes_through(monkeypatch):
    payload = {"data": {"search": {"nodes": []}}}
    _patch_requests(monkeypatch, payload)
    assert collect._post("tok", "query", {}) == payload
