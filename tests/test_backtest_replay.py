"""The backtest must score every replayed issue.

It silently scored none from the day the usage-quota gate landed until
2026-11-28: the gate reads plans from the database, the backtest has none, so
every issue came back "usage_unavailable". The replay app now runs
unmetered; this holds it there, and checks the replayed GitHub client feeds
the author-history lookup into the model like production does.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from ghic import backtest
from ghic.service.inference import Prediction
from ghic.service.settings import ServiceSettings


class _Predictor:
    model_name = "stub"

    @property
    def cfg(self) -> Any:
        from ghic.config import get_config

        return get_config(require_token=False)

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> Prediction:
        self.calls.append(kwargs)
        return Prediction(
            repo=kwargs["repo_full_name"], issue_number=kwargs["issue_number"],
            proba=0.7, threshold=0.5, predicted_label=1, model_name=self.model_name,
        )


ROW = {
    "repo_name": "o/r", "number": 2, "title": "Crash on save", "body": "Steps: ...",
    "created_at": "2024-06-01T00:00:00Z", "author_login": "alice",
    "author_created_at": "2020-01-01T00:00:00Z", "author_public_repos": 3,
    "author_followers": 1, "label": 1,
}


def _post(client: TestClient, row: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(backtest._payload(row)).encode()
    resp = client.post("/webhook", content=body, headers={
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": backtest._sign(body),
        "Content-Type": "application/json",
    })
    resp.raise_for_status()
    return resp.json()


def test_a_replayed_issue_is_scored_not_skipped_by_the_quota_gate() -> None:
    history = pd.DataFrame([
        {**ROW, "number": 1, "created_at": "2024-01-10T00:00:00Z"},
        ROW,
    ])
    gh = backtest.ReplayGitHub({}, history=history)
    gh.current_row = ROW
    predictor = _Predictor()
    settings = ServiceSettings(
        model_path=Path("unused.joblib"), webhook_secret=backtest.REPLAY_SECRET, dry_run=True,
        suggest_related=False, suggest_category=False, estimate_effort=False,
        suggest_assignees=False,
    )
    data = _post(TestClient(backtest.replay_app(settings, predictor, gh)), ROW)

    assert "prediction" in data, data
    # alice filed issue #1 earlier in the window, so she is not a first-timer.
    assert predictor.calls[-1]["author_has_prior_issues"] is True
