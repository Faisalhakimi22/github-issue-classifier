"""Tests for the deterministic, stateless layer of features.py.

The fittable preprocessor (TF-IDF / imputer / scaler) is covered implicitly by
the train pipeline; here we lock down the row-level transforms that must stay
non-leaky and webhook-reusable.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ghic import features
from ghic.config import get_config


@pytest.fixture(scope="module")
def cfg():
    return get_config(require_token=False)


def _row(**overrides):
    base = {
        "repo_name": "microsoft/vscode",
        "number": 1,
        "title": "App crashes on startup",
        "body": "### Steps to reproduce\nrun it\n```\ntraceback\n```\nexpected vs actual error",
        "created_at": "2024-03-04T15:00:00Z",          # Mon 15:00 UTC
        "closed_at": "2024-03-10T00:00:00Z",
        "author_login": "alice",
        "author_created_at": "2020-03-04T00:00:00Z",     # ~4 years before
        "author_public_repos": 12,
        "author_followers": 5,
        "label": 1,
    }
    base.update(overrides)
    return base


def _frame(rows):
    return pd.DataFrame([_row(**r) for r in rows])


def test_text_shape_and_keyword_features(cfg):
    out = features.engineer_features(_frame([{}]), cfg)
    r = out.iloc[0]
    assert r["title_len_chars"] == len("App crashes on startup")
    assert r["title_len_tokens"] == 4
    assert r["has_code_block"] == 1
    assert r["has_repro_steps"] == 1
    assert r["repro_keyword_hits"] >= 3          # steps to reproduce / expected / actual / error
    assert r["opened_via_template"] == 1         # markdown header present


def test_link_and_image_detection(cfg):
    rows = [
        {"body": "see http://example.com"},
        {"body": "![shot](x.png)"},
        {"body": "plain text only"},
    ]
    out = features.engineer_features(_frame(rows), cfg)
    assert list(out["has_link"]) == [1, 0, 0]
    assert list(out["has_image"]) == [0, 1, 0]


def test_account_age_is_point_in_time(cfg):
    out = features.engineer_features(_frame([{}]), cfg)
    # 2020-03-04 -> 2024-03-04 == 1461 days (includes leap day 2020/2024)
    assert out.iloc[0]["author_account_age_days"] == 1461


def test_first_time_contributor_is_chronological(cfg):
    rows = [
        {"number": 1, "author_login": "bob", "created_at": "2024-05-01T00:00:00Z"},
        {"number": 2, "author_login": "bob", "created_at": "2024-01-01T00:00:00Z"},  # earlier
        {"number": 3, "author_login": "carol", "created_at": "2024-02-01T00:00:00Z"},
    ]
    out = features.engineer_features(_frame(rows), cfg)
    flags = dict(zip(out["number"], out["author_is_first_time_contributor"]))
    assert flags[2] == 1   # bob's earliest issue
    assert flags[1] == 0   # bob's later issue
    assert flags[3] == 1   # carol's only issue


def test_missing_author_metadata_becomes_nan(cfg):
    out = features.engineer_features(
        _frame([{"author_created_at": None, "author_public_repos": None}]), cfg
    )
    assert pd.isna(out.iloc[0]["author_account_age_days"])
    assert pd.isna(out.iloc[0]["author_public_repos"])


def test_days_since_release_optional(cfg):
    df = _frame([{}])
    # Without release data the column is absent (logged warning, not a crash).
    out = features.engineer_features(df, cfg)
    assert features.OPTIONAL_FEATURE_DAYS_SINCE_RELEASE not in out.columns
    # With release data it is computed against the most recent prior release.
    out2 = features.engineer_features(
        df, cfg, repo_release_dates={"microsoft/vscode": ["2024-02-01T00:00:00Z"]}
    )
    assert out2.iloc[0][features.OPTIONAL_FEATURE_DAYS_SINCE_RELEASE] == 32


def test_text_column_present_for_tfidf(cfg):
    out = features.engineer_features(_frame([{}]), cfg)
    assert features.TEXT_COLUMN in out.columns
    assert "crashes" in out.iloc[0][features.TEXT_COLUMN]
