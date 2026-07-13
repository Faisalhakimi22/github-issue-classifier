"""Tests for the assignment recommender (ghic.assign) and effort target.

Live GraphQL collection is exercised only through its pure pieces (batch
query construction, node extraction) — the evaluation logic runs end-to-end
on a synthetic corpus with two clearly separated topic clusters, each
"owned" by a different maintainer.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from ghic.assign import (
    _build_batch_query,
    _decision_text,
    _extract,
    _is_human,
    evaluate_recommender,
)


class TestCollectionPieces:
    def test_batch_query_aliases_every_number(self):
        q = _build_batch_query([1, 7, 42])
        assert "i1: issue(number: 1)" in q
        assert "i7: issue(number: 7)" in q
        assert "i42: issue(number: 42)" in q
        assert "rateLimit" in q

    def test_extract_pulls_assignees_participants_closer(self):
        node = {
            "number": 5,
            "assignees": {"nodes": [{"login": "alice"}, {"login": "bob"}]},
            "participants": {"nodes": [{"login": "carol"}]},
            "timelineItems": {"nodes": [
                {"actor": {"login": "dave"}},
                {},                              # ClosedEvent without actor
            ]},
        }
        rec = _extract(node)
        assert rec == {"assignees": ["alice", "bob"],
                       "participants": ["carol"], "closer": "dave"}

    def test_extract_none_node(self):
        assert _extract(None) is None

    def test_is_human_filters_bots(self):
        assert _is_human("alice", set())
        assert not _is_human("dependabot[bot]", set())
        assert not _is_human("vs-code-triage-bot", {"vs-code-triage-bot"})
        assert not _is_human("", set())


class TestDecisionText:
    def test_similarity_wins(self):
        assert "similarity recommender wins" in _decision_text(0.60, 0.40)

    def test_baseline_wins(self):
        assert "baseline is the shipped feature" in _decision_text(0.40, 0.60)

    def test_tie_ships_similarity_for_evidence(self):
        assert "within noise" in _decision_text(0.50, 0.52)

    def test_no_data(self):
        assert "nothing ships" in _decision_text(None, None).lower()


def _synthetic_corpus(tmp_path):
    """30 issues, two topic clusters with distinct owners.

    'crash save file editor' issues are always assigned to alice;
    'slow render performance profiler' issues to bob. TF-IDF similarity
    should route new crash issues to alice at k=1.
    """
    crash = "Crash when saving file in the editor, crash save file editor stacktrace"
    slow = "Slow render performance in the profiler view, slow render performance profiler"
    rows, assignments = [], {}
    for i in range(30):
        is_crash = i % 2 == 0
        rows.append({
            "repo_name": "acme/widgets",
            "number": i + 1,
            "title": ("Crash on save %d" % i) if is_crash else ("Slow render %d" % i),
            "body": crash if is_crash else slow,
            "created_at": f"2024-01-{i + 1:02d}T00:00:00Z",
            "closed_at": f"2024-02-{i + 1:02d}T00:00:00Z",
            "author_login": "reporter",
        })
        assignments[str(i + 1)] = {
            "assignees": ["alice" if is_crash else "bob"],
            "participants": ["reporter"],
            "closer": "alice" if is_crash else "bob",
        }
    path = tmp_path / "combined.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, {"acme/widgets": assignments}


class TestEvaluateRecommender:
    def test_similarity_routes_clusters_to_their_owner(self, tmp_path):
        path, assignments = _synthetic_corpus(tmp_path)
        results = evaluate_recommender(path, assignments=assignments)
        block = results["per_repo"]["acme/widgets"]
        assert block["n_test_with_assignee"] == 6      # 20% chronological tail
        assert block["distinct_prior_assignees"] == 2
        # clusters are unambiguous -> similarity must hit at k=1
        assert block["similarity_hit_at_k"]["1"] == 1.0
        # with only two maintainers, top-3 frequency trivially contains truth
        assert block["most_active_hit_at_k"]["3"] == 1.0
        # overall block aggregates
        assert results["overall"]["n_test_with_assignee"] == 6

    def test_issues_without_assignee_are_not_scored(self, tmp_path):
        path, assignments = _synthetic_corpus(tmp_path)
        for rec in assignments["acme/widgets"].values():
            rec["assignees"] = []
        results = evaluate_recommender(path, assignments=assignments)
        assert results["overall"]["n_test_with_assignee"] == 0
        assert results["overall"]["similarity_hit_at_k"]["1"] is None


class TestEffortTarget:
    def test_log_days_to_close(self):
        from ghic.effort import _target_log_days
        import numpy as np

        df = pd.DataFrame([
            {"created_at": "2024-01-01T00:00:00Z", "closed_at": "2024-01-02T00:00:00Z"},
            {"created_at": "2024-01-01T00:00:00Z", "closed_at": "2024-01-01T00:00:00Z"},
            {"created_at": "2024-01-02T00:00:00Z", "closed_at": "2024-01-01T00:00:00Z"},
            {"created_at": "2024-01-01T00:00:00Z", "closed_at": None},
        ])
        t = _target_log_days(df)
        assert t.iloc[0] == pytest.approx(np.log1p(1.0))
        assert t.iloc[1] == 0.0
        assert t.iloc[2] == 0.0          # negative durations clip to zero
        assert t.iloc[3] != t.iloc[3]    # NaN


class TestAssignmentsRoundTrip:
    def test_load_assignments_reads_written_json(self, tmp_path, monkeypatch):
        from ghic import assign

        path = tmp_path / "assignments.json"
        payload = {"acme/widgets": {"1": {"assignees": ["a"], "participants": [], "closer": None}}}
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert assign.load_assignments(path) == payload
        assert assign.load_assignments(tmp_path / "missing.json") is None
