"""Phase 3: Engineering Intelligence.

The tests that matter most here are the ones asserting what the system
*refuses* to do -- claim without evidence, score without data, assert
causation, or render a heading with nothing under it. Those are the
properties that make the rest trustworthy, and they're the ones a later
refactor is most likely to quietly break.
"""
from __future__ import annotations

import time

import pytest

from ghic.engineering_intelligence import (
    Claim,
    ClaimStatus,
    ComponentAnalyzer,
    ComponentStats,
    Confidence,
    EngineeringAnalyzer,
    Evidence,
    EvidenceKind,
    Finding,
    classify_relationship,
    cluster_history,
    component_for_path,
    confidence_from_support,
    health_score,
    hedge,
    is_regression_text,
    parse_releases,
    render_engineering_sections,
    render_finding,
    render_timeline,
    summarize_repository,
)
from ghic.engineering_intelligence.clustering import (
    DUPLICATE,
    REGRESSION_FAMILY,
    RELATED,
)
from ghic.engineering_intelligence.components import MIN_ISSUES_FOR_SCORE
from ghic.repository_intelligence.models import (
    SOURCE_CODE,
    SOURCE_COMMIT,
    SOURCE_ISSUE,
    CodeChunk,
    RepositoryContext,
    RetrievedChunk,
)

NOW = time.time()
DAY = 86400.0


def code_chunk(path="src/importer/csv_parser.py", symbol="parse_csv", **kw):
    return CodeChunk(
        repo="acme/proj", path=path, language="Python",
        text=kw.pop("text", "def parse_csv(path):\n    return open(path).read()"),
        start_line=kw.pop("start_line", 1), end_line=kw.pop("end_line", 3),
        kind="function", symbol=symbol, source=SOURCE_CODE, **kw,
    )


def commit_chunk(reference="abc12345", subject="Fix UTF-8 decoding in parse_csv",
                 files="src/importer/csv_parser.py", age_days=3.0, **kw):
    return CodeChunk(
        repo="acme/proj", path=f"commit:{reference}", language="",
        text=f"Commit {reference} by Dev\n{subject}\nFiles changed: {files}",
        start_line=0, end_line=0, kind="commit", symbol=subject,
        source=SOURCE_COMMIT, reference=reference,
        timestamp=NOW - age_days * DAY, **kw,
    )


def issue_chunk(reference="#355", title="CSV import broke after upgrading",
                body="This used to work before.", age_days=20.0, **kw):
    return CodeChunk(
        repo="acme/proj", path=f"issue:{reference.lstrip('#')}", language="",
        text=f"Issue {reference}: {title}\nOutcome: closed as completed (resolved).\n{body}",
        start_line=0, end_line=0, kind="issue", symbol=title,
        source=SOURCE_ISSUE, reference=reference, timestamp=NOW - age_days * DAY, **kw,
    )


def make_context(chunks, scores=None):
    scores = scores or [0.6] * len(chunks)
    return RepositoryContext(
        repo="acme/proj", indexed=True,
        chunks=[RetrievedChunk(chunk=c, score=s) for c, s in zip(chunks, scores)],
    )


# ---------------------------------------------------------------------------
# The evidence guarantee (Feature 14)
# ---------------------------------------------------------------------------
class TestEvidenceGuarantee:
    def test_a_claim_without_evidence_cannot_be_constructed(self):
        """The single most important test in Phase 3: there is no code path
        that can produce an unfounded conclusion."""
        with pytest.raises(ValueError, match="no evidence"):
            Claim(statement="the parser is broken", status=ClaimStatus.INFERRED,
                  confidence=Confidence.HIGH, evidence=[])

    def test_an_empty_claim_is_rejected(self):
        with pytest.raises(ValueError):
            Claim(statement="   ", status=ClaimStatus.VERIFIED,
                  confidence=Confidence.LOW,
                  evidence=[Evidence(kind=EvidenceKind.CODE, reference="a.py")])

    def test_evidence_is_built_from_chunks_not_written(self):
        evidence = Evidence.from_chunk(commit_chunk(reference="deadbeef"))
        assert evidence.kind is EvidenceKind.COMMIT
        assert evidence.reference == "deadbeef"

    def test_code_evidence_carries_the_line_span(self):
        evidence = Evidence.from_chunk(code_chunk(start_line=10, end_line=42))
        assert evidence.reference == "src/importer/csv_parser.py:10-42"

    def test_verified_and_inferred_render_differently(self):
        item = Evidence(kind=EvidenceKind.CODE, reference="a.py")
        verified = Claim(statement="X", status=ClaimStatus.VERIFIED,
                         confidence=Confidence.HIGH, evidence=[item])
        inferred = Claim(statement="X", status=ClaimStatus.INFERRED,
                         confidence=Confidence.HIGH, evidence=[item])
        assert "Observed" in verified.render()
        assert "Inferred" in inferred.render()

    @pytest.mark.parametrize("sources,strong,expected", [
        (1, False, Confidence.LOW),
        (1, True, Confidence.MEDIUM),
        (2, False, Confidence.MEDIUM),
        (2, True, Confidence.HIGH),
        (3, False, Confidence.HIGH),
    ])
    def test_confidence_is_computed_from_support(self, sources, strong, expected):
        assert confidence_from_support(
            distinct_sources=sources, strong_signal=strong
        ) is expected

    def test_contradiction_caps_confidence(self):
        assert confidence_from_support(
            distinct_sources=5, strong_signal=True, contradicted=True
        ) is Confidence.LOW

    def test_hedging_vocabulary_never_asserts_certainty(self):
        for level in Confidence:
            assert hedge(level) in {"likely", "possibly", "may be"}

    def test_finding_confidence_is_its_best_claim(self):
        item = Evidence(kind=EvidenceKind.CODE, reference="a.py")
        finding = Finding(title="t", claims=[
            Claim(statement="a", status=ClaimStatus.INFERRED,
                  confidence=Confidence.LOW, evidence=[item]),
            Claim(statement="b", status=ClaimStatus.INFERRED,
                  confidence=Confidence.HIGH, evidence=[item]),
        ])
        assert finding.confidence is Confidence.HIGH


# ---------------------------------------------------------------------------
# Components + health (Features 4, 13)
# ---------------------------------------------------------------------------
class TestComponents:
    @pytest.mark.parametrize("path,expected", [
        ("src/auth/oauth/token.py", "auth/oauth"),
        ("ghic/service/app.py", "ghic/service"),
        ("lib/payments/stripe.go", "payments"),
        ("README.md", "README.md"),
        ("commit:abc123", ""),
        ("", ""),
    ])
    def test_component_inference(self, path, expected):
        assert component_for_path(path) == expected

    @pytest.mark.parametrize("text,expected", [
        ("This used to work in 2.6", True),
        ("broke after upgrading", True),
        ("regression in the parser", True),
        ("Please add dark mode", False),
        ("", False),
    ])
    def test_regression_marker_detection(self, text, expected):
        assert is_regression_text(text) is expected

    def test_no_score_below_the_minimum_sample(self):
        """A percentage from three data points is the false precision this
        project exists not to ship."""
        stats = ComponentStats(name="auth", issue_count=MIN_ISSUES_FOR_SCORE - 1)
        assert health_score(stats) is None
        assert stats.health_band == "insufficient history"

    def test_score_is_produced_with_enough_history(self):
        stats = ComponentStats(name="auth", issue_count=20, regression_count=0,
                               resolution_days=[1.0] * 20)
        score = health_score(stats)
        assert score is not None and 0 <= score <= 100

    def test_regressions_reduce_health(self):
        clean = ComponentStats(name="a", issue_count=20, regression_count=0)
        broken = ComponentStats(name="b", issue_count=20, regression_count=15)
        assert health_score(broken) < health_score(clean)

    def test_recent_pressure_reduces_health(self):
        calm = ComponentStats(name="a", issue_count=20, recent_issue_count=0)
        busy = ComponentStats(name="b", issue_count=20, recent_issue_count=18)
        assert health_score(busy) < health_score(calm)

    def test_slow_resolution_reduces_health(self):
        fast = ComponentStats(name="a", issue_count=20, resolution_days=[1.0] * 20)
        slow = ComponentStats(name="b", issue_count=20, resolution_days=[60.0] * 20)
        assert health_score(slow) < health_score(fast)

    def test_bands(self):
        assert ComponentStats(name="a", issue_count=20).health_band == "healthy"
        assert ComponentStats(
            name="b", issue_count=20, regression_count=20, recent_issue_count=20
        ).health_band == "fragile"

    def test_analyzer_links_issues_to_components_through_commits(self):
        """Attribution comes from evidence -- a commit names both its files
        and the issue it closed -- not from guessing at issue titles."""
        chunks = [
            code_chunk(),
            commit_chunk(subject="Fix parse_csv (fixes #355)"),
            issue_chunk(),
        ]
        stats = ComponentAnalyzer().analyze(chunks)
        assert "importer" in stats
        assert stats["importer"].issue_count == 1
        assert stats["importer"].commit_count == 1

    def test_unlinked_issues_are_not_attributed(self):
        stats = ComponentAnalyzer().analyze([
            code_chunk(), commit_chunk(subject="Unrelated change"), issue_chunk(),
        ])
        assert stats.get("importer", ComponentStats(name="x")).issue_count == 0

    def test_repository_summary_withholds_risk_without_history(self):
        summary = summarize_repository({"a": ComponentStats(name="a", issue_count=2)})
        assert summary["risk_index"] is None

    def test_repository_summary_reports_risk_with_history(self):
        summary = summarize_repository({
            "a": ComponentStats(name="a", issue_count=30, regression_count=25,
                                recent_issue_count=25),
        })
        assert summary["risk_index"] is not None
        assert "not a calibrated probability" in summary["risk_index_note"]


# ---------------------------------------------------------------------------
# Clustering (Feature 3)
# ---------------------------------------------------------------------------
class TestClustering:
    def test_regression_family_beats_duplicate(self):
        """Two reports of the same recurring break are not duplicates --
        closing the second loses the signal that it happened twice."""
        label = classify_relationship(
            similarity=0.9, issue_text="broke after upgrading",
            candidate_text="this used to work", issue_components={"importer"},
            candidate_components={"importer"},
        )
        assert label == REGRESSION_FAMILY

    def test_high_similarity_plus_component_is_a_duplicate(self):
        assert classify_relationship(
            similarity=0.8, issue_text="csv fails", candidate_text="csv fails",
            issue_components={"importer"}, candidate_components={"importer"},
        ) == DUPLICATE

    def test_moderate_similarity_is_only_related(self):
        assert classify_relationship(
            similarity=0.3, issue_text="csv fails", candidate_text="json fails",
            issue_components={"importer"}, candidate_components={"importer"},
        ) == RELATED

    def test_below_the_floor_is_not_a_relationship(self):
        assert classify_relationship(
            similarity=0.05, issue_text="a", candidate_text="b",
            issue_components=set(), candidate_components=set(),
        ) is None

    def test_clusters_group_by_component(self):
        chunks = [
            code_chunk(),
            commit_chunk(reference="c1", subject="fix (fixes #355)"),
            commit_chunk(reference="c2", subject="fix again (fixes #400)"),
            issue_chunk(reference="#355"),
            issue_chunk(reference="#400", title="CSV broke again",
                        body="regression, used to work"),
        ]
        clusters = cluster_history(chunks)
        assert clusters and clusters[0].component == "importer"
        assert clusters[0].size == 2

    def test_regression_family_detected_in_a_cluster(self):
        chunks = [
            commit_chunk(reference="c1", subject="fix (fixes #1)"),
            commit_chunk(reference="c2", subject="fix (fixes #2)"),
            issue_chunk(reference="#1", body="regression: used to work"),
            issue_chunk(reference="#2", body="broke after upgrading again"),
        ]
        clusters = cluster_history(chunks)
        assert clusters[0].is_regression_family

    def test_empty_input_clusters_to_nothing(self):
        assert cluster_history([]) == []


# ---------------------------------------------------------------------------
# Analysis (Features 1, 2, 5, 6, 7, 11)
# ---------------------------------------------------------------------------
class TestEngineeringAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return EngineeringAnalyzer()

    def test_root_cause_requires_file_overlap(self, analyzer):
        """A recent commit somewhere unrelated is not a lead."""
        context = make_context([
            code_chunk(),
            commit_chunk(files="docs/README.md", subject="Update docs"),
        ])
        analysis = analyzer.analyze("acme/proj", "csv fails", "parse_csv breaks", context,
                                    issue_created_at=NOW)
        assert analysis.root_cause is None

    def test_root_cause_found_with_overlap_and_recency(self, analyzer):
        context = make_context([code_chunk(), commit_chunk()])
        analysis = analyzer.analyze(
            "acme/proj", "CSV crashes", "parse_csv raises UnicodeDecodeError",
            context, issue_created_at=NOW,
        )
        assert analysis.root_cause is not None
        assert "abc12345" in analysis.root_cause.claims[0].statement

    def test_old_commits_are_not_suspects(self, analyzer):
        context = make_context([code_chunk(), commit_chunk(age_days=400)])
        analysis = analyzer.analyze("acme/proj", "csv fails", "parse_csv", context,
                                    issue_created_at=NOW)
        assert analysis.root_cause is None

    def test_root_cause_never_asserts_certainty(self, analyzer):
        context = make_context([code_chunk(), commit_chunk()])
        analysis = analyzer.analyze("acme/proj", "CSV crashes", "parse_csv", context,
                                    issue_created_at=NOW)
        for claim in analysis.root_cause.claims:
            assert claim.status is ClaimStatus.INFERRED
            assert any(word in claim.statement for word in ("likely", "possibly", "may be"))
            assert "caused by" not in claim.statement.lower()

    def test_reporter_words_are_the_verified_regression_signal(self, analyzer):
        context = make_context([code_chunk(), commit_chunk()])
        analysis = analyzer.analyze(
            "acme/proj", "CSV broken", "This used to work in 2.6.", context,
            issue_created_at=NOW,
        )
        verified = [c for c in analysis.regression.claims if c.status is ClaimStatus.VERIFIED]
        assert verified

    def test_release_window_correlation(self, analyzer):
        context = make_context([code_chunk(), commit_chunk()])
        analysis = analyzer.analyze(
            "acme/proj", "CSV broken", "fails", context, issue_created_at=NOW,
            releases=[{"name": "v2.8.0", "timestamp": NOW - 2 * DAY}],
        )
        assert any("v2.8.0" in c.statement for c in analysis.regression.claims)

    def test_release_outside_the_window_is_not_correlated(self, analyzer):
        context = make_context([code_chunk(), commit_chunk()])
        analysis = analyzer.analyze(
            "acme/proj", "CSV broken", "fails", context, issue_created_at=NOW,
            releases=[{"name": "v1.0.0", "timestamp": NOW - 300 * DAY}],
        )
        assert analysis.regression is None or not any(
            "v1.0.0" in c.statement for c in analysis.regression.claims
        )

    def test_investigation_plan_steps_all_carry_evidence(self, analyzer):
        context = make_context([code_chunk(), commit_chunk(), issue_chunk()])
        analysis = analyzer.analyze(
            "acme/proj", "CSV crashes", "parse_csv used to work", context,
            issue_created_at=NOW,
        )
        assert analysis.investigation_plan
        assert all(step.evidence for step in analysis.investigation_plan)

    def test_timeline_is_chronological_and_anchored(self, analyzer):
        context = make_context([
            code_chunk(), commit_chunk(age_days=5), issue_chunk(age_days=30),
        ])
        analysis = analyzer.analyze("acme/proj", "t", "b", context, issue_created_at=NOW)
        stamps = [e["timestamp"] for e in analysis.timeline]
        assert stamps == sorted(stamps)
        assert analysis.timeline[-1]["kind"] == "current_issue"

    def test_recurrence_needs_multiple_prior_issues(self, analyzer):
        one = make_context([code_chunk(), issue_chunk()])
        assert analyzer.analyze("acme/proj", "t", "b", one,
                                issue_created_at=NOW).recurrence is None

        many = make_context([
            code_chunk(), issue_chunk(reference="#1"), issue_chunk(reference="#2"),
        ])
        assert analyzer.analyze("acme/proj", "t", "b", many,
                                issue_created_at=NOW).recurrence is not None

    def test_impact_never_invents_customer_or_business_effects(self, analyzer):
        context = make_context([code_chunk(), issue_chunk()])
        analysis = analyzer.analyze("acme/proj", "t", "b", context, issue_created_at=NOW)
        if analysis.impact:
            blob = " ".join(c.statement.lower() for c in analysis.impact.claims)
            assert "customer" not in blob
            assert "revenue" not in blob

    def test_empty_context_yields_an_empty_analysis(self, analyzer):
        analysis = analyzer.analyze("acme/proj", "t", "b", None)
        assert analysis.is_empty

    def test_analysis_never_raises_on_malformed_input(self, analyzer):
        class Broken:
            is_empty = False

            @property
            def code_chunks(self):
                raise RuntimeError("boom")

        assert analyzer.analyze("acme/proj", "t", "b", Broken()).is_empty

    def test_release_parsing(self):
        releases = parse_releases("v2.8.0\t2026-08-01T00:00:00Z\nv2.7.0\t2026-07-01T00:00:00Z")
        assert [r["name"] for r in releases] == ["v2.8.0", "v2.7.0"]
        assert releases[0]["timestamp"] > releases[1]["timestamp"]

    def test_malformed_release_lines_are_skipped(self):
        assert parse_releases("garbage\n\nv1.0\tnot-a-date") == []


# ---------------------------------------------------------------------------
# Rendering (Feature 9)
# ---------------------------------------------------------------------------
class TestRendering:
    def test_empty_finding_renders_nothing(self):
        assert render_finding(Finding(title="Possible root cause")) == []
        assert render_finding(None) == []

    def test_empty_analysis_adds_no_sections(self):
        from ghic.engineering_intelligence.analysis import EngineeringAnalysis

        assert render_engineering_sections(EngineeringAnalysis(repo="a/b")) == []
        assert render_engineering_sections(None) == []

    def test_single_event_timeline_is_not_rendered(self):
        assert render_timeline([{"kind": "current_issue", "reference": "this issue",
                                 "timestamp": NOW, "label": "reported"}]) == []

    def test_timeline_anchor_row_is_not_duplicated(self):
        rows = render_timeline([
            {"kind": "commit", "reference": "abc", "timestamp": NOW - DAY, "label": "x"},
            {"kind": "current_issue", "reference": "this issue", "timestamp": NOW,
             "label": "reported"},
        ])
        anchor = [r for r in rows if "this issue" in r][0]
        assert anchor.count("this issue") == 1

    def test_report_shows_observed_and_inferred_labels(self):
        context = make_context([code_chunk(), commit_chunk()])
        analysis = EngineeringAnalyzer().analyze(
            "acme/proj", "CSV crashes", "parse_csv used to work", context,
            issue_created_at=NOW,
        )
        rendered = "\n".join(render_engineering_sections(analysis))
        assert "**Observed**" in rendered
        assert "**Inferred**" in rendered

    def test_every_rendered_reference_comes_from_evidence(self):
        """The Phase 3 form of the anti-hallucination guarantee."""
        context = make_context([code_chunk(), commit_chunk(reference="feedface")])
        analysis = EngineeringAnalyzer().analyze(
            "acme/proj", "CSV crashes", "parse_csv", context, issue_created_at=NOW,
        )
        rendered = "\n".join(render_engineering_sections(analysis))
        assert "feedface" in rendered
        # No other SHA-shaped token may appear.
        import re

        for token in re.findall(r"`([0-9a-f]{8})`", rendered):
            assert token == "feedface"

    def test_comment_includes_engineering_sections_when_supplied(self):
        from test_service import make_llm_analysis

        from ghic.service.inference import Prediction, format_llm_comment

        context = make_context([code_chunk(), commit_chunk()])
        analysis = EngineeringAnalyzer().analyze(
            "acme/proj", "CSV crashes", "parse_csv used to work", context,
            issue_created_at=NOW,
        )
        pred = Prediction(repo="acme/proj", issue_number=1, proba=0.9, threshold=0.5,
                          predicted_label=1, model_name="stub")
        comment = format_llm_comment(
            pred, make_llm_analysis(), repository_context=context,
            engineering_analysis=analysis,
        )
        assert "### Possible root cause" in comment
        assert "### Investigation plan" in comment
        assert "Observed" in comment and "Inferred" in comment

    def test_comment_unchanged_when_analysis_is_absent(self):
        from test_service import make_llm_analysis

        from ghic.service.inference import Prediction, format_llm_comment

        pred = Prediction(repo="acme/proj", issue_number=1, proba=0.9, threshold=0.5,
                          predicted_label=1, model_name="stub")
        comment = format_llm_comment(pred, make_llm_analysis(), engineering_analysis=None)
        assert "Possible root cause" not in comment
        assert "Investigation plan" not in comment
