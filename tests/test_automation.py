"""Phase 4: engineering automation.

The most important tests here are in `TestSafetyGuarantees`. Everything
else checks that suggestions are useful; those check that they can never
become actions. Feature 14 is a promise to a maintainer that a bot with
Issues write permission will not touch their repository, and a promise
like that is worth testing harder than the features it constrains.
"""
from __future__ import annotations

import time

import pytest

from ghic.automation import (
    ActionKind,
    AutomationFlags,
    AutomationService,
    Recommendation,
    build_pr_draft,
    build_weekly_digest,
    digest_inputs,
    render_automation_sections,
)
from ghic.engineering_intelligence import (
    ComponentStats,
    Confidence,
    EngineeringAnalyzer,
    Evidence,
    EvidenceKind,
)
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

ALL_ON = AutomationFlags(
    fix_suggestions=True, pr_drafts=True, test_plans=True, checklists=True,
    smart_labels=True, assignee_recommendations=True, duplicate_workflow=True,
)


def code_chunk(path="src/api/csv_import.py", symbol="parse_csv", **kw):
    return CodeChunk(
        repo="acme/proj", path=path, language="Python",
        text="def parse_csv(path):\n    return open(path).read()",
        start_line=1, end_line=3, kind="function", symbol=symbol,
        source=SOURCE_CODE, **kw,
    )


def commit_chunk(reference="abc12345", subject="Fix UTF-8 decoding in parse_csv",
                 files="src/api/csv_import.py", author="Dana Dev", age_days=3.0):
    return CodeChunk(
        repo="acme/proj", path=f"commit:{reference}", language="",
        text=f"Commit {reference} by {author}\n{subject}\nFiles changed: {files}",
        start_line=0, end_line=0, kind="commit", symbol=subject,
        source=SOURCE_COMMIT, reference=reference, timestamp=NOW - age_days * DAY,
    )


def issue_chunk(reference="#355", title="CSV import broke after upgrading",
                body="This used to work before.", age_days=20.0):
    return CodeChunk(
        repo="acme/proj", path=f"issue:{reference.lstrip('#')}", language="",
        text=f"Issue {reference}: {title}\nOutcome: closed as completed (resolved).\n{body}",
        start_line=0, end_line=0, kind="issue", symbol=title,
        source=SOURCE_ISSUE, reference=reference, timestamp=NOW - age_days * DAY,
    )


def make_context(chunks, scores=None):
    scores = scores or [0.6] * len(chunks)
    return RepositoryContext(
        repo="acme/proj", indexed=True,
        chunks=[RetrievedChunk(chunk=c, score=s) for c, s in zip(chunks, scores)],
    )


@pytest.fixture
def context():
    return make_context([code_chunk(), commit_chunk(), issue_chunk()])


@pytest.fixture
def analysis(context):
    return EngineeringAnalyzer().analyze(
        "acme/proj", "CSV import crashes",
        "parse_csv raises UnicodeDecodeError. This used to work in 2.6. "
        "Steps to reproduce: import a latin-1 file.",
        context, issue_number=999, issue_created_at=NOW,
    )


@pytest.fixture
def service():
    return AutomationService(ALL_ON, engine_version="test-1.0")


# ---------------------------------------------------------------------------
# Feature 14 -- the guarantees that matter most
# ---------------------------------------------------------------------------
class TestSafetyGuarantees:
    def test_no_destructive_action_kind_exists(self):
        """Not "we don't use MERGE" -- there is no MERGE to use. Adding one
        would be a visible, reviewable change rather than a one-line slip."""
        values = {a.value for a in ActionKind}
        for forbidden in ("merge", "close", "delete", "push", "write", "modify",
                          "force_push", "settings"):
            assert forbidden not in values

    def test_automation_service_holds_no_github_client(self, service):
        """A policy not to call a client is weaker than not having one."""
        for attribute in vars(service).values():
            assert not hasattr(attribute, "post_comment")
            assert not hasattr(attribute, "add_labels")

    def test_no_github_write_method_is_reachable_from_the_package(self):
        """Guards against a future contributor importing the client here."""
        import importlib
        import pkgutil

        import ghic.automation as package

        write_methods = (
            "post_comment", "add_labels", "add_issue_to_project",
            "installation_token", "merge_pull_request", "close_issue",
        )
        for module_info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"ghic.automation.{module_info.name}")
            source = getattr(module, "__file__", None)
            if not source:
                continue
            with open(source, encoding="utf-8") as handle:
                text = handle.read()
            for method in write_methods:
                assert f".{method}(" not in text, (
                    f"{module_info.name} calls {method} -- automation must stay advisory"
                )

    def test_every_recommendation_declares_itself_advisory(self, service, context, analysis):
        bundle = service.build("acme/proj", 999, "CSV crashes", "used to work",
                               analysis=analysis, repository_context=context)
        assert bundle.as_dict()["advisory_only"] is True
        for recommendation in bundle.all_recommendations:
            assert recommendation.is_advisory
            assert recommendation.as_dict()["advisory"] is True

    @pytest.mark.parametrize("title,body", [
        ("CSV import broke after upgrading", "This used to work before."),
        ("CSV import crashes", "Steps to reproduce: import a file.\n```\ntrace\n```"),
    ])
    def test_duplicate_workflow_never_presents_closing_as_its_own_action(
        self, service, analysis, title, body
    ):
        """GHIC may suggest a maintainer consider closing. It must never
        describe closing as something it does or will do."""
        ctx = make_context([code_chunk(), issue_chunk()], scores=[0.6, 0.9])
        bundle = service.build("acme/proj", 999, title, body,
                               analysis=analysis, repository_context=ctx)
        for recommendation in bundle.duplicate_workflow:
            assert recommendation.kind is ActionKind.LINK
            detail = recommendation.detail.lower()
            for forbidden in ("ghic will close", "automatically clos", "has been closed",
                              "closing this now", "i have closed"):
                assert forbidden not in detail

    def test_duplicate_recommendation_states_ghic_does_not_close(self, service, analysis):
        ctx = make_context([code_chunk(), issue_chunk(body="csv import fails")],
                           scores=[0.6, 0.95])
        bundle = service.build(
            "acme/proj", 999, "CSV import fails",
            "Steps to reproduce: import a file.\n```\ntrace\n```",
            analysis=analysis, repository_context=ctx,
        )
        duplicates = [r for r in bundle.duplicate_workflow if "Duplicate" in r.title]
        for recommendation in duplicates:
            assert "GHIC never closes" in recommendation.detail

    def test_comment_states_the_advisory_boundary(self, service, context, analysis):
        bundle = service.build("acme/proj", 999, "CSV crashes", "used to work",
                               analysis=analysis, repository_context=context)
        rendered = "\n".join(render_automation_sections(bundle))
        assert "does not apply labels" in rendered
        assert "open pull requests" in rendered

    def test_pr_draft_says_ghic_does_not_open_prs(self, context, analysis):
        draft = build_pr_draft("acme/proj", 999, "Fix CSV", analysis, context)
        assert "does not create or merge pull requests" in draft


# ---------------------------------------------------------------------------
# Feature 12 -- audit trail
# ---------------------------------------------------------------------------
class TestAuditTrail:
    def test_recommendation_requires_evidence(self):
        with pytest.raises(ValueError, match="no evidence"):
            Recommendation(
                kind=ActionKind.LABEL, title="bug", detail="d",
                confidence=Confidence.HIGH, reason="because", evidence=[],
            )

    def test_recommendation_requires_a_reason(self):
        with pytest.raises(ValueError, match="no reason"):
            Recommendation(
                kind=ActionKind.LABEL, title="bug", detail="d",
                confidence=Confidence.HIGH, reason="  ",
                evidence=[Evidence(kind=EvidenceKind.CODE, reference="a.py")],
            )

    def test_every_recommendation_carries_full_provenance(self, service, context, analysis):
        bundle = service.build("acme/proj", 999, "CSV crashes", "used to work",
                               analysis=analysis, repository_context=context)
        for recommendation in bundle.all_recommendations:
            record = recommendation.as_dict()
            assert record["created_at"] > 0
            assert record["engine_version"] == "test-1.0"
            assert record["inputs_digest"]
            assert record["confidence"]
            assert record["reason"]
            assert record["evidence"]

    def test_identical_inputs_produce_an_identical_digest(self):
        first = digest_inputs(repo="a/b", title="t", evidence=[1, 2])
        second = digest_inputs(evidence=[1, 2], title="t", repo="a/b")
        assert first == second

    def test_different_inputs_produce_a_different_digest(self):
        assert digest_inputs(repo="a/b", title="t") != digest_inputs(repo="a/b", title="u")

    def test_bundle_digest_is_stable_across_runs(self, context, analysis):
        service = AutomationService(ALL_ON, engine_version="v1")
        first = service.build("acme/proj", 1, "t", "b", analysis=analysis,
                              repository_context=context)
        second = service.build("acme/proj", 1, "t", "b", analysis=analysis,
                               repository_context=context)
        assert first.audit.inputs_digest == second.audit.inputs_digest


# ---------------------------------------------------------------------------
# Feature 13 -- independent flags
# ---------------------------------------------------------------------------
class TestFeatureFlags:
    def test_nothing_enabled_produces_nothing(self, context, analysis):
        service = AutomationService(AutomationFlags())
        assert service.build("acme/proj", 1, "t", "b", analysis=analysis,
                             repository_context=context) is None

    def test_capabilities_are_independent(self, context, analysis):
        service = AutomationService(AutomationFlags(smart_labels=True))
        bundle = service.build("acme/proj", 1, "CSV broke after upgrading",
                               "This used to work.", analysis=analysis,
                               repository_context=context)
        assert bundle.labels
        assert bundle.fix_plan == []
        assert bundle.test_plan == []
        assert bundle.pr_draft == ""

    def test_flags_read_from_environment(self, monkeypatch):
        monkeypatch.setenv("GHIC_SMART_LABELS", "true")
        monkeypatch.setenv("GHIC_PR_DRAFTS", "1")
        flags = AutomationFlags.from_env()
        assert flags.smart_labels and flags.pr_drafts
        assert not flags.fix_suggestions

    def test_all_flags_default_off(self):
        assert not AutomationFlags().any_enabled

    def test_no_evidence_means_no_automation(self):
        service = AutomationService(ALL_ON)
        empty = RepositoryContext(repo="acme/proj", indexed=True)
        assert service.build("acme/proj", 1, "t", "b", repository_context=empty) is None
        assert service.build("acme/proj", 1, "t", "b", repository_context=None) is None


# ---------------------------------------------------------------------------
# Features 1-7 -- the suggestions themselves
# ---------------------------------------------------------------------------
class TestSuggestions:
    def test_fix_plan_names_files_and_functions_not_patches(self, service, context, analysis):
        bundle = service.build("acme/proj", 1, "CSV crashes", "parse_csv fails",
                               analysis=analysis, repository_context=context)
        assert bundle.fix_plan
        blob = " ".join(r.detail for r in bundle.fix_plan)
        assert "src/api/csv_import.py" in blob
        # A plan, not a patch: no diff markers anywhere.
        assert "```diff" not in blob
        assert "\n+++" not in blob

    def test_fix_plan_requires_a_root_cause(self, service, context):
        """Without a hypothesis there is no evidence-backed opinion about
        what to change, and "review the code" is not a fix plan."""
        from ghic.engineering_intelligence.analysis import EngineeringAnalysis

        bundle = service.build("acme/proj", 1, "t", "b",
                               analysis=EngineeringAnalysis(repo="acme/proj"),
                               repository_context=context)
        assert bundle is None or bundle.fix_plan == []

    def test_regression_gets_a_regression_test_suggestion(self, service, context, analysis):
        bundle = service.build("acme/proj", 1, "CSV broke", "This used to work in 2.6.",
                               analysis=analysis, repository_context=context)
        assert any("Regression test" in r.title for r in bundle.test_plan)

    def test_edge_cases_from_the_report_are_suggested(self, service, context, analysis):
        bundle = service.build("acme/proj", 1, "CSV crash",
                               "Fails on utf-8 and empty files.",
                               analysis=analysis, repository_context=context)
        assert any("Edge cases" in r.title for r in bundle.test_plan)

    def test_checklist_asks_for_reproduction_when_missing(self, service, context, analysis):
        bundle = service.build("acme/proj", 1, "It is broken", "Please fix.",
                               analysis=analysis, repository_context=context)
        assert any(r.kind is ActionKind.REQUEST_INFO for r in bundle.checklist)

    def test_checklist_does_not_ask_when_reproduction_is_present(
        self, service, context, analysis
    ):
        bundle = service.build(
            "acme/proj", 1, "CSV crash",
            "Steps to reproduce: run it.\n```\nTraceback...\n```",
            analysis=analysis, repository_context=context,
        )
        assert not any(r.kind is ActionKind.REQUEST_INFO for r in bundle.checklist)

    @pytest.mark.parametrize("body,label", [
        ("There is a SQL injection vulnerability", "security"),
        ("The endpoint is very slow and times out", "performance"),
        ("This used to work in 2.6", "regression"),
        ("Please fix", "needs-reproduction"),
    ])
    def test_labels_are_evidence_backed(self, service, context, analysis, body, label):
        bundle = service.build("acme/proj", 1, "Issue", body,
                               analysis=analysis, repository_context=context)
        assert label in {r.title for r in bundle.labels}

    def test_every_label_carries_a_checkable_reason(self, service, context, analysis):
        bundle = service.build("acme/proj", 1, "Slow endpoint",
                               "The API endpoint is slow. This used to work.",
                               analysis=analysis, repository_context=context)
        for recommendation in bundle.labels:
            assert len(recommendation.reason) > 10
            assert recommendation.evidence

    def test_unsupported_labels_are_not_suggested(self, service, context, analysis):
        bundle = service.build("acme/proj", 1, "CSV crash",
                               "Steps to reproduce: run it.\n```\ntrace\n```",
                               analysis=analysis, repository_context=context)
        labels = {r.title for r in bundle.labels}
        assert "security" not in labels
        assert "database" not in labels

    def test_good_first_issue_needs_positive_evidence(self, service, analysis):
        """Guessing wrong sends a newcomer into a hard problem, so this
        requires narrowness to be shown, not merely un-contradicted."""
        wide = make_context([code_chunk(path=f"src/api/m{i}.py") for i in range(5)])
        bundle = service.build("acme/proj", 1, "Broken", "This used to work.",
                               analysis=analysis, repository_context=wide)
        assert bundle is None or "good-first-issue" not in {r.title for r in bundle.labels}

    def test_assignees_come_from_commit_authorship(self, service, context, analysis):
        bundle = service.build("acme/proj", 1, "CSV crash", "parse_csv fails",
                               analysis=analysis, repository_context=context)
        assert any("Dana Dev" in r.title for r in bundle.assignees)
        for recommendation in bundle.assignees:
            assert recommendation.kind is ActionKind.ASSIGN
            assert "does not assign" in recommendation.detail

    def test_bot_authors_are_excluded_from_assignee_suggestions(self, service, analysis):
        ctx = make_context([code_chunk(), commit_chunk(author="dependabot[bot]")])
        bundle = service.build("acme/proj", 1, "CSV crash", "parse_csv",
                               analysis=analysis, repository_context=ctx)
        assert bundle is None or bundle.assignees == []

    def test_regression_family_is_not_recommended_for_closure(self, service, analysis):
        ctx = make_context(
            [code_chunk(), issue_chunk(body="regression: this used to work")],
            scores=[0.6, 0.9],
        )
        bundle = service.build("acme/proj", 1, "Broke again",
                               "This used to work before the upgrade.",
                               analysis=analysis, repository_context=ctx)
        for recommendation in bundle.duplicate_workflow:
            if "Regression Family" in recommendation.title:
                assert "Keep this open" in recommendation.detail

    def test_one_broken_generator_does_not_lose_the_bundle(self, context, analysis, monkeypatch):
        import ghic.automation.service as service_module

        def boom(*a, **k):
            raise RuntimeError("generator exploded")

        monkeypatch.setattr(service_module, "build_label_suggestions", boom)
        bundle = AutomationService(ALL_ON).build(
            "acme/proj", 1, "CSV crash", "parse_csv fails",
            analysis=analysis, repository_context=context,
        )
        assert bundle is not None
        assert bundle.labels == []
        assert bundle.checklist


# ---------------------------------------------------------------------------
# Feature 2 -- PR drafts
# ---------------------------------------------------------------------------
class TestPullRequestDraft:
    def test_draft_has_every_required_section(self, context, analysis):
        draft = build_pr_draft("acme/proj", 999, "Fix CSV encoding", analysis, context)
        for heading in ("### Summary", "### Problem", "### Likely cause",
                        "### Suggested changes", "### Testing checklist",
                        "### Breaking changes", "### Related issues",
                        "### Related commits"):
            assert heading in draft

    def test_draft_leaves_human_only_fields_blank(self, context, analysis):
        """A draft that looks finished invites being submitted unread."""
        draft = build_pr_draft("acme/proj", 999, "Fix CSV", analysis, context)
        assert "<!--" in draft
        assert "Breaking changes" in draft

    def test_draft_links_the_issue(self, context, analysis):
        assert "Closes #999" in build_pr_draft("acme/proj", 999, "t", analysis, context)

    def test_draft_references_only_retrieved_artifacts(self, context, analysis):
        draft = build_pr_draft("acme/proj", 999, "t", analysis, context)
        assert "abc12345" in draft
        import re

        for sha in re.findall(r"`([0-9a-f]{8})`", draft):
            assert sha == "abc12345"


# ---------------------------------------------------------------------------
# Feature 9 -- weekly digest
# ---------------------------------------------------------------------------
class TestWeeklyDigest:
    @pytest.fixture
    def stats(self):
        return {
            "api": ComponentStats(name="api", issue_count=20, regression_count=12,
                                  commit_count=40, recent_issue_count=8,
                                  resolution_days=[10.0] * 20),
            "ui": ComponentStats(name="ui", issue_count=3, commit_count=5),
        }

    def test_digest_reports_measured_counts(self, stats):
        digest = build_weekly_digest("acme/proj", stats, [])
        assert "acme/proj" in digest
        assert "Components tracked: **2**" in digest

    def test_unscored_components_are_not_ranked_as_fragile(self, stats):
        digest = build_weekly_digest("acme/proj", stats, [])
        fragile_section = digest.split("## Most fragile components")[1].split("##")[0]
        assert "`api`" in fragile_section
        assert "`ui`" not in fragile_section

    def test_digest_says_so_when_there_is_no_data(self):
        digest = build_weekly_digest("acme/proj", {}, [])
        assert "not computed" in digest
        assert "No component has enough history" in digest

    def test_digest_labels_indices_as_heuristics(self, stats):
        assert "not calibrated probabilities" in build_weekly_digest("acme/proj", stats, [])

    def test_plain_text_format_strips_markdown(self, stats):
        text = build_weekly_digest("acme/proj", stats, [], fmt="text")
        assert "**" not in text
        assert "# " not in text


# ---------------------------------------------------------------------------
# Comment rendering
# ---------------------------------------------------------------------------
class TestRendering:
    def test_nothing_rendered_without_a_bundle(self):
        assert render_automation_sections(None) == []

    def test_suggestions_are_collapsed(self, service, context, analysis):
        bundle = service.build("acme/proj", 1, "CSV crash", "parse_csv fails",
                               analysis=analysis, repository_context=context)
        rendered = "\n".join(render_automation_sections(bundle))
        assert "<details>" in rendered

    def test_comment_includes_automation_when_supplied(self, service, context, analysis):
        from test_service import make_llm_analysis

        from ghic.service.inference import Prediction, format_llm_comment

        bundle = service.build("acme/proj", 1, "CSV crash", "parse_csv fails",
                               analysis=analysis, repository_context=context)
        pred = Prediction(repo="acme/proj", issue_number=1, proba=0.9, threshold=0.5,
                          predicted_label=1, model_name="stub")
        comment = format_llm_comment(pred, make_llm_analysis(),
                                     repository_context=context, automation=bundle)
        assert "### Suggested next steps" in comment
        assert "advisory" in comment.lower()

    def test_comment_unchanged_without_automation(self):
        from test_service import make_llm_analysis

        from ghic.service.inference import Prediction, format_llm_comment

        pred = Prediction(repo="acme/proj", issue_number=1, proba=0.9, threshold=0.5,
                          predicted_label=1, model_name="stub")
        comment = format_llm_comment(pred, make_llm_analysis(), automation=None)
        assert "Suggested next steps" not in comment
