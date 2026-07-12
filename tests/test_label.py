"""Unit tests for ghic.label rule logic.

Every rule branch has at least one fixture issue that exercises it. Precedence
tests check ordering across rules. R1b regex tests cover the closing-keyword
variants you'd actually see in PR bodies (this is where I shipped a regex bug
once already — full keyword coverage is non-negotiable).

Run from project root: python -m pytest tests/ -v
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from ghic.config import get_config
from ghic.label import (
    DROP_BOT,
    DROP_LOCKED,
    DROP_NO_AUTHOR,
    DROP_QUESTION,
    R1_MERGED_PR,
    R1B_MERGED_PR_TEXTUAL,
    R2_BUG_LABEL_COMPLETED,
    R3_NON_ACTIONABLE_LABEL,
    R3B_STATE_NOT_PLANNED,
    R4_DEFAULT,
    LabelResult,
    label_dataset,
    label_issue,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def labeling():
    """Real labeling config from config.yaml. No token needed."""
    return get_config(require_token=False).labeling


def make_issue(**overrides: Any) -> dict[str, Any]:
    """Default issue with all required keys; tests override specific fields."""
    base: dict[str, Any] = {
        "number": 1,
        "author_login": "alice",
        "locked": False,
        "state_reason": None,
        "labels_at_close": [],
        "closed_by_merged_pr": False,
        "cross_referenced_prs": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------
class TestExclusions:
    def test_drop_no_author(self, labeling):
        issue = make_issue(author_login=None)
        assert label_issue(issue, labeling) == LabelResult(None, DROP_NO_AUTHOR)

    def test_drop_empty_author_string(self, labeling):
        issue = make_issue(author_login="")
        assert label_issue(issue, labeling) == LabelResult(None, DROP_NO_AUTHOR)

    def test_drop_bot_beats_R1(self, labeling):
        """Bot exclusion fires even when R1 would otherwise classify as Class 1."""
        issue = make_issue(author_login="dependabot[bot]", closed_by_merged_pr=True)
        assert label_issue(issue, labeling) == LabelResult(None, DROP_BOT)

    def test_drop_locked_beats_R1_when_enabled(self, labeling):
        """With drop_locked_issues=True the exclusion outranks R1."""
        labeling_lock_drop = replace(labeling, drop_locked_issues=True)
        issue = make_issue(locked=True, closed_by_merged_pr=True)
        assert label_issue(issue, labeling_lock_drop) == LabelResult(None, DROP_LOCKED)

    def test_locked_issues_kept_by_default(self, labeling):
        """Project config disables drop_locked (the vscode auto-lock finding):
        locked issues fall through to normal classification."""
        assert labeling.drop_locked_issues is False
        issue = make_issue(locked=True, closed_by_merged_pr=True)
        assert label_issue(issue, labeling) == LabelResult(1, R1_MERGED_PR)

    def test_drop_question_beats_R1(self, labeling):
        """Question label drops even with merged PR — questions are a different population."""
        issue = make_issue(labels_at_close=["question"], closed_by_merged_pr=True)
        assert label_issue(issue, labeling) == LabelResult(None, DROP_QUESTION)


# ---------------------------------------------------------------------------
# Class 1 rules
# ---------------------------------------------------------------------------
class TestClass1:
    def test_R1_merged_pr_closer(self, labeling):
        issue = make_issue(closed_by_merged_pr=True)
        assert label_issue(issue, labeling) == LabelResult(1, R1_MERGED_PR)

    @pytest.mark.parametrize("keyword", [
        "Fix", "Fixes", "Fixed",
        "Close", "Closes", "Closed",
        "Resolve", "Resolves", "Resolved",
        "fix", "fixes", "fixed",
        "CLOSES", "RESOLVED",
    ])
    def test_R1b_all_closing_keywords(self, labeling, keyword):
        """The bug I shipped once: 'Fixes' and 'Fixed' must match, not just 'Fix'."""
        issue = make_issue(
            number=42,
            cross_referenced_prs=[{"number": 100, "merged": True, "body": f"{keyword} #42"}],
        )
        assert label_issue(issue, labeling) == LabelResult(1, R1B_MERGED_PR_TEXTUAL)

    def test_R1b_unmerged_pr_does_not_count(self, labeling):
        issue = make_issue(
            number=42,
            cross_referenced_prs=[{"number": 100, "merged": False, "body": "Fixes #42"}],
        )
        assert label_issue(issue, labeling) == LabelResult(0, R4_DEFAULT)

    def test_R1b_different_issue_number_does_not_count(self, labeling):
        """PR fixes #1234, but this is issue #123 — \\b boundary must prevent the match."""
        issue = make_issue(
            number=123,
            cross_referenced_prs=[{"number": 100, "merged": True, "body": "Fixes #1234"}],
        )
        assert label_issue(issue, labeling) == LabelResult(0, R4_DEFAULT)

    def test_R1b_casual_mention_does_not_count(self, labeling):
        """No closing keyword — should not match."""
        issue = make_issue(
            number=42,
            cross_referenced_prs=[{"number": 100, "merged": True, "body": "See also #42 for context"}],
        )
        assert label_issue(issue, labeling) == LabelResult(0, R4_DEFAULT)

    def test_R1b_suffix_word_does_not_count(self, labeling):
        """'Suffixes #42' must not match because \\b prevents mid-word matches."""
        issue = make_issue(
            number=42,
            cross_referenced_prs=[{"number": 100, "merged": True, "body": "Suffixes #42"}],
        )
        assert label_issue(issue, labeling) == LabelResult(0, R4_DEFAULT)

    def test_R1b_multiple_prs_any_match_wins(self, labeling):
        issue = make_issue(
            number=42,
            cross_referenced_prs=[
                {"number": 100, "merged": True, "body": "Mentions #42"},        # no keyword
                {"number": 101, "merged": False, "body": "Fixes #42"},          # unmerged
                {"number": 102, "merged": True, "body": "Fixes #42 properly"},  # match
            ],
        )
        assert label_issue(issue, labeling) == LabelResult(1, R1B_MERGED_PR_TEXTUAL)

    def test_R2_bug_label_completed(self, labeling):
        issue = make_issue(labels_at_close=["bug"], state_reason="COMPLETED")
        assert label_issue(issue, labeling) == LabelResult(1, R2_BUG_LABEL_COMPLETED)

    def test_R2_label_case_insensitive(self, labeling):
        issue = make_issue(labels_at_close=["BUG"], state_reason="COMPLETED")
        assert label_issue(issue, labeling) == LabelResult(1, R2_BUG_LABEL_COMPLETED)

    def test_R2_alternate_bug_label_form(self, labeling):
        """'type:bug' is a real label form in vscode."""
        issue = make_issue(labels_at_close=["type:bug"], state_reason="COMPLETED")
        assert label_issue(issue, labeling) == LabelResult(1, R2_BUG_LABEL_COMPLETED)

    def test_R2_requires_completed_state(self, labeling):
        """Bug label + NOT_PLANNED state — R2 must not fire; falls to R3b."""
        issue = make_issue(labels_at_close=["bug"], state_reason="NOT_PLANNED")
        assert label_issue(issue, labeling) == LabelResult(0, R3B_STATE_NOT_PLANNED)


# ---------------------------------------------------------------------------
# Class 0 rules
# ---------------------------------------------------------------------------
class TestClass0:
    @pytest.mark.parametrize("label", ["duplicate", "invalid", "wontfix", "stale", "not-a-bug"])
    def test_R3_non_actionable_labels(self, labeling, label):
        issue = make_issue(labels_at_close=[label])
        assert label_issue(issue, labeling) == LabelResult(0, R3_NON_ACTIONABLE_LABEL)

    def test_R3b_not_planned_only(self, labeling):
        issue = make_issue(state_reason="NOT_PLANNED")
        assert label_issue(issue, labeling) == LabelResult(0, R3B_STATE_NOT_PLANNED)

    def test_R4_default_with_no_signals(self, labeling):
        issue = make_issue()
        assert label_issue(issue, labeling) == LabelResult(0, R4_DEFAULT)

    def test_R4_default_with_completed_state_no_bug_label(self, labeling):
        """Closed-completed but no bug label, no merged PR — defaults to Class 0."""
        issue = make_issue(state_reason="COMPLETED")
        assert label_issue(issue, labeling) == LabelResult(0, R4_DEFAULT)


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------
class TestPrecedence:
    def test_R1_beats_R3(self, labeling):
        """Merged PR + non-actionable label — R1 wins (PR-based signal is stronger)."""
        issue = make_issue(closed_by_merged_pr=True, labels_at_close=["duplicate"])
        assert label_issue(issue, labeling) == LabelResult(1, R1_MERGED_PR)

    def test_R1_beats_R1b(self, labeling):
        """R1 short-circuits before R1b regex is even checked."""
        issue = make_issue(
            number=42,
            closed_by_merged_pr=True,
            cross_referenced_prs=[{"number": 100, "merged": True, "body": "Fixes #42"}],
        )
        assert label_issue(issue, labeling) == LabelResult(1, R1_MERGED_PR)

    def test_R3_beats_R3b(self, labeling):
        """Non-actionable label + NOT_PLANNED — R3 fires before R3b."""
        issue = make_issue(labels_at_close=["duplicate"], state_reason="NOT_PLANNED")
        assert label_issue(issue, labeling) == LabelResult(0, R3_NON_ACTIONABLE_LABEL)


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------
class TestDataset:
    def test_audit_totals_match_input(self):
        """Every input issue increments exactly one audit counter."""
        issues = [
            make_issue(number=1, closed_by_merged_pr=True),
            make_issue(number=2, author_login="dependabot[bot]"),
            make_issue(number=3, labels_at_close=["duplicate"]),
            make_issue(number=4),
            make_issue(number=5, labels_at_close=["question"]),
        ]
        kept, audit = label_dataset(issues)
        assert sum(audit.values()) == len(issues)
        assert len(kept) == 3  # bot + question dropped, 3 classified

    def test_kept_rows_have_label_and_rule_and_preserve_original_fields(self):
        issues = [make_issue(number=42, title="hello", closed_by_merged_pr=True)]
        kept, _ = label_dataset(issues)
        assert kept[0]["label"] == 1
        assert kept[0]["label_rule"] == R1_MERGED_PR
        assert kept[0]["number"] == 42
        assert kept[0]["title"] == "hello"

    def test_dropped_rows_excluded_from_kept(self):
        issues = [
            make_issue(number=1, author_login="dependabot[bot]"),
            make_issue(number=2, labels_at_close=["question"]),
        ]
        kept, audit = label_dataset(issues)
        assert kept == []
        assert audit[DROP_BOT] == 1
        assert audit[DROP_QUESTION] == 1
