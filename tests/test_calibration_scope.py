"""The statistical score is only shown where it means something.

The classifier learned from three large repositories. Asked about anything
else it still returns a confident-looking percentage, computed from author
metadata and text patterns that carry different information outside the
population it was fitted on. It scored a real, reproducible bug at 26% --
"Likely not actionable" -- on a small personal repository, which is the
failure these tests exist to prevent: not a wrong number, but a wrong number
presented as a finding.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ghic.service.inference import Prediction, format_llm_comment
from ghic.service.settings import ServiceSettings, parse_repo_list

MODEL = Path("models/champion.joblib")


def settings(**kwargs) -> ServiceSettings:
    return ServiceSettings(model_path=MODEL, **kwargs)


def prediction(proba: float = 0.26, label: int = 0) -> Prediction:
    return Prediction(
        repo="acme/widgets", issue_number=1, proba=proba, threshold=0.5,
        predicted_label=label, model_name="champion",
    )


def analysis(**kwargs):
    base = dict(
        summary="A summary.", recommended_action="Do the thing.",
        reasoning=["Because."], missing_information=[], recommended_labels=[],
        severity="high", risk_level="high", category="bug", priority="high",
        confidence=0.9, risk_reasons=["A risk."], business_impact="",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestCalibrationScope:
    def test_nothing_is_calibrated_by_default(self):
        # Fail closed. A deployment claims calibration explicitly; the product
        # does not assume it and publish a number it cannot support.
        s = settings()
        assert s.score_is_calibrated_for("microsoft/vscode") is False
        assert s.score_is_calibrated_for("acme/widgets") is False

    def test_an_explicitly_listed_repository_is_calibrated(self):
        s = settings(calibrated_repos=frozenset({"facebook/react"}))
        assert s.score_is_calibrated_for("facebook/react") is True
        assert s.score_is_calibrated_for("acme/widgets") is False

    def test_a_verified_threshold_counts_as_calibration(self):
        # A per-repo threshold only exists because ghic.backtest measured that
        # repository's held-out issues and the result beat the default, which
        # is precisely the evidence this question asks for.
        s = settings(repo_thresholds={"microsoft/vscode": 0.18})
        assert s.score_is_calibrated_for("microsoft/vscode") is True

    @pytest.mark.parametrize("name", ["", "   ", None])
    def test_a_missing_repository_name_is_not_calibrated(self, name):
        assert settings().score_is_calibrated_for(name) is False

    def test_parse_repo_list(self):
        assert parse_repo_list("") == set()
        assert parse_repo_list("a/b, c/d ,") == {"a/b", "c/d"}


class TestUncalibratedComment:
    def test_the_verdict_is_withheld(self):
        out = format_llm_comment(prediction(), analysis(), [], score_is_calibrated=False)
        assert "Needs maintainer review" in out.splitlines()[0]
        assert "Likely not actionable" not in out
        assert "Likely actionable" not in out

    def test_the_score_is_replaced_by_the_reason(self):
        out = format_llm_comment(prediction(), analysis(), [], score_is_calibrated=False)
        # Not the number with a footnote: a percentage with a caveat still
        # anchors the reader on the percentage.
        assert "26%" not in out
        assert "outside the set the classifier was calibrated on" in out

    def test_the_review_itself_is_untouched(self):
        # Only the model's contribution is withheld. The LLM read of the issue
        # is what the maintainer opened the comment for, and it stays whole.
        out = format_llm_comment(
            prediction(),
            analysis(summary="A summary.", recommended_action="Do the thing."),
            [],
            score_is_calibrated=False,
        )
        for expected in ["A summary.", "Do the thing.", "Because.", "A risk."]:
            assert expected in out

    def test_no_disagreement_banner_without_a_trustworthy_score(self):
        # The banner says two signals conflict. With no score worth trusting
        # there is nothing to conflict with, and saying so would imply the
        # score carried weight after all.
        out = format_llm_comment(
            prediction(), analysis(), [], disagreement=True, score_is_calibrated=False
        )
        assert "disagree here" not in out
        assert "Needs maintainer review" in out.splitlines()[0]


class TestCalibratedComment:
    def test_the_score_and_verdict_are_shown(self):
        out = format_llm_comment(prediction(), analysis(), [], score_is_calibrated=True)
        assert "Likely not actionable" in out.splitlines()[0]
        assert "26%" in out
        assert "outside the set" not in out

    def test_an_actionable_verdict_still_renders(self):
        out = format_llm_comment(
            prediction(proba=0.81, label=1), analysis(), [], score_is_calibrated=True
        )
        assert "Likely actionable" in out.splitlines()[0]
        assert "81%" in out

    def test_the_disagreement_banner_still_works(self):
        out = format_llm_comment(
            prediction(), analysis(), [], disagreement=True, score_is_calibrated=True
        )
        assert "disagree here" in out
        assert "Needs maintainer review" in out.splitlines()[0]

    def test_calibrated_is_the_default_for_existing_callers(self):
        # Every other caller in the codebase and its tests predates this flag.
        out = format_llm_comment(prediction(), analysis(), [])
        assert "26%" in out
