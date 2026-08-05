"""Tests for ghic.service.explain -- the raw-feature-name humanizer and
natural-language explanation generator behind the webhook comment."""
from __future__ import annotations

from ghic.service.explain import confidence_bar, explain_prediction, humanize_feature


class TestHumanizeFeature:
    def test_known_structured_column(self):
        assert humanize_feature("numeric__has_code_block") == "including a code block"

    def test_unknown_structured_column_returns_none(self):
        assert humanize_feature("numeric__some_future_column") is None

    def test_text_ngram_quotes_the_term(self):
        assert humanize_feature("text__null pointer") == 'the text mentioning "null pointer"'

    def test_char_ngram_never_exposes_raw_fragment(self):
        result = humanize_feature("char__ash ")
        assert result is not None
        assert "ash" not in result  # the raw fragment itself must not leak

    def test_column_without_prefix_separator_returns_none(self):
        assert humanize_feature("malformed") is None


class TestExplainPrediction:
    def test_unsigned_never_claims_direction(self):
        text = explain_prediction(
            [{"feature": "numeric__has_code_block", "value": 0.4}], signed=False,
        )
        assert text is not None
        assert "toward" not in text
        assert "away" not in text
        assert "not which direction" in text

    def test_signed_splits_supporting_and_against(self):
        text = explain_prediction(
            [
                {"feature": "numeric__has_code_block", "value": 0.5},
                {"feature": "numeric__author_is_first_time_contributor", "value": -0.3},
            ],
            signed=True,
        )
        assert "Leaning toward actionable: including a code block." in text
        assert "Leaning away from it: the reporter being a first-time contributor." in text

    def test_no_explainable_features_returns_none(self):
        assert explain_prediction([], signed=False) is None
        assert explain_prediction(
            [{"feature": "unmapped_internal_col", "value": 0.1}], signed=False,
        ) is None

    def test_deduplicates_same_human_label(self):
        # created_hour_sin/cos both map to the same phrase -- must not repeat it.
        text = explain_prediction(
            [
                {"feature": "numeric__created_hour_sin", "value": 0.2},
                {"feature": "numeric__created_hour_cos", "value": 0.15},
            ],
            signed=False,
        )
        assert text.count("time of day") == 1


class TestConfidenceBar:
    def test_full_and_empty(self):
        assert confidence_bar(1.0, width=10) == "█" * 10
        assert confidence_bar(0.0, width=10) == "░" * 10

    def test_midpoint(self):
        bar = confidence_bar(0.5, width=10)
        assert bar.count("█") == 5
        assert bar.count("░") == 5

    def test_clamps_out_of_range_values(self):
        assert confidence_bar(1.5, width=4) == "████"
        assert confidence_bar(-0.2, width=4) == "░░░░"
