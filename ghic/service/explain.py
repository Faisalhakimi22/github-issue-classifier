"""Human-readable explanations for the webhook comment.

The model scores ~5-8k engineered feature columns (sklearn ColumnTransformer
output: numeric__<name>, text__<ngram>, char__<ngram>). Those raw column
names mean nothing to an issue reporter. This module maps them to plain
language and composes a short paragraph.

Honesty constraint, carried over from evaluate.top_contributions()'s own
docstring: an unsigned explanation (magnitude only, e.g. Random Forest) must
never claim a feature pushed the prediction up or down -- only that it was
influential. A signed explanation (e.g. Logistic Regression) can say which
side each feature leaned toward, because that's the actual math.
"""
from __future__ import annotations

from typing import Any

_STRUCTURED_LABELS: dict[str, str] = {
    "title_len_chars": "the title's length",
    "title_len_tokens": "the number of words in the title",
    "body_len_chars": "the description's length",
    "body_len_tokens": "the number of words in the description",
    "has_code_block": "including a code block",
    "has_link": "including a link",
    "has_image": "including a screenshot or image",
    "repro_keyword_hits": "mentioning reproduction-related language",
    "has_repro_steps": "including reproduction steps",
    "author_account_age_days": "the reporter's account age",
    "author_public_repos": "the reporter's public repository count",
    "author_followers": "the reporter's follower count",
    "author_is_first_time_contributor": "the reporter being a first-time contributor",
    "created_hour_sin": "the time of day the issue was opened",
    "created_hour_cos": "the time of day the issue was opened",
    "created_dow_sin": "the day of the week the issue was opened",
    "created_dow_cos": "the day of the week the issue was opened",
    "opened_via_template": "using the repository's issue template",
    "days_since_last_release": "days since the repository's last release",
}


def humanize_feature(name: str) -> str | None:
    """Plain-language phrase for one raw feature column.

    None means "don't surface this one" -- an unmapped internal column, not
    a fabricated guess at what it means.
    """
    if "__" not in name:
        return None
    prefix, rest = name.split("__", 1)
    if prefix == "numeric":
        return _STRUCTURED_LABELS.get(rest)
    if prefix == "text":
        return f'the text mentioning "{rest}"'
    if prefix == "char":
        # Character n-grams are sub-word fragments (stack-trace shapes,
        # version strings) -- meaningful to the model, not to a reader.
        return "distinctive wording in the text"
    return None


def _join_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def explain_prediction(
    top_features: list[dict[str, Any]], signed: bool, max_items: int = 4,
) -> str | None:
    """Short natural-language paragraph from the model's top features.

    None if nothing explainable survived humanization -- never fabricates
    an explanation from data it doesn't have.
    """
    if not signed:
        seen: set[str] = set()
        phrases: list[str] = []
        for item in top_features:
            label = humanize_feature(item["feature"])
            if not label or label in seen:
                continue
            seen.add(label)
            phrases.append(label)
            if len(phrases) >= max_items:
                break
        if not phrases:
            return None
        return (
            f"The most influential factors in this prediction: {_join_list(phrases)}. "
            "This model reports which factors mattered, not which direction each "
            "one pushed the prediction."
        )

    supporting: list[str] = []
    against: list[str] = []
    for item in top_features:
        label = humanize_feature(item["feature"])
        if not label:
            continue
        bucket = supporting if item["value"] > 0 else against
        if label not in bucket:
            bucket.append(label)
    supporting, against = supporting[:max_items], against[:max_items]
    if not supporting and not against:
        return None
    parts = []
    if supporting:
        parts.append(f"Leaning toward actionable: {_join_list(supporting)}.")
    if against:
        parts.append(f"Leaning away from it: {_join_list(against)}.")
    return " ".join(parts)


def confidence_bar(proba: float, width: int = 10) -> str:
    """Unicode block bar -- no external image request, renders instantly."""
    proba = max(0.0, min(1.0, proba))
    filled = round(proba * width)
    return "█" * filled + "░" * (width - filled)
