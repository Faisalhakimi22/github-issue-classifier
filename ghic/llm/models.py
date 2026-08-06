"""Domain types for the LLM layer.

Plain frozen dataclasses + a manual validator, matching the rest of the
service package (see service/inference.py's `Prediction`) rather than
introducing a new modeling pattern (e.g. pydantic) for one feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .exceptions import LLMResponseError

_VALID_LEVELS = frozenset({"low", "medium", "high", "critical"})
_REQUIRED_STRING_FIELDS = ("category", "summary", "recommended_action")
_LEVEL_FIELDS = ("priority", "severity", "risk_level")
_MAX_LABELS = 5
_MAX_REASONING_POINTS = 6
_MAX_RISK_REASONS = 5
_MAX_BUSINESS_IMPACT = 5
_MAX_MISSING_INFORMATION = 5

# Below this, the comment adds an explicit low-confidence disclaimer rather
# than presenting the analysis as settled -- see format_llm_comment().
LOW_CONFIDENCE_THRESHOLD = 0.4


@dataclass(frozen=True)
class IssueContext:
    """Everything the LLM needs to reason about one issue, including the
    ML head's own verdict -- passed as supporting evidence, never
    recomputed (see prompts.py)."""
    repo: str
    title: str
    body: str
    labels: list[str]
    author: str
    metadata: dict[str, Any]
    ml_probability: float
    ml_predicted_label: int
    # Retrieved code evidence from the Repository Intelligence Engine, or
    # None when the repo isn't indexed / retrieval found nothing confident.
    # Typed loosely to keep ghic.llm independent of
    # ghic.repository_intelligence -- the LLM layer works identically with
    # or without that subsystem installed. See prompts.build_user_prompt().
    repository_context: Any = None


@dataclass(frozen=True)
class IssueAnalysis:
    """The LLM's structured judgment. Priority/severity/risk are the
    model's opinion, not a statistically validated prediction -- see
    models/LLM_ANALYSIS_CARD.md.

    `summary` doubles as the comment's lead executive-summary paragraph --
    kept to 1-2 plain-language sentences by the prompt, not a separate
    field, so there's exactly one place asking "what is this issue and how
    important is it" instead of two that can drift apart.

    `reasoning` is short factual bullet points ("Clear reproduction steps
    were provided."), not a paragraph -- it's rendered directly as a list
    in the comment, so it has to already be list-shaped and free of model-
    internal language when it arrives. `risk_reasons` and `business_impact`
    are the same shape for the same reason.

    `confidence` is kept for API consumers (the dashboard, /api/predict-
    style callers) but is deliberately not rendered next to the ML score in
    the comment -- two competing confidence numbers on one message is
    exactly the ambiguity the comment redesign removes. Below
    LOW_CONFIDENCE_THRESHOLD it still surfaces, as a plain-language
    disclaimer rather than a number (see format_llm_comment()).
    """
    category: str
    priority: str
    severity: str
    confidence: float
    summary: str
    reasoning: list[str]
    recommended_action: str
    risk_level: str
    risk_reasons: list[str]
    missing_information: list[str] = field(default_factory=list)
    recommended_labels: list[str] = field(default_factory=list)
    business_impact: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "priority": self.priority,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "summary": self.summary,
            "reasoning": self.reasoning,
            "recommended_action": self.recommended_action,
            "risk_level": self.risk_level,
            "risk_reasons": self.risk_reasons,
            "missing_information": self.missing_information,
            "recommended_labels": self.recommended_labels,
            "business_impact": self.business_impact,
        }


def parse_issue_analysis(raw: Any) -> IssueAnalysis:
    """Validate a provider's parsed-JSON response and build an IssueAnalysis.

    Raises LLMResponseError with a specific, logged-safe reason for every
    way the response can be wrong -- never silently coerces a malformed
    response into something that looks valid.
    """
    if not isinstance(raw, dict):
        raise LLMResponseError(f"expected a JSON object, got {type(raw).__name__}")

    for key in _REQUIRED_STRING_FIELDS:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise LLMResponseError(f"field {key!r} must be a non-empty string")

    priority, severity, risk_level = (_level(raw, name) for name in _LEVEL_FIELDS)

    confidence = raw.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise LLMResponseError("field 'confidence' must be a number")
    confidence = float(confidence)
    if not (0.0 <= confidence <= 1.0):
        raise LLMResponseError(f"field 'confidence' must be in [0, 1], got {confidence}")

    reasoning = _string_list(raw.get("reasoning"), "reasoning")
    if not reasoning:
        raise LLMResponseError("field 'reasoning' must have at least one item")
    reasoning = reasoning[:_MAX_REASONING_POINTS]

    risk_reasons = _string_list(raw.get("risk_reasons"), "risk_reasons")
    if not risk_reasons:
        raise LLMResponseError("field 'risk_reasons' must have at least one item")
    risk_reasons = risk_reasons[:_MAX_RISK_REASONS]

    business_impact = _string_list(raw.get("business_impact", []), "business_impact")
    business_impact = business_impact[:_MAX_BUSINESS_IMPACT]

    missing_information = _string_list(raw.get("missing_information", []), "missing_information")
    missing_information = missing_information[:_MAX_MISSING_INFORMATION]

    recommended_labels = _string_list(raw.get("recommended_labels", []), "recommended_labels")
    recommended_labels = recommended_labels[:_MAX_LABELS]

    return IssueAnalysis(
        category=raw["category"].strip(),
        priority=priority,
        severity=severity,
        confidence=confidence,
        summary=raw["summary"].strip(),
        reasoning=reasoning,
        recommended_action=raw["recommended_action"].strip(),
        risk_level=risk_level,
        risk_reasons=risk_reasons,
        missing_information=missing_information,
        recommended_labels=recommended_labels,
        business_impact=business_impact,
    )


def _level(raw: dict[str, Any], field_name: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise LLMResponseError(f"field {field_name!r} must be a non-empty string")
    level = value.strip().lower()
    if level not in _VALID_LEVELS:
        raise LLMResponseError(
            f"field {field_name!r} must be one of {sorted(_VALID_LEVELS)}, got {value!r}"
        )
    return level


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise LLMResponseError(f"field {field_name!r} must be a list of strings")
    return [v.strip() for v in value if v.strip()]
