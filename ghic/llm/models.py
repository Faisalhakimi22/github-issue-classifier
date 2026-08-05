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
_REQUIRED_STRING_FIELDS = ("category", "priority", "severity", "summary", "reasoning")


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


@dataclass(frozen=True)
class IssueAnalysis:
    """The LLM's structured judgment. Priority/severity are the model's
    opinion, not a statistically validated prediction -- see
    models/LLM_ANALYSIS_CARD.md."""
    category: str
    priority: str
    severity: str
    confidence: float
    summary: str
    reasoning: str
    missing_information: list[str] = field(default_factory=list)
    recommended_labels: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "priority": self.priority,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "summary": self.summary,
            "reasoning": self.reasoning,
            "missing_information": self.missing_information,
            "recommended_labels": self.recommended_labels,
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

    priority = raw["priority"].strip().lower()
    if priority not in _VALID_LEVELS:
        raise LLMResponseError(
            f"field 'priority' must be one of {sorted(_VALID_LEVELS)}, got {raw['priority']!r}"
        )
    severity = raw["severity"].strip().lower()
    if severity not in _VALID_LEVELS:
        raise LLMResponseError(
            f"field 'severity' must be one of {sorted(_VALID_LEVELS)}, got {raw['severity']!r}"
        )

    confidence = raw.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise LLMResponseError("field 'confidence' must be a number")
    confidence = float(confidence)
    if not (0.0 <= confidence <= 1.0):
        raise LLMResponseError(f"field 'confidence' must be in [0, 1], got {confidence}")

    missing_information = _string_list(raw.get("missing_information", []), "missing_information")
    recommended_labels = _string_list(raw.get("recommended_labels", []), "recommended_labels")

    return IssueAnalysis(
        category=raw["category"].strip(),
        priority=priority,
        severity=severity,
        confidence=confidence,
        summary=raw["summary"].strip(),
        reasoning=raw["reasoning"].strip(),
        missing_information=missing_information,
        recommended_labels=recommended_labels,
    )


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise LLMResponseError(f"field {field_name!r} must be a list of strings")
    return [v.strip() for v in value if v.strip()]
