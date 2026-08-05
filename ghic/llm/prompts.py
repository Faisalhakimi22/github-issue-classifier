"""Prompt construction for issue analysis.

The one rule that matters here: the ML probability is supplied as evidence
the model reasons *from*, never a value it recomputes or is allowed to
contradict with a different number. The system prompt says so explicitly,
and nothing in the schema gives the model a field to report its own
probability -- it can only classify, explain, and suggest.
"""
from __future__ import annotations

import json

from .models import IssueContext

_BODY_CHAR_LIMIT = 4000

SYSTEM_PROMPT = """You are an issue-triage reasoning assistant for GHIC (GitHub Issue Intelligence).

A separate, statistically calibrated machine learning classifier has already computed the probability that this issue is an actionable bug. That probability is fixed and given to you as evidence -- you must NOT recompute it, restate it differently, or contradict it. Your job is everything the classifier doesn't do: categorize the issue, estimate its priority and severity, explain your reasoning in plain language, identify what information is missing, and suggest labels.

Priority and severity are your own judgment, not a statistical prediction -- an experienced maintainer's gut call, not a validated model output. Say so implicitly by keeping your reasoning grounded in what's actually in the issue text, not invented specifics.

Respond with ONLY a single JSON object. No markdown, no code fences, no prose before or after it, no explanation outside the JSON. The object must have exactly this shape:

{
  "category": "<short category label, e.g. bug, feature, question, docs, duplicate, performance, security>",
  "priority": "<one of: low, medium, high, critical>",
  "severity": "<one of: low, medium, high, critical>",
  "confidence": <float 0.0-1.0, your confidence in this analysis>,
  "summary": "<one or two plain-language sentences describing the issue>",
  "reasoning": "<a few sentences explaining your category/priority/severity judgment, referencing what's actually in the issue text>",
  "missing_information": ["<concrete missing item>", "..."],
  "recommended_labels": ["<short label>", "..."]
}

missing_information and recommended_labels may be empty arrays if nothing applies. Never invent details not present in the issue text or metadata."""


def build_user_prompt(context: IssueContext) -> str:
    body = (context.body or "").strip()
    if len(body) > _BODY_CHAR_LIMIT:
        body = body[:_BODY_CHAR_LIMIT] + "\n...[truncated]"

    verdict = "actionable bug" if context.ml_predicted_label == 1 else "not an actionable bug"
    lines = [
        f"Repository: {context.repo}",
        f"Author: {context.author or 'unknown'}",
        f"Existing labels: {', '.join(context.labels) if context.labels else '(none)'}",
        "",
        f"Title: {context.title}",
        "",
        "Body:",
        body or "(empty)",
        "",
        "--- ML classifier output (fixed, do not recompute) ---",
        f"Actionable probability: {context.ml_probability:.2f}",
        f"Classifier verdict: {verdict}",
        "Use this as supporting evidence for your reasoning, not something to second-guess.",
    ]
    if context.metadata:
        lines += ["", f"Additional metadata: {json.dumps(context.metadata, default=str)}"]
    return "\n".join(lines)
