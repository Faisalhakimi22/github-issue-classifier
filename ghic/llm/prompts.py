"""Prompt construction for issue analysis.

The one rule that matters here: the ML probability is supplied as evidence
the model reasons *from*, never a value it recomputes or is allowed to
contradict with a different number. The system prompt says so explicitly,
and nothing in the schema gives the model a field to report its own
probability -- it can only classify, explain, and suggest.

`reasoning` is requested as short factual bullet points, not prose -- the
comment renders it directly as a list, so asking the model to already
produce it list-shaped is far more reliable than splitting a paragraph
after the fact.
"""
from __future__ import annotations

import json
from typing import Any

from .models import IssueContext

_BODY_CHAR_LIMIT = 4000

SYSTEM_PROMPT = """You are an issue-triage reasoning assistant for GHIC (GitHub Issue Intelligence), writing analysis that maintainers read directly on GitHub. The output should read like a first-party GitHub feature -- comparable to GitHub Copilot, Linear, or Microsoft's own product writing -- not an AI experiment. A maintainer should be able to understand the issue in under 10 seconds from your `summary` alone.

A separate, statistically calibrated machine learning classifier has already computed the probability that this issue is an actionable bug. That probability is fixed and given to you as evidence -- you must NOT recompute it, restate it differently, or contradict it. Your job is everything the classifier doesn't do: categorize the issue, judge its priority/severity/overall risk, explain your reasoning in plain language, estimate business impact, recommend one next step, identify what information is missing, and suggest labels.

Priority, severity, and risk_level are your own judgment, not a statistical prediction -- an experienced maintainer's gut call, not a validated model output. Keep every judgment grounded in what's actually in the issue text, never an invented specific. If the issue text is too thin to support a confident read, say so plainly rather than filling the gap with a guess -- a low `confidence` score is expected and fine when the evidence is thin.

Voice: write like a sharp, calm colleague, not a system. Say "the issue appears to..." or "the description suggests...", never "the classifier indicates..." or "the model predicts...". Never mention features, embeddings, probabilities-as-a-concept, or any other ML/engineering internals -- a maintainer reading this should never be able to tell there's a model behind it at all, only that the analysis is well-reasoned. Be concise: no filler, no hedging for its own sake, no restating the question back at the reader, never robotic.

Match tone and emphasis to the issue's category:
- bug: emphasize severity, reproducibility, and concrete impact.
- feature: emphasize the use case and value being requested, not urgency.
- question: recommend documentation or the support channel, not code investigation.
- duplicate: keep it brief -- GHIC's related-issues feature handles cross-referencing separately, don't invent a specific issue number yourself.
- security: be direct about urgency; do not speculate about exploitability beyond what's stated.
- performance: describe the performance impact concretely (e.g. what got slower, under what condition), not just "this is a performance issue."
- docs: focus on what documentation is missing or wrong, not code severity.
Regressions (something that used to work and now doesn't) deserve an explicit mention that it's a regression, in whichever category they otherwise fall under.

Respond with ONLY a single JSON object. No markdown, no code fences, no prose before or after it, no explanation outside the JSON. The object must have exactly this shape:

{
  "category": "<short category label, e.g. bug, feature, question, docs, duplicate, performance, security>",
  "priority": "<one of: low, medium, high, critical>",
  "severity": "<one of: low, medium, high, critical>",
  "confidence": <float 0.0-1.0, your confidence in this analysis>,
  "summary": "<what this issue is and how important it is, in AT MOST 2 sentences, no jargon -- this is the executive summary a maintainer reads first>",
  "reasoning": ["<one short factual observation>", "<another>", "2 to 5 items total"],
  "recommended_action": "<exactly one sentence: the single next step a maintainer should take>",
  "risk_level": "<one of: low, medium, high, critical -- your overall risk read, may differ from priority/severity if they pull in different directions>",
  "risk_reasons": ["<one short concrete reason for the risk_level>", "...", "1 to 5 items, only the reasons that actually apply"],
  "business_impact": ["<one short, concretely-evidenced operational/business consequence>", "...", "empty array if none is reasonably inferable from the issue text -- never invent one just to fill the section"],
  "missing_information": ["<concrete missing item>", "...", "at most 5"],
  "recommended_labels": ["<short label>", "...", "at most 5, ordered most-important-first"]
}

Each `reasoning` item is one short, concrete, evidence-grounded observation (e.g. "Includes a full stack trace and reproduction steps.") -- never a restatement of the priority/severity/category you already gave, never a reference to how you arrived at it. `recommended_action` names one concrete action ("Investigate immediately -- this affects a core workflow.", "Route to the support team -- this reads as a usage question.", "Ask the reporter for a stack trace before triaging further."), not a generic "review this issue". `risk_reasons` justify `risk_level` specifically (e.g. "Blocks a production workflow", "Reproducible", "Regression after a recent update") -- only include ones that genuinely apply to this issue. `business_impact` must be grounded in what the issue actually says (e.g. "Data import unavailable" when the issue is literally about a broken import) -- an empty array is the correct answer far more often than not; never pad it. `missing_information` and `recommended_labels` may also be empty arrays when nothing applies -- do not pad them. Never invent details not present in the issue text or metadata."""


_MAX_PROMPT_CHUNKS = 6
_MAX_HISTORY_CHUNKS = 3
_MAX_CHUNK_CHARS = 1_200


def build_repository_section(repo_context: Any) -> str:
    """Render retrieved repository evidence for the prompt, or "" for none.

    This is the *only* path by which repository facts reach the model. It
    contains nothing but chunks the retriever actually returned, and it
    states its own limits explicitly ("this is a partial view", "do not
    assume anything not shown"). The model cannot describe a file it was
    never shown, because it was never shown a file list -- only these
    chunks.
    """
    if repo_context is None or getattr(repo_context, "is_empty", True):
        return ""

    lines = ["", "--- Repository code retrieved for this issue ---"]

    metadata = getattr(repo_context, "metadata", None)
    if metadata is not None:
        facts = [
            f"Primary language: {metadata.primary_language}" if metadata.primary_language else "",
            f"Frameworks: {', '.join(metadata.frameworks)}" if metadata.frameworks else "",
            f"Project type: {metadata.project_type}" if metadata.project_type else "",
            f"Entry points: {', '.join(metadata.entry_points)}" if metadata.entry_points else "",
        ]
        lines += [f for f in facts if f]
        if metadata.readme_summary:
            lines.append(f"README summary: {metadata.readme_summary}")

    code_chunks = getattr(repo_context, "code_chunks", list(repo_context.chunks))
    history_chunks = getattr(repo_context, "history_chunks", [])

    if code_chunks:
        lines.append("")
        lines.append(
            "The following code was retrieved by semantic search against this "
            "repository. It is a partial view -- the most relevant fragments, not "
            "the whole codebase."
        )
        for retrieved in list(code_chunks)[:_MAX_PROMPT_CHUNKS]:
            chunk = retrieved.chunk
            text = chunk.text
            if len(text) > _MAX_CHUNK_CHARS:
                text = text[:_MAX_CHUNK_CHARS] + "\n... [truncated]"
            header = f"{chunk.path}:{chunk.start_line}-{chunk.end_line}"
            if chunk.qualified_symbol:
                header += f" ({chunk.kind} {chunk.qualified_symbol})"
            lines += ["", header, "```" + _fence_language(chunk.language), text, "```"]

    if history_chunks:
        # Framed as history, not as code, and explicitly as *possibly*
        # related: a similar past issue is a lead, and presenting it as an
        # established fact ("this was fixed in #412") is the kind of
        # confident wrongness this whole design avoids.
        lines += [
            "",
            "Related project history, retrieved by similarity. These are "
            "candidates a maintainer may find relevant -- not confirmed "
            "matches, and not necessarily the same root cause. Refer to them "
            "only as possibilities, never as established fact, and never cite "
            "a commit, pull request, or issue that does not appear below.",
        ]
        for retrieved in list(history_chunks)[:_MAX_HISTORY_CHUNKS]:
            chunk = retrieved.chunk
            label = {
                "commit": "Commit", "pull_request": "Pull request", "issue": "Issue",
            }.get(chunk.kind, "Record")
            text = chunk.text
            if len(text) > _MAX_CHUNK_CHARS:
                text = text[:_MAX_CHUNK_CHARS] + "\n... [truncated]"
            lines += ["", f"{label} {chunk.reference}", text]

    lines += [
        "",
        "Ground any statement you make about this repository in the code above. "
        "Refer to files and functions by their exact names as shown. Do NOT "
        "mention or infer the existence of any file, function, class, or module "
        "that does not appear above -- if the retrieved code doesn't answer "
        "something, say the available code doesn't show it rather than "
        "guessing. Absence of a file here does not mean it doesn't exist in the "
        "repository, only that it wasn't retrieved; never claim something is "
        "missing from the codebase on this basis.",
    ]
    return "\n".join(lines)


def _fence_language(language: str) -> str:
    return {
        "Python": "python", "JavaScript": "javascript", "TypeScript": "typescript",
        "Go": "go", "Rust": "rust", "Java": "java", "C#": "csharp",
        "Markdown": "markdown", "reStructuredText": "rst",
    }.get(language, "")


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

    repository_section = build_repository_section(context.repository_context)
    if repository_section:
        lines.append(repository_section)
    else:
        # Said explicitly rather than left silent: a model given no code
        # section and no instruction about it will happily speculate about
        # the codebase from the issue text alone.
        lines += [
            "",
            "--- Repository code ---",
            "No repository code was retrieved for this issue. Reason about the "
            "issue text alone, and do not speculate about specific files, "
            "functions, classes, or the project's internal structure.",
        ]
    return "\n".join(lines)
