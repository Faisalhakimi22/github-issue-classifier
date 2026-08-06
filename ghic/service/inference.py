"""Single-issue inference: webhook payload -> feature row -> prediction.

Reuses the exact training-time feature code (ghic.features.engineer_features)
so there is no train/serve skew: the same regexes, the same cyclical encodings,
the same imputation (inside the fitted pipeline) that handled missing author
fields at training time handles them at serving time.

Known serving-time degradations, mirrored from the training docs:
  - author_is_first_time_contributor degrades to 1 for a single issue (no
    history in the frame); documented limitation.
  - days_since_last_release comes from the repo's latest release at event
    time; NaN (imputed) when the repo has no releases or enrichment is off.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import evaluate, features, utils
from ..config import Config, get_config

logger = utils.get_logger(__name__)


@dataclass(frozen=True)
class Prediction:
    repo: str
    issue_number: int
    proba: float
    threshold: float
    predicted_label: int                       # 1 = actionable bug
    model_name: str
    top_features: list[dict[str, Any]] = field(default_factory=list)
    signed_contributions: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "issue_number": self.issue_number,
            "proba_actionable_bug": round(self.proba, 4),
            "threshold": self.threshold,
            "predicted_label": self.predicted_label,
            "predicted_class": (
                "actionable-bug" if self.predicted_label == 1 else "non-actionable"
            ),
            "model": self.model_name,
            "top_features": self.top_features,
            "signed_contributions": self.signed_contributions,
        }


def build_feature_frame(
    cfg: Config,
    *,
    repo_full_name: str,
    issue_number: int,
    title: str,
    body: str,
    created_at: str,
    author_login: str = "",
    author_created_at: str | None = None,
    author_public_repos: int | None = None,
    author_followers: int | None = None,
    latest_release_iso: str | None = None,
) -> pd.DataFrame:
    """One issue -> the engineered single-row frame every model head consumes.

    Shared by the actionability and category predictors so an issue is
    feature-engineered exactly once per event.
    """
    raw = pd.DataFrame([{
        "repo_name": repo_full_name,
        "number": issue_number,
        "title": title or "",
        "body": body or "",
        "created_at": created_at,
        "author_login": author_login or "unknown",
        "author_created_at": author_created_at,
        "author_public_repos": (
            np.nan if author_public_repos is None else author_public_repos
        ),
        "author_followers": (
            np.nan if author_followers is None else author_followers
        ),
    }])
    # Always pass a release mapping so the days_since_last_release column
    # exists (the fitted ColumnTransformer requires it); an empty list
    # yields NaN, which the pipeline's median imputer absorbs.
    release_dates = {
        repo_full_name: [latest_release_iso] if latest_release_iso else []
    }
    return features.engineer_features(raw, cfg, repo_release_dates=release_dates)


class IssuePredictor:
    """Loads a fitted pipeline once and scores single issues from dicts."""

    def __init__(self, model_path: Path, threshold: float = 0.5,
                 cfg: Config | None = None) -> None:
        import joblib

        self.model_path = model_path
        self.model_name = model_path.stem
        self.threshold = threshold
        self.cfg = cfg or get_config(require_token=False)
        self._lock = threading.Lock()  # sklearn predict is thread-safe; joblib load is not
        logger.info("loading model %s", model_path)
        self.pipeline = joblib.load(model_path)

    def predict(
        self,
        *,
        repo_full_name: str,
        issue_number: int,
        title: str,
        body: str,
        created_at: str,
        author_login: str = "",
        author_created_at: str | None = None,
        author_public_repos: int | None = None,
        author_followers: int | None = None,
        latest_release_iso: str | None = None,
        explain: bool = True,
        threshold: float | None = None,
    ) -> Prediction:
        threshold = self.threshold if threshold is None else threshold
        feats = build_feature_frame(
            self.cfg,
            repo_full_name=repo_full_name,
            issue_number=issue_number,
            title=title,
            body=body,
            created_at=created_at,
            author_login=author_login,
            author_created_at=author_created_at,
            author_public_repos=author_public_repos,
            author_followers=author_followers,
            latest_release_iso=latest_release_iso,
        )

        with self._lock:
            proba = float(self.pipeline.predict_proba(feats)[:, 1][0])
            items: list[tuple[str, float]] = []
            signed = False
            if explain:
                items, signed = evaluate.top_contributions(self.pipeline, feats)

        return Prediction(
            repo=repo_full_name,
            issue_number=issue_number,
            proba=proba,
            threshold=threshold,
            predicted_label=int(proba >= threshold),
            model_name=self.model_name,
            top_features=[
                {"feature": f, "value": round(v, 4)} for f, v in items
            ],
            signed_contributions=signed,
        )


def format_comment(
    pred: Prediction,
    related: list[dict[str, Any]] | None = None,
    category: dict[str, Any] | None = None,
) -> str:
    """Markdown comment the bot posts on a scored issue."""
    from .explain import confidence_bar, explain_prediction

    verdict = (
        "likely an **actionable bug**"
        if pred.predicted_label == 1
        else "likely **non-actionable** (duplicate / question / won't-fix territory)"
    )
    lines = [
        "### 🤖 Issue triage prediction",
        "",
        f"This issue is {verdict}.",
        "",
        "| | |",
        "|---|---|",
        f"| Confidence | `{confidence_bar(pred.proba)}` **{pred.proba:.0%}** |",
        f"| Decision threshold | {pred.threshold:.0%} |",
        f"| Model | `{pred.model_name}` |",
    ]
    if category:
        lines.append(
            f"| Suggested category | **{category['predicted']}** "
            f"(confidence {category['confidence']:.0%}) |"
        )
    explanation = explain_prediction(pred.top_features, pred.signed_contributions)
    if explanation:
        lines += ["", explanation]
    if pred.top_features:
        kind = (
            "signed contribution" if pred.signed_contributions else "importance (magnitude)"
        )
        lines += ["", f"<details><summary>Technical details ({kind})</summary>", ""]
        lines += [
            f"- `{item['feature']}`: {item['value']:+.3f}"
            if pred.signed_contributions
            else f"- `{item['feature']}`: {item['value']:.3f}"
            for item in pred.top_features
        ]
        lines += ["", "</details>"]
    if related:
        lines += ["", "**Possibly related prior issues** (by text similarity — please verify):"]
        lines += [
            f"- #{r['number']} — {r['title']} (similarity {r['similarity']:.2f})"
            for r in related
        ]
    lines += [
        "",
        "_Automated prediction from issue text and metadata at open time — "
        "it can be wrong. A maintainer's judgement always wins._",
    ]
    return "\n".join(lines)


_MAX_EVIDENCE_FILES = 5
_MAX_EVIDENCE_HISTORY = 3


def _repository_evidence_lines(repository_context: Any) -> list[str]:
    """Render the Repository Evidence section from retrieved chunks.

    Built in Python from `RetrievedChunk` metadata -- deliberately never
    from LLM output. That is what makes "GHIC never hallucinates a file"
    a structural property instead of a request the model can ignore: every
    path and symbol printed here came out of the index, so it exists in the
    repository at the indexed commit by construction.

    Renders nothing at all when the engine is off or the repo isn't indexed
    (no section is better than an empty one), and renders the honest
    "nothing found" line when the engine ran but found no confident match.
    """
    if repository_context is None:
        return []
    if getattr(repository_context, "is_empty", True):
        if not getattr(repository_context, "indexed", False):
            return []  # engine off or repo not yet indexed -- say nothing
        from ..repository_intelligence.models import EMPTY_CONTEXT_NOTE

        return ["", "---", "", "### Repository Evidence", "", f"_{EMPTY_CONTEXT_NOTE}_"]

    lines = ["", "---", "", "### Repository Evidence", ""]

    code_chunks = getattr(repository_context, "code_chunks", list(repository_context.chunks))
    history_chunks = getattr(repository_context, "history_chunks", [])

    if code_chunks:
        by_file: dict[str, list[Any]] = {}
        for retrieved in code_chunks:
            by_file.setdefault(retrieved.chunk.path, []).append(retrieved.chunk)

        lines.append("**Relevant files**")
        lines.append("")
        for path, chunks in list(by_file.items())[:_MAX_EVIDENCE_FILES]:
            symbols = [c.qualified_symbol for c in chunks if c.qualified_symbol]
            spans = ", ".join(f"{c.start_line}-{c.end_line}" for c in chunks[:2])
            suffix = f" — `{'`, `'.join(symbols[:2])}`" if symbols else ""
            lines.append(f"- `{path}` (lines {spans}){suffix}")

    if history_chunks:
        # Rendered from retrieved chunk metadata, like the file list --
        # so a cited commit SHA or issue number is one that exists in the
        # index, not one the model produced.
        lines += ["", "**Related history**", ""]
        for retrieved in history_chunks[:_MAX_EVIDENCE_HISTORY]:
            chunk = retrieved.chunk
            label = {
                "commit": "commit", "pull_request": "PR", "issue": "issue",
            }.get(chunk.kind, "record")
            title = (chunk.symbol or "").strip()
            reference = f"[{chunk.reference}]({chunk.url})" if chunk.url else chunk.reference
            lines.append(
                f"- {label} {reference}" + (f" — {title}" if title else "")
            )
        lines += [
            "",
            "_Surfaced by similarity to this issue — possible leads, not "
            "confirmed matches._",
        ]

    metadata = getattr(repository_context, "metadata", None)
    if metadata is not None and (metadata.primary_language or metadata.frameworks):
        facts = []
        if metadata.primary_language:
            facts.append(metadata.primary_language)
        facts.extend(metadata.frameworks[:2])
        lines += ["", f"_Repository context: {' · '.join(facts)}._"]

    lines += [
        "",
        "_Files identified by semantic search over the repository at its last "
        "indexed commit — a starting point for investigation, not a diagnosis._",
    ]
    return lines


def format_llm_comment(
    pred: Prediction,
    analysis: Any,
    related: list[dict[str, Any]] | None = None,
    disagreement: bool = False,
    generated_at: datetime | None = None,
    repository_context: Any = None,
    engineering_analysis: Any = None,
    automation: Any = None,
) -> str:
    """Polished, first-party-feeling markdown comment built from an LLM
    IssueAnalysis on top of the ML prediction. `analysis` is a
    ghic.llm.IssueAnalysis -- typed as Any here to avoid a
    service/inference -> llm import at module load time for callers that
    never use this path.

    Answers four questions in order, so a maintainer can stop reading as
    soon as they have what they need: what is this (executive summary),
    how important is it (the fact table + Risk Assessment), why does GHIC
    think that (reasoning), what should happen next (recommended action).
    Never includes raw feature names, TF-IDF terms, feature-importance
    values, prompts, providers, token counts, or any other ML/engineering
    internals -- those stay in logs only (see app.py). format_comment()
    above is the fallback when no LLM analysis is available at all; it
    keeps its own collapsed technical-details section for maintainers who
    want it, which this format never shows.

    `disagreement=True` means ghic.llm.detect_disagreement() found the
    LLM's own priority/severity/reasoning strongly implying an actionable
    bug while the ML classifier called it not actionable. The Actionability
    line reads "Needs Maintainer Review" rather than naming the underlying
    mechanism -- "the models disagree" is an implementation detail, not
    something a maintainer needs framed that way. Every other section
    (summary, risk, reasoning, next step) still renders, since those stay
    useful regardless of which verdict a maintainer ends up trusting.

    Below models.LOW_CONFIDENCE_THRESHOLD, a plain-language disclaimer is
    added rather than trusting the model to remember to say so itself --
    deterministic where it needs to be, same reasoning as consistency.py.

    Tone/emphasis differences per issue category (bug vs. feature vs.
    question vs. ...) are steered entirely through the prompt (see
    prompts.py's SYSTEM_PROMPT), not branched on here -- the free-text
    fields (summary, reasoning, business_impact) are where that shows up,
    and hardcoding per-category templates here would fight the model's own
    read of the issue rather than express it.
    """
    from ..llm.models import LOW_CONFIDENCE_THRESHOLD

    if disagreement:
        actionability = "Needs Maintainer Review"
    elif pred.predicted_label == 1:
        actionability = "Likely Actionable"
    else:
        actionability = "Likely Not Actionable"

    lines = [
        "## 🤖 GHIC Analysis",
        "",
        analysis.summary,
        "",
        "---",
        "",
        "| | |",
        "|---|---|",
        f"| **Classification** | {analysis.category} |",
        f"| **Actionability** | {actionability} |",
        f"| **Statistical Risk Score** | {pred.proba:.0%} |",
        f"| **Priority** | {analysis.priority.title()} |",
        f"| **Severity** | {analysis.severity.title()} |",
        "",
        "_This score is based on historical issue patterns._",
    ]
    if disagreement:
        lines += [
            "",
            "_The statistical prediction and the AI review reached different "
            "conclusions on this one. Given the evidence below, GHIC recommends "
            "a maintainer take a direct look before this gets triaged on the "
            "score alone._",
        ]

    lines += [
        "",
        "---",
        "",
        "### Risk Assessment",
        "",
        f"**{analysis.risk_level.title()}**",
        "",
    ]
    lines += [f"- {reason}" for reason in analysis.risk_reasons]

    if analysis.business_impact:
        lines += ["", "---", "", "### Business Impact", "", "**Potential Impact**", ""]
        lines += [f"- {item}" for item in analysis.business_impact]

    lines += ["", "---", "", "### Why GHIC Reached This Conclusion", ""]
    lines += [f"- {point}" for point in analysis.reasoning]
    if analysis.confidence < LOW_CONFIDENCE_THRESHOLD:
        lines += [
            "",
            "_The available information is insufficient to reach a "
            "high-confidence assessment._",
        ]

    if analysis.missing_information:
        lines += [
            "", "---", "", "### Missing Information", "",
            "To speed up investigation, consider adding:", "",
        ]
        lines += [f"- {item}" for item in analysis.missing_information]

    if analysis.recommended_labels:
        lines += ["", "---", "", "### Suggested Labels", ""]
        lines += [" ".join(f"`{label}`" for label in analysis.recommended_labels)]

    lines += [
        "",
        "---",
        "",
        "### Recommended Next Step",
        "",
        analysis.recommended_action,
    ]

    lines += _repository_evidence_lines(repository_context)

    # Phase 3 engineering sections: root cause, regression, recurrence,
    # impact, timeline, investigation plan. Every factual reference in
    # them is rendered from an Evidence object built out of a retrieved
    # chunk -- the model never writes one. Renders nothing when the
    # analysis found nothing it could support.
    if engineering_analysis is not None:
        from ..engineering_intelligence import (
            render_attribution_note,
            render_engineering_sections,
        )

        engineering_lines = render_engineering_sections(engineering_analysis)
        if engineering_lines:
            lines += engineering_lines
            lines += render_attribution_note()

    # Phase 4 automation: advisory suggestions, collapsed behind <details>
    # so the analysis stays the thing a maintainer reads first.
    if automation is not None:
        from ..automation import render_automation_sections

        lines += render_automation_sections(automation)

    if related:
        lines += ["", "---", "", "**Possibly related prior issues** (by text similarity — please verify):"]
        lines += [
            f"- #{r['number']} — {r['title']} (similarity {r['similarity']:.2f})"
            for r in related
        ]

    timestamp = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    lines += [
        "",
        "---",
        "",
        "### Analysis Details",
        "",
        "| | |",
        "|---|---|",
        "| **Status** | AI Analysis Completed |",
        "| **Pipeline** | Machine Learning + AI Review |",
        f"| **Generated** | {timestamp} |",
        "",
        "> GHIC combines statistical machine learning with AI reasoning to assist "
        "maintainers. Final triage decisions always remain with project maintainers.",
    ]
    return "\n".join(lines)
