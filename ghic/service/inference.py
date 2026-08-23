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
    repository_context: Any = None,
) -> str:
    """Markdown comment the bot posts on a scored issue."""
    from .explain import confidence_bar, explain_prediction

    verdict = (
        "likely an **actionable bug**"
        if pred.predicted_label == 1
        else "likely **non-actionable** (duplicate / question / won't-fix territory)"
    )
    lines = [
        "### GHIC · Issue triage prediction",
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
    lines += _repository_evidence_lines(repository_context)
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

    Renders nothing when the engine is off. When the engine was enabled but
    could not access an index, it says that plainly without exposing internal
    infrastructure details. A successful search with no confident match gets
    the separate honest "nothing found" line.
    """
    if repository_context is None:
        return []
    if getattr(repository_context, "is_empty", True):
        from ..repository_intelligence.models import (
            EMPTY_CONTEXT_NOTE,
            UNAVAILABLE_CONTEXT_NOTE,
        )

        default_note = (
            EMPTY_CONTEXT_NOTE
            if getattr(repository_context, "indexed", False)
            else UNAVAILABLE_CONTEXT_NOTE
        )
        note = getattr(repository_context, "note", "") or default_note
        return ["", "---", "", "### Repository Evidence", "", f"_{note}_"]

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
    score_is_calibrated: bool = True,
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

    # An uncalibrated repository gets the same treatment as a disagreement:
    # the maintainer is told a decision is theirs to make, rather than handed
    # a verdict the model is not entitled to. The classifier learned from a
    # fixed set of large repositories, and on anything else its output is a
    # confident-looking number about a population it never saw. Leading with
    # "Likely not actionable" there is not a hedge, it is a claim -- and one
    # wrong call on a real bug teaches maintainers to discount the whole
    # comment, including the parts that are sound.
    if disagreement or not score_is_calibrated:
        verdict = "Needs maintainer review"
    elif pred.predicted_label == 1:
        verdict = "Likely actionable"
    else:
        verdict = "Likely not actionable"

    # The verdict, category and priority go in the heading, so the whole
    # triage decision is legible from GitHub's notification list and the
    # collapsed-comment preview without opening anything.
    headline = f"## GHIC · {verdict} · {analysis.category} · {analysis.priority.title()} priority"

    lines = [headline, "", analysis.summary, ""]

    if disagreement and score_is_calibrated:
        lines += [
            "> ⚠️ The statistical score and the AI review disagree here. Worth a "
            "direct look before triaging on the score alone.",
            "",
        ]

    # The single most useful line in the comment, so it sits directly under
    # the summary in bold rather than eight sections down under its own
    # heading.
    lines += [f"**→ {analysis.recommended_action}**", ""]

    # Reasoning as a tight list. No heading: three bullets under a summary
    # read as "why" without being told, and "Why GHIC Reached This
    # Conclusion" was longer than some of the bullets under it.
    lines += [f"- {point}" for point in analysis.reasoning]
    if analysis.confidence < LOW_CONFIDENCE_THRESHOLD:
        lines.append(
            "- _Thin evidence — not enough here for a high-confidence read._"
        )
    lines.append("")

    if analysis.missing_information:
        needs = "; ".join(analysis.missing_information)
        lines += [f"**Would help:** {needs}", ""]

    if analysis.recommended_labels:
        labels = " ".join(f"`{label}`" for label in analysis.recommended_labels)
        lines += [f"**Labels:** {labels}", ""]

    # Everything below is supporting detail: true, occasionally useful, and
    # not what the maintainer opened the issue to find out. Collapsed so
    # the comment is ~15 lines on arrival instead of 74.
    detail: list[str] = [
        "| | |",
        "|---|---|",
        (
            f"| Statistical risk score | {pred.proba:.0%} (from historical issue patterns) |"
            if score_is_calibrated
            else "| Statistical risk score | Not shown — this repository is outside "
            "the set the classifier was calibrated on, so the score would not "
            "mean what it appears to mean. The review above is unaffected. |"
        ),
        f"| Severity | {analysis.severity.title()} |",
        f"| Overall risk | {analysis.risk_level.title()} |",
        "",
        "**Risk factors**",
        "",
    ]
    detail += [f"- {reason}" for reason in analysis.risk_reasons]
    if analysis.business_impact:
        detail += ["", "**Potential impact**", ""]
        detail += [f"- {item}" for item in analysis.business_impact]

    lines += ["<details><summary>Scoring detail</summary>", ""]
    lines += detail
    lines += ["", "</details>"]

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

    # One <sub> line instead of an "Analysis Details" table. The old table
    # carried three rows: "Status: AI Analysis Completed" (self-evident --
    # you are reading the result), "Pipeline: Machine Learning + AI Review"
    # (internal architecture, which this comment is meant not to expose),
    # and the timestamp, which is the only part worth keeping.
    timestamp = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    lines += [
        "",
        f"<sub>GHIC · assistive triage, a maintainer decides · {timestamp}</sub>",
    ]
    return "\n".join(lines)
