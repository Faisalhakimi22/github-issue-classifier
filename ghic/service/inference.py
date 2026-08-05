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


def format_llm_comment(
    pred: Prediction,
    analysis: Any,
    related: list[dict[str, Any]] | None = None,
    disagreement: bool = False,
) -> str:
    """Polished markdown comment built from an LLM IssueAnalysis on top of
    the ML prediction. `analysis` is a ghic.llm.IssueAnalysis -- typed as
    Any here to avoid a service/inference -> llm import at module load time
    for callers that never use this path.

    Never includes raw feature names, TF-IDF terms, feature-importance
    values, or any other ML-internal language -- those stay in logs only
    (see app.py). The existing format_comment() above is the fallback when
    the LLM analysis isn't available at all; it keeps its own collapsed
    technical-details section for maintainers who want it. This format
    never shows that section, by design -- a maintainer reading this
    should not be able to tell there's a model behind it.

    `disagreement=True` means ghic.llm.detect_disagreement() found the
    LLM's own priority/severity/reasoning strongly implying an actionable
    bug while the ML classifier called it not actionable. Rather than pick
    a side, the Actionability line says so plainly -- every other section
    (summary, reasoning, next step) still renders, since those stay useful
    regardless of which verdict a maintainer ends up trusting.
    """
    if disagreement:
        actionability = "⚠️ Model Disagreement"
    elif pred.predicted_label == 1:
        actionability = "Likely Actionable"
    else:
        actionability = "Likely Not Actionable"

    lines = [
        "## 🤖 GHIC Analysis",
        "",
        "| | |",
        "|---|---|",
        f"| **Classification** | {analysis.category} |",
        f"| **Actionability** | {actionability} |",
        f"| **ML Actionability Score** | {pred.proba:.0%} |",
        f"| **Priority** | {analysis.priority.title()} |",
        f"| **Severity** | {analysis.severity.title()} |",
    ]
    if disagreement:
        lines += [
            "",
            "_The AI's own reasoning below points more strongly toward an actionable "
            "bug than the ML score suggests. Shown as a disagreement rather than a "
            "single verdict — worth a maintainer's own look._",
        ]

    lines += [
        "",
        "---",
        "",
        "### Summary",
        "",
        analysis.summary,
        "",
        "---",
        "",
        "### Why GHIC Reached This Conclusion",
        "",
    ]
    lines += [f"- {point}" for point in analysis.reasoning]

    if analysis.missing_information:
        lines += ["", "---", "", "### Missing Information", ""]
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

    if related:
        lines += ["", "---", "", "**Possibly related prior issues** (by text similarity — please verify):"]
        lines += [
            f"- #{r['number']} — {r['title']} (similarity {r['similarity']:.2f})"
            for r in related
        ]

    lines += [
        "",
        "---",
        "",
        "> GHIC combines statistical machine learning with AI reasoning to assist "
        "maintainers. Final triage decisions always remain with project maintainers.",
    ]
    return "\n".join(lines)
