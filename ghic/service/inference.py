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
        feats = features.engineer_features(raw, self.cfg, repo_release_dates=release_dates)

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


def format_comment(pred: Prediction) -> str:
    """Markdown comment the bot posts on a scored issue."""
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
        f"| P(actionable bug) | **{pred.proba:.2f}** |",
        f"| Decision threshold | {pred.threshold:.2f} |",
        f"| Model | `{pred.model_name}` |",
    ]
    if pred.top_features:
        kind = (
            "signed contribution" if pred.signed_contributions else "importance (magnitude)"
        )
        lines += ["", f"<details><summary>Top model features ({kind})</summary>", ""]
        lines += [
            f"- `{item['feature']}`: {item['value']:+.3f}"
            if pred.signed_contributions
            else f"- `{item['feature']}`: {item['value']:.3f}"
            for item in pred.top_features
        ]
        lines += ["", "</details>"]
    lines += [
        "",
        "_Automated prediction from issue text and metadata at open time — "
        "it can be wrong. A maintainer's judgement always wins._",
    ]
    return "\n".join(lines)
