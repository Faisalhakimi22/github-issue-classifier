"""Feature engineering: collected/labeled issue rows -> model-ready matrix.

Two layers, deliberately separated so the webhook service can reuse them:

  1. engineer_features(df, cfg) -- deterministic, stateless row transforms that
     turn raw issue fields into numeric/boolean feature columns plus a single
     `text_combined` column for TF-IDF. No fitting, no vocabulary, no leakage
     from the test split. This is the part the webhook calls per-issue.

  2. build_preprocessor(cfg) -- an UNFITTED sklearn ColumnTransformer that
     TF-IDF-vectorizes `text_combined` and imputes+scales the structured
     columns. train.py fits it on the training split ONLY, then reuses it.

Leakage discipline:
  - `number_of_labels_at_open` is intentionally absent (production sees zero
    labels at inference; see config.yaml note).
  - TF-IDF vocabulary, imputer medians, and scaler stats are all fit on train
    only -- that is the ColumnTransformer's job, not this module's.
  - `author_is_first_time_contributor` is computed chronologically *within the
    provided dataframe*. At inference the webhook must supply prior-issue
    history; with a single issue it degrades to 1 (treat as first-time).

Honest limitations carried over from collection:
  - author_public_repos / author_followers are CURRENT API snapshots, not
    point-in-time at issue open (documented in config.yaml).
  - days_since_last_release is computed only when release dates are supplied
    (collect.py writes them to data/processed/releases.json, which train.py
    loads); otherwise the column is omitted and a warning is logged.
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import utils
from .config import Config, get_config

logger = utils.get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature column registries. train.py / evaluate.py import these instead of
# hardcoding column lists, so adding a feature here propagates automatically.
# ---------------------------------------------------------------------------
TEXT_COLUMN = "text_combined"

# Structured (non-text) numeric/boolean feature columns produced by
# engineer_features(). days_since_last_release is appended dynamically only
# when release data is available, so it is not listed statically here.
STRUCTURED_FEATURES: tuple[str, ...] = (
    # textual shape
    "title_len_chars",
    "title_len_tokens",
    "body_len_chars",
    "body_len_tokens",
    "has_code_block",
    "has_link",
    "has_image",
    "repro_keyword_hits",
    "has_repro_steps",
    # author
    "author_account_age_days",
    "author_public_repos",
    "author_followers",
    "author_is_first_time_contributor",
    # temporal (cyclical encodings of UTC open time)
    "created_hour_sin",
    "created_hour_cos",
    "created_dow_sin",
    "created_dow_cos",
    # structural
    "opened_via_template",
)

OPTIONAL_FEATURE_DAYS_SINCE_RELEASE = "days_since_last_release"

# Columns carried through engineer_features() unchanged for splitting/auditing.
PASSTHROUGH_COLUMNS: tuple[str, ...] = ("repo_name", "number", "created_at", "label")

_TOKEN_RE = re.compile(r"\w+")
_CODE_BLOCK_RE = re.compile(r"```|~~~|\n {4,}\S")          # fenced or indented code
_LINK_RE = re.compile(r"https?://", re.IGNORECASE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(|<img\b", re.IGNORECASE)  # md image or <img>
# Template heuristic: GitHub issue templates leave behind markdown section
# headers ("### Steps to reproduce") and/or HTML comment scaffolding.
_TEMPLATE_RE = re.compile(r"(?m)^#{1,4}\s+\S|<!--", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Scalar helpers (pure, unit-testable, webhook-safe)
# ---------------------------------------------------------------------------
def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def _parse_iso(ts: Any) -> datetime | None:
    if not ts or (isinstance(ts, float) and math.isnan(ts)):
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _repro_keyword_hits(text: str, keywords: Sequence[str]) -> int:
    low = (text or "").lower()
    return sum(1 for kw in keywords if kw.lower() in low)


def _cyclical(value: float, period: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * (value / period)
    return math.sin(angle), math.cos(angle)


# ---------------------------------------------------------------------------
# Layer 1: deterministic, stateless feature engineering
# ---------------------------------------------------------------------------
def engineer_features(
    df: pd.DataFrame,
    cfg: Config | None = None,
    repo_release_dates: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Return a new frame: passthrough columns + structured features + text.

    `df` must contain the columns produced by collect.normalize_issue() plus
    `label` from label.py. `repo_release_dates` optionally maps repo slug ->
    sorted ISO release timestamps, enabling days_since_last_release.
    """
    cfg = cfg or get_config(require_token=False)
    text_cfg = cfg.features.section("text")
    repro_keywords = text_cfg.get("repro_keywords", [])

    title = df["title"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    blob = (title + "\n\n" + body)

    out = pd.DataFrame(index=df.index)
    for col in PASSTHROUGH_COLUMNS:
        if col in df.columns:
            out[col] = df[col]

    # --- text shape ---
    out["title_len_chars"] = title.str.len()
    out["title_len_tokens"] = title.map(_token_count)
    out["body_len_chars"] = body.str.len()
    out["body_len_tokens"] = body.map(_token_count)
    out["has_code_block"] = body.str.contains(_CODE_BLOCK_RE).astype(int)
    out["has_link"] = blob.str.contains(_LINK_RE).astype(int)
    out["has_image"] = blob.str.contains(_IMAGE_RE).astype(int)
    out["repro_keyword_hits"] = blob.map(lambda t: _repro_keyword_hits(t, repro_keywords))
    out["has_repro_steps"] = (out["repro_keyword_hits"] > 0).astype(int)

    # --- author ---
    created = df["created_at"].map(_parse_iso)
    author_created = df["author_created_at"].map(_parse_iso) if "author_created_at" in df else pd.Series([None] * len(df), index=df.index)
    out["author_account_age_days"] = [
        (c - a).days if (c is not None and a is not None) else np.nan
        for c, a in zip(created, author_created)
    ]
    out["author_public_repos"] = pd.to_numeric(df.get("author_public_repos"), errors="coerce")
    out["author_followers"] = pd.to_numeric(df.get("author_followers"), errors="coerce")
    out["author_is_first_time_contributor"] = _first_time_contributor_flags(df, created)

    # --- temporal (cyclical, UTC) ---
    hours = pd.Series([c.hour if c else 0 for c in created], index=df.index)
    dows = pd.Series([c.weekday() if c else 0 for c in created], index=df.index)
    out["created_hour_sin"], out["created_hour_cos"] = zip(*hours.map(lambda h: _cyclical(h, 24)))
    out["created_dow_sin"], out["created_dow_cos"] = zip(*dows.map(lambda d: _cyclical(d, 7)))

    # --- structural ---
    out["opened_via_template"] = body.str.contains(_TEMPLATE_RE).astype(int)

    # --- optional: days since last release ---
    if repo_release_dates:
        out[OPTIONAL_FEATURE_DAYS_SINCE_RELEASE] = _days_since_last_release(
            df.get("repo_name"), created, repo_release_dates
        )
    elif cfg.features.section("temporal").get("days_since_last_release"):
        logger.warning(
            "days_since_last_release is enabled in config but no release dates "
            "were supplied -- omitting the feature. Run `python -m ghic.collect` "
            "to produce data/processed/releases.json, or pass repo_release_dates=."
        )

    # --- text corpus for TF-IDF ---
    out[TEXT_COLUMN] = blob
    return out


def _first_time_contributor_flags(df: pd.DataFrame, created: "pd.Series[Any]") -> list[int]:
    """1 if this row is the author's earliest issue in its repo within `df`.

    Computed chronologically so it is non-leaky: it reflects only information
    knowable at open time given the dataset's own history.
    """
    if "author_login" not in df or "repo_name" not in df:
        return [1] * len(df)
    order = pd.DataFrame({
        "repo": df["repo_name"].values,
        "author": df["author_login"].values,
        "ts": [c.timestamp() if c else float("inf") for c in created],
        "pos": range(len(df)),
    })
    earliest = (
        order.sort_values("ts")
        .groupby(["repo", "author"], dropna=False)["pos"]
        .first()
    )
    first_positions = set(earliest.values)
    return [1 if i in first_positions else 0 for i in range(len(df))]


def _days_since_last_release(
    repo_names: "pd.Series[Any] | None",
    created: "pd.Series[Any]",
    repo_release_dates: Mapping[str, Sequence[str]],
) -> list[float]:
    parsed = {
        repo: sorted(d for d in (_parse_iso(r) for r in dates) if d is not None)
        for repo, dates in repo_release_dates.items()
    }
    repos = repo_names if repo_names is not None else pd.Series([None] * len(created))
    result: list[float] = []
    for repo, c in zip(repos, created):
        rels = parsed.get(repo, [])
        if c is None or not rels:
            result.append(np.nan)
            continue
        prior = [r for r in rels if r <= c]
        result.append((c - prior[-1]).days if prior else np.nan)
    return result


# ---------------------------------------------------------------------------
# Layer 2: the fittable sklearn preprocessor
# ---------------------------------------------------------------------------
def structured_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Structured columns actually present in `frame` (handles optional ones)."""
    cols = [c for c in STRUCTURED_FEATURES if c in frame.columns]
    if OPTIONAL_FEATURE_DAYS_SINCE_RELEASE in frame.columns:
        cols.append(OPTIONAL_FEATURE_DAYS_SINCE_RELEASE)
    return cols


def build_preprocessor(cfg: Config, structured_cols: Sequence[str]) -> ColumnTransformer:
    """Unfitted ColumnTransformer: TF-IDF on text + impute/scale on numerics.

    StandardScaler(with_mean=False) keeps the combined matrix sparse so it
    coexists with the TF-IDF block without densifying (important at 5k+ dims).
    """
    text_cfg = cfg.features.section("text")
    ngram = tuple(text_cfg.get("tfidf_ngram_range", [1, 2]))
    max_features = int(text_cfg.get("tfidf_max_features", 5000))

    text_pipe = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram,  # type: ignore[arg-type]
        min_df=5,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
    )
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])
    transformers = [
        ("text", text_pipe, TEXT_COLUMN),
        ("numeric", numeric_pipe, list(structured_cols)),
    ]
    # Character n-grams catch what word tokens miss in GitHub text: stack-trace
    # shapes, version strings, path fragments, camelCase identifiers. char_wb
    # stays within word boundaries so the vocabulary doesn't explode.
    if text_cfg.get("tfidf_char_ngrams", False):
        char_pipe = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=int(text_cfg.get("tfidf_char_max_features", 3000)),
            min_df=5,
            sublinear_tf=True,
            lowercase=True,
        )
        transformers.insert(1, ("char", char_pipe, TEXT_COLUMN))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )


def feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    """Output feature names from a FITTED preprocessor (for importance plots)."""
    return preprocessor.get_feature_names_out()
