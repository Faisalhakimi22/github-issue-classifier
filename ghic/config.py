"""Typed configuration loader.

Reads config.yaml + .env once and exposes a frozen Config dataclass via
`get_config()`. Other modules import this rather than parsing YAML themselves,
so feature flags, label sets, and the GitHub token come from one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml
from dotenv import load_dotenv

from . import utils

logger = utils.get_logger(__name__)

CONFIG_PATH = utils.PROJECT_ROOT / "config.yaml"
ENV_PATH = utils.PROJECT_ROOT / ".env"


# ---------------------------------------------------------------------------
# Typed config sections
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CollectionConfig:
    api: str
    page_size: int
    rate_limit_floor: int
    backoff_max_attempts: int
    backoff_base_seconds: float
    date_window_months: int
    cache_dir: str
    max_runtime_minutes: int


@dataclass(frozen=True)
class RepoConfig:
    owner: str
    name: str
    created_after: str
    created_before: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass(frozen=True)
class LabelingConfig:
    bot_logins: frozenset[str]
    bug_labels: frozenset[str]
    non_actionable_labels: frozenset[str]
    question_labels: frozenset[str]
    drop_question_labeled_issues: bool
    drop_locked_issues: bool
    pr_fix_issue_regex_template: str

    def pr_fix_regex(self, issue_number: int) -> str:
        """Return the per-issue regex by substituting {issue_number}."""
        return self.pr_fix_issue_regex_template.format(issue_number=issue_number)


@dataclass(frozen=True)
class FeaturesConfig:
    # Granular access lives in features.py; this is a transparent passthrough so
    # we don't churn this dataclass every time a feature flag is added.
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})


@dataclass(frozen=True)
class Config:
    random_seed: int
    collection: CollectionConfig
    repos: tuple[RepoConfig, ...]
    labeling: LabelingConfig
    features: FeaturesConfig
    github_token: str


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _normalize_labels(items: list[str]) -> frozenset[str]:
    return frozenset(s.strip().lower() for s in items)


def _load_token() -> str:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError(
            "GH_TOKEN is not set. Add it to .env (see .env.example) or export "
            "it in your shell before running collection."
        )
    return token


def _load_yaml() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_CACHED: Config | None = None


def get_config(*, require_token: bool = True) -> Config:
    """Load and cache the project config.

    Pass require_token=False for unit tests that don't need a live API token
    (e.g. label.py tests that operate on fixture issues).
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    raw = _load_yaml()

    collection = CollectionConfig(**raw["collection"])
    repos = tuple(RepoConfig(**r) for r in raw["repos"])

    lab = raw["labeling"]
    labeling = LabelingConfig(
        bot_logins=_normalize_labels(lab["bot_logins"]),
        bug_labels=_normalize_labels(lab["bug_labels"]),
        non_actionable_labels=_normalize_labels(lab["non_actionable_labels"]),
        question_labels=_normalize_labels(lab["question_labels"]),
        drop_question_labeled_issues=bool(lab["drop_question_labeled_issues"]),
        drop_locked_issues=bool(lab["drop_locked_issues"]),
        pr_fix_issue_regex_template=lab["pr_fix_issue_regex_template"],
    )
    features = FeaturesConfig(raw=raw["features"])

    token = _load_token() if require_token else ""

    _CACHED = Config(
        random_seed=int(raw["project"]["random_seed"]),
        collection=collection,
        repos=repos,
        labeling=labeling,
        features=features,
        github_token=token,
    )
    logger.info(
        "Loaded config: %d repos, %d bot logins, %d bug labels, "
        "%d non-actionable labels, %d question labels (drop=%s)",
        len(_CACHED.repos),
        len(_CACHED.labeling.bot_logins),
        len(_CACHED.labeling.bug_labels),
        len(_CACHED.labeling.non_actionable_labels),
        len(_CACHED.labeling.question_labels),
        _CACHED.labeling.drop_question_labeled_issues,
    )
    return _CACHED


def reset_cache() -> None:
    """Drop the cached Config. Used in tests to force reload after monkeypatching."""
    global _CACHED
    _CACHED = None
