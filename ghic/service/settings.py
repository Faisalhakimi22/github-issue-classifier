"""Service configuration from environment variables (12-factor style).

Every knob is a GHIC_* env var so the same image runs unchanged in dev,
staging, and production. Secrets (webhook secret, private key) are never read
from config.yaml — that file only holds ML/labeling configuration.

Safety defaults: DRY_RUN=true and no write action enabled, so a fresh deploy
can never spam repositories until the operator explicitly opts in.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .. import utils

logger = utils.get_logger(__name__)

_TRUE = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in _TRUE


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def parse_repo_thresholds(raw: str) -> dict[str, float]:
    """Parse GHIC_REPO_THRESHOLDS: "owner/repo=0.35,other/repo=0.6"."""
    out: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        repo, _, value = part.partition("=")
        if not repo or not value:
            raise ValueError(
                f"Malformed GHIC_REPO_THRESHOLDS entry {part!r}; "
                "expected owner/repo=0.35"
            )
        out[repo.strip()] = float(value)
    return out


@dataclass(frozen=True)
class ServiceSettings:
    # Model
    model_path: Path
    threshold: float = 0.5
    # Per-repo overrides — the research showed one global cutoff is wrong for
    # repos with different score distributions. Calibrate with ghic.backtest.
    repo_thresholds: dict = field(default_factory=dict)

    # Webhook security
    webhook_secret: str = ""
    allow_unsigned: bool = False          # dev only; never enable in production

    # GitHub App credentials (needed for enrichment + write actions)
    app_id: str = ""
    private_key_pem: str = ""

    # Actions on scored issues
    dry_run: bool = True                  # log the decision, never write to GitHub
    post_comment: bool = False            # comment the prediction on the issue
    apply_label: bool = False             # add `label_name` when P >= threshold
    label_name: str = "predicted:actionable-bug"

    # Enrichment (author metadata + latest release via the GitHub API)
    enrich: bool = True
    api_base_url: str = "https://api.github.com"
    request_timeout: float = 15.0

    # Online-evaluation ledger (predictions graded at issue close). None
    # keeps the ledger in memory only (tests, backtests).
    ledger_path: Path | None = None

    # Surface likely-duplicate prior issues (requires models/dup_index.joblib,
    # built by `python -m ghic.dupdetect --build-index`).
    suggest_related: bool = True
    related_min_similarity: float = 0.55

    # Draft a "missing information" comment for under-specified issues.
    # Uses the Anthropic API when a key is configured; otherwise a
    # deterministic template. Never touches the classifier's decision.
    draft_missing_info: bool = False

    extras: dict = field(default_factory=dict)

    @property
    def can_call_github(self) -> bool:
        return bool(self.app_id and self.private_key_pem)

    def threshold_for(self, repo_full_name: str) -> float:
        return self.repo_thresholds.get(repo_full_name, self.threshold)

    def validate(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Train one with "
                "`python -m ghic.train` or point GHIC_MODEL_PATH at a .joblib pipeline."
            )
        if not self.webhook_secret and not self.allow_unsigned:
            raise RuntimeError(
                "GHIC_WEBHOOK_SECRET is not set. Set it to the secret configured "
                "on the GitHub App, or set GHIC_ALLOW_UNSIGNED=true (dev only)."
            )
        if (self.post_comment or self.apply_label) and not self.dry_run and not self.can_call_github:
            raise RuntimeError(
                "Write actions are enabled but GHIC_APP_ID / GHIC_PRIVATE_KEY(_PATH) "
                "are missing — the service could not authenticate to GitHub."
            )
        if not self.webhook_secret and self.allow_unsigned:
            logger.warning(
                "Running with GHIC_ALLOW_UNSIGNED=true and no webhook secret. "
                "Do NOT expose this instance to the internet."
            )


def _load_private_key() -> str:
    pem = os.environ.get("GHIC_PRIVATE_KEY", "")
    if pem:
        return pem
    key_path = os.environ.get("GHIC_PRIVATE_KEY_PATH", "")
    if key_path and Path(key_path).exists():
        return Path(key_path).read_text(encoding="utf-8")
    return ""


def default_model_path() -> Path:
    """Prefer the calibrated champion; fall back to the v1 best model."""
    champion = utils.PROJECT_ROOT / "models" / "champion.joblib"
    if champion.exists():
        return champion
    return utils.PROJECT_ROOT / "models" / "rf_balanced.joblib"


def load_settings() -> ServiceSettings:
    """Build ServiceSettings from GHIC_* environment variables."""
    # `or` (not a get() default) so empty strings from .env fall back too.
    return ServiceSettings(
        model_path=Path(os.environ.get("GHIC_MODEL_PATH") or default_model_path()),
        threshold=_env_float("GHIC_THRESHOLD", 0.5),
        repo_thresholds=parse_repo_thresholds(os.environ.get("GHIC_REPO_THRESHOLDS", "")),
        webhook_secret=os.environ.get("GHIC_WEBHOOK_SECRET", ""),
        allow_unsigned=_env_bool("GHIC_ALLOW_UNSIGNED", False),
        app_id=os.environ.get("GHIC_APP_ID", ""),
        private_key_pem=_load_private_key(),
        dry_run=_env_bool("GHIC_DRY_RUN", True),
        post_comment=_env_bool("GHIC_POST_COMMENT", False),
        apply_label=_env_bool("GHIC_APPLY_LABEL", False),
        label_name=os.environ.get("GHIC_LABEL_NAME", "predicted:actionable-bug"),
        enrich=_env_bool("GHIC_ENRICH", True),
        api_base_url=os.environ.get("GHIC_API_BASE_URL", "https://api.github.com"),
        request_timeout=_env_float("GHIC_REQUEST_TIMEOUT", 15.0),
        ledger_path=_load_ledger_path(),
        suggest_related=_env_bool("GHIC_SUGGEST_RELATED", True),
        related_min_similarity=_env_float("GHIC_RELATED_MIN_SIM", 0.55),
        draft_missing_info=_env_bool("GHIC_DRAFT_MISSING_INFO", False),
    )


def _load_ledger_path() -> Path | None:
    """GHIC_LEDGER: unset -> data/predictions.jsonl; empty string -> disabled."""
    raw = os.environ.get("GHIC_LEDGER")
    if raw is None:
        return utils.PROJECT_ROOT / "data" / "predictions.jsonl"
    return Path(raw) if raw.strip() else None
