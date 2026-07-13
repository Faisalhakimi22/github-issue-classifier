"""Effort estimation — experiment with a pre-declared ship bar.

The only effort proxy available in this corpus is **time-to-close**, and it
is a known-weak one (stated up front, per the build plan): time-to-close
conflates the work an issue takes with how long it sat in the backlog, with
triage priority, and with stale-bot policies. The honest way to handle a
weak proxy is to declare the bar before running the experiment and ship
nothing if the bar isn't met:

    SHIP BAR (declared before the first run):
      Spearman rank correlation >= 0.30 on the chronological test set, AND
      >= 10% MAE(log-days) improvement over the constant-median baseline.

If the model clears the bar it ships as a coarse bucket estimate (days /
weeks / months), never a point estimate. If it doesn't, the negative result
is recorded in `models/EFFORT_CARD.md` and nothing ships — the same way the
duplicate-detection negative result was handled.

Protocol: per-repo chronological split, walk-forward CV between candidates
(selection by mean Spearman), single test evaluation.

CLI:
  python -m ghic.effort --evaluate
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import evaluate, features, utils
from .config import Config, get_config
from .train import _load_release_dates, chronological_split, walk_forward_folds

logger = utils.get_logger(__name__)

CARD_PATH = utils.PROJECT_ROOT / "models" / "EFFORT_CARD.md"
REPORT_PATH = evaluate.REPORTS_DIR / "effort.json"
MODEL_PATH = utils.PROJECT_ROOT / "models" / "effort.joblib"

SHIP_BAR_SPEARMAN = 0.30
SHIP_BAR_MAE_IMPROVEMENT = 0.10


def _target_log_days(df: pd.DataFrame) -> pd.Series:
    created = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    closed = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    days = (closed - created).dt.total_seconds() / 86400.0
    return np.log1p(days.clip(lower=0))


def load_effort_dataset(
    combined_path: Path | None = None, cfg: Config | None = None,
) -> pd.DataFrame:
    cfg = cfg or get_config(require_token=False)
    df = pd.read_csv(combined_path or (utils.DATA_PROCESSED / "combined.csv"))
    df = df[df["author_login"].notna() & (df["author_login"] != "")]
    df = df[~df["author_login"].isin(cfg.labeling.bot_logins)]
    df = df.assign(log_days_to_close=_target_log_days(df))
    df = df[df["log_days_to_close"].notna()].copy()
    logger.info("effort dataset: %d issues; median days-to-close per repo: %s",
                len(df),
                df.groupby("repo_name")["log_days_to_close"]
                  .apply(lambda s: round(float(np.expm1(s.median())), 1)).to_dict())
    return df


def _candidates(cfg: Config) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    seed = cfg.random_seed
    return {
        "ridge": Ridge(alpha=1.0, random_state=seed),
        # max_features="sqrt": the regression default (all features) scans the
        # full ~8k-dim sparse block at every split — intractably slow here.
        "rf_regressor": RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=seed,
            min_samples_leaf=5, max_features="sqrt",
        ),
    }


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import spearmanr

    rho = spearmanr(y_true, y_pred).statistic
    return float(rho) if rho == rho else 0.0


def evaluate_effort(
    cfg: Config | None = None, *, combined_path: Path | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error
    from sklearn.pipeline import Pipeline

    cfg = cfg or get_config(require_token=False)
    df = load_effort_dataset(combined_path, cfg)
    feats = features.engineer_features(df, cfg, repo_release_dates=_load_release_dates())
    feats["log_days_to_close"] = df["log_days_to_close"]
    train_feats, test_feats = chronological_split(feats)
    structured_cols = features.structured_feature_columns(feats)

    # --- walk-forward selection by mean Spearman -----------------------------
    folds = walk_forward_folds(train_feats)
    cv_results: dict[str, dict[str, Any]] = {}
    for name, estimator in _candidates(cfg).items():
        scores: list[float] = []
        for k, (tr, va) in enumerate(folds, 1):
            pipe = Pipeline([
                ("pre", features.build_preprocessor(cfg, structured_cols)),
                ("reg", copy.deepcopy(estimator)),
            ])
            pipe.fit(tr, tr["log_days_to_close"].to_numpy())
            rho = _spearman(va["log_days_to_close"].to_numpy(), pipe.predict(va))
            scores.append(rho)
            logger.info("cv %-14s fold %d/%d: spearman=%.4f", name, k, len(folds), rho)
        cv_results[name] = {
            "fold_spearman": scores,
            "mean_spearman": float(np.mean(scores)),
        }
    winner = max(cv_results, key=lambda n: cv_results[n]["mean_spearman"])

    # --- single test evaluation ----------------------------------------------
    pipe = Pipeline([
        ("pre", features.build_preprocessor(cfg, structured_cols)),
        ("reg", _candidates(cfg)[winner]),
    ])
    y_train = train_feats["log_days_to_close"].to_numpy()
    y_test = test_feats["log_days_to_close"].to_numpy()
    pipe.fit(train_feats, y_train)
    pred = pipe.predict(test_feats)

    baseline = np.full_like(y_test, float(np.median(y_train)))
    mae_model = float(mean_absolute_error(y_test, pred))
    mae_base = float(mean_absolute_error(y_test, baseline))
    rho_test = _spearman(y_test, pred)
    improvement = (mae_base - mae_model) / mae_base if mae_base else 0.0
    shipped = rho_test >= SHIP_BAR_SPEARMAN and improvement >= SHIP_BAR_MAE_IMPROVEMENT

    result = {
        "proxy": "log1p(days from open to close) — conflates effort with backlog "
                 "priority and stale-bot policy; declared-weak up front",
        "ship_bar": {
            "spearman_min": SHIP_BAR_SPEARMAN,
            "mae_improvement_min": SHIP_BAR_MAE_IMPROVEMENT,
        },
        "winner": winner,
        "cv": cv_results,
        "test": {
            "n": int(len(y_test)),
            "spearman": round(rho_test, 4),
            "mae_log_days": round(mae_model, 4),
            "baseline_mae_log_days": round(mae_base, 4),
            "mae_improvement": round(improvement, 4),
        },
        "shipped": shipped,
    }
    if shipped:
        import joblib

        joblib.dump(pipe, MODEL_PATH)
        logger.info("ship bar met — wrote %s", MODEL_PATH)
    else:
        logger.info("ship bar NOT met (spearman=%.3f, improvement=%.1f%%) — nothing ships",
                    rho_test, improvement * 100)
    utils.write_json(REPORT_PATH, result)
    write_card(result)
    return result


# ---------------------------------------------------------------------------
# Serving (only reachable when a run met the ship bar and wrote the artifact)
# ---------------------------------------------------------------------------
# Buckets are deliberately coarse: the validated claim is rank-informativeness
# (Spearman), not point accuracy (MAE in log-days is ~1.5 ≈ a 4x typical
# factor), so anything finer than these four bands would imply precision the
# evaluation does not support.
BUCKETS: tuple[tuple[float, str], ...] = (
    (3.0, "a few days"),
    (21.0, "1–3 weeks"),
    (90.0, "1–3 months"),
    (float("inf"), "3+ months"),
)


def bucket_for_days(days: float) -> str:
    for upper, label in BUCKETS:
        if days <= upper:
            return label
    return BUCKETS[-1][1]


class EffortPredictor:
    """Coarse resolution-time estimate from the shipped effort pipeline."""

    def __init__(self, model_path: Path | None = None) -> None:
        import joblib

        self.model_path = model_path or MODEL_PATH
        self.pipeline = joblib.load(self.model_path)

    def predict_frame(self, feats: pd.DataFrame) -> dict[str, Any]:
        log_days = float(self.pipeline.predict(feats)[0])
        days = float(np.expm1(max(0.0, log_days)))
        return {
            "bucket": bucket_for_days(days),
            "basis": "historical time-to-close of similar issues; "
                     "rank-informative, not a commitment",
        }


def load_effort_predictor(model_path: Path | None = None) -> EffortPredictor | None:
    path = model_path or MODEL_PATH
    if not path.exists():
        return None
    return EffortPredictor(path)


def write_card(result: dict[str, Any]) -> Path:
    t = result["test"]
    verdict = (
        "**Shipped** as a coarse bucket estimate (a few days / 1–3 weeks /\n"
        "1–3 months / 3+ months) — the bar was met. Surfaced in the API\n"
        "response only, never in the public comment: a time estimate shown\n"
        "to issue reporters reads as a commitment, and the validated claim\n"
        "is rank-informativeness, not point accuracy."
        if result["shipped"] else
        "**Not shipped.** The bar was not met; per the build discipline the "
        "negative result is the deliverable. Predicting time-to-close from "
        "open-time text at this accuracy would be decoration, not signal."
    )
    lines = [
        "# Card — effort estimation (time-to-close proxy)",
        "",
        f"**Proxy:** {result['proxy']}.",
        "",
        "## Pre-declared ship bar",
        f"Spearman ≥ {result['ship_bar']['spearman_min']:.2f} on the chronological",
        f"test set AND ≥ {result['ship_bar']['mae_improvement_min']:.0%} MAE(log-days)",
        "improvement over the constant-median baseline. Declared before the",
        "first run; not adjusted afterward.",
        "",
        "## Walk-forward CV (mean Spearman)",
        "",
        "| candidate | mean Spearman |",
        "|---|---|",
    ]
    for name, r in sorted(result["cv"].items(), key=lambda kv: -kv[1]["mean_spearman"]):
        marker = " ← selected" if name == result["winner"] else ""
        lines.append(f"| `{name}`{marker} | {r['mean_spearman']:.4f} |")
    lines += [
        "",
        f"## Test result (n={t['n']}, single evaluation)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| Spearman | {t['spearman']:.4f} |",
        f"| MAE (log-days) | {t['mae_log_days']:.4f} |",
        f"| constant-median baseline MAE | {t['baseline_mae_log_days']:.4f} |",
        f"| MAE improvement | {t['mae_improvement']:.1%} |",
        "",
        "## Decision",
        verdict,
        "",
        "What would change this: a real effort signal — linked-PR diff size,",
        "number of files touched, review rounds — collected per issue. That",
        "is roadmap data collection, not modeling cleverness.",
        "",
        "_Auto-generated by `python -m ghic.effort --evaluate`._",
    ]
    CARD_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", CARD_PATH)
    return CARD_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Effort estimation experiment.")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)
    if not args.evaluate:
        parser.print_help()
        return 0
    result = evaluate_effort()
    print(f"winner={result['winner']} spearman={result['test']['spearman']} "
          f"mae_improvement={result['test']['mae_improvement']:.1%} "
          f"shipped={result['shipped']}")
    return 0


if __name__ == "__main__":
    from ghic.effort import main as _main

    sys.exit(_main())
