"""Model training and evaluation orchestration.

Pipeline:
  labeled.csv -> engineer_features -> per-repo chronological 80/20 split ->
  fit ColumnTransformer on TRAIN only -> train 4 models (LogReg & RandomForest,
  each class_weight None and 'balanced') -> evaluate overall + per-repo ->
  persist fitted pipelines + metrics + plots.

Split rationale: a random split leaks the future into the past.
Issue resolution patterns drift (label conventions change, triage bots come and
go), and at deployment the webhook only ever sees issues newer than its training
data. So we sort each repo by created_at and take the first 80% as train, last
20% as test -- per repo, then concatenate, so every repo is represented in both
splits in time order.

class_weight is reported both ways (None vs 'balanced') rather than using SMOTE:
the positive class is the minority and we care about recall on it, but we also
want to show the precision/recall trade the reweighting buys. No synthetic
oversampling -- it would fabricate issue text that never existed.

CLI:
  python -m ghic.train                      # input: data/processed/labeled.csv
  python -m ghic.train --no-plots           # skip figure generation
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import evaluate, features, utils
from .config import Config, get_config

logger = utils.get_logger(__name__)

MODELS_DIR: Path = utils.PROJECT_ROOT / "models"
TEST_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_release_dates() -> dict[str, list[str]] | None:
    """Release dates from collect.py, if collected. Enables days_since_last_release."""
    path = utils.DATA_PROCESSED / "releases.json"
    if not path.exists():
        logger.info("No releases.json found; days_since_last_release will be skipped.")
        return None
    data = utils.read_json(path)
    logger.info("Loaded release dates for %d repos", len(data))
    return data


def load_labeled(path: Path) -> pd.DataFrame:
    """Read labeled.csv with dtypes that survive the round-trip from label.py."""
    df = pd.read_csv(path, dtype={"label": "Int64"})
    if "label" not in df.columns:
        raise ValueError(f"{path} has no `label` column -- run `python -m ghic.label` first.")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Chronological per-repo split
# ---------------------------------------------------------------------------
def chronological_split(
    frame: pd.DataFrame, test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per repo: sort by created_at, last `test_fraction` of rows -> test."""
    train_parts, test_parts = [], []
    for repo, grp in frame.groupby("repo_name", sort=True):
        grp = grp.sort_values("created_at", kind="stable")
        cut = int(round(len(grp) * (1 - test_fraction)))
        cut = min(max(cut, 1), len(grp) - 1) if len(grp) > 1 else len(grp)
        train_parts.append(grp.iloc[:cut])
        test_parts.append(grp.iloc[cut:])
        logger.info("split %s: %d train / %d test", repo, cut, len(grp) - cut)
    return pd.concat(train_parts), pd.concat(test_parts)


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def build_models(cfg: Config) -> dict[str, Any]:
    """The four estimators to compare. Seeded for reproducibility."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    seed = cfg.random_seed
    return {
        "logreg": LogisticRegression(max_iter=2000, random_state=seed, class_weight=None),
        "logreg_balanced": LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed, class_weight=None),
        "rf_balanced": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed, class_weight="balanced"),
    }


def build_models_v2(cfg: Config) -> dict[str, Any]:
    """The champion-protocol candidate zoo: balanced v1 models + gradient
    boosting over an LSA reduction + a soft-voting ensemble of all three
    families. Every estimator is seeded; selection happens via walk-forward
    CV in train_champion, never on the final test set."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    seed = cfg.random_seed

    def lr():
        return LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced")

    def rf():
        return RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=seed, class_weight="balanced"
        )

    def svd_hgb():
        # HistGradientBoosting needs dense input; LSA (TruncatedSVD) reduces
        # the sparse TF-IDF+numeric block to 256 dense components first.
        return Pipeline([
            ("svd", TruncatedSVD(n_components=256, random_state=seed)),
            ("hgb", HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                class_weight="balanced", random_state=seed,
            )),
        ])

    return {
        "logreg_balanced": lr(),
        "rf_balanced": rf(),
        "svd_hgb": svd_hgb(),
        "ensemble": VotingClassifier(
            estimators=[("lr", lr()), ("rf", rf()), ("hgb", svd_hgb())],
            voting="soft",
            n_jobs=1,
        ),
    }


# ---------------------------------------------------------------------------
# Walk-forward temporal cross-validation
# ---------------------------------------------------------------------------
def walk_forward_folds(
    frame: pd.DataFrame, n_folds: int = 3, val_fraction: float = 0.10,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window CV over time, per repo.

    Fold i trains on everything before its validation window and validates on
    the next `val_fraction` slice, so every fold respects causality: the model
    never sees an issue newer than the ones it is judged on. With n_folds=3
    and val_fraction=0.1, validation windows are the [70-80%), [80-90%), and
    [90-100%) slices of the given frame (which should be the TRAINING portion
    only — the final test set stays untouched).
    """
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_folds):
        upper = 1.0 - val_fraction * (n_folds - 1 - i)
        lower = upper - val_fraction
        train_parts, val_parts = [], []
        for _, grp in frame.groupby("repo_name", sort=True):
            grp = grp.sort_values("created_at", kind="stable")
            n = len(grp)
            lo, hi = int(round(n * lower)), int(round(n * upper))
            train_parts.append(grp.iloc[:lo])
            val_parts.append(grp.iloc[lo:hi])
        folds.append((pd.concat(train_parts), pd.concat(val_parts)))
    return folds


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------
@dataclass
class TrainedModel:
    name: str
    pipeline: Any                       # fitted Pipeline(preprocessor -> estimator)
    overall: evaluate.Metrics
    per_repo: dict[str, evaluate.Metrics]


def _evaluate_per_repo(
    pipeline: Any, test_feats: pd.DataFrame, y_test: np.ndarray, threshold: float,
) -> dict[str, evaluate.Metrics]:
    out: dict[str, evaluate.Metrics] = {}
    for repo, idx in test_feats.groupby("repo_name").groups.items():
        sub = test_feats.loc[idx]
        proba = pipeline.predict_proba(sub)[:, 1]
        out[str(repo)] = evaluate.compute_metrics(
            y_test[test_feats.index.get_indexer(idx)], proba, threshold
        )
    return out


def train_all(
    cfg: Config | None = None,
    *,
    labeled_path: Path | None = None,
    make_plots: bool = True,
    threshold: float = evaluate.DEFAULT_THRESHOLD,
) -> dict[str, TrainedModel]:
    from sklearn.pipeline import Pipeline

    cfg = cfg or get_config(require_token=False)
    labeled_path = labeled_path or (utils.DATA_PROCESSED / "labeled.csv")
    logger.info("Loading %s", labeled_path)
    df = load_labeled(labeled_path)

    feats = features.engineer_features(df, cfg, repo_release_dates=_load_release_dates())
    train_feats, test_feats = chronological_split(feats)
    structured_cols = features.structured_feature_columns(feats)
    logger.info(
        "Features: %d structured + TF-IDF; train=%d test=%d (overall C1 ratio=%.2f%%)",
        len(structured_cols), len(train_feats), len(test_feats), 100 * feats["label"].mean(),
    )

    y_train = train_feats["label"].to_numpy()
    y_test = test_feats["label"].to_numpy()

    results: dict[str, TrainedModel] = {}
    for name, estimator in build_models(cfg).items():
        preprocessor = features.build_preprocessor(cfg, structured_cols)
        pipe = Pipeline([("pre", preprocessor), ("clf", estimator)])
        logger.info("fitting %s", name)
        pipe.fit(train_feats, y_train)

        proba = pipe.predict_proba(test_feats)[:, 1]
        overall = evaluate.compute_metrics(y_test, proba, threshold)
        per_repo = _evaluate_per_repo(pipe, test_feats, y_test, threshold)
        results[name] = TrainedModel(name, pipe, overall, per_repo)

        _persist(pipe, name)
        if make_plots:
            _plots_for(name, pipe, test_feats, y_test, proba, overall)

    _write_metrics_report(results)
    print(evaluate.metrics_table({n: r.overall for n, r in results.items()}))
    return results


def _persist(pipeline: Any, name: str) -> Path:
    import joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    logger.info("wrote %s", path)
    return path


def _plots_for(name, pipe, test_feats, y_test, proba, overall) -> None:
    evaluate.plot_confusion_matrix(overall, f"Confusion -- {name}", f"cm_{name}.png")
    if len(np.unique(y_test)) > 1:
        evaluate.plot_roc_curve(y_test, proba, f"ROC -- {name}", f"roc_{name}.png")
        evaluate.plot_pr_curve(y_test, proba, f"PR -- {name}", f"pr_{name}.png")
    names = features.feature_names(pipe.named_steps["pre"])
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        evaluate.plot_top_features(names, clf.coef_[0], f"Top coefficients -- {name}",
                                   f"coef_{name}.png", signed=True)
    elif hasattr(clf, "feature_importances_"):
        evaluate.plot_top_features(names, clf.feature_importances_, f"Top importances -- {name}",
                                   f"importance_{name}.png", signed=False)


def _write_metrics_report(results: dict[str, TrainedModel]) -> None:
    payload = {
        name: {
            "overall": r.overall.as_dict(),
            "per_repo": {repo: m.as_dict() for repo, m in r.per_repo.items()},
        }
        for name, r in results.items()
    }
    path = evaluate.REPORTS_DIR / "metrics.json"
    utils.write_json(path, payload)
    logger.info("wrote %s", path)


# ---------------------------------------------------------------------------
# Champion protocol: walk-forward selection -> isotonic calibration -> one
# final evaluation on the untouched test set -> models/champion.joblib
# ---------------------------------------------------------------------------
CHAMPION_NAME = "champion"


def train_champion(
    cfg: Config | None = None,
    *,
    labeled_path: Path | None = None,
    threshold: float = evaluate.DEFAULT_THRESHOLD,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Rigorous selection: candidates are compared by mean PR-AUC across
    walk-forward folds (inside the training window only). The winner is
    refit on the earlier 85% of train, isotonically calibrated on the last
    15% (still older than every test issue), and evaluated exactly once on
    the held-out test set. The calibrated pipeline is saved as champion.joblib
    with an auto-generated model card."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.pipeline import Pipeline

    cfg = cfg or get_config(require_token=False)
    labeled_path = labeled_path or (utils.DATA_PROCESSED / "labeled.csv")
    df = load_labeled(labeled_path)
    feats = features.engineer_features(df, cfg, repo_release_dates=_load_release_dates())
    train_feats, test_feats = chronological_split(feats)
    structured_cols = features.structured_feature_columns(feats)

    # --- 1. walk-forward model selection (training window only) -------------
    folds = walk_forward_folds(train_feats)
    cv_results: dict[str, dict[str, Any]] = {}
    for name, estimator in build_models_v2(cfg).items():
        import copy

        fold_pr_aucs: list[float] = []
        for k, (tr, va) in enumerate(folds, 1):
            pipe = Pipeline([
                ("pre", features.build_preprocessor(cfg, structured_cols)),
                ("clf", copy.deepcopy(estimator)),
            ])
            pipe.fit(tr, tr["label"].to_numpy())
            proba = pipe.predict_proba(va)[:, 1]
            m = evaluate.compute_metrics(va["label"].to_numpy(), proba, threshold)
            fold_pr_aucs.append(m.pr_auc)
            logger.info("cv %-16s fold %d/%d: pr_auc=%.4f f1=%.4f", name, k, len(folds), m.pr_auc, m.f1)
        cv_results[name] = {
            "fold_pr_aucs": fold_pr_aucs,
            "mean_pr_auc": float(np.nanmean(fold_pr_aucs)),
            "std_pr_auc": float(np.nanstd(fold_pr_aucs)),
        }
        logger.info(
            "cv %-16s mean pr_auc=%.4f (+/- %.4f)",
            name, cv_results[name]["mean_pr_auc"], cv_results[name]["std_pr_auc"],
        )

    winner = max(cv_results, key=lambda n: cv_results[n]["mean_pr_auc"])
    logger.info("champion by walk-forward PR-AUC: %s", winner)

    # --- 2. refit winner, isotonic calibration on a newer-than-train slice --
    fit_feats, cal_feats = chronological_split(train_feats, test_fraction=0.15)
    champion_pipe = Pipeline([
        ("pre", features.build_preprocessor(cfg, structured_cols)),
        ("clf", build_models_v2(cfg)[winner]),
    ])
    champion_pipe.fit(fit_feats, fit_feats["label"].to_numpy())
    calibrated = CalibratedClassifierCV(FrozenEstimator(champion_pipe), method="isotonic")
    calibrated.fit(cal_feats, cal_feats["label"].to_numpy())

    # --- 3. the one and only look at the test set ---------------------------
    y_test = test_feats["label"].to_numpy()
    proba_raw = champion_pipe.predict_proba(test_feats)[:, 1]
    proba_cal = calibrated.predict_proba(test_feats)[:, 1]
    overall_raw = evaluate.compute_metrics(y_test, proba_raw, threshold)
    overall_cal = evaluate.compute_metrics(y_test, proba_cal, threshold)
    per_repo = _evaluate_per_repo(calibrated, test_feats, y_test, threshold)

    logger.info("champion (%s) uncalibrated: f1=%.3f pr_auc=%.3f brier=%.4f",
                winner, overall_raw.f1, overall_raw.pr_auc, overall_raw.brier)
    logger.info("champion (%s) calibrated:   f1=%.3f pr_auc=%.3f brier=%.4f",
                winner, overall_cal.f1, overall_cal.pr_auc, overall_cal.brier)

    if make_plots:
        evaluate.plot_calibration_curve(
            y_test,
            {"uncalibrated": proba_raw, "isotonic": proba_cal},
            f"Calibration — champion ({winner})",
            "calibration_champion.png",
        )
        evaluate.plot_pr_curve(y_test, proba_cal, f"PR — champion ({winner})", "pr_champion.png")
        evaluate.plot_roc_curve(y_test, proba_cal, f"ROC — champion ({winner})", "roc_champion.png")

    _persist(calibrated, CHAMPION_NAME)
    result = {
        "winner": winner,
        "cv": cv_results,
        "test_uncalibrated": overall_raw.as_dict(),
        "test_calibrated": overall_cal.as_dict(),
        "per_repo": {repo: m.as_dict() for repo, m in per_repo.items()},
        "protocol": {
            "selection": "walk-forward CV (3 expanding folds) by mean PR-AUC, training window only",
            "calibration": "isotonic on the newest 15% of train (older than all test issues)",
            "test": "single evaluation on the untouched chronological test set",
        },
    }
    utils.write_json(evaluate.REPORTS_DIR / "champion.json", result)
    _write_model_card(result, threshold)
    print(evaluate.metrics_table({
        f"champion={winner} (raw)": overall_raw,
        f"champion={winner} (cal)": overall_cal,
        **per_repo,
    }))
    return result


def _write_model_card(result: dict[str, Any], threshold: float) -> None:
    """Auto-generate models/MODEL_CARD.md from the champion run."""
    cal = result["test_calibrated"]
    lines = [
        "# Model card — issue triage champion",
        "",
        f"**Model:** `{result['winner']}` (selected by walk-forward temporal CV), "
        "isotonically calibrated. Artifact: `models/champion.joblib`.",
        "",
        "## Intended use",
        "Rank/triage newly opened GitHub issues by probability of resolving as an",
        "actionable bug (fixed via merged PR or confirmed+completed). Assistive",
        "signal for maintainers — never an auto-close mechanism.",
        "",
        "## Training data",
        "5,885 closed issues from microsoft/vscode, facebook/react,",
        "tensorflow/tensorflow (calendar 2024), labeled by deterministic rules",
        "(see README). Class 1 base rate: 27.8%.",
        "",
        "## Selection protocol",
        f"- {result['protocol']['selection']}",
        f"- {result['protocol']['calibration']}",
        f"- {result['protocol']['test']}",
        "",
        "## Walk-forward CV (mean PR-AUC ± std)",
        "",
        "| candidate | mean PR-AUC | std |",
        "|---|---|---|",
    ]
    for name, r in sorted(result["cv"].items(), key=lambda kv: -kv[1]["mean_pr_auc"]):
        marker = " ← **champion**" if name == result["winner"] else ""
        lines.append(f"| `{name}`{marker} | {r['mean_pr_auc']:.4f} | {r['std_pr_auc']:.4f} |")
    lines += [
        "",
        f"## Final test set (n={cal['n']}, threshold {threshold})",
        "",
        "| metric | uncalibrated | calibrated |",
        "|---|---|---|",
    ]
    raw = result["test_uncalibrated"]
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "brier"):
        lines.append(f"| {key} | {raw[key]:.4f} | {cal[key]:.4f} |")
    lines += [
        "",
        "## Known limitations",
        "- Trained on three large, well-triaged repos; calibration on other",
        "  communities is unverified — run `python -m ghic.backtest` and start",
        "  in dry-run mode.",
        "- Author public_repos/followers are collection-time snapshots.",
        "- first_time_contributor degrades to 'first-time' at single-issue inference.",
        "- Labels are rule-derived, not human-annotated; Rule 4 (conservative",
        "  default) contributes 31% of Class 0.",
        "",
        "_This card is auto-generated by `python -m ghic.train --champion`._",
    ]
    path = MODELS_DIR / "MODEL_CARD.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate issue-outcome models.")
    parser.add_argument("--input", type=Path, default=utils.DATA_PROCESSED / "labeled.csv")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--threshold", type=float, default=evaluate.DEFAULT_THRESHOLD)
    parser.add_argument(
        "--champion", action="store_true",
        help="run the rigorous protocol: walk-forward CV selection + isotonic "
             "calibration -> models/champion.joblib + MODEL_CARD.md",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        logger.error("Input not found: %s. Run `python -m ghic.label` first.", args.input)
        return 1

    if args.champion:
        train_champion(
            labeled_path=args.input,
            threshold=args.threshold,
            make_plots=not args.no_plots,
        )
        return 0

    train_all(
        labeled_path=args.input,
        make_plots=not args.no_plots,
        threshold=args.threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
