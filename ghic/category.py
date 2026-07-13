"""Issue category classification: bug / feature / question / docs / duplicate / invalid.

Unlike priority or severity (which have no ground truth in this corpus), the
category head trains on real ground truth: the category label a maintainer
eventually applied to the issue. The causal rule is the same as everywhere
else in this repo — predict from what is visible at open time (title, body,
author profile, timing), target is the *eventual* label.

Ground-truth derivation:
  - Each repo names categories differently (vscode `bug`, tensorflow
    `type:bug`, react `type: bug`); CATEGORY_LABEL_MAP normalizes them.
  - When an issue carries labels mapping to several classes (77 of 2,753 in
    the corpus), CLASS_PRIORITY resolves the conflict: the more specific
    triage verdict wins (an issue labeled both `bug` and `*duplicate` was
    ultimately triaged as a duplicate).
  - Issues with no category label have no ground truth and are excluded
    from training/evaluation — the model card reports coverage honestly.
  - The original spec also lists `security` and `regression` classes:
    security has zero occurrences in the corpus and vscode's `regression`
    (27 issues) is mapped into `bug` (a regression is a bug subtype).
    Neither is trainable as its own class; documented, not faked.

Protocol mirrors the champion model (train.py): per-repo chronological
split, walk-forward CV for candidate selection (macro-F1), refit + sigmoid
calibration on the newest train slice, exactly one test-set evaluation,
auto-generated model card with the full confusion matrix.

CLI:
  python -m ghic.category --train        # -> models/category.joblib + CATEGORY_CARD.md
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import evaluate, features, utils
from .config import Config, get_config
from .train import _load_release_dates, chronological_split, walk_forward_folds

logger = utils.get_logger(__name__)

MODEL_PATH = utils.PROJECT_ROOT / "models" / "category.joblib"
CARD_PATH = utils.PROJECT_ROOT / "models" / "CATEGORY_CARD.md"
REPORT_PATH = evaluate.REPORTS_DIR / "category.json"

# Canonical classes, and the priority order that resolves multi-label
# conflicts (first match wins; specific triage verdicts beat generic types).
CLASS_PRIORITY: tuple[str, ...] = (
    "duplicate", "invalid", "question", "docs", "feature", "bug",
)

# repo label (lowercased) -> canonical class. Only labels that unambiguously
# denote a category are mapped; component/status/version labels are not.
CATEGORY_LABEL_MAP: dict[str, str] = {
    # microsoft/vscode
    "bug": "bug",
    "regression": "bug",            # regression is a bug subtype; see module doc
    "feature-request": "feature",
    "*question": "question",
    "*dev-question": "question",
    "*duplicate": "duplicate",
    "invalid": "invalid",
    # tensorflow/tensorflow
    "type:bug": "bug",
    "type:feature": "feature",
    "type:support": "question",
    "type:docs-bug": "docs",
    "type:docs-feature": "docs",
    # facebook/react
    "type: bug": "bug",
    "type: question": "question",
    "type: enhancement": "feature",
    "type: documentation": "docs",
    # generic conventions (default GitHub label set)
    "duplicate": "duplicate",
    "documentation": "docs",
    "question": "question",
    "enhancement": "feature",
}


def derive_category(labels_at_close: Iterable[str]) -> str | None:
    """Map an issue's final labels to its canonical category, or None."""
    classes = {
        CATEGORY_LABEL_MAP[low]
        for lab in labels_at_close or []
        if (low := (lab or "").strip().lower()) in CATEGORY_LABEL_MAP
    }
    for cls in CLASS_PRIORITY:
        if cls in classes:
            return cls
    return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_category_dataset(
    combined_path: Path | None = None, cfg: Config | None = None,
) -> pd.DataFrame:
    """combined.csv rows that have a derivable category, bots/no-author dropped."""
    cfg = cfg or get_config(require_token=False)
    combined_path = combined_path or (utils.DATA_PROCESSED / "combined.csv")
    df = pd.read_csv(combined_path)

    df = df[df["author_login"].notna() & (df["author_login"] != "")]
    df = df[~df["author_login"].isin(cfg.labeling.bot_logins)]

    labels = df["labels_at_close"].map(lambda v: json.loads(v) if isinstance(v, str) and v else [])
    df = df.assign(category=labels.map(derive_category))
    kept = df[df["category"].notna()].copy()
    logger.info(
        "category dataset: %d of %d issues have ground truth (%s)",
        len(kept), len(df),
        dict(kept["category"].value_counts()),
    )
    return kept


def _engineered(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    feats = features.engineer_features(df, cfg, repo_release_dates=_load_release_dates())
    feats["category"] = df["category"]
    return feats


# ---------------------------------------------------------------------------
# Training — champion protocol, multi-class
# ---------------------------------------------------------------------------
def _candidates(cfg: Config) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    seed = cfg.random_seed
    return {
        "logreg_balanced": LogisticRegression(
            max_iter=2000, random_state=seed, class_weight="balanced"
        ),
        "rf_balanced": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=seed, class_weight="balanced"
        ),
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def train_category(
    cfg: Config | None = None, *, combined_path: Path | None = None,
) -> dict[str, Any]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline

    cfg = cfg or get_config(require_token=False)
    df = load_category_dataset(combined_path, cfg)
    feats = _engineered(df, cfg)
    train_feats, test_feats = chronological_split(feats)
    structured_cols = features.structured_feature_columns(feats)

    # --- 1. walk-forward selection by macro-F1 (training window only) -------
    folds = walk_forward_folds(train_feats)
    cv_results: dict[str, dict[str, Any]] = {}
    for name, estimator in _candidates(cfg).items():
        fold_scores: list[float] = []
        for k, (tr, va) in enumerate(folds, 1):
            pipe = Pipeline([
                ("pre", features.build_preprocessor(cfg, structured_cols)),
                ("clf", copy.deepcopy(estimator)),
            ])
            pipe.fit(tr, tr["category"].to_numpy())
            score = _macro_f1(va["category"].to_numpy(), pipe.predict(va))
            fold_scores.append(score)
            logger.info("cv %-16s fold %d/%d: macro_f1=%.4f", name, k, len(folds), score)
        cv_results[name] = {
            "fold_macro_f1": fold_scores,
            "mean_macro_f1": float(np.mean(fold_scores)),
            "std_macro_f1": float(np.std(fold_scores)),
        }
    winner = max(cv_results, key=lambda n: cv_results[n]["mean_macro_f1"])
    logger.info("category winner by walk-forward macro-F1: %s", winner)

    # --- 2. refit + calibrate on the newest train slice ---------------------
    fit_feats, cal_feats = chronological_split(train_feats, test_fraction=0.15)
    pipe = Pipeline([
        ("pre", features.build_preprocessor(cfg, structured_cols)),
        ("clf", _candidates(cfg)[winner]),
    ])
    pipe.fit(fit_feats, fit_feats["category"].to_numpy())

    # Sigmoid (not isotonic): the rare classes (docs, invalid) have single-digit
    # support in the calibration slice, where isotonic overfits badly. If a
    # class is entirely absent from the slice, calibration cannot be fit at
    # all — ship the uncalibrated pipeline and say so on the card.
    calibrated: Any = pipe
    calibration = "uncalibrated (calibration slice missing classes)"
    if set(pipe.classes_) <= set(cal_feats["category"]):
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid")
        calibrated.fit(cal_feats, cal_feats["category"].to_numpy())
        calibration = "sigmoid on the newest 15% of train (older than all test issues)"

    # --- 3. the one look at the test set -------------------------------------
    y_test = test_feats["category"].to_numpy()
    y_pred = calibrated.predict(test_feats)
    classes = [c for c in CLASS_PRIORITY if c in set(feats["category"])]
    report = classification_report(
        y_test, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    result = {
        "winner": winner,
        "cv": cv_results,
        "calibration": calibration,
        "classes": classes,
        "class_support_total": {c: int((feats["category"] == c).sum()) for c in classes},
        "test": {
            "n": int(len(y_test)),
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "per_class": {
                c: {
                    "precision": float(report[c]["precision"]),
                    "recall": float(report[c]["recall"]),
                    "f1": float(report[c]["f1-score"]),
                    "support": int(report[c]["support"]),
                }
                for c in classes
            },
            "confusion_matrix": cm,
        },
        "coverage": {
            "issues_with_ground_truth": int(len(df)),
            "corpus_total": 6175,
        },
    }

    import joblib

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, MODEL_PATH)
    logger.info("wrote %s", MODEL_PATH)
    utils.write_json(REPORT_PATH, result)
    write_card(result)
    return result


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------
def write_card(result: dict[str, Any]) -> None:
    classes = result["classes"]
    t = result["test"]
    lines = [
        "# Model card — issue category classifier",
        "",
        f"**Model:** `{result['winner']}` (selected by walk-forward temporal CV,",
        f"macro-F1); calibration: {result['calibration']}.",
        "Artifact: `models/category.joblib`.",
        "",
        "## Task and ground truth",
        "Multi-class prediction of the category label a maintainer eventually",
        "applies, from information visible at open time. Ground truth is the",
        "real applied label, normalized across repo conventions",
        "(`CATEGORY_LABEL_MAP` in `ghic/category.py`); multi-label conflicts",
        f"resolve by priority `{' > '.join(CLASS_PRIORITY)}`.",
        "",
        f"Coverage: {result['coverage']['issues_with_ground_truth']:,} of",
        f"{result['coverage']['corpus_total']:,} collected issues carry a",
        "category label — the rest have no ground truth and are excluded.",
        "`security` (0 occurrences) and standalone `regression` (27, mapped",
        "into `bug`) are not trainable in this corpus; adding them requires",
        "data from repos that use those labels.",
        "",
        "## Class support (whole dataset)",
        "",
        "| class | n |",
        "|---|---|",
    ]
    lines += [f"| {c} | {result['class_support_total'][c]} |" for c in classes]
    lines += [
        "",
        "## Walk-forward CV (mean macro-F1 ± std)",
        "",
        "| candidate | mean | std |",
        "|---|---|---|",
    ]
    for name, r in sorted(result["cv"].items(), key=lambda kv: -kv[1]["mean_macro_f1"]):
        marker = " ← **selected**" if name == result["winner"] else ""
        lines.append(f"| `{name}`{marker} | {r['mean_macro_f1']:.4f} | {r['std_macro_f1']:.4f} |")
    lines += [
        "",
        f"## Final test set (n={t['n']}, chronological, single evaluation)",
        "",
        f"Accuracy **{t['accuracy']:.3f}**, macro-F1 **{t['macro_f1']:.3f}**.",
        "",
        "| class | precision | recall | F1 | support |",
        "|---|---|---|---|---|",
    ]
    for c in classes:
        pc = t["per_class"][c]
        lines.append(
            f"| {c} | {pc['precision']:.3f} | {pc['recall']:.3f} "
            f"| {pc['f1']:.3f} | {pc['support']} |"
        )
    lines += [
        "",
        "## Confusion matrix (rows = truth, columns = predicted)",
        "",
        "| | " + " | ".join(classes) + " |",
        "|---|" + "---|" * len(classes),
    ]
    for c, row in zip(classes, t["confusion_matrix"]):
        lines.append(f"| **{c}** | " + " | ".join(str(v) for v in row) + " |")
    lines += [
        "",
        "## Limitations",
        "- Facebook/react contributes only ~106 labeled issues; per-repo",
        "  performance there is effectively unvalidated.",
        "- Ground truth is triager behavior: repos that mislabel categories",
        "  teach the model those mistakes.",
        "- Rare classes (docs, invalid) have small test support — treat their",
        "  per-class numbers as indicative, not established.",
        "- The service surfaces the category as an assistive suggestion; it",
        "  never applies category labels automatically.",
        "",
        "_Auto-generated by `python -m ghic.category --train`._",
    ]
    CARD_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", CARD_PATH)


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------
class CategoryPredictor:
    """Loads the fitted category pipeline and scores single issues.

    Reuses the exact same engineered-feature frame the actionability
    predictor builds, so serving adds one predict_proba call, not a second
    feature pass.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        import joblib

        self.model_path = model_path or MODEL_PATH
        self.pipeline = joblib.load(self.model_path)
        self.classes: list[str] = list(self.pipeline.classes_)

    def predict_frame(self, feats: pd.DataFrame) -> dict[str, Any]:
        proba = self.pipeline.predict_proba(feats)[0]
        by_class = {c: round(float(p), 4) for c, p in zip(self.classes, proba)}
        top = max(by_class, key=by_class.__getitem__)
        return {"predicted": top, "confidence": by_class[top], "proba": by_class}


def load_category_predictor(model_path: Path | None = None) -> CategoryPredictor | None:
    """CategoryPredictor if the artifact exists, else None (feature off)."""
    path = model_path or MODEL_PATH
    if not path.exists():
        return None
    return CategoryPredictor(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the issue category classifier.")
    parser.add_argument("--train", action="store_true", help="train + write card")
    parser.add_argument("--input", type=Path, default=None,
                        help="combined.csv path (default: data/processed/combined.csv)")
    args = parser.parse_args(argv)

    if not args.train:
        parser.print_help()
        return 0
    result = train_category(combined_path=args.input)
    print(f"winner={result['winner']} test macro-F1={result['test']['macro_f1']:.3f} "
          f"accuracy={result['test']['accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    from ghic.category import main as _main  # re-import so pickles carry the package path

    sys.exit(_main())
