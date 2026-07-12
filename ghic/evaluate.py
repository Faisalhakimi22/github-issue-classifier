"""Evaluation: metrics, plots, and per-prediction explanations.

Kept separate from train.py so the same metric/plotting helpers can be called
from a notebook, from train.py, or from a future monitoring job on the webhook
without dragging in the training loop.

Class imbalance is the headline concern, so accuracy is reported but never
emphasised -- precision/recall/F1 on the positive class (Actionable Bug = 1)
plus ROC AUC and the full precision-recall curve are the real evaluation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import utils

logger = utils.get_logger(__name__)

REPORTS_DIR: Path = utils.PROJECT_ROOT / "reports"
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Metrics:
    n: int
    positives: int
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    accuracy: float
    tn: int
    fp: int
    fn: int
    tp: int
    # Brier score: mean squared error of the probabilities themselves.
    # Low Brier = calibrated probabilities = thresholds you can trust.
    brier: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> Metrics:
    """Threshold the probabilities and compute the full metric set.

    ROC AUC and PR AUC (average precision) are computed on probabilities
    (threshold-independent). Both are set to NaN when only one class is present,
    which happens on tiny per-repo test slices -- the caller should report them
    as 'n/a' rather than crash.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0, labels=[0, 1]
    )
    single_class = len(np.unique(y_true)) < 2
    try:
        auc = float("nan") if single_class else float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = float("nan") if single_class else float(average_precision_score(y_true, y_proba))
    except ValueError:
        pr_auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    brier = float(brier_score_loss(y_true, y_proba))
    return Metrics(
        brier=brier,
        n=len(y_true),
        positives=int(y_true.sum()),
        threshold=threshold,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=auc,
        pr_auc=pr_auc,
        accuracy=float(accuracy_score(y_true, y_pred)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def metrics_table(rows: dict[str, Metrics]) -> str:
    """Render {model_or_repo_name: Metrics} as an aligned text table."""
    header = (
        f"{'subset':28s} {'n':>6} {'pos':>6} {'prec':>6} {'rec':>6} {'f1':>6} "
        f"{'auc':>6} {'prauc':>6} {'acc':>6}"
    )
    lines = [header, "-" * len(header)]
    for name, m in rows.items():
        auc = "  n/a" if m.roc_auc != m.roc_auc else f"{m.roc_auc:6.3f}"  # NaN check
        prauc = "  n/a" if m.pr_auc != m.pr_auc else f"{m.pr_auc:6.3f}"
        lines.append(
            f"{name:28s} {m.n:6d} {m.positives:6d} "
            f"{m.precision:6.3f} {m.recall:6.3f} {m.f1:6.3f} {auc} {prauc} {m.accuracy:6.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots (saved to reports/; matplotlib imported lazily so importing this module
# is cheap and headless-safe)
# ---------------------------------------------------------------------------
# Shared style so every figure is legible at print size. Applied once per
# figure in _new_ax.
_PLOT_RC = {
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
}
_SAVE_DPI = 150


def _new_ax(figsize=(6, 5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(_PLOT_RC)
    fig, ax = plt.subplots(figsize=figsize)
    return plt, fig, ax


def _save(plt, fig, name: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def plot_confusion_matrix(m: Metrics, title: str, filename: str) -> Path:
    plt, fig, ax = _new_ax((5, 4.5))
    mat = np.array([[m.tn, m.fp], [m.fn, m.tp]])
    im = ax.imshow(mat, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="issue count")
    ax.set_xticks([0, 1], ["Pred: Non-Actionable (0)", "Pred: Actionable Bug (1)"])
    ax.set_yticks([0, 1], ["True: Non-Actionable (0)", "True: Actionable Bug (1)"])
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center", fontsize=9)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > mat.max() / 2 else "black", fontsize=15,
                fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    return _save(plt, fig, filename)


def plot_roc_curve(y_true, y_proba, title: str, filename: str) -> Path:
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt, fig, ax = _new_ax()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _save(plt, fig, filename)


def plot_pr_curve(y_true, y_proba, title: str, filename: str) -> Path:
    from sklearn.metrics import average_precision_score, precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    base = np.asarray(y_true).mean()
    plt, fig, ax = _new_ax()
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.axhline(base, ls="--", color="k", alpha=0.4, label=f"baseline = {base:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return _save(plt, fig, filename)


def plot_calibration_curve(
    y_true, curves: dict[str, Any], title: str, filename: str,
) -> Path:
    """Reliability diagram: predicted probability vs observed frequency.

    `curves` maps a label (e.g. "uncalibrated" / "isotonic") to its probability
    vector. A perfectly calibrated model follows the diagonal.
    """
    from sklearn.calibration import calibration_curve

    plt, fig, ax = _new_ax()
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    for label, proba in curves.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.set_xlabel("Mean predicted P(actionable bug)")
    ax.set_ylabel("Observed fraction of actionable bugs")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return _save(plt, fig, filename)


def plot_top_features(
    names: Sequence[str],
    importances: Sequence[float],
    title: str,
    filename: str,
    top_n: int = 25,
    signed: bool = False,
) -> Path:
    """Horizontal bar of top features.

    signed=True (logistic regression coefficients) ranks by absolute value but
    plots the signed value, so the direction of each effect is visible.
    signed=False (random forest importances) ranks and plots magnitude.
    """
    names = np.asarray(names)
    importances = np.asarray(importances, dtype=float)
    rank_key = np.abs(importances) if signed else importances
    idx = np.argsort(rank_key)[-top_n:]
    plt, fig, ax = _new_ax((8, max(4.5, 0.40 * len(idx))))
    colors = ["#c44" if v < 0 else "#268" for v in importances[idx]] if signed else "#268"
    ax.barh(range(len(idx)), importances[idx], color=colors)
    ax.set_yticks(range(len(idx)), names[idx], fontsize=9)
    ax.set_ylim(-0.6, len(idx) - 0.4)
    ax.set_xlabel("Coefficient (signed)" if signed else "Importance (Gini)")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    if signed:
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#268", label="pushes toward Actionable Bug (1)"),
            Patch(color="#c44", label="pushes toward Non-Actionable (0)"),
        ], loc="lower right", fontsize=9)
    return _save(plt, fig, filename)


# ---------------------------------------------------------------------------
# High-value summary plots for the report
# ---------------------------------------------------------------------------
def plot_feature_comparison(
    lr_names: Sequence[str],
    lr_coefs: Sequence[float],
    rf_names: Sequence[str],
    rf_importances: Sequence[float],
    filename: str = "feature_comparison.png",
    top_n: int = 15,
) -> Path:
    """Side-by-side: top LR coefficients (by magnitude) and top RF importances.

    Two panels share the figure so a reader can compare what each model relies
    on. LR bars are signed (red = pushes toward Non-Actionable); RF bars are
    magnitudes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update(_PLOT_RC)

    lr_names = np.asarray(lr_names)
    lr_coefs = np.asarray(lr_coefs, dtype=float)
    rf_names = np.asarray(rf_names)
    rf_importances = np.asarray(rf_importances, dtype=float)

    lr_idx = np.argsort(np.abs(lr_coefs))[-top_n:]
    rf_idx = np.argsort(rf_importances)[-top_n:]

    fig, (ax_lr, ax_rf) = plt.subplots(1, 2, figsize=(14, max(5, 0.42 * top_n)))

    lr_colors = ["#c44" if v < 0 else "#268" for v in lr_coefs[lr_idx]]
    ax_lr.barh(range(top_n), lr_coefs[lr_idx], color=lr_colors)
    ax_lr.set_yticks(range(top_n), lr_names[lr_idx], fontsize=9)
    ax_lr.set_ylim(-0.6, top_n - 0.4)
    ax_lr.set_xlabel("Coefficient (signed)")
    ax_lr.set_title(f"Logistic Regression: top {top_n} coefficients")
    ax_lr.grid(axis="x", alpha=0.3)
    ax_lr.legend(handles=[
        Patch(color="#268", label="toward Actionable Bug (1)"),
        Patch(color="#c44", label="toward Non-Actionable (0)"),
    ], loc="lower right", fontsize=8)

    ax_rf.barh(range(top_n), rf_importances[rf_idx], color="#268")
    ax_rf.set_yticks(range(top_n), rf_names[rf_idx], fontsize=9)
    ax_rf.set_ylim(-0.6, top_n - 0.4)
    ax_rf.set_xlabel("Importance (Gini)")
    ax_rf.set_title(f"Random Forest: top {top_n} importances")
    ax_rf.grid(axis="x", alpha=0.3)

    fig.suptitle("Feature importance comparison: Logistic Regression vs Random Forest "
                 "(rf_balanced)", fontsize=15, fontweight="bold")
    return _save(plt, fig, filename)


def plot_per_repo_f1(
    repo_f1: dict[str, float],
    filename: str = "per_repo_f1_rf_balanced.png",
    model_label: str = "rf_balanced",
) -> Path:
    """Bar chart of per-repo F1 for the chosen model, with value labels."""
    repos = list(repo_f1.keys())
    vals = [repo_f1[r] for r in repos]
    plt, fig, ax = _new_ax((7, 5))
    bars = ax.bar(range(len(repos)), vals, color=["#268", "#3a8", "#c44"][: len(repos)])
    ax.set_xticks(range(len(repos)), repos, rotation=10, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 score (Actionable Bug class)")
    ax.set_xlabel("Repository")
    ax.set_title(f"Per-repository F1 at threshold 0.5 ({model_label})")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    return _save(plt, fig, filename)


def plot_drop_locked_effect(
    before: int,
    after: int,
    filename: str = "drop_locked_effect.png",
) -> Path:
    """Before/after bar chart of merged-PR Class 1 labels around the drop_locked fix."""
    plt, fig, ax = _new_ax((6.5, 5))
    labels = ["Locked filter ON\n(original config)", "Locked filter OFF\n(corrected)"]
    vals = [before, after]
    bars = ax.bar(range(2), vals, color=["#c44", "#268"])
    ax.set_xticks(range(2), labels)
    ax.set_ylabel("Issues labeled Class 1 by the merged-PR rule")
    ax.set_title("Effect of the drop_locked fix on the merged-PR signal")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
                ha="center", va="bottom", fontsize=13, fontweight="bold")
    return _save(plt, fig, filename)


# ---------------------------------------------------------------------------
# Per-prediction explanation (one JSON per issue) -- webhook-shaped output
# ---------------------------------------------------------------------------
def unwrap_pipeline(model: Any) -> Any | None:
    """Return the underlying Pipeline(pre -> clf), looking through calibration.

    CalibratedClassifierCV (and the FrozenEstimator it wraps) hide the fitted
    pipeline; explanations need the raw preprocessor + estimator. Direction and
    ranking of contributions are unchanged by monotone calibration.
    """
    if hasattr(model, "named_steps"):
        return model
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        est = calibrated[0].estimator
        est = getattr(est, "estimator", est)  # FrozenEstimator wraps once more
        if hasattr(est, "named_steps"):
            return est
    return None


def top_contributions(pipe: Any, row: Any, k: int = 6) -> tuple[list[tuple[str, float]], bool]:
    """Per-feature explanation for one issue (single-row DataFrame).

    Returns (items, signed). For Logistic Regression `signed=True` and each
    value is the exact contribution (coef * standardized feature value), so its
    sign is the direction the feature pushed this prediction. For the Random
    Forest there is no per-sample linear term, so `signed=False` and each value
    is the global Gini importance of a feature that is active for this issue --
    a magnitude, not a direction. Callers must label the two cases differently
    so the output never implies a direction the model cannot give.
    """
    from . import features

    pipe = unwrap_pipeline(pipe)
    if pipe is None:
        return [], False
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names = features.feature_names(pre)
    x = pre.transform(row)
    x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()

    if hasattr(clf, "coef_"):
        contrib = clf.coef_[0] * x                      # exact, signed LR contribution
        order = np.argsort(np.abs(contrib))[::-1]
        return [(str(names[i]), float(contrib[i])) for i in order[:k] if contrib[i] != 0], True
    if hasattr(clf, "feature_importances_"):
        # RF: rank active features by global importance (magnitude only).
        imp = clf.feature_importances_ * (x != 0)
        order = np.argsort(imp)[::-1]
        return [(str(names[i]), float(imp[i])) for i in order[:k] if imp[i] > 0], False
    # Ensembles / boosted models over reduced dimensions have no per-input-
    # feature attribution to offer honestly — return nothing rather than lie.
    return [], False


def explain_prediction(
    issue_number: int,
    repo_name: str,
    proba: float,
    threshold: float,
    top_contributions: list[tuple[str, float]],
) -> dict[str, Any]:
    """Build the per-issue explanation record the webhook will emit."""
    return {
        "repo_name": repo_name,
        "issue_number": issue_number,
        "predicted_proba_actionable_bug": round(float(proba), 4),
        "threshold": threshold,
        "predicted_label": int(proba >= threshold),
        "top_contributions": [
            {"feature": f, "contribution": round(float(c), 4)} for f, c in top_contributions
        ],
    }


def write_explanation(explanation: dict[str, Any], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (REPORTS_DIR / "explanations")
    path = out_dir / f"{explanation['repo_name'].replace('/', '__')}__{explanation['issue_number']}.json"
    utils.write_json(path, explanation)
    return path
