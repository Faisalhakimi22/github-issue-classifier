"""Interactive demo: load the trained model and evaluate it live.

Three things it shows, in order:

  1. Held-out evaluation. Reload the saved model, rebuild the exact per-repo
     chronological test split, and print the metrics table (overall + per repo).
     These match reports/metrics.json -- proof the saved model is the real one.

  2. Worked examples. Walk through a few real test issues: show the title, the
     true label, the predicted probability, and the top features that pushed the
     prediction up or down. This is the "why did it decide that" story.

  3. Live prediction. Score an arbitrary issue typed on the command line
     (--title / --body), the same path the webhook service calls per issue.

Run (from the project root):
  python -m ghic.demo                          # full demo on the best model
  python -m ghic.demo --model rf_balanced      # pick a model
  python -m ghic.demo --examples 5             # number of worked examples
  python -m ghic.demo --title "App crashes on startup" \
                     --body "Steps to reproduce: 1. open 2. crash. Stack trace..."

Nothing here retrains or hits the API. It reads models/*.joblib and the
processed dataset only.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd

from . import evaluate, features, train, utils
from .config import Config, get_config

logger = utils.get_logger(__name__)

DEFAULT_MODEL = "rf_balanced"
_BAR = "=" * 72


def _load_model(name: str) -> Any:
    import joblib
    path = train.MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        avail = sorted(p.stem for p in train.MODELS_DIR.glob("*.joblib"))
        raise SystemExit(
            f"Model '{name}' not found at {path}.\n"
            f"Available: {avail or '(none -- run `python -m ghic.train` first)'}"
        )
    return joblib.load(path)


def _rebuild_test_split(cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    """Reproduce train.py's test split exactly, so metrics line up."""
    labeled = utils.DATA_PROCESSED / "labeled.csv"
    if not labeled.exists():
        raise SystemExit(
            f"Missing {labeled}. Run `python -m ghic.label` (or notebook 01) first."
        )
    df = train.load_labeled(labeled)
    feats = features.engineer_features(
        df, cfg, repo_release_dates=train._load_release_dates()
    )
    _, test_feats = train.chronological_split(feats)
    y_test = test_feats["label"].to_numpy()
    return test_feats, y_test


# ---------------------------------------------------------------------------
# Section 1: held-out evaluation
# ---------------------------------------------------------------------------
def show_evaluation(pipe: Any, name: str, test_feats: pd.DataFrame, y_test: np.ndarray,
                    threshold: float) -> np.ndarray:
    proba = pipe.predict_proba(test_feats)[:, 1]
    overall = evaluate.compute_metrics(y_test, proba, threshold)
    per_repo = train._evaluate_per_repo(pipe, test_feats, y_test, threshold)

    print(_BAR)
    print(f"  1. HELD-OUT EVALUATION  --  model: {name}  (threshold {threshold})")
    print(_BAR)
    print(f"Test set: {overall.n} issues, {overall.positives} actual Actionable Bugs "
          f"({overall.positives / overall.n:.1%} positive)\n")
    print("Overall:")
    print(evaluate.metrics_table({name: overall}))
    print("\nPer repository (does it generalize across codebases?):")
    print(evaluate.metrics_table(per_repo))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("             pred 0   pred 1")
    print(f"  true 0  {overall.tn:8d} {overall.fp:8d}")
    print(f"  true 1  {overall.fn:8d} {overall.tp:8d}")
    print(f"\nReading it: of {overall.tp + overall.fn} real bugs the model caught "
          f"{overall.tp} (recall {overall.recall:.2f}); of {overall.tp + overall.fp} "
          f"it flagged, {overall.tp} were right (precision {overall.precision:.2f}).")
    return proba


# ---------------------------------------------------------------------------
# Section 2: per-issue explanation (why this prediction?)
# ---------------------------------------------------------------------------
def _top_contributions(pipe: Any, row: pd.DataFrame, k: int = 6) -> tuple[list[tuple[str, float]], bool]:
    """Shared with the webhook service; see evaluate.top_contributions."""
    return evaluate.top_contributions(pipe, row, k)


def _print_contributions(contribs: list[tuple[str, float]], signed: bool) -> None:
    if not contribs:
        print("    (no non-zero contributing features)")
        return
    if signed:
        for feat, val in contribs:
            arrow = "+ toward Bug    " if val > 0 else "- toward Non-Bug"
            print(f"    {arrow}  {val:+.3f}  {feat}")
    else:
        print("    (Random Forest: most influential active features, by importance;")
        print("     importances are magnitudes, not signed directions)")
        for feat, val in contribs:
            print(f"      importance {val:.3f}  {feat}")


def show_examples(pipe: Any, name: str, test_feats: pd.DataFrame, y_test: np.ndarray,
                  proba: np.ndarray, threshold: float, n: int) -> None:
    print("\n" + _BAR)
    print("  2. WORKED EXAMPLES  --  real test issues and why the model scored them")
    print(_BAR)

    # Pick a varied, convincing set: most confident correct bug, most confident
    # correct non-bug, and the worst miss -- so the demo shows wins and a failure.
    pred = (proba >= threshold).astype(int)
    correct = pred == y_test
    picks: list[tuple[str, int]] = []
    pos_correct = np.where(correct & (y_test == 1))[0]
    neg_correct = np.where(correct & (y_test == 0))[0]
    wrong = np.where(~correct)[0]
    if len(pos_correct):
        picks.append(("correct: caught a real bug", pos_correct[np.argmax(proba[pos_correct])]))
    if len(neg_correct):
        picks.append(("correct: correctly dismissed", neg_correct[np.argmin(proba[neg_correct])]))
    if len(wrong):
        picks.append(("wrong: the model's biggest miss",
                      wrong[np.argmax(np.abs(proba[wrong] - y_test[wrong]))]))
    # Top up with more confident-correct positives if the user asked for more.
    for idx in pos_correct[np.argsort(proba[pos_correct])[::-1]]:
        if len(picks) >= n:
            break
        if idx not in [p[1] for p in picks]:
            picks.append(("correct: caught a real bug", idx))
    picks = picks[:n]

    reset = test_feats.reset_index(drop=True)
    for tag, i in picks:
        row = reset.iloc[[i]]
        repo = row["repo_name"].iloc[0]
        num = row["number"].iloc[0]
        title = str(row["text_combined"].iloc[0]).splitlines()[0][:90]
        true_lab = "Actionable Bug" if y_test[i] == 1 else "Non-Actionable"
        pred_lab = "Actionable Bug" if pred[i] == 1 else "Non-Actionable"
        verdict = "RIGHT" if pred[i] == y_test[i] else "WRONG"
        print(f"\n[{tag}]")
        print(f"  {repo} #{num}: {title}")
        print(f"  true = {true_lab:16s} predicted = {pred_lab:16s} "
              f"P(bug) = {proba[i]:.3f}  -> {verdict}")
        print(f"  top features for this issue ({name}):")
        items, signed = _top_contributions(pipe, row)
        _print_contributions(items, signed)


# ---------------------------------------------------------------------------
# Section 3: live prediction on a typed-in issue
# ---------------------------------------------------------------------------
def predict_one(pipe: Any, name: str, cfg: Config, title: str, body: str,
                threshold: float, repo_name: str = "live/demo") -> None:
    print("\n" + _BAR)
    print(f"  3. LIVE PREDICTION  --  scoring a new issue ({name})")
    print(_BAR)
    raw = pd.DataFrame([{
        "repo_name": repo_name,
        "number": 0,
        "title": title,
        "body": body,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "author_created_at": None,
        "author_public_repos": np.nan,
        "author_followers": np.nan,
        "author_login": "demo-user",
        "label": 0,
    }])
    feats = features.engineer_features(raw, cfg, repo_release_dates=train._load_release_dates())
    proba = float(pipe.predict_proba(feats)[:, 1][0])
    pred = "Actionable Bug" if proba >= threshold else "Non-Actionable"
    print(f"\n  Title: {title}")
    print(f"  Body : {body[:120]}{'...' if len(body) > 120 else ''}")
    print(f"\n  -> P(Actionable Bug) = {proba:.3f}   prediction = {pred} "
          f"(threshold {threshold})")
    print("  top features driving this prediction:")
    items, signed = _top_contributions(pipe, feats)
    _print_contributions(items, signed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Demo: evaluate the trained issue-outcome model live.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"model name in models/ (default: {DEFAULT_MODEL})")
    parser.add_argument("--examples", type=int, default=3,
                        help="number of worked examples to show (default: 3)")
    parser.add_argument("--threshold", type=float, default=evaluate.DEFAULT_THRESHOLD)
    parser.add_argument("--title", default=None, help="title of a custom issue to score")
    parser.add_argument("--body", default="", help="body of the custom issue")
    parser.add_argument("--no-eval", action="store_true",
                        help="skip the held-out evaluation section")
    parser.add_argument("--no-examples", action="store_true",
                        help="skip the worked-examples section")
    args = parser.parse_args(argv)

    cfg = get_config(require_token=False)
    pipe = _load_model(args.model)

    if not (args.no_eval and args.no_examples):
        test_feats, y_test = _rebuild_test_split(cfg)

    if not args.no_eval:
        proba = show_evaluation(pipe, args.model, test_feats, y_test, args.threshold)
    if not args.no_examples:
        if args.no_eval:
            proba = pipe.predict_proba(test_feats)[:, 1]
        show_examples(pipe, args.model, test_feats, y_test, proba,
                      args.threshold, max(1, args.examples))

    if args.title:
        predict_one(pipe, args.model, cfg, args.title, args.body, args.threshold)
    else:
        print("\n(tip: add --title \"...\" --body \"...\" to score your own issue live)")

    print("\n" + _BAR)
    print("  demo complete.")
    print(_BAR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
