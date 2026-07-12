"""Backtest the production webhook against held-out real issues — in minutes,
not a week of live soak testing.

What it does:
  1. Rebuilds the exact chronological test split (the issues the model never
     trained on).
  2. Replays every one of them through the REAL service path: a signed
     `issues.opened` webhook POST into the FastAPI app, author + release
     enrichment served from the dataset (standing in for the GitHub API),
     feature engineering, model, decision. This is the request production
     will receive, byte-for-byte in shape.
  3. Scores the service's answers against the known ground-truth labels.
  4. Calibrates a per-repo decision threshold on the FIRST half of each
     repo's test slice and verifies it on the SECOND half (never tuned and
     evaluated on the same issues), then prints the exact GHIC_* env lines
     to deploy with.

Because the replay goes through the service (not the offline eval code), it
also exercises the serving-time degradations — e.g. first_time_contributor
collapsing to 1 for single issues — so the numbers it reports are what the
deployed bot will actually do, which the offline metrics in reports/ are not.

Run:
  python -m ghic.backtest                 # full replay + calibration report
  python -m ghic.backtest --limit 200     # quick smoke (subsample per repo)
Requires: pip install -e ".[service,dev]"  (fastapi + httpx) and a trained
model + data/processed/labeled.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import evaluate, train, utils
from .service.inference import IssuePredictor
from .service.settings import ServiceSettings

logger = utils.get_logger(__name__)

REPLAY_SECRET = "backtest-replay-secret"
THRESHOLD_GRID = np.round(np.arange(0.10, 0.91, 0.02), 2)


# ---------------------------------------------------------------------------
# Replay enrichment: serves what the GitHub API would, from the dataset
# ---------------------------------------------------------------------------
class ReplayGitHub:
    """Stands in for GitHubAppClient. Author fields come from the collected
    dataset (the same snapshot the API would return); the latest release is
    the newest one at or before the issue's creation time, from releases.json.
    Write methods refuse loudly — a backtest must never touch GitHub."""

    def __init__(self, releases: dict[str, list[str]] | None) -> None:
        self.releases = releases or {}
        self.current_row: dict[str, Any] = {}

    def get_user(self, login: str, installation_id: int) -> dict[str, Any]:
        row = self.current_row
        return {
            "created_at": _clean(row.get("author_created_at")),
            "public_repos": _clean_int(row.get("author_public_repos")),
            "followers": _clean_int(row.get("author_followers")),
        }

    def get_latest_release_date(self, full_name: str, installation_id: int) -> str | None:
        created = self.current_row.get("created_at")
        prior = [r for r in self.releases.get(full_name, []) if r <= created]
        return prior[-1] if prior else None

    def post_comment(self, *a: Any, **k: Any) -> None:
        raise AssertionError("backtest tried to write to GitHub")

    add_labels = post_comment


def _clean(v: Any) -> Any:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _clean_int(v: Any) -> int | None:
    v = _clean(v)
    return None if v is None else int(v)


# ---------------------------------------------------------------------------
# Replay loop
# ---------------------------------------------------------------------------
def _sign(body: bytes) -> str:
    return "sha256=" + hmac.new(REPLAY_SECRET.encode(), body, hashlib.sha256).hexdigest()


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": "opened",
        "issue": {
            "number": int(row["number"]),
            "title": str(_clean(row.get("title")) or ""),
            "body": str(_clean(row.get("body")) or ""),
            "created_at": row["created_at"],
            "user": {"login": str(_clean(row.get("author_login")) or "unknown")},
        },
        "repository": {"full_name": row["repo_name"]},
        "installation": {"id": 1},
    }


def replay(model_path: Path, limit_per_repo: int | None = None) -> list[dict[str, Any]]:
    """Send every test-split issue through the service; return scored records."""
    from fastapi.testclient import TestClient

    from .service.app import create_app

    labeled = utils.DATA_PROCESSED / "labeled.csv"
    if not labeled.exists():
        raise SystemExit(f"Missing {labeled} — run `python -m ghic.label` first.")
    df = train.load_labeled(labeled)
    _, test_df = train.chronological_split(df)
    if limit_per_repo:
        test_df = test_df.groupby("repo_name", group_keys=False).head(limit_per_repo)

    settings = ServiceSettings(
        model_path=model_path,
        webhook_secret=REPLAY_SECRET,
        dry_run=True,
    )
    gh = ReplayGitHub(train._load_release_dates())
    predictor = IssuePredictor(model_path, settings.threshold)
    client = TestClient(create_app(settings, predictor=predictor, gh_client=gh))

    records: list[dict[str, Any]] = []
    started = time.time()
    rows = test_df.to_dict("records")
    logger.info("replaying %d held-out issues through the webhook ...", len(rows))
    for i, row in enumerate(rows, 1):
        gh.current_row = row
        body = json.dumps(_payload(row)).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "issues",
                "X-Hub-Signature-256": _sign(body),
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        if "prediction" not in data:            # e.g. author login ends in -bot
            continue
        records.append({
            "repo": row["repo_name"],
            "number": int(row["number"]),
            "created_at": row["created_at"],
            "y_true": int(row["label"]),
            "proba": float(data["prediction"]["proba_actionable_bug"]),
        })
        if i % 200 == 0:
            logger.info("  %d/%d (%.0fs elapsed)", i, len(rows), time.time() - started)
    logger.info("replay done: %d scored, %.0fs", len(records), time.time() - started)
    return records


# ---------------------------------------------------------------------------
# Calibration: tune on the first half per repo, verify on the second half
# ---------------------------------------------------------------------------
def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold on the grid maximizing F1; 0.5 if the slice is degenerate."""
    if len(np.unique(y)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        m = evaluate.compute_metrics(y, p, float(t))
        if m.f1 > best_f1:
            best_t, best_f1 = float(t), m.f1
    return best_t


def calibrate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-repo: pick a threshold on the earlier half, report both halves."""
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_repo.setdefault(r["repo"], []).append(r)

    out: dict[str, Any] = {"repos": {}, "overall": {}}
    all_y = np.array([r["y_true"] for r in records])
    all_p = np.array([r["proba"] for r in records])

    for repo, rows in sorted(by_repo.items()):
        rows.sort(key=lambda r: r["created_at"])       # keep time order
        cut = len(rows) // 2
        cal, val = rows[:cut], rows[cut:]
        y_cal = np.array([r["y_true"] for r in cal])
        p_cal = np.array([r["proba"] for r in cal])
        y_val = np.array([r["y_true"] for r in val])
        p_val = np.array([r["proba"] for r in val])

        t = best_threshold(y_cal, p_cal)
        out["repos"][repo] = {
            "n": len(rows),
            "recommended_threshold": t,
            "val_at_default": evaluate.compute_metrics(y_val, p_val, 0.5).as_dict(),
            "val_at_recommended": evaluate.compute_metrics(y_val, p_val, t).as_dict(),
        }

    out["overall"] = {
        "n": len(records),
        "at_default": evaluate.compute_metrics(all_y, all_p, 0.5).as_dict(),
    }
    return out


def format_report(result: dict[str, Any]) -> str:
    lines = ["", "=" * 76, "  BACKTEST — held-out issues replayed through the production webhook",
             "=" * 76]
    o = result["overall"]["at_default"]
    lines.append(
        f"\nOverall (threshold 0.5): n={o['n']}  precision={o['precision']:.3f}  "
        f"recall={o['recall']:.3f}  f1={o['f1']:.3f}  roc_auc={o['roc_auc']:.3f}"
    )
    lines.append("\nPer-repo calibration (tuned on earlier half, verified on later half):")
    lines.append(f"{'repository':30s} {'n':>5} {'thr':>5} {'F1@0.5':>8} {'F1@thr':>8} {'rec@thr':>8}")
    lines.append("-" * 76)
    env_parts = []
    for repo, r in result["repos"].items():
        d, v = r["val_at_default"], r["val_at_recommended"]
        lines.append(
            f"{repo:30s} {r['n']:5d} {r['recommended_threshold']:5.2f} "
            f"{d['f1']:8.3f} {v['f1']:8.3f} {v['recall']:8.3f}"
        )
        env_parts.append(f"{repo}={r['recommended_threshold']}")
    lines += [
        "",
        "Deploy with:",
        f"  GHIC_REPO_THRESHOLDS={','.join(env_parts)}",
        "",
        "Notes: numbers here are the SERVICE's behavior (single-issue feature",
        "degradations included), so they are the deployment truth — expect them",
        "to differ slightly from the offline numbers in reports/metrics.json.",
        "For repos outside the training set, start from the global threshold",
        "and re-run this backtest after collecting their issues.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay held-out issues through the webhook and calibrate thresholds."
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="fitted .joblib to backtest (default: champion, else rf_balanced)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="max issues per repo (quick smoke run)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=evaluate.REPORTS_DIR / "backtest.json",
    )
    args = parser.parse_args(argv)

    if args.model is None:
        from .service.settings import default_model_path

        args.model = default_model_path()
    if not args.model.exists():
        logger.error("Model not found: %s — run `python -m ghic.train` first.", args.model)
        return 1

    records = replay(args.model, limit_per_repo=args.limit)
    result = calibrate(records)
    utils.write_json(args.output, result)
    logger.info("wrote %s", args.output)
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
