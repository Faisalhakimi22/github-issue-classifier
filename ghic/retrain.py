"""Retraining pipeline: one command from labeled data to registered models.

Runs the stages that already exist, in dependency order, then snapshots the
run so model history is versioned:

  1. label      — refresh labeled.csv from combined.csv (rules may change)
  2. champion   — walk-forward selection + calibration -> champion.joblib
  3. backtest   — held-out replay through the real webhook + threshold calib
  4. category   — the multi-class category head
  5. dup index  — rebuild the serving similarity index
  6. snapshot   — copy this run's cards + metrics to reports/runs/<utc-ts>/
                  and append a row to models/REGISTRY.md

Versioning philosophy (stated because Phase 20 asks): at one model family
and one operator, a experiment-tracking server is process for its own sake.
The registry row (when / what won / headline metric / artifact hash) plus
the immutable per-run snapshot directory answer every question a registry
answers — "what changed, when, and what did it score" — with zero infra.
Revisit when several people train concurrently.

Scheduling: this is operator-triggered by design. The training data
(data/processed/*) is deliberately not in the repo, so CI cannot retrain;
a cron/scheduled-Actions wrapper is a deployment choice documented in
docs/DEPLOYMENT.md, not code shipped here.

CLI:
  python -m ghic.retrain            # full pipeline (~25 min, mostly champion CV)
  python -m ghic.retrain --quick    # skip backtest replay (fast iteration)
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import evaluate, utils

logger = utils.get_logger(__name__)

RUNS_DIR = evaluate.REPORTS_DIR / "runs"
REGISTRY_PATH = utils.PROJECT_ROOT / "models" / "REGISTRY.md"
MODELS_DIR = utils.PROJECT_ROOT / "models"

_REGISTRY_HEADER = [
    "# Model registry",
    "",
    "One row per retraining run (`python -m ghic.retrain`). Snapshots live",
    "in `reports/runs/<timestamp>/`; artifacts are content-addressed by",
    "sha256 so a deployed model can always be traced to its run.",
    "",
    "| run (UTC) | champion | test PR-AUC (cal) | category macro-F1 | champion sha256 | snapshot |",
    "|---|---|---|---|---|---|",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_run(ts: str) -> Path:
    """Copy this run's cards and metric reports into an immutable run dir."""
    run_dir = RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("MODEL_CARD.md", "CATEGORY_CARD.md"):
        src = MODELS_DIR / name
        if src.exists():
            shutil.copy2(src, run_dir / name)
    for name in ("champion.json", "category.json", "backtest.json"):
        src = evaluate.REPORTS_DIR / name
        if src.exists():
            shutil.copy2(src, run_dir / name)
    logger.info("snapshot -> %s", run_dir)
    return run_dir


def append_registry(ts: str, champion: dict[str, Any], category: dict[str, Any] | None) -> None:
    champ_path = MODELS_DIR / "champion.joblib"
    sha = _sha256(champ_path)[:12] if champ_path.exists() else "missing"
    pr_auc = champion.get("test_calibrated", {}).get("pr_auc")
    macro = (category or {}).get("test", {}).get("macro_f1")
    row = (f"| {ts} | `{champion.get('winner', '?')}` "
           f"| {'%.4f' % pr_auc if pr_auc is not None else 'n/a'} "
           f"| {'%.4f' % macro if macro is not None else 'n/a'} "
           f"| `{sha}` | `reports/runs/{ts}/` |")
    if REGISTRY_PATH.exists():
        content = REGISTRY_PATH.read_text(encoding="utf-8").rstrip("\n")
    else:
        content = "\n".join(_REGISTRY_HEADER)
    REGISTRY_PATH.write_text(content + "\n" + row + "\n", encoding="utf-8")
    logger.info("registry += %s", row)


def run(quick: bool = False) -> dict[str, Any]:
    from . import category as category_mod
    from . import dupdetect, label, train

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    logger.info("retraining run %s (quick=%s)", ts, quick)

    # 1. labels
    rc = label.main([])
    if rc != 0:
        raise RuntimeError("labeling failed — is data/processed/combined.csv present?")
    # 2. champion protocol
    champion = train.train_champion()
    # 3. service-path backtest (skippable for fast iterations)
    if not quick:
        from . import backtest

        backtest.main([])
    # 4. category head
    category = category_mod.train_category()
    # 5. serving similarity index
    dupdetect.build_index()
    # 6. snapshot + registry
    run_dir = snapshot_run(ts)
    append_registry(ts, champion, category)
    return {"run": ts, "snapshot": str(run_dir),
            "champion": champion.get("winner"),
            "pr_auc": champion.get("test_calibrated", {}).get("pr_auc")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full retraining pipeline + registry.")
    parser.add_argument("--quick", action="store_true",
                        help="skip the backtest replay stage")
    args = parser.parse_args(argv)
    result = run(quick=args.quick)
    print(f"retrained: run={result['run']} champion={result['champion']} "
          f"pr_auc={result['pr_auc']}")
    return 0


if __name__ == "__main__":
    from ghic.retrain import main as _main

    sys.exit(_main())
