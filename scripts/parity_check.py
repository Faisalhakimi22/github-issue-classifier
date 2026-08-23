"""Offline vs served, on the same issues at the same threshold.

The earlier comparison was not like-for-like: reports/champion.json scores the
whole chronological test slice, while the backtest table reports only the
later half it verifies thresholds on. This replays the full test slice through
the service and scores it exactly as the offline evaluation does.
"""
import json
import logging
from pathlib import Path

import numpy as np

logging.disable(logging.INFO)

from ghic import evaluate, train, utils  # noqa: E402
from ghic.backtest import replay  # noqa: E402
from ghic.service.settings import default_model_path  # noqa: E402

records = replay(default_model_path())

by_repo: dict[str, list[dict]] = {}
for r in records:
    by_repo.setdefault(r["repo"], []).append(r)

offline = json.load(open("reports/champion.json", encoding="utf-8"))["per_repo"]

print()
print("=" * 92)
print("  OFFLINE vs SERVED — same test slice, threshold 0.5")
print("=" * 92)
print(f"{'repository':26} {'n off':>6} {'n srv':>6} "
      f"{'F1 off':>7} {'F1 srv':>7} {'rec off':>8} {'rec srv':>8} {'AUC off':>8} {'AUC srv':>8}")
print("-" * 92)

served_all_y, served_all_p = [], []
for repo, rows in sorted(by_repo.items()):
    y = np.array([r["y_true"] for r in rows])
    p = np.array([r["proba"] for r in rows])
    served_all_y.extend(y.tolist())
    served_all_p.extend(p.tolist())
    srv = evaluate.compute_metrics(y, p, 0.5).as_dict()
    off = offline.get(repo, {})
    print(f"{repo:26} {off.get('n', 0):>6} {srv['n']:>6} "
          f"{off.get('f1', 0):>7.3f} {srv['f1']:>7.3f} "
          f"{off.get('recall', 0):>8.3f} {srv['recall']:>8.3f} "
          f"{off.get('roc_auc', 0):>8.3f} {srv['roc_auc']:>8.3f}")

y = np.array(served_all_y)
p = np.array(served_all_p)
srv = evaluate.compute_metrics(y, p, 0.5).as_dict()
champ = json.load(open("reports/champion.json", encoding="utf-8"))["test_calibrated"]
print("-" * 92)
print(f"{'OVERALL':26} {champ['n']:>6} {srv['n']:>6} "
      f"{champ['f1']:>7.3f} {srv['f1']:>7.3f} "
      f"{champ['recall']:>8.3f} {srv['recall']:>8.3f} "
      f"{champ['roc_auc']:>8.3f} {srv['roc_auc']:>8.3f}")

print()
print("Interpretation: identical n means the same issues reached the model.")
print("A gap in AUC is the model seeing different features at serving time;")
print("a gap only in recall/F1 at equal AUC would be a threshold effect.")
