# Model registry

One row per retraining run (`python -m ghic.retrain`). Snapshots live
in `reports/runs/<timestamp>/`; artifacts are content-addressed by
sha256 so a deployed model can always be traced to its run.

| run (UTC) | champion | test PR-AUC (cal) | category macro-F1 | champion sha256 | snapshot |
|---|---|---|---|---|---|
| 2026-07-13T19-41-30Z | `rf_balanced` | 0.7172 | 0.4704 | `5538c3115ba7` | `reports/runs/2026-07-13T19-41-30Z/` |
