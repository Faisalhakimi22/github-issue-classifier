# Benchmarks — every number here was produced by a gate

Nothing on this page is projected, extrapolated, or invented. Each section
names the artifact that produced it (all regenerable from source), and the
negative results are listed with the same prominence as the positive ones —
they carry the same information value.

Dataset for all model numbers: 6,175 closed issues from microsoft/vscode,
facebook/react, tensorflow/tensorflow (calendar 2024); chronological
per-repo 80/20 split; every evaluation below is a single look at the
untouched test slice after walk-forward selection inside the training
window.

## 1. Actionability champion (the core model)

Source: `models/MODEL_CARD.md`, `reports/champion.json`
(`python -m ghic.train --champion`).

Walk-forward CV (mean PR-AUC ± std) — selection, training window only:

| candidate | mean PR-AUC | std |
|---|---|---|
| `rf_balanced` ← **champion** | 0.7669 | 0.0102 |
| `ensemble` (LR+RF+HGB soft vote) | 0.7582 | 0.0058 |
| `svd_hgb` | 0.7256 | 0.0126 |
| `logreg_balanced` | 0.7102 | 0.0210 |

Honest finding: the fancier candidates did **not** beat the plain balanced
random forest.

Final test set (n=1,177, threshold 0.5):

| metric | uncalibrated | isotonic-calibrated |
|---|---|---|
| precision | 0.816 | 0.798 |
| recall | 0.544 | 0.575 |
| F1 | 0.653 | 0.668 |
| ROC-AUC | 0.887 | 0.881 |
| PR-AUC | 0.759 | 0.717 |
| Brier | 0.101 | 0.104 |

## 2. Service-path backtest (the deployment-shaped number)

Source: `reports/backtest.json` (`python -m ghic.backtest` — replays all
1,177 held-out issues through the real webhook endpoint, signed payloads,
real feature path).

- Overall: **ROC-AUC 0.902**, PR-AUC 0.773, F1 0.655 at the default 0.5
  threshold.
- Calibrated per-repo thresholds: react **0.38**, vscode **0.30**,
  tensorflow **0.40** (`GHIC_REPO_THRESHOLDS`).
- Known weak spot, stated: vscode recall at the default threshold is poor
  (0.10 at 0.5; the lower per-repo threshold exists precisely for this).

## 3. Category head (bug / feature / question / docs / duplicate / invalid)

Source: `models/CATEGORY_CARD.md`, `reports/category.json`
(`python -m ghic.category --train`); ground truth = the category label a
maintainer eventually applied (2,747 issues).

Test (n=549): accuracy **0.583**, macro-F1 **0.470**.

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| bug | 0.813 | 0.603 | 0.692 | 282 |
| feature | 0.621 | 0.678 | 0.648 | 87 |
| question | 0.460 | 0.505 | 0.482 | 91 |
| duplicate | 0.346 | 0.545 | 0.424 | 66 |
| docs | 0.316 | 0.545 | 0.400 | 11 |
| invalid | 0.136 | 0.250 | 0.176 | 12 |

`security` (0 corpus occurrences) and standalone `regression` (27) are not
trainable in this corpus — documented on the card, not faked.

## 4. Duplicate detection — negative result

Source: `models/DUPLICATE_CARD.md`, `reports/dupdetect.json`
(`python -m ghic.dupdetect --evaluate`).

Task: predict duplicate-labeled closure from max similarity to prior
same-repo issues (test n=1,177, 60 positives, validated on vscode only —
duplicate labels barely exist elsewhere).

| representation | ROC-AUC (overall) | PR-AUC (vs 7.3% base rate) |
|---|---|---|
| TF-IDF cosine | 0.555 | 0.058 |
| MiniLM (all-MiniLM-L6-v2) | 0.527 | 0.056 |

**Near chance, and embeddings don't beat TF-IDF.** Consequence applied:
similarity candidates ship as assistive "possibly related" context only; no
duplicate flag exists anywhere in the product.

## 5. Effort head — shipped after clearing a pre-declared bar

Source: `models/EFFORT_CARD.md`, `reports/effort.json`
(`python -m ghic.effort --evaluate`). Proxy: log1p(days-to-close),
declared weak up front; ship bar (Spearman ≥ 0.30 AND ≥ 10% MAE
improvement) declared before the first run.

Test (n=1,227): Spearman **0.492**, MAE 1.477 log-days vs baseline 1.674
(**11.8%** improvement) — bar met. Ships as four coarse buckets in the API
response only.

## 6. Priority & severity — correctly not shipped

Source: `models/PRIORITY_CARD.md`, `models/SEVERITY_CARD.md`. No ground
truth exists in the corpus (zero priority/severity labels), and every
candidate proxy fails for a documented reason (the severity keyword-proxy
is circular: it would train the model to detect its own labeling rule).
No numbers exist because none could honestly be produced.

## 7. Maintainer assignment — similarity recommender beats the baseline

Source: `models/ASSIGNMENT_CARD.md`, `reports/assign.json`
(`python -m ghic.assign --collect && --evaluate`, 2026-07-15). Ground
truth: the issue's eventual human assignee(s); 968 test issues scored.

| recommender | hit@1 | hit@3 | hit@5 |
|---|---|---|---|
| similarity (assignees/closers of 20 most similar priors) | **0.115** | **0.389** | **0.485** |
| most-active-assignee baseline (causal) | 0.000 | 0.261 | 0.406 |

Per the plan's decision rule, the similarity mechanism ships (a learned
ranker would have to beat 0.389 hit@3 on this protocol first). Suggestions
are response-level only — the service never assigns anyone and never
@-mentions suggested names in comments. Known caveats on the card:
assignee state is close-time (slightly acausal for late assignments), and
"most-active-committer" is approximated by most-active-assignee.

## 8. Service performance — measured, single uvicorn worker, Windows dev box

Source: `reports/loadtest*.json` (`python scripts/loadtest.py` against the
live service, calibrated champion, 2026-07-13).

| scenario | throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| 30 requests, concurrency 1 | 1.68 req/s | 596 ms | 626 ms | 637 ms |
| 200 requests, concurrency 8 | 1.65 req/s | 4,811 ms | 5,309 ms | 5,875 ms |

Reading: one prediction costs ~600 ms CPU on this path; a single worker
serializes concurrent requests (flat throughput, queueing latency). Ample
for any single repo's issue rate and GitHub's 10 s webhook deadline;
scaling levers documented in `docs/PRD.md` §8.

**Full webhook path, all heads enabled** — measured from the 2026-07-14
retraining run's backtest, which replays all 1,177 held-out issues through
the real `/webhook` endpoint with the champion + category + effort heads
and the duplicate index loaded (`reports/webhook_latency_allheads.json`;
in-process TestClient, sequential, dry-run):

| n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| 1,177 | 357 ms | 300 ms | 683 ms | 988 ms | 2,367 ms |

Why this is *faster* than the single-head `/api/predict` row above: the API
path computes per-feature explanations (`explain=True`), which dominates
its ~600 ms; the webhook path in dry-run skips explanations, and the three
extra heads plus the cosine lookup together cost less than that. Numbers
are honest to their configurations, which is why both are stated.
