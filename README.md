# GitHub Issue Triage Bot

[![CI](https://github.com/Faisalhakimi22/github-issue-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Faisalhakimi22/github-issue-classifier/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)

**The moment an issue is opened, this GitHub App predicts whether it will end
as an actionable bug — fixed by a merged PR — or die as a duplicate,
a won't-fix, or silence.** Maintainers of busy repos triage hundreds of
reports a week; most never lead to a code change. This bot reads the issue
the way a triager would — is there a stack trace? reproduction steps? who is
the author? — and posts a calibrated probability before a human has spent a
minute on it.

```
issue opened ──▶ webhook (HMAC-verified) ──▶ enrich ──▶ featurize ──▶ score
                                                                        │
              P(actionable bug) = 0.87 ────────────────────────────────┤
              likely duplicates of prior issues (assistive) ───────────┤
              "missing info" draft when under-specified (LLM, scoped) ─┤
                                                                        ▼
                                              comment + label on the issue
issue closed ──▶ outcome derived from close labels ──▶ the bot grades its
                                                       own prediction, live
```

Three properties distinguish it from the usual classifier-behind-a-webhook:

1. **No leakage, anywhere.** The model sees only what exists at the moment
   an issue is opened. Label counts are excluded (a new issue has zero
   labels in production), splits are chronological (train on the past,
   predict the future), the contributor-history feature is computed
   causally, and the TF-IDF vocabulary is fit on the training window only.
2. **Calibrated probabilities, verified end-to-end.** The shipped model is
   selected by walk-forward temporal cross-validation and isotonically
   calibrated — and a replay of all 1,177 held-out issues through the real
   webhook shows it: facebook/react's empirically optimal decision threshold
   lands at exactly 0.50. When this bot says 0.7, it means roughly 70%.
3. **It grades itself in production.** Every prediction enters a
   restart-safe ledger; when the issue is eventually closed, the service
   derives the true outcome from the close payload using the same rules
   that built the training set, and `GET /stats` reports live precision and
   recall. Offline metrics are a claim; this is the receipt.

## Quick start

```bash
pip install -e ".[service]"

# score an issue locally, no GitHub App required
GHIC_ALLOW_UNSIGNED=true python -m ghic.service.app
curl -X POST localhost:8000/api/predict -H "Content-Type: application/json" \
  -d '{"title": "Crash on startup", "body": "Steps to reproduce: ..."}'
```

Production runs as a GitHub App in Docker — registration, staged rollout,
and the Marketplace checklist are in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md):

```bash
docker build -t ghic .
docker run -p 8000:8000 -v ghic-data:/app/data \
  -e GHIC_WEBHOOK_SECRET=... -e GHIC_APP_ID=... \
  -e GHIC_PRIVATE_KEY="$(cat private-key.pem)" \
  -e GHIC_DRY_RUN=false -e GHIC_POST_COMMENT=true \
  ghic
```

**A fresh deploy cannot hurt anyone.** `GHIC_DRY_RUN=true` is the default:
the service scores and logs but writes nothing to GitHub until the operator
explicitly enables actions.

| Variable | Default | Purpose |
|---|---|---|
| `GHIC_WEBHOOK_SECRET` | — required | HMAC secret; unsigned requests are rejected |
| `GHIC_APP_ID` / `GHIC_PRIVATE_KEY(_PATH)` | — | GitHub App credentials |
| `GHIC_MODEL_PATH` | `champion.joblib`, else `rf_balanced` | fitted pipeline to serve |
| `GHIC_THRESHOLD` | `0.5` | global decision threshold |
| `GHIC_REPO_THRESHOLDS` | — | per-repo overrides; produced by `ghic-backtest` |
| `GHIC_DRY_RUN` | `true` | score and log only |
| `GHIC_POST_COMMENT` / `GHIC_APPLY_LABEL` | `false` | write actions, individually gated |
| `GHIC_ENRICH` | `true` | fetch author profile + latest release |
| `GHIC_LEDGER` | `data/predictions.jsonl` | online-evaluation ledger (`""` = in-memory) |
| `GHIC_SUGGEST_RELATED` | `true` | surface likely-duplicate prior issues (needs `dup_index.joblib`) |
| `GHIC_DRAFT_MISSING_INFO` | `false` | LLM-drafted info request on vague issues (template without a key) |

## How the ground truth was built

There is no human-annotated dataset here — the labels come from
deterministic rules over what actually happened to 5,885 closed issues from
microsoft/vscode, facebook/react, and tensorflow/tensorflow (all of 2024).
Rules fire in priority order; an audit histogram reports how many issues hit
each rule, so every label is traceable to a reason:

| Rule | Class | Trigger |
|---|---|---|
| R1 | bug | GitHub recorded a merged PR as the closer |
| R1b | bug | a merged PR body says `Fixes #N` / `Closes #N` / `Resolves #N` |
| R2 | bug | bug/defect/regression label **and** closed as completed |
| R3 | non-actionable | duplicate / invalid / wontfix / cannot-reproduce label |
| R3b | non-actionable | GitHub's own `NOT_PLANNED` close reason |
| R4 | non-actionable | conservative default: closed with no fix signal |

Bots, question-labeled issues, and deleted authors are dropped — support
requests are a different population, not a third class. Class 1 base rate:
27.8%, so accuracy is a vanity metric here; everything is reported as
precision/recall/F1/PR-AUC.

## How the model is chosen

`python -m ghic.train --champion` runs a protocol in which **no decision
ever touches the final test set**:

1. **Walk-forward temporal CV.** Four candidates — balanced logistic
   regression, balanced random forest, gradient boosting over an LSA-256
   reduction, and a soft-voting ensemble — compete on mean PR-AUC across
   three expanding time folds inside the training window. Every fold trains
   on the past and validates on the future, because that is the only
   direction deployment runs.
2. **Isotonic calibration** on the newest 15% of the training window
   (still older than every test issue).
3. **One evaluation** on the untouched chronological test set, then
   [`models/MODEL_CARD.md`](models/MODEL_CARD.md) is auto-generated with the
   protocol, CV table, metrics (including Brier score), and limitations.

Features: word TF-IDF (5k, uni+bigrams) **and** char 3–5-gram TF-IDF (stack
traces, version strings, identifiers), plus 19 structured signals — text
shape, reproduction-keyword hits, author account age/repos/followers,
causal first-time-contributor flag, cyclical open-time encodings, and days
since the repo's last release.

### Results, including the negative ones

| Candidate (walk-forward CV) | mean PR-AUC | std |
|---|---|---|
| **Random Forest (balanced)** ← champion | **0.767** | 0.010 |
| Soft-voting ensemble (LR+RF+HGB) | 0.758 | 0.006 |
| HistGradientBoosting over LSA-256 | 0.726 | 0.013 |
| LogReg (balanced) | 0.710 | 0.021 |

Three findings we report because they cost us something to learn:

- **The fancy models lost.** Gradient boosting and the ensemble did not
  beat a balanced random forest under temporal CV, so the random forest is
  what ships. Complexity has to pay rent.
- **A "reasonable" filter nearly poisoned the dataset.** Our first config
  dropped locked issues; the audit revealed vscode auto-locks 90.8% of its
  closed issues as routine hygiene (react: 0.8%). That one filter silently
  deleted 61% of one repo's data and most of its merged-PR signal. Only the
  per-rule audit caught it — the model would have scored fine on its own
  broken test set.
- **One threshold is the wrong number of thresholds.** The same 0.5 cutoff
  yields F1 0.85 on tensorflow and 0.17 on vscode's most recent issues —
  the score distributions sit differently per repo. Hence per-repo
  calibration below.

Service-path replay of the full held-out set: ROC-AUC 0.885, precision 0.80
at threshold 0.5 (`reports/backtest_champion.json`). The uncalibrated v1
model keeps a small ranking edge (0.902) from seeing 15% more training data
and ships alongside the champion; we default to calibrated probabilities
because comments and thresholds depend on them meaning what they say.

## Beyond the classifier

**Duplicate candidates** (`ghic/dupdetect.py`): every new issue is compared
against all prior same-repo issues by exact cosine search — at ~6k issues a
normalized matrix product beats any vector database, and the query interface
is the seam where an ANN index would slot in at 100× the corpus. Candidates
above `GHIC_RELATED_MIN_SIM` are surfaced in the comment as *assistive*
suggestions, never acted on automatically — and the evaluation is why:
`python -m ghic.dupdetect --evaluate` tested both MiniLM embeddings and a
TF-IDF baseline causally against rule-derived duplicate labels, and **both
came out near chance** at predicting duplicate closure (ROC ≈ 0.53, with
MiniLM failing to beat TF-IDF). So no "likely duplicate" flag ships off this
score; the full negative result, mechanism, and the ground-truth work that
would change it are in
[models/DUPLICATE_CARD.md](models/DUPLICATE_CARD.md).

**Scoped LLM drafting** (`ghic/service/drafting.py`): when a deterministic,
tested trigger says an issue is under-specified (no repro steps, no
trace/code, near-empty body), Claude drafts the "could you add…" comment a
triager would write, grounded in similar prior issues from the duplicate
index. The LLM never makes the actionability decision — that stays with the
calibrated classifier — and everything degrades to a deterministic template
without an API key. Off by default.

**Operations**: structured request logs, per-endpoint latency percentiles
and 5xx counts in `/stats`, a read-only `/dashboard`, an audit record for
every GitHub write in the ledger, and the OpenAPI spec exported to
[docs/openapi.json](docs/openapi.json) (`python -m ghic.service.app
--openapi`).

**Measured performance** (real load test, single worker —
`reports/loadtest.json`): one prediction costs ~600 ms CPU; p95 is 626 ms at
concurrency 1 and a single worker sustains ~1.7 predictions/s. Concurrency
beyond that queues (p50 4.8 s at c=8, throughput flat) — still far above any
single repo's issue rate; the scaling path and the honest roadmap
(multi-tenancy, Kubernetes, billing — each gated on a named usage signal)
live in [docs/PRD.md](docs/PRD.md).

## Validation is a command, not a waiting period

```bash
python -m ghic.backtest      # ~2 minutes
```

Replays every held-out issue through the production webhook — signed HTTP
request, enrichment, features, model, decision — scores the answers against
ground truth, calibrates a per-repo threshold on the earlier half of each
repo's test slice, verifies it on the later half, and prints the exact
`GHIC_REPO_THRESHOLDS=` line to deploy with. Because the replay exercises
the service (single-issue feature degradations included), its numbers are
the deployment truth, not the notebook truth.

After deploying: `GET /stats` (token-gated) shows totals, positive rate, the
last 20 predictions, and the live-graded confusion matrix. Merged-PR links
are invisible to webhooks, so live recall is reported as a lower bound —
the approximation is documented, not hidden.

## Repository map

```
ghic/
  collect.py     GraphQL collection: content-addressed page cache, rate-limit
                 gating, batched author enrichment, release history — idempotent
  label.py       the ground-truth rules + audit histogram (stdlib-only)
  features.py    feature engineering; the same code trains and serves
  train.py       v1 comparison zoo + the champion protocol
  backtest.py    held-out replay through the real webhook + threshold calibration
  evaluate.py    metrics incl. Brier, plots incl. reliability diagram, explanations
  demo.py        CLI walkthrough: metrics, worked examples, live scoring
  dupdetect.py   duplicate detection: index, query, honest evaluation
  service/
    app.py         FastAPI: /webhook, /healthz, /stats, /dashboard, /api/predict
    github_app.py  App auth (JWT → installation token) + REST helpers
    inference.py   single-issue prediction + top-feature explanations
    tracking.py    the self-grading ledger + GitHub-write audit trail
    drafting.py    scoped LLM comment drafting (never the decision)
    settings.py    GHIC_* env config, safe-by-default
scripts/         loadtest.py — real latency percentiles against a live instance
notebook/        the pipeline as three narrative notebooks
tests/           107 tests: labeling, features, collection, service, duplicates, drafting
reports/         metrics, figures, backtest/champion/loadtest artifacts
docs/            DEPLOYMENT.md · PRD.md (roadmap, design decisions) · openapi.json
```

Rebuild everything from a bare clone and a GitHub token:

```bash
pip install -e ".[dev]"
cp .env.example .env                    # add a read-only PAT
python -m ghic.collect                  # cached, resumable, budget-aware
python -m ghic.label                    # prints the audit histogram
python -m ghic.train --champion         # CV -> calibration -> model card
python -m ghic.backtest                 # verify through the service path
```

## Honest limitations

- Trained on three large, professionally triaged repos. Transfer to small
  or differently run projects is unvalidated — deploy in dry-run first and
  read `/stats` before enabling writes.
- vscode-style repos (huge volume, house triage conventions) remain the
  weak spot even after calibration. The fix is more training data, and the
  collector is built to scale to it: add repos to `config.yaml`, re-run.
- Author repo/follower counts are collection-time snapshots; the
  first-time-contributor flag degrades to "first-time" at single-issue
  inference. Both are quantified in the backtest rather than assumed away.
- Labels are rule-derived. Rule 4 (conservative default) contributes 31% of
  the negative class — traceable in the audit, and the first thing better
  data would improve.

## License

[MIT](LICENSE). Built by **Faisal Hakimi**.
