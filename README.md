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
              suggested category: bug (assistive, never auto-applied) ─┤
              likely duplicates of prior issues (assistive) ───────────┤
              coarse resolution-time bucket (API only) ────────────────┤
              "missing info" draft when under-specified (LLM, scoped) ─┤
                                                                        ▼
                                    comment · label · project (opt-in each)
issue edited ──▶ re-scored on the improved text (never re-posted)
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
| `GHIC_SUGGEST_CATEGORY` | `true` | assistive category suggestion (needs `category.joblib`) |
| `GHIC_ESTIMATE_EFFORT` | `true` | coarse resolution-time bucket, API response only (needs `effort.joblib`) |
| `GHIC_PROJECT_ID` | — | add predicted-actionable issues to a Projects v2 board |
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

**Category suggestion** (`ghic/category.py`): a second head predicts the
category label a maintainer will eventually apply (bug / feature / question
/ docs / duplicate / invalid) — real ground truth, 2,747 labeled issues,
same walk-forward protocol. Test macro-F1 is a modest 0.470 (bug F1 0.69,
per-class table and full confusion matrix in
[models/CATEGORY_CARD.md](models/CATEGORY_CARD.md)), which is why it ships
as an assistive suggestion in the comment and **never applies a label
itself**. Priority and severity heads were deliberately *not* built — the
corpus has zero ground truth for either, and the tempting keyword-proxy for
severity is circular; the reasoning is recorded in
[models/PRIORITY_CARD.md](models/PRIORITY_CARD.md) and
[models/SEVERITY_CARD.md](models/SEVERITY_CARD.md).

**Resolution-time estimate** (`ghic/effort.py`): time-to-close is a weak
effort proxy, so the experiment ran against a pre-declared ship bar
(Spearman ≥ 0.30 + ≥ 10% MAE improvement over a constant baseline) and only
shipped because it cleared it — Spearman 0.492 on the chronological test.
It surfaces as four coarse buckets in the API response only, never in the
public comment: the validated claim is rank-informativeness, not a promise
([models/EFFORT_CARD.md](models/EFFORT_CARD.md)).

**Scoped LLM drafting** (`ghic/service/drafting.py`): when a deterministic,
tested trigger says an issue is under-specified (no repro steps, no
trace/code, near-empty body), Claude drafts the "could you add…" comment a
triager would write, grounded in similar prior issues from the duplicate
index. The LLM never makes the actionability decision — that stays with the
calibrated classifier — and everything degrades to a deterministic template
without an API key. Off by default.

**LLM-assisted analysis** (`ghic/llm/`): on top of the ML actionability
probability, an LLM produces a category, priority, severity, a 1–2
sentence executive summary, a risk assessment (level + reasons),
business-impact notes (only when genuinely inferable from the issue text),
plain-language reasoning, missing-information suggestions, and ordered
label suggestions — the polished, first-party-feeling comment format in
`format_llm_comment()`, tuned to answer "what is this / how important is
it / why / what next" in under 10 seconds. Tone adapts per issue category
(bug, feature, question, duplicate, security, performance, docs, plus an
explicit regression callout) entirely through prompt guidance, not
per-category templates in code. Two providers, priority chain: Groq
(`openai/gpt-oss-120b`) is the fast primary — measured ~1.9s per call —
with OpenRouter (`nvidia/nemotron-3-ultra-550b-a55b:free`, measured ~17s)
as the fallback for when Groq itself fails; 17s alone is over a webhook's
realistic response budget, which is why it's the backup and not the
primary despite being the originally-specified model (numbers and
reasoning in the card). The ML probability is passed in as fixed evidence
the model reasons from, never something it recomputes; category/priority/
severity/risk here are the LLM's judgment, explicitly not a statistically
validated prediction the way the classifier's probability is (that
distinction is why priority/severity weren't shipped as a *classifier*
head — see the card). A deterministic consistency check
(`ghic/llm/consistency.py`) flags the rare case where the LLM's own
priority/severity/reasoning strongly implies an actionable bug the ML
classifier called non-actionable, rendering `Needs Maintainer Review`
instead of silently showing two contradictory conclusions — described in
plain language, never as "the models disagree." Below a low-confidence
threshold, the comment adds an explicit disclaimer rather than presenting
a thin-evidence read as settled. Every failure mode (timeout, rate limit,
malformed JSON, no API key) degrades to the original ML-only comment,
never breaks the webhook. `ghic/llm/provider.py` is a one-method abstract interface, and
`ghic/llm/_chat_completions.py` a narrower shared base for any OpenAI-
compatible API (which both shipped providers are), so another backend
(OpenAI, Anthropic, Gemini, Ollama) is a small new class, not a rewrite.
Off by default (`GHIC_USE_LLM_ANALYSIS=false`); full design/testing
notes in [models/LLM_ANALYSIS_CARD.md](models/LLM_ANALYSIS_CARD.md).

**Repository Intelligence** (`ghic/repository_intelligence/`): retrieval-
augmented reasoning over the repository an issue was opened on. The repo is
shallow-cloned and indexed in the background (structural chunking — Python
via `ast`, Markdown by heading, C-family via a declaration/brace scanner —
then embedded into a vector index); when an issue arrives, a query built
from its title and identifier-like tokens retrieves the most relevant code,
which is passed to the LLM as evidence and surfaced as a **Repository
Evidence** section naming the files and functions involved.

The anti-hallucination guarantee is structural, not a prompt request: the
prompt contains only chunks that were actually retrieved, and the comment's
file list is rendered in Python from retrieved-chunk metadata rather than
from model output — so a path in a GHIC comment exists in the index by
construction. When nothing clears the confidence floor, every layer says so
("No directly related source files were confidently identified") instead of
guessing.

Retrieval never blocks and never fails a webhook: `get_context()` only reads
an existing index, an unindexed repo queues a background job through the
same QStash queue and returns empty, and every failure mode (git missing,
clone denied, index corrupt, embeddings down) degrades to text-only
analysis. The default embedder is lexical (offline, no API key, no cost);
`GHIC_REPO_INTEL_EMBEDDING_PROVIDER=openai` swaps in real semantic
embeddings without touching anything else. Production infrastructure is provider-based: vectors in Postgres/pgvector
(reusing the same `DATABASE_URL` as the ledger — no new infrastructure) or
a local index for development; durable repository state with a real
lifecycle (`queued → indexing → ready`, plus `failed`/`updating`) rather
than state inferred from the filesystem; a swappable index queue; and
incremental re-indexing that diffs `git` and re-embeds only what changed,
falling back to a full rebuild whenever the diff can't be trusted. Secrets
never enter an index — credential-shaped files are excluded by path and
token-shaped values are redacted from file content before chunking.

On an ephemeral filesystem (Vercel, Lambda, Cloud Run) with no persistent
vector store configured, the engine **disables itself and says why** rather
than re-indexing on every cold start and discarding the result; `/healthz`
reports whether state and queue are actually durable. Off by default
(`GHIC_USE_REPO_INTELLIGENCE=false`); index and query it directly with
`python -m ghic.repo_index`, inspect status at `/repositories` and counters
at `/repositories/metrics`. Full design notes — including the ranking
corrections that came out of real output, the recommended production stack,
and which providers are deliberately *not* implemented — in
[models/REPOSITORY_INTELLIGENCE_CARD.md](models/REPOSITORY_INTELLIGENCE_CARD.md).

**Async webhook processing** (`ghic/service/qstash.py`,
`ghic/service/idempotency.py`): `/webhook` can validate, idempotency-check,
and return 200 in well under a second, deferring the actual scoring/LLM/
comment work to a callback via Upstash QStash — a real durable queue,
not a home-grown workaround, because the obvious alternatives don't
actually work on this project's serverless target: FastAPI's
`BackgroundTasks` is confirmed unreliable on Vercel's Python runtime (no
guarantee the function keeps running after it responds), Vercel's own
`waitUntil()` continuation is documented for Node.js/Edge only, and
Vercel's Cron Jobs are capped at once/day on the Hobby plan — useless as a
poller for a triage bot. Idempotency (dedup on GitHub's own
`X-GitHub-Delivery` header, so a retried delivery never produces a second
comment) is independent of the queue and always active. Off by default —
`/webhook` processes every issue inline exactly as before otherwise. Full
platform research and design notes in
[models/ASYNC_PROCESSING_CARD.md](models/ASYNC_PROCESSING_CARD.md).

**Operations**: structured request logs, per-endpoint latency percentiles
and 5xx counts in `/stats`, a read-only `/dashboard` with six analytics
facets computed from real ledger data (issue trends, duplicate rate,
resolution analytics, confidence histogram, label stats, per-repo
analytics), an audit record for every GitHub write in the ledger, maintainer
label events recorded live as future ground truth, and the OpenAPI spec
exported to [docs/openapi.json](docs/openapi.json) (`python -m
ghic.service.app --openapi`). Retraining is one command
(`python -m ghic.retrain`) that snapshots every run and appends to
[models/REGISTRY.md](models/REGISTRY.md) with artifact hashes. All
benchmark numbers live in one place: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

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
  category.py    category head: train, card, serving (assistive suggestion)
  effort.py      resolution-time head, gated by a pre-declared ship bar
  assign.py      assignment recommender: collection + causal hit@k evaluation
  retrain.py     one-command retraining + run snapshots + model registry
  cli.py         the unified `ghic` CLI (train/predict/explain/benchmark/serve/…)
  service/
    app.py         FastAPI: /webhook, /healthz, /stats, /dashboard, /api/predict
    github_app.py  App auth (JWT → installation token) + REST/GraphQL helpers
    inference.py   single-issue prediction + top-feature explanations
    explain.py     raw-feature-name humanizer + natural-language explanation
    tracking.py    the self-grading ledger + audit trail + dashboard analytics
    pg_ledger.py   Postgres ledger backend (serverless deploys, no local disk)
    idempotency.py dedup GitHub deliveries by X-GitHub-Delivery (Postgres/file/memory)
    qstash.py      async queue publish + Upstash-Signature JWT verification
    drafting.py    scoped LLM comment drafting (never the decision)
    settings.py    GHIC_* env config, safe-by-default
  llm/             LLM-assisted issue analysis (see "LLM-assisted analysis" above)
    provider.py         abstract LLMProvider interface (one method, swappable backends)
    _chat_completions.py shared base for OpenAI-compatible APIs: httpx, retry/backoff, timeout
    groq.py             Groq (openai/gpt-oss-120b) — the fast primary, ~1.9s measured
    openrouter.py        OpenRouter (nemotron, free tier) — the fallback, ~17s measured
    fallback.py          FallbackLLMProvider: tries providers in order, itself an LLMProvider
    prompts.py           system/user prompt construction
    models.py            IssueContext / IssueAnalysis + strict JSON-schema validation
    service.py           cache lookup → provider call → cache write, never raises
    exceptions.py        LLMError hierarchy
scripts/         loadtest.py — real latency percentiles against a live instance
notebook/        the pipeline as three narrative notebooks
tests/           253 tests: labeling, features, collection, service, heads, CLI, analytics, llm, async
reports/         metrics, figures, backtest/champion/loadtest artifacts, runs/
docs/            DEPLOYMENT.md · PRD.md · BENCHMARKS.md · openapi.json · assets/
```

Rebuild everything from a bare clone and a GitHub token:

```bash
pip install -e ".[dev]"
cp .env.example .env                    # add a read-only PAT
python -m ghic.collect                  # cached, resumable, budget-aware
python -m ghic.retrain                  # label -> champion -> backtest ->
                                        # category -> dup index; snapshots the
                                        # run and appends models/REGISTRY.md
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
