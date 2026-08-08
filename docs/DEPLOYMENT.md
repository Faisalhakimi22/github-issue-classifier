# Deployment guide: GitHub App + Marketplace

This walks through taking the triage bot from a trained model to a live GitHub
App that anyone can install — and, optionally, a GitHub Marketplace listing.

## 1. Prerequisites

- Trained model artifacts in `models/` — one `python -m ghic.retrain` run
  produces all of them. `champion.joblib`, `category.joblib`, and
  `dup_index.joblib` (the three the deployed service actually loads) are
  tracked directly in git, so a fresh clone already has them. The rest
  (`rf_balanced.joblib`, `effort.joblib`, and the other retraining
  artifacts) are gitignored — rebuild them locally, or copy from a GitHub
  Release if you publish one.
- A host with a public HTTPS URL (any of: Fly.io, Render, Railway, a VPS
  behind a reverse proxy, or a tunnel like `smee.io` / `ngrok` for testing).

## 2. Register the GitHub App

GitHub → Settings → Developer settings → **GitHub Apps** → *New GitHub App*:

| Field | Value |
|---|---|
| App name | e.g. `issue-triage-bot` (globally unique) |
| Homepage URL | your repo URL |
| Webhook URL | `https://<your-host>/webhook` |
| Setup URL | `https://<your-marketing-site>/github/setup` |
| Webhook secret | generate one: `python -c "import secrets; print(secrets.token_hex(32))"` |
| **Repository permissions** | Issues: **Read & write** · Metadata: Read-only |
| **Subscribe to events** | Issues |

**Why each permission** (Marketplace review asks): *Issues read* — receive
`issues.opened`/`edited`/`closed` and label events, and read labels for the
online-evaluation loop; *Issues write* — post the prediction comment and
apply the triage label; *Metadata read* — GitHub's implicit baseline for
any App. Nothing else is requested: no code, PR, or member access. One
exception, opt-in: setting `GHIC_PROJECT_ID` (add triaged issues to a
Projects v2 board) additionally requires Organization/Repository
**Projects: Read & write** — leave it unset and the permission is not
needed.
| Where can it be installed? | Any account (required for Marketplace) |

After creating the app:

1. Note the **App ID** (top of the app settings page).
2. Scroll to *Private keys* → **Generate a private key** → downloads a `.pem`.
3. Keep the webhook secret you entered.

These three values are the service's credentials:

```bash
GHIC_APP_ID=123456
GHIC_PRIVATE_KEY_PATH=/secrets/issue-triage-bot.pem   # or GHIC_PRIVATE_KEY inline
GHIC_WEBHOOK_SECRET=<the secret>
```

The Setup URL is the post-install handoff, not a webhook. After someone
installs or updates the App on a repository, GitHub redirects the browser to
that URL with `installation_id` and `setup_action`; the marketing site then
forwards the user to the GHIC hub. If this field is blank or points at the
wrong deployment, installation succeeds but the user will not land in the hub.

## 3. Deploy the service

```bash
docker build -t ghic .
docker run -d -p 8000:8000 \
  -e GHIC_WEBHOOK_SECRET=... \
  -e GHIC_APP_ID=... \
  -e GHIC_PRIVATE_KEY="$(cat issue-triage-bot.pem)" \
  ghic
```

Check it: `curl https://<your-host>/healthz` should return the model name and
`"dry_run": true`.

### Validate, then roll out — same day

0. **Backtest first (minutes).** Before anything touches GitHub, run
   `python -m ghic.backtest`. It replays every held-out real issue through
   the actual webhook path, scores the results against ground truth, and
   prints the calibrated `GHIC_REPO_THRESHOLDS=` line. If those numbers look
   right, the service logic is validated — no soak period needed.
1. **Dry run (default).** Deploy, install the app on one of your own repos,
   open 2–3 test issues (one detailed bug report with a traceback, one vague
   feature request), and check `GET /stats` — you should see them scored with
   sensible probabilities and nothing written to GitHub.
2. **Comments on.** Redeploy with `GHIC_DRY_RUN=false GHIC_POST_COMMENT=true`.
   The bot now posts one prediction comment per newly opened issue.
3. **Labels on.** Add `GHIC_APPLY_LABEL=true`. Issues scoring above the
   threshold get the `predicted:actionable-bug` label (create the label in
   the repo first, or GitHub creates it with a random color).

### Tuning per deployment

The research found one global threshold is wrong for repos with different
score distributions (vscode needed a lower cutoff than tensorflow). Use the
`GHIC_REPO_THRESHOLDS` values from the backtest for the training repos; for
your own repos, start from the global `GHIC_THRESHOLD` and lower it to catch
more bugs (more false positives) or raise it for higher precision. `/stats`
shows the live probability distribution to tune against.

### Alternative: deploy on Vercel

The Docker path above assumes a host with a persistent disk for the
online-evaluation ledger. Vercel's Python functions have none — every
invocation may be a fresh container — so this path swaps the ledger to
Postgres and trims the model bundle to fit Vercel's size limit. Everything
else (webhook logic, models, dashboard) is unchanged.

**What's different from Docker:**

- **Ledger → Postgres.** `data/predictions.jsonl` doesn't survive between
  invocations on Vercel. Add a Postgres database (Vercel's own Postgres
  integration, or any external one — Neon, Supabase, RDS) and set
  `DATABASE_URL`; `ghic/service/pg_ledger.py` takes over automatically (see
  `settings.py` — `GHIC_DATABASE_URL` / `DATABASE_URL` / `POSTGRES_URL`, in
  that precedence order). Without one, the service still runs, but online
  precision/recall reset on every cold start.
- **Smaller model bundle, and no custom install command.** Vercel's
  documented Python limit is 500MB uncompressed, but that's only reachable
  through Vercel's own automatic bundle optimizer — a custom
  `installCommand` disables it and drops the effective cap to ~225MB (found
  the hard way: a first attempt with a custom `installCommand` hit 315.92MB
  and failed). So `vercel.json` has no `installCommand`; Vercel auto-detects
  and installs the root `requirements.txt` itself. `.vercelignore` also
  drops `rf.joblib`/`rf_balanced.joblib`/`logreg*.joblib` (unused —
  `champion.joblib` is what's actually loaded) and, for bundle size,
  `effort.joblib` (21MB — `GHIC_ESTIMATE_EFFORT` degrades to off
  automatically when its artifact is missing, no env var changes needed).
  What ships: `champion.joblib` (required), `category.joblib` (676KB), and
  `dup_index.joblib` (14MB — powers the "possibly related" comment section
  and assignment suggestions; measured pre-optimization bundle with it
  included is ~295MB, well clear of the 500MB cap). These three are the
  only model files tracked in git (see `.gitignore`) — deliberately, because
  Vercel's GitHub integration auto-deploys from a plain `git push`, building
  from the git tree rather than the local filesystem. A model that only
  exists on disk locally deploys fine via `vercel --prod` (which uploads the
  working directory) but 500s on the next `git push`-triggered build with
  `FileNotFoundError: Model not found` — this bit the project once in
  production; don't regitignore these three. The unused/oversized heads
  (`rf*.joblib`, `logreg*.joblib`, `effort.joblib`) stay untracked and
  rebuildable via `python -m ghic.train` / `python -m ghic.retrain`.
- **`pyproject.toml` is hidden from Vercel** (`.vercelignore`) so
  `requirements.txt` wins dependency detection instead — Vercel prefers
  `pyproject.toml` when both exist, and its base `[project.dependencies]`
  deliberately excludes the service extras (fastapi etc.) to keep a plain
  CLI install light. `requirements.txt` itself was repointed at this
  deployment (Docker still installs from `pyproject.toml`'s `[service]`
  extra, unaffected); the previous fully-pinned research-reproduction
  manifest — including matplotlib/pytest/httpx, which have no business in a
  serverless bundle — moved to `requirements-freeze.txt`.
- **Cold starts re-load everything.** Each cold invocation re-imports
  scikit-learn/pandas and re-`joblib.load`s ~20MB of models. `maxDuration`
  is set to 10s in `vercel.json` to match GitHub's own webhook timeout and
  the Hobby plan's default cap — raise it (Pro/Fluid compute) if cold
  starts run long in practice; measure with `/healthz` timing before
  assuming it's needed.

**Deploy:**

```bash
npm i -g vercel        # or: npx vercel <command>
vercel login           # opens a browser; needs your Vercel account
vercel link             # from the repo root — creates/links the Vercel project
vercel env add GHIC_WEBHOOK_SECRET production
vercel env add GHIC_APP_ID production
vercel env add GHIC_PRIVATE_KEY production   # paste the .pem contents
vercel env add DATABASE_URL production        # from your Postgres provider
vercel --prod
```

The webhook URL to register on the GitHub App is `https://<project>.vercel.app/webhook`.
Everything from the "Validate, then roll out" checklist below still applies —
backtest first, dry-run, then flip on comments/labels via the same
`GHIC_DRY_RUN`/`GHIC_POST_COMMENT`/`GHIC_APPLY_LABEL` env vars, just set
through `vercel env add` instead of `docker run -e`.

`api/index.py` is the entrypoint Vercel's Python runtime loads (it puts the
repo root on `sys.path` and calls `create_app()` — the `ghic` package isn't
pip-installed here, it's imported straight from source).

## 4. Install the app on repositories

App settings → *Install App* → choose the account → select repositories.
Every `issues.opened` event from those repos now flows to your webhook.

## 5. (Optional) List on GitHub Marketplace

Marketplace requirements ([docs](https://docs.github.com/en/apps/github-marketplace)),
checked against the actual repo state (2026-07-14):

- [ ] The app is owned by an **organization** you own, or your personal
      account — *blocked on deployment: the App isn't registered yet*
- [ ] It's installed on at least **1 account** other than your own —
      *blocked on deployment*
- [ ] Webhook events are processed over HTTPS with a verified domain —
      *blocked on deployment (host choice)*
- [x] Logo — `docs/assets/logo.svg` (original artwork)
- [x] Description / feature card — drafted in `docs/assets/listing.md`
- [ ] At least one screenshot — *blocked on deployment; per
      `docs/assets/listing.md` screenshots come from a real deployment,
      never mockups*
- [x] Support and privacy-policy URLs — `SUPPORT.md` + `PRIVACY.md`
- [x] Customer data handling statement — `PRIVACY.md`; the service stores
      predictions/outcomes in its own ledger (repo, issue number, score —
      no issue text) and sends only the comment/label back to GitHub

Then: App settings → *List in Marketplace* → draft the listing (category:
**Project management** or **Utilities**), submit for review. Start with a
**free plan**; paid plans require the extra verification tier.

## 6. Operations

- `/healthz` — liveness for load balancers; reports model + thresholds + dry-run.
- `/dashboard` — read-only operator view (enter the webhook secret in the
  page; data is fetched client-side with the token).
- `/stats` — token-gated (`X-GHIC-Token: <webhook secret>`): totals, positive
  rate, mean probability, the last 20 scored issues, **and the online
  evaluation block** — live precision/recall computed by grading each
  prediction when its issue is eventually closed. First stop after any
  deploy, and the long-term health signal: if live precision drifts down,
  retrain.
- Prediction ledger — every prediction/outcome is appended to
  `data/predictions.jsonl` (override with `GHIC_LEDGER`, empty string
  disables), so online metrics survive restarts. Mount a volume for it in
  Docker.
- Logs — one line per scored issue with repo, number, probability, decision.
- The webhook responds in well under GitHub's 10s limit (model inference is
  ~50ms; the two enrichment API calls dominate). If GitHub reports delivery
  timeouts, set `GHIC_ENRICH=false` — the pipeline imputes the missing fields.
- Retrain periodically: issue-triage vocabulary drifts. One command:
  `python -m ghic.retrain` (label → champion protocol → backtest →
  category head → duplicate index). Each run snapshots its cards/metrics to
  `reports/runs/<timestamp>/` and appends a row to `models/REGISTRY.md`, so
  any deployed artifact traces back to the run that produced it (sha256).
  Retraining is operator-triggered by design — the training data is not in
  the repo, so CI can't do it; schedule it with cron on the box that holds
  `data/` if you want it periodic.
- Capacity: one prediction costs ~600 ms CPU; a single worker sustains
  ~1.7 predictions/s (measured — `reports/loadtest.json`), far above any
  single repo's issue rate. If you ever need more, run multiple workers or
  replicas — but move the ledger off JSONL first (see docs/PRD.md §9).
- Optional features: `GHIC_SUGGEST_RELATED` surfaces likely-duplicate prior
  issues (ship `models/dup_index.joblib`); `GHIC_SUGGEST_CATEGORY` adds an
  assistive category suggestion (ship `models/category.joblib`; never
  auto-labeled); `GHIC_DRAFT_MISSING_INFO=true` + `ANTHROPIC_API_KEY`
  drafts a "missing information" request on under-specified issues
  (template fallback without a key); `GHIC_PROJECT_ID` files predicted-
  actionable issues onto a Projects v2 board; `GHIC_USE_LLM_ANALYSIS=true`
  + `OPENROUTER_API_KEY` replaces the comment with a full LLM-assisted
  report — see §7 below.
- Edited issues are re-scored automatically (never re-commented), so a
  report improved after the bot's nudge is graded on its improved text.
  Maintainer label events are recorded to the ledger as future ground
  truth (`label_events_observed` in `/stats`).

## 7. LLM-assisted analysis (optional)

Adds category / priority / severity / plain-language summary / missing-info
/ suggested-labels to the comment, on top of the (unchanged) ML
actionability probability. See [models/LLM_ANALYSIS_CARD.md](../models/LLM_ANALYSIS_CARD.md)
for what this is and isn't (short version: priority/severity here are the
LLM's judgment, not a validated prediction — that's a real distinction in
this project, see the card).

Ships as a priority chain of two providers, not one — see
[models/LLM_ANALYSIS_CARD.md](../models/LLM_ANALYSIS_CARD.md) for the
measured latency comparison that drove this: OpenRouter's free nemotron
model (the originally-specified one) measured ~17s per call, over a
webhook's realistic response budget, so Groq (`openai/gpt-oss-120b`,
~1.9s measured) is the primary and OpenRouter is the fallback for when
Groq itself fails.

**Setup:**

1. Get a free Groq key at
   [console.groq.com/keys](https://console.groq.com/keys) (primary,
   recommended) and/or a free OpenRouter key at
   [openrouter.ai/keys](https://openrouter.ai/keys) (fallback). Either
   alone works; both gives you the fallback chain.
2. Set env vars (Vercel: `vercel env add <NAME> production --value "..."`;
   Docker: pass with `-e`):
   ```bash
   GHIC_USE_LLM_ANALYSIS=true
   GROQ_API_KEY=gsk_...
   # OPENROUTER_API_KEY=sk-or-v1-...     # optional fallback
   ```
3. Redeploy. `GET /healthz` doesn't report LLM status directly, but startup
   logs a line: `llm analysis enabled (GroqProvider)` (single provider) or
   `llm analysis enabled (GroqProvider -> OpenRouterProvider)` (chain).
4. Dry-run first, same as any other rollout — the analysis is computed and
   returned in `/api/predict`'s and the webhook's JSON response
   (`llm_analysis` key) regardless of dry-run, so you can inspect it before
   any comment goes out for real.

**Troubleshooting:**

| Symptom | Likely cause | Fix |
|---|---|---|
| Comment still uses the old format (no "GHIC Analysis" header) | `GHIC_USE_LLM_ANALYSIS` unset/false, or neither key is set | `settings.can_use_llm` needs the flag *and* at least one key. Check startup logs for the "llm analysis enabled" line; its absence means the feature never activated. |
| `llm_analysis` is `null` in the API/webhook response even with a key set | Every configured provider failed and was caught — check logs for `llm provider ... failed (...)` and the final `llm analysis failed (...)` | Usually a bad/expired key (`LLMProviderError`, HTTP 401) or a rate limit on the free tier (`LLMProviderError`, HTTP 429 after retries exhausted). Verify the key at the provider's console; add the second provider's key if you're only running one. |
| Comments arrive slower than before | Expected — this adds one synchronous LLM call per issue | With Groq alone this should stay under ~2-3s. If it's consistently much slower, Groq may be degraded/rate-limited and falling through to OpenRouter's ~17s path on every request — check logs for `llm provider 1/2 (GroqProvider) failed`. |
| Free-tier model unavailable / retired | Provider catalogs change over time | Groq: check [console.groq.com/docs/models](https://console.groq.com/docs/models). OpenRouter: check `https://openrouter.ai/api/v1/models` for current `:free` slugs. Set `LLM_MODEL` / `OPENROUTER_MODEL` accordingly — no code change needed. |

## 8. Asynchronous webhook processing (optional)

`/webhook` validates and returns 200 in well under a second instead of
waiting on the LLM call, deferring the actual work to a queued callback.
See [models/ASYNC_PROCESSING_CARD.md](../models/ASYNC_PROCESSING_CARD.md)
for why this needs a real queue (Upstash QStash) rather than FastAPI
`BackgroundTasks` or Vercel Cron — both were checked directly and neither
works for this on Vercel's Python runtime / Hobby plan.

**Setup:**

1. Create a free account at [console.upstash.com](https://console.upstash.com) →
   QStash tab. Copy the **QStash Token** and both **Signing Keys** (current
   and next).
2. Set env vars:
   ```bash
   GHIC_USE_ASYNC_PROCESSING=true
   QSTASH_TOKEN=...
   QSTASH_CURRENT_SIGNING_KEY=...
   QSTASH_NEXT_SIGNING_KEY=...
   GHIC_PUBLIC_BASE_URL=https://<your-deployed-domain>   # no trailing slash
   ```
3. Redeploy. Open a test issue on an installed repo and check `/webhook`'s
   response — with async active it returns `{"ok": true, "queued": true,
   "message_id": "..."}` immediately; the comment appears a few seconds
   later once QStash's callback to `/internal/process-issue` completes.
4. If any of the four vars above is missing, `/webhook` silently processes
   inline instead (same as async being off) — check startup logs for a
   warning naming which one.

**Idempotency** (dedup by `X-GitHub-Delivery`) is independent of this and
always on — no setup needed, works whether or not the queue is configured.
On Vercel it shares `DATABASE_URL`/`GHIC_DATABASE_URL` with the ledger (no
separate connection string); on Docker/Fly it's a file at
`GHIC_IDEMPOTENCY_FILE` (default `data/idempotency.json`, same volume as
the ledger).

## 9. Repository Intelligence (optional)

Semantic code retrieval over the repository an issue was opened on. Design
notes in
[models/REPOSITORY_INTELLIGENCE_CARD.md](../models/REPOSITORY_INTELLIGENCE_CARD.md);
this section is the operational part.

### What it needs

Unlike every other feature here, this one has real infrastructure
requirements — it clones repositories, stores vectors, and runs jobs that
take minutes:

| Need | Why | Provider |
|---|---|---|
| Persistent vector storage | Indexes must outlive a process | `GHIC_VECTOR_PROVIDER=postgres` (pgvector) or a real disk |
| Durable state | Lifecycle, not "is there a directory?" | Auto: Postgres → file → memory |
| A job queue | Indexing must never touch the webhook path | QStash (already configured if async processing is on) |
| `git` on PATH | Cloning | — |

The vector and state stores both reuse `DATABASE_URL`, so a deploy that
already has Postgres for the ledger needs no new services.

### Deployment matrix

| Platform | Works? | Configuration |
|---|---|---|
| Docker / Fly.io / Railway / Render | **Recommended** | A volume for `GHIC_REPO_INTEL_CACHE_DIR`; `local` vectors are fine, Postgres is better with >1 replica |
| Kubernetes | Yes | Postgres vectors (pods are cattle); a PVC only for the clone scratch dir |
| Vercel / Lambda / Cloud Run | Only with Postgres | **Auto-disables itself** without `GHIC_VECTOR_PROVIDER=postgres` — see below |
| Local development | Yes | Defaults are fine; `GHIC_INDEX_QUEUE_PROVIDER=inline` to index without a queue |

### The Vercel auto-disable

On a platform with an ephemeral filesystem, `build_service()` returns None
and logs the reason unless a persistent vector store is configured. This is
deliberate: a local index there is rebuilt on every cold start and thrown
away minutes later — slow, expensive against a paid embedding provider, and
producing no working feature while appearing to be enabled. To run it on
Vercel:

```bash
vercel env add GHIC_USE_REPO_INTELLIGENCE production   # true
vercel env add GHIC_VECTOR_PROVIDER production         # postgres
# DATABASE_URL is already set by the Postgres integration
```

Check it took effect: `/healthz` reports a `repository_intelligence` block
with `state_durable` and `queue_durable`. Both false means the deploy is
running without real persistence — the failure that otherwise looks
identical to a healthy one.

### Full-free indexing on GitHub Actions

Vercel can serve retrieval from Postgres, but it cannot build the index when
`git` is missing from the runtime. The free path is to run indexing out of
band in GitHub Actions and write vectors into the same Postgres database the
Vercel API reads from.

The repo includes `.github/workflows/index-repositories.yml` for that:

1. In this repository, open **Settings -> Secrets and variables -> Actions**.
2. Add one database secret. Use whichever name matches your deploy:
   `DATABASE_URL`, `GHIC_DATABASE_URL`, or `POSTGRES_URL`.
3. Optional for private repos: add `GHIC_INDEX_GITHUB_TOKEN`, a fine-grained
   GitHub token that can read repository contents for the repos you want to
   index.
4. Optional for scheduled indexing: add `GHIC_INDEX_REPOSITORIES` with
   comma-, space-, or newline-separated repo names, for example
   `owner/api, owner/web`.
5. Run **Actions -> Index repositories -> Run workflow**. Pass `owner/name`
   in the `repo` input for one repo, or leave it empty to use
   `GHIC_INDEX_REPOSITORIES`.

The workflow also accepts `repository_dispatch` events with type
`ghic-index-repository` and payload `{"repo":"owner/name","force":false}`.
That is the hook to trigger indexing from another service later without
changing the worker.

### Semantic embeddings

The default embedding provider is still `hashing-512`: free, deterministic,
offline, and usable without secrets. For better semantic matching, the same
pipeline can use an OpenAI-compatible embeddings endpoint:

```bash
GHIC_REPO_INTEL_EMBEDDING_PROVIDER=openai
GHIC_REPO_INTEL_EMBEDDING_MODEL=<embedding-model>
GHIC_REPO_INTEL_EMBEDDING_DIMENSIONS=<dimension>   # use 0 only when the endpoint chooses
OPENAI_API_KEY=<secret>
```

`GHIC_REPO_INTEL_EMBEDDING_BASE_URL` can point at any endpoint that implements
OpenAI's `/v1/embeddings` response shape. `GHIC_REPO_INTEL_EMBEDDING_API_KEY`
also works and takes precedence over `OPENAI_API_KEY`.

Codestral Embed through OpenRouter uses this repository configuration:

```bash
GHIC_REPO_INTEL_EMBEDDING_PROVIDER=openai
GHIC_REPO_INTEL_EMBEDDING_MODEL=mistralai/codestral-embed-2505
GHIC_REPO_INTEL_EMBEDDING_DIMENSIONS=1536
GHIC_REPO_INTEL_EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
GHIC_REPO_INTEL_EMBEDDING_API_KEY=<openrouter-secret>
```

Embedding provider, model, dimension, and indexed commit SHA are persisted in
repository state. If any embedding identity changes, retrieval refuses to use
the old vectors and indexing performs a full rebuild for that repository. This
is required: vectors from different embedding models are not comparable, even
when raw similarity scores look plausible.

Postgres uses one fixed-width pgvector column. A forced full rebuild can
change that width only when the chunks table contains no other repository;
the replacement is transactional and rolls back to the old vectors if any
new vector cannot be stored. Without `--force`, a dimension mismatch fails
instead of falling back to JSON or mixing embeddings.

To rebuild safely through GitHub Actions:

1. Set repository variables:
   `GHIC_REPO_INTEL_EMBEDDING_PROVIDER=openai`,
   `GHIC_REPO_INTEL_EMBEDDING_MODEL=<embedding-model>`, and
   `GHIC_REPO_INTEL_EMBEDDING_DIMENSIONS=<dimension>`.
2. Set a repository secret: `OPENAI_API_KEY` or
   `GHIC_REPO_INTEL_EMBEDDING_API_KEY`.
3. Run **Actions -> Index repositories -> Run workflow** with the target
   repo and `force=true`.

For a local or CI worker using the same Postgres database:

```bash
GHIC_USE_REPO_INTELLIGENCE=true \
GHIC_VECTOR_PROVIDER=postgres \
GHIC_STATE_PROVIDER=postgres \
GHIC_INDEX_QUEUE_PROVIDER=none \
GHIC_REPO_AUTO_INDEX=false \
GHIC_REPO_INTEL_EMBEDDING_PROVIDER=openai \
GHIC_REPO_INTEL_EMBEDDING_MODEL=<embedding-model> \
GHIC_REPO_INTEL_EMBEDDING_DIMENSIONS=<dimension> \
OPENAI_API_KEY=<secret> \
DATABASE_URL=<same-postgres-url> \
python -m ghic.repo_index --repo owner/name --force
```

The `.github/workflows/retrieval-benchmark.yml` workflow is read-only. It
reports hashing retrieval against the current index and reports semantic
retrieval only when the stored vectors were built with the requested semantic
provider/model/dimension. Otherwise it reports `requires_reindex` instead of
mixing incompatible vectors.

### Rollout

```bash
# 1. Index one repository by hand and look at what retrieval returns.
python -m ghic.repo_index --repo owner/name
python -m ghic.repo_index --repo owner/name --query "csv import crashes"

# 2. Turn it on. Retrieval quality varies a lot by codebase, so step 1
#    first is worth the two minutes.
GHIC_USE_REPO_INTELLIGENCE=true

# 3. Watch it.
curl -H "X-GHIC-Token: $SECRET" https://<host>/repositories
curl -H "X-GHIC-Token: $SECRET" https://<host>/repositories/metrics
```

Feature flags let you narrow the rollout without redeploying:
`GHIC_VECTOR_SEARCH=false` (kill switch for retrieval),
`GHIC_REPO_AUTO_INDEX=false` (index only via the CLI),
`GHIC_INCREMENTAL_INDEXING=false` (always full rebuilds).

### Operations

- **Secrets never enter an index.** Credential-shaped files (`.env*`,
  `*.pem`, `id_rsa`, `credentials*`, kubeconfig, terraform state) are
  excluded by path, and token-shaped values are redacted from file content
  before chunking. Both layers are tested; see
  `tests/test_repository_production.py::TestSecretHandling`.
- **Storage grows silently.** A repository indexed once and never queried
  keeps its chunks. `service.cleanup()` drops indexes untouched for
  `GHIC_REPO_STALE_DAYS` (default 30) — run it from a cron job or a
  maintenance task.
- **A wedged repository recovers on its own.** If a worker dies mid-index,
  the repository sits in `indexing` until `STALE_IN_FLIGHT_SECONDS` (1
  hour) passes, then re-queues.
- **Failure never reaches the webhook.** Git missing, clone denied, index
  corrupt, embeddings down, database unreachable — all degrade to
  text-only analysis with no evidence section. That path is covered by
  `TestFallback` and by a webhook-level test that injects a component
  violating its own non-raising contract.

## Security posture

- HMAC (`X-Hub-Signature-256`) verified on every webhook with a constant-time
  compare; unsigned requests are rejected unless `GHIC_ALLOW_UNSIGNED=true`
  (dev only).
- `/internal/process-issue` (the QStash callback target, when async
  processing is enabled) verifies the `Upstash-Signature` JWT — HS256 via
  the configured signing key(s), checking issuer, destination URL, and a
  hash of the request body — before doing anything. Returns 503 if async
  processing isn't configured, so the route can't be probed into doing
  anything on a deploy that never enabled it.
- `/api/predict` is gated by the same secret (`X-GHIC-Token` header) so the
  model is not a public scoring oracle.
- The container runs as a non-root user; no state is persisted.
- The GitHub App needs only Issues read/write + Metadata read — no code access.
