# Deployment guide: GitHub App + Marketplace

This walks through taking the triage bot from a trained model to a live GitHub
App that anyone can install — and, optionally, a GitHub Marketplace listing.

## 1. Prerequisites

- Trained model artifacts in `models/` — one `python -m ghic.retrain` run
  produces all of them (`champion.joblib` + `rf_balanced.joblib` required;
  `category.joblib`, `effort.joblib`, `dup_index.joblib` optional heads), or
  copy them from a GitHub Release. They are gitignored, so publish them as
  Release assets when you push the repo.
- A host with a public HTTPS URL (any of: Fly.io, Render, Railway, a VPS
  behind a reverse proxy, or a tunnel like `smee.io` / `ngrok` for testing).

## 2. Register the GitHub App

GitHub → Settings → Developer settings → **GitHub Apps** → *New GitHub App*:

| Field | Value |
|---|---|
| App name | e.g. `issue-triage-bot` (globally unique) |
| Homepage URL | your repo URL |
| Webhook URL | `https://<your-host>/webhook` |
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
  included is ~295MB, well clear of the 500MB cap).
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

## Security posture

- HMAC (`X-Hub-Signature-256`) verified on every webhook with a constant-time
  compare; unsigned requests are rejected unless `GHIC_ALLOW_UNSIGNED=true`
  (dev only).
- `/api/predict` is gated by the same secret (`X-GHIC-Token` header) so the
  model is not a public scoring oracle.
- The container runs as a non-root user; no state is persisted.
- The GitHub App needs only Issues read/write + Metadata read — no code access.
