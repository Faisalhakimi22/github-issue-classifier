# PROGRESS — build-plan execution log

Phase tracking for `BUILD_PLAN_2.md` (Phases 14–22). Note on provenance:
the `BUILD_PLAN.md` / `CLAUDE.md` files that plan references were never
files in this repo — Phases 0–13 correspond to the work landed in commits
`cc51401` and `fc2b56b` (champion protocol, backtest, service, duplicate
detection, drafting, dashboard, load test, PRD). The governing
non-negotiables are unchanged: never fabricate numbers, negative results
are shipped as model cards, [DESIGN] items say "planned — not implemented".

## Phase 14 — Gap Audit (2026-07-14)

Audit of the repo at `fc2b56b` against the original 22-line spec.
Verified against code, not against `docs/PRD.md`'s own table.

| # | Spec item | Status | Evidence |
|---|---|---|---|
| 1 | Issue category (bug/enhancement/question/duplicate/docs/invalid/security/regression) | **MISSING** | No category head exists. Ground truth IS available in `labels_at_close`: bug≈1514, feature≈435, question/support≈421, duplicate≈369, docs≈48, invalid≈40 across repos (per-repo naming differs); security ≈0 and regression=27 (vscode only) — those two classes have no trainable support and will be documented, not faked. → Phase 15 |
| 2 | Actionable vs non-actionable | **CONFIRMED BUILT** | `ghic/train.py --champion` (walk-forward CV + isotonic calibration); validation: `models/MODEL_CARD.md`, `reports/champion.json`, service-path backtest `ghic/backtest.py` |
| 3 | Duplicate detection | **CONFIRMED BUILT — negative result** | `ghic/dupdetect.py`; validation: `models/DUPLICATE_CARD.md` (ROC≈0.52–0.56 ≈ chance; MiniLM does not beat TF-IDF). Ships assistive-only in `app.py` (`suggest_related`) |
| 4 | Priority prediction | **MISSING** | No head, and no ground truth: the corpus has zero P0–P4 labels; nearest signal is `important` (31 issues, vscode only). → Phase 17 verify-or-negative-card |
| 5 | Severity prediction | **MISSING** | Same evidence as #4 — no severity labels exist in the three collected repos. → Phase 17 |
| 6 | Effort estimation | **MISSING** | No head. Only available proxy is time-to-close (known-weak: conflates effort with backlog priority). → Phase 17, negative card acceptable per plan |
| 7 | Missing-info detection + comment drafting | **CONFIRMED BUILT** | `ghic/service/drafting.py` (deterministic trigger; Claude draft w/ template fallback); validation: `tests/test_service.py::TestDrafting` |
| 8 | Label recommendation | **PARTIAL** | Only the single actionability label is applied (`app.py` `apply_label` → `settings.label_name`). Category-label recommendation lands with Phase 15's category head |
| 9 | Maintainer assignment recommendation | **MISSING** | `collect.py` does not fetch assignees/closers/participants (verified: `ISSUE_QUERY` has no such fields) — collection extension required first. → Phase 16 |
| 10 | Confidence scores | **CONFIRMED BUILT** | Isotonic-calibrated probabilities (`train.py` champion protocol); reliability curve in `reports/`; surfaced in every prediction dict (`inference.py`) |
| 11 | Natural-language explanation | **CONFIRMED BUILT** | `evaluate.py::top_contributions` → `inference.py` `explain=` → `format_comment`; tested in `tests/test_service.py` |
| 12 | REST API | **CONFIRMED BUILT** | `/api/predict`, `/stats`, `/healthz`, `/webhook`, `/dashboard`; spec exported to `docs/openapi.json` (test asserts path coverage) |
| 13 | CLI (train/predict/explain/dashboard/benchmark/serve) | **PARTIAL** | `ghic-collect/label/train/demo/backtest/serve` exist (`pyproject.toml [project.scripts]`); no `predict`, `explain`, `dashboard`, `benchmark` subcommands, no unified `ghic` entry. → Phase 19 |
| 14 | GitHub App: OAuth + installation flow | **CONFIRMED BUILT** | `ghic/service/github_app.py` — App JWT (RS256) → per-installation token, cached. Note: a GitHub *App* install flow is hosted by GitHub itself; a separate user-OAuth flow is only needed for a settings UI, which is roadmap (`docs/PRD.md` §10 multi-tenancy) |
| 15 | Webhooks: opened / edited / closed, label events | **PARTIAL** | `opened` + `closed` handled (`app.py:187-190`); `edited` and `labeled`/`unlabeled` fall through to ignored. → Phase 18 |
| 16 | Labels / Issues / Comments API | **CONFIRMED BUILT** | `github_app.py::post_comment`, `add_labels`, issue enrichment reads; exercised in `tests/test_service.py` webhook action tests |
| 17 | Checks API | **MISSING** | Nothing in the repo touches Checks. Honest scoping note: check runs require a commit `head_sha` — they attach to commits/PRs, not issues. → Phase 18 decides build-vs-documented-inapplicability |
| 18 | Projects API | **MISSING** | No Projects (v2 GraphQL) integration. → Phase 18 |
| 19 | Dashboard: six facets | **PARTIAL** | `/dashboard` covers confidence metrics (live precision/recall, positive rate) + latency + recent predictions. Missing: issue trends, duplicate rate, resolution analytics, label stats, component analytics. → Phase 21 |
| 20 | MLOps: Docker/CI, versioning, experiment tracking, retraining, monitoring | **PARTIAL** | Docker + CI **BUILT** (`Dockerfile`, `.github/workflows/ci.yml`); monitoring **BUILT** (`/stats`, ledger); versioning/tracking = model cards + `reports/champion.json` (lightweight-by-design); retraining pipeline not documented as a runnable procedure. → Phase 20 |
| 21 | Marketplace listing prep | **PARTIAL** | `docs/DEPLOYMENT.md` has the registration + Marketplace checklist; no logo/feature-card/screenshots. → Phase 22 |
| 22 | Architecture diagrams + consolidated benchmarks doc | **MISSING** | `docs/PRD.md` §2 has a text tree, no diagrams; benchmark numbers scattered across `models/*.md`, `reports/*.json`, PRD §8. → Phase 22 |

**Data facts the later phases depend on** (measured from
`data/processed/combined.csv`, 6,175 rows):

- Category ground truth per repo: vscode `bug`/`feature-request`/`*question`/`*duplicate`/`invalid`/`regression`; tensorflow `type:bug`/`type:feature`/`type:support`/`type:docs-bug`/`invalid`; react `type: bug`/`type: question` (sparse: ~113 total). Mapping table lives in Phase 15.
- No priority/severity labels anywhere in the corpus (checked all labels ≥5 occurrences per repo).
- Assignee/closer/commenter logins are not collected; `GH_TOKEN` is present in `.env`, so a supplemental collection pass is feasible (Phase 16).

**Gate: PASSED** — this table. Committed before Phase 15 work started.

## Phase 15 — Issue Category Classification (2026-07-14)

Built `ghic/category.py`: 6-class head (bug / feature / question / docs /
duplicate / invalid) trained on real ground truth — the category label a
maintainer eventually applied (2,747 of 6,175 issues), normalized across
repo conventions, multi-label conflicts resolved by documented priority.
Same protocol as the champion: per-repo chronological split, walk-forward
CV (macro-F1), one test evaluation. `security` (0 occurrences) and
standalone `regression` (27, folded into bug) are documented as
untrainable in this corpus, not faked.

Measured (chronological test, n=549, `models/CATEGORY_CARD.md`):
winner `logreg_balanced`, accuracy 0.583, macro-F1 0.470. Per-class F1:
bug 0.69, feature 0.65, question 0.48, duplicate 0.42, docs 0.40,
invalid 0.18. Note the duplicate row independently corroborates the
Phase 2 negative finding. Calibration: shipped uncalibrated — the
newest-15% calibration slice was missing rare classes entirely (the
guard in `train_category` refuses to fit a partial calibrator).

Service wiring (spec item 8, label recommendation): category ships as an
assistive suggestion in the webhook response and comment
(`GHIC_SUGGEST_CATEGORY`, on by default when `models/category.joblib`
exists); **no category label is ever applied automatically** — a test
asserts that. Feature frame is shared with the actionability head via
`inference.build_feature_frame` (one engineering pass per event).

**Gate: PASSED** — `models/CATEGORY_CARD.md` with real per-class
metrics + confusion matrix; 115 tests green (8 new). Committed.

## Phase 16 — Maintainer Assignment (2026-07-14) — ⚠ BLOCKED on credentials

Built `ghic/assign.py` end-to-end:

- **Collection extension** (`--collect`): assignees + participants + closing
  actor per issue, aliased 50-issue GraphQL batches, cached/resumable, same
  rate-limit discipline as `collect.py`. Output `data/processed/assignments.json`.
- **Causal evaluation** (`--evaluate`): similarity recommender (assignees/
  closers of the 20 most similar prior issues, TF-IDF, similarity-weighted)
  vs the naive most-active-assignee baseline; hit@{1,3,5} on the
  chronological test slice; auto card `models/ASSIGNMENT_CARD.md` whose
  Decision section applies the plan's rule (baseline wins → baseline ships).
- 12 tests, including a synthetic two-cluster corpus where the similarity
  recommender must route each cluster to its owner at k=1 (passes).

~~Gate initially blocked: the original `GH_TOKEN` was revoked (401 on
`/rate_limit`).~~ **Unblocked 2026-07-15** with a fresh user-provided
token; collection ran (~7 min, 124 cached batches), evaluation produced
real numbers (n=968 test issues with a human assignee):

| recommender | hit@1 | hit@3 | hit@5 |
|---|---|---|---|
| **similarity** (shipped) | 0.115 | **0.389** | 0.485 |
| most-active baseline | 0.000 | 0.261 | 0.406 |

Per the plan's decision rule the similarity mechanism ships — assistive,
response-level only (`suggested_assignees`), never an assignment action,
never an @-mention in a comment. A learned ranker is not built: it would
have to beat hit@3 0.389 on this protocol first.

**Gate: PASSED** — `models/ASSIGNMENT_CARD.md` with real top-k vs the
naive baseline; serving wired + tested (160 tests).

## Phase 17 — Priority / Severity / Effort (2026-07-14)

One card per head, as the gate requires — two correctly-not-shipped, one
shipped after clearing a pre-declared bar:

- **Priority — not shipped** (`models/PRIORITY_CARD.md`): zero priority
  labels in the corpus; every candidate proxy (time-to-close, response
  latency, reactions) fails for a stated reason. Buildable with repos that
  use P0–P3 labels (e.g. flutter/flutter) — a data problem, not a model one.
- **Severity — not shipped** (`models/SEVERITY_CARD.md`): no severity
  ground truth, and the tempting keyword-proxy is circular (labels derived
  from the model's own input text ⇒ near-tautological accuracy). Refused.
- **Effort — SHIPPED, bar met** (`models/EFFORT_CARD.md`): proxy =
  log1p(days-to-close), declared weak up front, with a pre-declared ship
  bar (Spearman ≥ 0.30 AND ≥ 10% MAE improvement over constant-median).
  Result on the chronological test (n=1,227): winner `rf_regressor`,
  **Spearman 0.492**, MAE improvement **11.8%** — bar met. Ships as four
  coarse buckets in the **API response only**, never the public comment
  (a time estimate shown to reporters reads as a commitment; the validated
  claim is rank-informativeness, MAE ≈ 1.5 log-days ≈ 4× typical factor).
  Engineering note: sklearn's RF-regression default `max_features=1.0`
  was intractable on the ~8k-dim sparse block; `sqrt` (set before any
  test-set look) made it tractable.

**Gate: PASSED** — three cards with real numbers/decisions. 155 tests.

## Phase 18 — GitHub API Surface (2026-07-14)

- **`issues.edited`** — BUILT: re-scores with the edited text and updates
  the pending ledger entry (so close-time grading judges the text
  maintainers actually triaged). Never posts/labels on edit — one issue,
  at most one comment. Surfaced as `rescored_after_edit` in `/stats`.
- **Label events (`labeled`/`unlabeled`)** — BUILT: recorded to the ledger
  (`type: "label_event"`), replayed on restart, counted in `/stats`. This
  is deliberate data collection: category labels are early ground truth
  for the category head, and timestamped duplicate labels are exactly the
  pairwise signal `DUPLICATE_CARD.md` names as missing.
- **Projects v2** — BUILT: `GHIC_PROJECT_ID` (board node ID) adds issues
  predicted actionable to the board via the GraphQL mutation (v2 has no
  REST API). Dry-run-respecting, audited like every write.
- **Checks API — documented as inapplicable, not built.** A check run
  requires a commit `head_sha`; it attaches to commits/PRs, not issues.
  For an issues-only App the only way to "show a check" is to pin issue
  predictions onto unrelated commits — misleading UX and permission scope
  creep (checks:write). If the product ever scores PRs, this is the first
  thing to build; for issues it is the wrong API, and building it anyway
  to tick a checklist line is exactly what the ground rules prohibit.

**Gate: PASSED** — a live test per event/API (edited re-score + pending
update, label-event record + replay, project add on positive/dry-run/
negative paths). 135 tests green. Committed.

## Phase 19 — CLI (2026-07-14)

`ghic` console entry (`ghic/cli.py`): `train` / `collect` / `label` /
`benchmark` (→ the backtest replay) / `serve` pass through to the existing
module CLIs; `predict` / `explain` score one issue end-to-end against the
local model artifact; `dashboard` opens the running service's dashboard.
No new logic — routing only. The `ghic-*` per-stage scripts remain.

**Gate: PASSED** — subcommand tests incl. real end-to-end predict/explain
against the trained champion (auto-skip when no artifact). Committed.

## Phase 20 — MLOps (2026-07-14)

- **Retraining pipeline**: `python -m ghic.retrain` = label → champion
  protocol → service-path backtest → category head → duplicate index →
  snapshot → registry. Operator-triggered by design (training data is
  deliberately not in the repo, so CI can't retrain); cron wrapper is a
  deployment choice, documented in DEPLOYMENT.md.
- **Versioning / experiment tracking**: per-run immutable snapshot in
  `reports/runs/<utc-ts>/` (both cards + all metric JSONs) plus one row in
  `models/REGISTRY.md` — run time, winner, headline metrics, artifact
  sha256. The lightest tool that answers "what changed, when, what did it
  score"; an experiment-tracking server at one model family and one
  operator would be process for its own sake.
- **The documented run** (the gate): run `2026-07-13T19-41-30Z`, executed
  end-to-end (~2 h wall, of which ~1.5 h was one CV fold stalled by
  machine throttling — pipeline recovered unaided). Produced a fresh
  `models/MODEL_CARD.md`, registry row (`rf_balanced`, test PR-AUC 0.7172,
  sha `5538c3115ba7`), and the snapshot dir. Metrics reproduced the
  original run exactly — seeded determinism verified.

**Gate: PASSED** — documented retraining run + new model card + registry
entry + snapshot; registry/snapshot mechanics covered by tests.

## Phase 21 — Dashboard (2026-07-14)

All six facets render from real ledger data (`tracker.analytics()`,
surfaced in `/stats` and drawn by `/dashboard`):

1. **Issue trends** — predictions/day (ledger records now carry
   timestamps; old lines replay without a trend point, never invented).
2. **Duplicate rate** — share of predictions with similar-prior
   candidates (`related_count` now recorded) + duplicate labels observed
   live via label events.
3. **Resolution analytics** — resolved actionable / non-actionable /
   awaiting, from graded outcomes.
4. **Confidence metrics** — P(actionable) decile histogram + mean.
5. **Label stats** — top maintainer-applied labels, observed live.
6. **Component analytics** — per-repo scored / positive rate / mean P
   (the component boundary this service actually has; finer component
   labels appear in label stats as maintainers apply them).

Facets a fresh deploy hasn't earned data for show zeros/empty — no
placeholders. **Gate: PASSED** — tests assert each facet reports real
counted numbers from webhook activity, incl. rebuild-from-ledger.

## Phase 22 — Docs, Benchmarks, Marketplace (2026-07-14)

- **`docs/BENCHMARKS.md`**: every number from Phases 2, 15–17 and the load
  tests in one place, each section naming the artifact that produced it;
  negative results listed with equal prominence. Includes a new real
  measurement: full-webhook-path latency with ALL heads enabled (p50
  300 ms / p95 683 ms / p99 988 ms over the 1,177-issue replay) — and the
  explanation of why that is *faster* than the explain-enabled API path.
- **Architecture diagrams**: two mermaid diagrams in `docs/PRD.md` §2
  (component flow + event sequence) reflecting the actual current
  structure, tree updated with the new modules.
- **Marketplace assets**: original logo (`docs/assets/logo.svg`), listing
  copy / feature card (`docs/assets/listing.md` — claims cross-checked
  against BENCHMARKS), checklist in DEPLOYMENT.md checked against real
  repo state — logo/description/support/privacy ✅; registration,
  installs, HTTPS domain, and screenshots explicitly *blocked on
  deployment* (screenshots come from a real deployment, never mockups).
- **Doc corrections found during the pass**: PRIVACY.md now discloses the
  optional Anthropic transmission when LLM drafting is enabled (was
  wrongly absolute before) and the ledger's action/label-event records;
  Dockerfile ships the new heads (`.dockerignore`); README/PRD/DEPLOYMENT
  updated for category/effort/edited/label-events/Projects/CLI/retrain.

**Gate: PASSED** — docs render (plain GFM + mermaid), checklist fully
checked or explicitly blocked-on-deployment. 155 tests, ruff clean.

---

## Post-plan status (2026-07-15)

**Phases 14–22 complete, all gates green.** The Phase 16 live evaluation
ran after the user supplied a fresh `GH_TOKEN`; the similarity recommender
won and is wired into the service. 160 tests, ruff clean.
