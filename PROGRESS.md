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

**Gate: NOT passed — blocked.** The live collection failed: `GH_TOKEN` in
`.env` returns 401 on a bare `/rate_limit` probe (unauthenticated requests
succeed → the token is revoked/expired, not a network/code issue). Real
top-k numbers cannot be produced without it, and per the ground rules they
will not be invented. **To unblock:** refresh `GH_TOKEN` in `.env`, then
`python -m ghic.assign --collect && python -m ghic.assign --evaluate`
(~5 min). Until then no assignment feature ships, and nothing in the
service references one.

## Phase 17 — Priority / Severity / Effort — pending

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

## Phase 19 — CLI — pending

## Phase 20 — MLOps — pending

## Phase 21 — Dashboard — pending

## Phase 22 — Docs / Benchmarks / Marketplace — pending
