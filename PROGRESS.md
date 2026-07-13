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

## Phase 15 — Issue Category Classification — pending

## Phase 16 — Maintainer Assignment — pending

## Phase 17 — Priority / Severity / Effort — pending

## Phase 18 — GitHub API Surface — pending

## Phase 19 — CLI — pending

## Phase 20 — MLOps — pending

## Phase 21 — Dashboard — pending

## Phase 22 — Docs / Benchmarks / Marketplace — pending
