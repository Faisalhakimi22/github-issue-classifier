# BUILD_PLAN_2.md — Gap Closure (Phases 14–22)

Continues `BUILD_PLAN.md` (Phases 0–13, already executed). The same
`CLAUDE.md` non-negotiables govern everything below — don't restate
them, follow them. Same working process as before: one phase per
session, gate before advancing, update `PROGRESS.md` every time.

---

## Phase 14 — Gap Audit (do this before building anything)

Cross-check the repo against the full original spec below — not just
`docs/PRD.md`'s own BUILT/Planned table, since that table is exactly
where two items already went missing once and a self-referential check
would miss them again. For each line, mark **CONFIRMED BUILT** (cite the
file/module and where its validation lives), **CONFIRMED PLANNED-ONLY**
(intentional — cite the roadmap section and the usage signal it's gated
on), or **MISSING**.

Original spec to check against:
1. Issue category (bug/enhancement/question/duplicate/documentation/
   invalid/security/regression)
2. Actionable vs non-actionable
3. Duplicate detection
4. Priority prediction
5. Severity prediction
6. Effort estimation
7. Missing-info detection + comment drafting
8. Label recommendation
9. Maintainer assignment recommendation
10. Confidence scores
11. Natural-language explanation
12. REST API
13. CLI (train / predict / explain / dashboard / benchmark / serve)
14. GitHub App: OAuth + installation flow
15. Webhooks: issue opened / edited / closed, label events
16. Labels / Issues / Comments API
17. Checks API
18. Projects API
19. Dashboard: issue trends, duplicate rate, resolution analytics,
    confidence metrics, label stats, component analytics (six facets)
20. MLOps: Docker/CI, model versioning, experiment tracking, retraining
    pipeline, monitoring
21. Marketplace listing prep (assets, checklist)
22. Architecture diagrams + a consolidated benchmarks doc

**Gate:** `PROGRESS.md` updated with an explicit table mapping every
line above to BUILT / PLANNED / MISSING, each with a file reference.
Commit before starting Phase 15 — every phase after this one executes
against that table, not against assumption.

## Phase 15 — Issue Category Classification
- New module (e.g. `category_prediction/`), reusing `feature_engineering/`.
  Unlike priority/severity/effort, this one has real ground truth: the
  eventual bug/enhancement/question/documentation label a maintainer
  actually applied. Use it — same causal rule as everywhere else:
  predict from what's visible at open time, target is the eventual
  label.
- Multi-class, not binary — report a full confusion matrix and
  per-class precision/recall, not just aggregate accuracy. Walk-forward
  CV, calibration, held-out eval, model card, same protocol as the
  champion model.
- **Gate:** model card with real per-class metrics. Commit.

## Phase 16 — Maintainer Assignment Recommendation
- Check first: does `collect.py` capture historical assignee data? If
  not, extend it — same rate-limit/caching discipline as the rest of
  collection — before modeling anything.
- Start with the honest baseline, not a model: surface maintainers who
  closed or commented on the most similar prior issues, reusing the
  Phase 2 similarity index. Validate it — does the real historical
  assignee show up in the top-k suggestions more often than a
  most-active-committer baseline? Report the real number.
- Only build a learned ranker on top of this if the baseline actually
  underperforms it by a meaningful margin. If the baseline wins, the
  baseline is the shipped feature — that's a fine outcome, not a
  shortfall.
- Assistive only, never an automatic assignment action — same posture
  as duplicate detection, for the same reason: a wrong automatic
  assignment affects a real person's workload, not just a label.
- **Gate:** a model/baseline card with the real top-k number vs. the
  naive baseline. Commit.

## Phase 17 — Priority, Severity, Effort (verify-or-build)
- For each: if Phase 14 shows it already built, verify it has a
  documented proxy label, a held-out metric, and a model card. Missing
  any of the three means it isn't actually done — finish it.
- If not built: original Phase 2 spec — proxy label documented as an
  explicit assumption, walk-forward CV, calibration, held-out eval.
- Effort estimation specifically is the weakest natural proxy of the
  three (time-to-close conflates effort with backlog priority; PR size
  conflates effort with scope creep). If it doesn't calibrate
  meaningfully, don't ship it — write the negative result into a model
  card and move on, the same way duplicate detection's negative result
  got handled. A missing feature with an honest card beats a shipped one
  nobody validated.
- **Gate:** a model card per head, including any correctly not-shipped.
  Commit.

## Phase 18 — GitHub API Surface Completion
- Verify-or-build: Checks API — a Check Run showing the prediction
  inline. Do this one regardless of what else in this phase is missing;
  it's small scope and reads as native GitHub UX rather than bolted-on.
- Verify-or-build: Projects API integration.
- Verify-or-build: webhook handling for `issues.edited`, `issues.closed`,
  `label` events — Phase 1 asked for this; confirm it's actually wired,
  not just planned.
- **Gate:** a test per event/API confirming it's live, not just present
  in code. Commit.

## Phase 19 — CLI
- Verify-or-build: `ghic train`, `predict`, `explain`, `dashboard`,
  `benchmark`, `serve` — thin wrappers over what already exists, no new
  logic.
- **Gate:** each subcommand has an end-to-end test. Commit.

## Phase 20 — MLOps Completion
- Verify-or-build: model versioning beyond the existing model cards,
  experiment tracking (lightest tool that works — a versioned directory
  of model cards is a legitimate answer, don't reach for heavier infra
  without a stated reason), a retraining pipeline (scheduled or
  triggered).
- **Gate:** a documented retraining run (even a manually triggered one)
  that produces a new model card. Commit.

## Phase 21 — Dashboard Completion
- Check which of the six facets (issue trends, duplicate rate,
  resolution analytics, confidence metrics, label stats, component
  analytics) the existing dashboard actually covers.
- Build whichever are missing, backed by real ledger data.
- **Gate:** all six render with real numbers, not placeholders. Commit.

## Phase 22 — Docs, Benchmarks, Marketplace Assets
- Assemble the scattered benchmark numbers (Phase 5's load test, every
  model card from Phases 2 and 15–17) into one `docs/BENCHMARKS.md` —
  no number in it that wasn't produced by an actual gate.
- Architecture diagrams (mermaid is fine) reflecting the real, current
  structure — not the aspirational one.
- Marketplace listing assets: logo, feature card, screenshots, and the
  submission checklist from `BUILD_PLAN.md` Phase 13, checked off
  against actual repo state.
- **Gate:** docs build/render cleanly; checklist fully checked or
  explicitly marked blocked-on-deployment. Commit.

---

Same rule as before: update `PROGRESS.md` after every phase, don't carry
a red gate forward, and if a feature doesn't validate, that's a model
card, not a missing commit.
