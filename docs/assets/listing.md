# Marketplace listing copy (draft assets)

Everything here describes only what the product measurably does — claims
are backed by `docs/BENCHMARKS.md`, and the negative results those numbers
include are why some obvious-sounding features are deliberately absent.

## Name

**Issue Triage Bot**

## Tagline (≤80 chars)

> Calibrated triage the moment an issue is opened — and it grades itself live.

## Feature card (listing description)

**The moment an issue is opened, this App predicts whether it will end as an
actionable bug** — fixed by a merged PR — or die as a duplicate, a won't-fix,
or silence, and posts a calibrated probability with the evidence behind it.

- **Calibrated, not vibes.** Walk-forward temporal validation, isotonic
  calibration, and a replay of 1,177 held-out issues through the production
  webhook path. When it says 0.7, it means roughly 70%.
- **It grades itself in production.** Every prediction is scored against the
  real outcome when the issue closes; live precision/recall are on your
  dashboard, not in a slide deck.
- **Assistive, never destructive.** Dry-run by default; comments, labels,
  and project placement are individually opt-in. Category suggestions and
  "possibly related" prior issues are surfaced for a human to confirm —
  never acted on automatically.
- **Asks for what's missing.** Under-specified reports get a courteous,
  specific request for repro steps/traces (deterministic trigger; optional
  LLM drafting with a template fallback).
- **Minimum permissions.** Issues read/write + metadata. No code access, no
  PRs, no members.

## Categories

Project management · Utilities

## Screenshot checklist (taken from a real deployment — never mocked)

- [ ] `/dashboard` with live analytics after a week on a real repo
- [ ] a prediction comment on a real issue (dry-run lifted)
- [ ] a "missing information" request on a vague issue

## Logo

`docs/assets/logo.svg` (128×128, original artwork: issue dot → triage
funnel → verified tick).
