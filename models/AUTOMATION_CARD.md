# Card — Engineering Automation (Phase 4)

**Task:** turn the analysis from Phases 1–3 into artifacts a maintainer
would otherwise write by hand — an implementation plan, a triage checklist,
a draft PR description, suggested tests, labels, reviewers, and a weekly
digest.

Everything is **advisory**. GHIC does not merge, close, delete, assign,
label, modify code, or change repository settings.

## How the safety guarantee is enforced

Feature 14 is a promise to a maintainer that a bot holding Issues write
permission will not touch their repository. A promise like that deserves
enforcement stronger than a policy, so it's enforced three ways:

**1. Destructive actions are unrepresentable.** `ActionKind` has no `MERGE`,
`CLOSE`, `DELETE`, or `WRITE`. There is nothing to select. Adding
automation that closes an issue would require adding the enum member
first — a visible, reviewable diff rather than a one-line slip inside a
larger change.

**2. `AutomationService` holds no GitHub client.** It cannot act on its own
suggestions because the capability isn't wired in. Not having a client is a
stronger guarantee than a policy of not calling one.

**3. A test walks the package source** and fails if any GitHub write method
(`post_comment`, `add_labels`, `installation_token`, …) is ever called from
it. That one catches the future contributor who imports the client "just
for a moment".

Plus: every `Recommendation` requires evidence *and* a reason — the
constructor raises without them — so an untraceable suggestion cannot reach
a maintainer's screen.

## Plans, not patches

The spec says not to generate full code patches. That's also the right call
independently: generating a diff for a codebase the tool has seen only
fragments of produces code that looks right and doesn't compile, and
reviewing a wrong patch costs more than writing a right one.

What saves a maintainer time is knowing **which function**, **why**, and
**what to verify**. So a fix plan names the file, the line span, the
symbol, and the commit worth diffing against — and stops there. A test
asserts no diff markers ever appear in a fix plan.

Same reasoning shapes the PR draft: it is **deliberately incomplete where
only a human can fill it in**. "Suggested changes" names the files and
leaves the change description as an HTML comment; "Breaking changes" is
blank because GHIC cannot determine that from an issue. A draft that looks
finished invites being submitted unread, which is the failure mode this
whole phase exists to avoid — the gaps are load-bearing.

## Evidence-backed suggestions only

| Capability | Fires on |
|---|---|
| Fix plan | a Phase 3 root-cause hypothesis + retrieved code (both required) |
| Test plan | retrieved code; regression markers; recurrence finding; edge-case terms in the report |
| Checklist | presence/absence of reproduction markers; retrieved code; root cause; regression |
| Labels | a phrase in the report, a path in retrieved code, or a Phase 3 finding |
| Assignees | commit authorship over the implicated files |
| Duplicate workflow | Phase 3's `classify_relationship` |

**A fix plan requires a root cause.** Without one there is no
evidence-backed opinion about what to change, and "review the code" is the
absence of a plan dressed up as advice.

**`good-first-issue` needs positive evidence of narrowness** — one or two
files retrieved, reproduction included, no regression or recurrence
signals — not merely the absence of complexity markers. Guessing wrong
sends a newcomer into a hard problem, which is worse than staying silent.

**A regression family is never recommended for closure.** Two reports of
the same recurring break are not duplicates; closing the second destroys
the signal that it happened twice. The recommendation explicitly says
"keep this open and link".

**Bot authors are excluded from reviewer suggestions.** Suggesting
`dependabot[bot]` review a bug wastes the one thing this phase is meant to
save.

## Reproducibility (Feature 12)

Every recommendation carries `inputs_digest`, `engine_version`,
`created_at`, `confidence`, `reason`, and its evidence — on the object, not
in a side log, because a recommendation separated from its justification is
what nobody can review later.

`inputs_digest` is a stable hash of the inputs (sorted keys,
`default=str`), so "why did GHIC say that last Tuesday" is answerable: the
same digest means the same inputs, and a different result from the same
digest means the engine changed, with `engine_version` pinning which one.
It answers "were these the same inputs?" — not "has this been tampered
with?"; it isn't a security primitive.

## Feature flags (Feature 13)

Each capability is independent and **defaults off**:

```bash
GHIC_FIX_SUGGESTIONS=false
GHIC_PR_DRAFTS=false
GHIC_TEST_PLANS=false
GHIC_CHECKLISTS=false
GHIC_SMART_LABELS=false
GHIC_ASSIGNEE_RECOMMENDATIONS=false
GHIC_DUPLICATE_WORKFLOW=false
```

Automation with no retrieved evidence produces nothing at all, rather than
generic advice.

## API (Feature 10)

```
POST /automation/analyze              advisory bundle for one issue
GET  /automation/weekly-digest        ?repo=&fmt=markdown|text
GET  /automation/repository-analytics ?repo=  (dashboard widgets)
```

**One `/automation/analyze` rather than five endpoints.** `/fix-plan`,
`/pr-draft`, `/test-plan`, and `/checklist` share every input and the same
retrieval + analysis pipeline; separate endpoints would re-retrieve the
same evidence up to four times for one issue. Callers get the whole bundle
and read the part they want. All are token-gated and read-only.

Documented via the existing OpenAPI generation — FastAPI derives it from
the route signatures, and `test_openapi_spec_covers_all_endpoints` already
guards coverage.

## What isn't built

**Feature 8's release risk summary is partial.** It needs a merged-PR
webhook event to be genuinely useful, and GHIC currently subscribes to
issues only. The component-level inputs exist (`ComponentStats`,
`summarize_repository`); wiring them to a `pull_request.closed` handler is
the remaining work, and it needs a permissions change to the App.

**Feature 11's dashboard widgets are API-only.** `/automation/repository-analytics`
returns hotspots, clusters, component health, and the risk summary. The
React dashboard is a separate repository; rendering them there is separate
work.

**No Slack/Teams integration.** `build_weekly_digest(fmt="text")` produces
something readable when pasted. A real integration would use Block Kit or
an Adaptive Card and a webhook URL — that's a delivery mechanism this
project has no credentials for and couldn't test.
