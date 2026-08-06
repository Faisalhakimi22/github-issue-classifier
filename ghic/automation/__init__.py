"""Engineering automation: advisory artifacts that save maintainer time.

Phase 4 turns the analysis of Phases 1-3 into things a maintainer would
otherwise write by hand -- an implementation plan, a triage checklist, a
draft PR description, suggested tests, labels, reviewers, and a weekly
digest.

## The safety guarantee, and how it's enforced

Everything here is **advisory**. GHIC does not merge, close, delete,
assign, label, modify code, or change repository settings. That is
enforced structurally rather than by policy:

  **`ActionKind` has no destructive members.** There is no `MERGE`,
  `CLOSE`, or `DELETE` to select. Adding automation that closes an issue
  would require adding the enum member first -- a visible, reviewable
  change rather than a one-line slip.

  **`AutomationService` holds no GitHub client.** It cannot act on its own
  suggestions because the capability isn't wired in. A policy not to call
  a client is weaker than not having one.

  **Every `Recommendation` requires evidence and a reason.** The
  constructor raises without them, so an untraceable suggestion cannot
  reach a maintainer's screen.

`tests/test_automation.py` asserts these against the real client surface,
including a test that fails if any GitHub write method is ever reachable
from this package.

## Reproducibility

Every recommendation carries `inputs_digest` (a stable hash of what
produced it), `engine_version`, `created_at`, `confidence`, `reason`, and
its evidence. Same digest means same inputs; a different result from the
same digest means the engine changed, and `engine_version` says which one.

See models/AUTOMATION_CARD.md.
"""
from .models import (
    ActionKind,
    AuditRecord,
    AutomationBundle,
    Recommendation,
    digest_inputs,
)
from .pr_draft import build_pr_draft, build_weekly_digest, render_automation_sections
from .service import AutomationFlags, AutomationService
from .suggestions import (
    build_assignee_suggestions,
    build_checklist,
    build_duplicate_workflow,
    build_fix_plan,
    build_label_suggestions,
    build_test_plan,
)

__all__ = [
    "ActionKind",
    "AuditRecord",
    "AutomationBundle",
    "Recommendation",
    "digest_inputs",
    "AutomationService",
    "AutomationFlags",
    "build_fix_plan",
    "build_test_plan",
    "build_checklist",
    "build_label_suggestions",
    "build_assignee_suggestions",
    "build_duplicate_workflow",
    "build_pr_draft",
    "build_weekly_digest",
    "render_automation_sections",
]
