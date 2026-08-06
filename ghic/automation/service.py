"""Automation orchestration: flags in, advisory bundle out.

One entry point (`AutomationService.build`) that assembles whichever
capabilities are enabled. Each is independently flagged (Feature 13), so a
deployment can run label suggestions without PR drafts, or checklists
without fix plans, and a bundle containing only one of them is a normal
outcome rather than a partial failure.

Never raises: automation is the least essential thing GHIC does, and a
generator failing must not cost the analysis above it, let alone the
webhook. Each capability is wrapped individually so one broken generator
doesn't take the rest of the bundle with it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import utils
from .models import AuditRecord, AutomationBundle, digest_inputs
from .pr_draft import build_pr_draft
from .suggestions import (
    build_assignee_suggestions,
    build_checklist,
    build_duplicate_workflow,
    build_fix_plan,
    build_label_suggestions,
    build_test_plan,
)

logger = utils.get_logger(__name__)


@dataclass(frozen=True)
class AutomationFlags:
    """Per-capability switches (Feature 13).

    All default False. Automation is additive and opinionated, and a
    deployment should turn on the pieces it wants rather than discovering
    them in a comment.
    """
    fix_suggestions: bool = False
    pr_drafts: bool = False
    test_plans: bool = False
    checklists: bool = False
    smart_labels: bool = False
    assignee_recommendations: bool = False
    duplicate_workflow: bool = False

    @property
    def any_enabled(self) -> bool:
        return any([
            self.fix_suggestions, self.pr_drafts, self.test_plans, self.checklists,
            self.smart_labels, self.assignee_recommendations, self.duplicate_workflow,
        ])

    @classmethod
    def from_env(cls) -> AutomationFlags:
        import os

        def flag(name: str) -> bool:
            return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            fix_suggestions=flag("GHIC_FIX_SUGGESTIONS"),
            pr_drafts=flag("GHIC_PR_DRAFTS"),
            test_plans=flag("GHIC_TEST_PLANS"),
            checklists=flag("GHIC_CHECKLISTS"),
            smart_labels=flag("GHIC_SMART_LABELS"),
            assignee_recommendations=flag("GHIC_ASSIGNEE_RECOMMENDATIONS"),
            duplicate_workflow=flag("GHIC_DUPLICATE_WORKFLOW"),
        )


class AutomationService:
    """Builds the advisory bundle for one issue.

    Holds no GitHub client, by design. It has no way to act on anything it
    suggests -- the capability simply isn't wired in, which is a stronger
    guarantee than a policy not to use one.
    """

    def __init__(
        self, flags: AutomationFlags | None = None, *, engine_version: str = "",
    ) -> None:
        self.flags = flags or AutomationFlags()
        self.engine_version = engine_version or _default_version()

    def build(
        self,
        repo: str,
        issue_number: int,
        title: str,
        body: str,
        *,
        analysis: Any = None,
        repository_context: Any = None,
    ) -> AutomationBundle | None:
        """None when nothing is enabled or there's no evidence to work from."""
        if not self.flags.any_enabled:
            return None
        if repository_context is None or getattr(repository_context, "is_empty", True):
            # Automation with no retrieved evidence would be generic advice,
            # which is the filler this project keeps refusing to ship.
            return None

        audit = AuditRecord(
            repo=repo,
            issue_number=issue_number,
            engine_version=self.engine_version,
            inputs_digest=digest_inputs(
                repo=repo, issue_number=issue_number, title=title, body=body,
                evidence=[
                    (rc.chunk.path, rc.chunk.reference, round(rc.score, 4))
                    for rc in repository_context.chunks
                ],
                flags=self.flags.__dict__,
            ),
        )
        stamp = {
            "engine_version": audit.engine_version,
            "inputs_digest": audit.inputs_digest,
        }
        bundle = AutomationBundle(audit=audit)

        if self.flags.smart_labels:
            bundle.labels = self._safe(
                "labels", build_label_suggestions,
                analysis, repository_context, title, body, stamp=stamp,
            )
        if self.flags.checklists:
            bundle.checklist = self._safe(
                "checklist", build_checklist,
                analysis, repository_context, title, body, stamp=stamp,
            )
        if self.flags.fix_suggestions:
            bundle.fix_plan = self._safe(
                "fix plan", build_fix_plan, analysis, repository_context, stamp=stamp,
            )
        if self.flags.test_plans:
            bundle.test_plan = self._safe(
                "test plan", build_test_plan,
                analysis, repository_context, title, body, stamp=stamp,
            )
        if self.flags.duplicate_workflow:
            bundle.duplicate_workflow = self._safe(
                "duplicate workflow", build_duplicate_workflow,
                repository_context, title, body, stamp=stamp,
            )
        if self.flags.assignee_recommendations:
            bundle.assignees = self._safe(
                "assignees", build_assignee_suggestions, repository_context, stamp=stamp,
            )
        if self.flags.pr_drafts:
            try:
                bundle.pr_draft = build_pr_draft(
                    repo, issue_number, title, analysis, repository_context,
                    test_plan=bundle.test_plan,
                )
            except Exception as e:
                logger.warning("pr draft failed for %s#%s: %s", repo, issue_number, e)

        return None if bundle.is_empty else bundle

    def _safe(self, label: str, fn: Any, *args: Any, **kwargs: Any) -> list[Any]:
        """One generator failing must not cost the rest of the bundle."""
        try:
            return fn(*args, **kwargs) or []
        except Exception as e:
            logger.warning("automation generator %r failed: %s", label, e)
            return []


def _default_version() -> str:
    try:
        from ..service.app import __version__

        return __version__
    except Exception:
        return "unknown"
