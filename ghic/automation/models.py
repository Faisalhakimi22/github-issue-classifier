"""Audit-carrying recommendation types.

Phase 4 produces things a maintainer might act on, which raises the bar
from Phase 3: a wrong analysis wastes a few minutes of reading, while a
wrong recommendation that someone follows wastes an afternoon. So every
recommendation carries its own provenance and every one is *advisory*.

`Recommendation` is the unit. It carries the audit fields Feature 12 asks
for -- timestamp, inputs digest, evidence, confidence, engine version,
reason -- on the object itself rather than in a side log, because a
recommendation separated from its justification is exactly what nobody can
review later.

**`inputs_digest` is what makes a recommendation reproducible.** It's a
hash of the inputs that produced it, so "why did GHIC say that last
Tuesday" is answerable: same digest means same inputs, and a different
result from the same digest means the engine changed (which
`engine_version` then pins down).

## The safety property

`ActionKind` has no destructive members, by construction. There is no
`MERGE`, no `CLOSE`, no `DELETE`, no `WRITE_FILE` -- not because the code
declines to use them, but because they cannot be expressed. A future
contributor adding automation cannot accidentally produce a merge
recommendation through this type; they would have to add the enum member
first, which is a visible, reviewable change rather than a one-line slip.

Everything here produces *text for a human to read*. Nothing in this
package calls a GitHub write endpoint, and `tests/test_automation.py`
asserts that against the real client surface.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..engineering_intelligence.evidence import Confidence, Evidence


class ActionKind(str, Enum):
    """What a maintainer might do with a recommendation.

    Deliberately advisory-only. See the module docstring: the absence of
    destructive members is the enforcement mechanism, not an oversight.
    """
    REVIEW = "review"                  # read something
    INVESTIGATE = "investigate"        # follow a debugging step
    LABEL = "label"                    # consider applying a label
    ASSIGN = "assign"                  # consider a reviewer
    LINK = "link"                      # cross-reference another issue
    TEST = "test"                      # add or run a test
    DOCUMENT = "document"              # write a PR description, docs
    REQUEST_INFO = "request_info"      # ask the reporter for something


@dataclass(frozen=True)
class Recommendation:
    """One advisory suggestion, with everything needed to audit it.

    Like `Claim` in Phase 3, this refuses to exist without evidence -- a
    recommendation a maintainer can't trace is one they can't evaluate, and
    an untraceable suggestion in a triage comment is worse than no
    suggestion at all.
    """
    kind: ActionKind
    title: str
    detail: str
    confidence: Confidence
    reason: str
    evidence: list[Evidence] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    engine_version: str = ""
    inputs_digest: str = ""

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("a recommendation must have a title")
        if not self.reason.strip():
            raise ValueError(
                f"recommendation {self.title!r} has no reason. Every "
                "recommendation must say why it was made -- that is what "
                "makes it reviewable rather than an instruction."
            )
        if not self.evidence:
            raise ValueError(
                f"recommendation {self.title!r} has no evidence. A suggestion "
                "a maintainer cannot trace is one they cannot evaluate."
            )

    @property
    def is_advisory(self) -> bool:
        """Always True. Present so the property can be asserted in tests
        and read at a call site, making the guarantee explicit rather than
        implicit in the absence of write calls."""
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "title": self.title,
            "detail": self.detail,
            "confidence": self.confidence.value,
            "reason": self.reason,
            "evidence": [e.as_dict() for e in self.evidence],
            "created_at": self.created_at,
            "engine_version": self.engine_version,
            "inputs_digest": self.inputs_digest,
            "advisory": True,
        }


@dataclass(frozen=True)
class AuditRecord:
    """The provenance shared by every recommendation in one run.

    Built once per analysis and stamped onto each recommendation, so a
    bundle is internally consistent: everything in it demonstrably came
    from the same inputs at the same moment.
    """
    repo: str
    issue_number: int
    engine_version: str
    inputs_digest: str
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "issue_number": self.issue_number,
            "engine_version": self.engine_version,
            "inputs_digest": self.inputs_digest,
            "created_at": self.created_at,
        }


def digest_inputs(**inputs: Any) -> str:
    """A stable hash of the inputs behind a recommendation.

    Sorted keys and `default=str` so the digest is deterministic across
    runs and processes -- the property the whole audit trail rests on. Not
    a security primitive; it answers "were these the same inputs?", not
    "has someone tampered with this?".
    """
    payload = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class AutomationBundle:
    """Everything Phase 4 produced for one issue.

    Every field is optional and defaults empty: each capability is
    independently flagged (Feature 13), so a bundle with only labels in it
    is a normal, expected outcome rather than a partial failure.
    """
    audit: AuditRecord
    fix_plan: list[Recommendation] = field(default_factory=list)
    test_plan: list[Recommendation] = field(default_factory=list)
    checklist: list[Recommendation] = field(default_factory=list)
    labels: list[Recommendation] = field(default_factory=list)
    assignees: list[Recommendation] = field(default_factory=list)
    duplicate_workflow: list[Recommendation] = field(default_factory=list)
    pr_draft: str = ""

    @property
    def is_empty(self) -> bool:
        return not any([
            self.fix_plan, self.test_plan, self.checklist, self.labels,
            self.assignees, self.duplicate_workflow, self.pr_draft,
        ])

    @property
    def all_recommendations(self) -> list[Recommendation]:
        return [
            *self.fix_plan, *self.test_plan, *self.checklist,
            *self.labels, *self.assignees, *self.duplicate_workflow,
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "audit": self.audit.as_dict(),
            "advisory_only": True,
            "fix_plan": [r.as_dict() for r in self.fix_plan],
            "test_plan": [r.as_dict() for r in self.test_plan],
            "checklist": [r.as_dict() for r in self.checklist],
            "labels": [r.as_dict() for r in self.labels],
            "assignees": [r.as_dict() for r in self.assignees],
            "duplicate_workflow": [r.as_dict() for r in self.duplicate_workflow],
            "pr_draft": self.pr_draft,
        }
