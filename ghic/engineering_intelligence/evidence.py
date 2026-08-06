"""Evidence attribution: the spine every Phase 3 conclusion hangs from.

This module exists first and everything else depends on it, because the
difference between an engineering intelligence system and a plausible-
sounding text generator is entirely here: whether a statement can be traced
to something that was actually retrieved.

Two ideas do the work.

**`Evidence` is a pointer, never prose.** It names a thing that exists in
the index -- a commit SHA, an issue number, a file and line span, a release
tag, or a counted statistic -- and it is constructed from retrieved chunks,
not written by a model. If it's in an `Evidence`, it was in the index.

**A `Claim` cannot exist without evidence.** The constructor enforces it.
There is no way to express "the CSV parser is probably broken" in this
system without attaching the chunk that suggests it, which is what makes
the guarantee mechanical rather than aspirational. `Claim.status`
distinguishes what was *observed* (`VERIFIED` -- a commit touched this file,
this issue was closed as completed) from what was *reasoned* (`INFERRED` --
this commit may be related to this issue), and both render differently in
the report so a maintainer always knows which they're reading.

Confidence is computed from the evidence, deterministically -- never
self-reported by a model. A model asked "how confident are you?" produces
a number that correlates with fluency, not correctness; counting
independent supporting artifacts at least measures something real.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceKind(str, Enum):
    """What kind of artifact backs a claim. str-valued so it serializes
    into the API response without an adapter."""
    CODE = "code"
    COMMIT = "commit"
    ISSUE = "issue"
    PULL_REQUEST = "pull_request"
    RELEASE = "release"
    STATISTIC = "statistic"      # a count over indexed history
    ISSUE_TEXT = "issue_text"    # a quoted phrase from the issue being triaged


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def label(self) -> str:
        return self.value.capitalize()


class ClaimStatus(str, Enum):
    """Whether a statement was observed or reasoned to.

    The distinction a maintainer needs and almost never gets: "commit
    abc123 modified csv_parser.py" is a fact from the index, while "that
    commit may have introduced this bug" is an inference from timing and
    overlap. Both are useful; conflating them is how a tool loses trust the
    first time it's confidently wrong.
    """
    VERIFIED = "verified"
    INFERRED = "inferred"


@dataclass(frozen=True)
class Evidence:
    """A pointer to one retrieved artifact."""
    kind: EvidenceKind
    reference: str                 # "abc1234", "#412", "src/csv.py:10-40", "v2.8.0"
    detail: str = ""               # a short human-readable descriptor
    url: str = ""
    timestamp: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "reference": self.reference,
            "detail": self.detail,
            "url": self.url,
            "timestamp": self.timestamp,
        }

    def render(self) -> str:
        """Markdown for the report. Links when a URL is known."""
        label = f"`{self.reference}`" if self.kind != EvidenceKind.STATISTIC else self.reference
        if self.url:
            label = f"[{self.reference}]({self.url})"
        return f"{label} — {self.detail}" if self.detail else label

    @classmethod
    def from_chunk(cls, chunk: Any) -> Evidence:
        """Build from a retrieved `CodeChunk`.

        The only sanctioned way to make code/commit/issue evidence: it
        cannot name an artifact the retriever didn't return, because it has
        nothing to read but the chunk.
        """
        source = getattr(chunk, "source", "code")
        if source == "commit":
            return cls(
                kind=EvidenceKind.COMMIT,
                reference=chunk.reference or chunk.path,
                detail=(chunk.symbol or "").strip()[:120],
                url=chunk.url, timestamp=chunk.timestamp,
            )
        if source == "pull_request":
            return cls(
                kind=EvidenceKind.PULL_REQUEST,
                reference=chunk.reference or chunk.path,
                detail=(chunk.symbol or "").strip()[:120],
                url=chunk.url, timestamp=chunk.timestamp,
            )
        if source == "issue":
            return cls(
                kind=EvidenceKind.ISSUE,
                reference=chunk.reference or chunk.path,
                detail=(chunk.symbol or "").strip()[:120],
                url=chunk.url, timestamp=chunk.timestamp,
            )
        span = f"{chunk.path}:{chunk.start_line}-{chunk.end_line}"
        return cls(
            kind=EvidenceKind.CODE,
            reference=span,
            detail=chunk.qualified_symbol or "",
        )


@dataclass(frozen=True)
class Claim:
    """A statement with its backing. Cannot be constructed without evidence.

    `__post_init__` raising is the enforcement point for the entire
    phase: no code path anywhere can produce an unsupported conclusion,
    because the type refuses to hold one.
    """
    statement: str
    status: ClaimStatus
    confidence: Confidence
    evidence: list[Evidence] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("a claim must say something")
        if not self.evidence:
            raise ValueError(
                f"claim {self.statement!r} has no evidence. Every Phase 3 "
                "conclusion must point at a retrieved artifact -- if there is "
                "nothing to point at, the claim must not be made."
            )

    @property
    def is_verified(self) -> bool:
        return self.status is ClaimStatus.VERIFIED

    def as_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "status": self.status.value,
            "confidence": self.confidence.value,
            "evidence": [e.as_dict() for e in self.evidence],
        }

    def render(self, *, show_status: bool = True) -> str:
        prefix = ""
        if show_status:
            prefix = "**Observed** — " if self.is_verified else "**Inferred** — "
        return prefix + self.statement


def confidence_from_support(
    *, distinct_sources: int, strong_signal: bool = False, contradicted: bool = False
) -> Confidence:
    """Confidence from countable support, not from a model's self-report.

    `distinct_sources` is the number of *independent* artifact kinds
    agreeing (a commit and an issue and a file beats three commits, which
    could all be one refactor). `strong_signal` marks an explicit textual
    marker -- the reporter saying "this worked in 2.7" is worth more than
    a similarity score. `contradicted` caps the result when something
    disagrees, because "several signals, one of which says otherwise" is
    exactly when a tool should hedge rather than assert.

    Deliberately coarse: three buckets are all this evidence supports, and
    a finer scale would imply a precision that isn't there.
    """
    if contradicted:
        return Confidence.LOW
    if distinct_sources >= 3 or (distinct_sources >= 2 and strong_signal):
        return Confidence.HIGH
    if distinct_sources == 2 or strong_signal:
        return Confidence.MEDIUM
    return Confidence.LOW


# Hedging vocabulary, applied at the point a claim is written rather than
# left to a prompt. The spec is explicit that certainty is never claimed;
# doing it in code means it can't be forgotten under a different prompt.
_HEDGE_BY_CONFIDENCE = {
    Confidence.HIGH: "likely",
    Confidence.MEDIUM: "possibly",
    Confidence.LOW: "may be",
}


def hedge(confidence: Confidence) -> str:
    """The verb qualifier matching a confidence level."""
    return _HEDGE_BY_CONFIDENCE[confidence]


@dataclass(frozen=True)
class Finding:
    """A titled group of claims -- one section of the engineering report.

    A finding with no claims is not rendered at all, which is how "we found
    nothing here" stays silent instead of producing an empty heading with a
    confident-looking title above it.
    """
    title: str
    claims: list[Claim] = field(default_factory=list)
    summary: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.claims

    @property
    def confidence(self) -> Confidence:
        """The finding's confidence is its best-supported claim's."""
        if not self.claims:
            return Confidence.LOW
        order = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}
        return max((c.confidence for c in self.claims), key=lambda c: order[c])

    @property
    def all_evidence(self) -> list[Evidence]:
        seen: dict[tuple[str, str], Evidence] = {}
        for claim in self.claims:
            for item in claim.evidence:
                seen.setdefault((item.kind.value, item.reference), item)
        return list(seen.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "confidence": self.confidence.value,
            "claims": [c.as_dict() for c in self.claims],
        }
