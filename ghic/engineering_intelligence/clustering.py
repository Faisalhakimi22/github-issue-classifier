"""Duplicate intelligence: relationship classification and issue clustering.

Semantic similarity alone answers "are these two issues worded alike",
which is not the question a maintainer has. They want to know *which kind*
of relationship this is, because the correct action differs completely:

  **Probable duplicate** -- close it and link. Very high text similarity
  plus the same component.
  **Related issue** -- cross-reference it. Same component, related
  symptoms, different specifics.
  **Recurring problem** -- the area keeps producing issues; the fix may be
  structural rather than another patch.
  **Regression family** -- this broke, was fixed, and broke again. The
  most actionable finding of the four and the one similarity alone will
  never surface, because a regression and its original report often share
  little vocabulary.

Classification uses signals beyond text: component overlap, shared labels,
regression markers, and time. Thresholds are deliberately conservative --
the cost of a wrong "probable duplicate" is a maintainer closing a real
bug, so that label requires the strongest evidence and everything
ambiguous degrades to the weaker, safer one.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .. import utils
from .components import component_for_path, is_regression_text

logger = utils.get_logger(__name__)

# Relationship labels, strongest claim first.
DUPLICATE = "probable duplicate"
REGRESSION_FAMILY = "regression family"
RECURRING = "recurring problem"
RELATED = "related issue"

# Retrieval similarity required before "duplicate" is even considered.
# High on purpose: this label invites closing an issue.
_DUPLICATE_SIMILARITY = 0.55
_RELATED_SIMILARITY = 0.25

_DAY = 86400.0


@dataclass
class IssueCluster:
    """A group of historical issues sharing a component and a theme."""
    component: str
    references: list[str] = field(default_factory=list)
    regression_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def size(self) -> int:
        return len(self.references)

    @property
    def is_regression_family(self) -> bool:
        """Two or more regressions in one component is a pattern, not luck."""
        return self.regression_count >= 2

    @property
    def span_days(self) -> float:
        if not (self.first_seen and self.last_seen):
            return 0.0
        return max(0.0, (self.last_seen - self.first_seen) / _DAY)

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "size": self.size,
            "references": self.references,
            "regression_count": self.regression_count,
            "is_regression_family": self.is_regression_family,
            "span_days": round(self.span_days, 1),
            "label": RECURRING if self.size >= 3 else RELATED,
        }


def classify_relationship(
    *,
    similarity: float,
    issue_text: str,
    candidate_text: str,
    issue_components: set[str],
    candidate_components: set[str],
    shared_labels: int = 0,
) -> str | None:
    """What kind of relationship one retrieved issue has to this one.

    None means "not related enough to mention" -- the same discipline as
    the retrieval similarity floor. Ordering matters: a regression family
    is checked before duplicate, because two reports of the same recurring
    break are *not* duplicates (closing the second loses the signal that it
    happened twice).
    """
    if similarity < _RELATED_SIMILARITY:
        return None

    component_overlap = bool(issue_components & candidate_components)
    both_regressions = is_regression_text(issue_text) and is_regression_text(candidate_text)

    if both_regressions and component_overlap:
        return REGRESSION_FAMILY
    if similarity >= _DUPLICATE_SIMILARITY and component_overlap:
        return DUPLICATE
    if similarity >= _DUPLICATE_SIMILARITY and shared_labels >= 2:
        return DUPLICATE
    if component_overlap or shared_labels >= 1 or similarity >= _RELATED_SIMILARITY:
        return RELATED
    return None


def cluster_history(chunks: list[Any], *, min_cluster_size: int = 2) -> list[IssueCluster]:
    """Group historical issues by the component their fixes touched.

    Clusters on component rather than text, which is the point: two issues
    described completely differently that were both fixed in the same
    module belong together, and text clustering never finds them. Component
    attribution comes from `components.ComponentAnalyzer`'s
    commit-to-issue linkage, so a chunk that was never linked to a
    component is simply not clustered rather than guessed at.
    """
    from .components import ComponentAnalyzer

    stats = ComponentAnalyzer().analyze(chunks)
    if not stats:
        return []

    # Rebuild the issue -> component mapping the analyzer established, so
    # clustering agrees with the health numbers by construction.
    issue_chunks = {
        c.reference: c for c in _iter_chunks(chunks)
        if getattr(c, "source", "") in ("issue", "pull_request") and c.reference
    }
    by_component: dict[str, IssueCluster] = defaultdict(lambda: IssueCluster(component=""))

    for chunk in _iter_chunks(chunks):
        if getattr(chunk, "source", "") != "commit":
            continue
        from .components import referenced_issue_numbers

        touched = _commit_paths(chunk.text)
        components = {component_for_path(p) for p in touched} - {""}
        for number in referenced_issue_numbers(f"{chunk.symbol} {chunk.text}"):
            reference = f"#{number}"
            issue_chunk = issue_chunks.get(reference)
            if issue_chunk is None:
                continue
            for component in components:
                cluster = by_component[component]
                cluster.component = component
                if reference not in cluster.references:
                    cluster.references.append(reference)
                    if is_regression_text(f"{issue_chunk.symbol} {issue_chunk.text}"):
                        cluster.regression_count += 1
                    stamp = issue_chunk.timestamp
                    if stamp:
                        cluster.first_seen = min(cluster.first_seen or stamp, stamp)
                        cluster.last_seen = max(cluster.last_seen, stamp)

    clusters = [c for c in by_component.values() if c.size >= min_cluster_size]
    clusters.sort(key=lambda c: (-c.regression_count, -c.size))
    return clusters


def _iter_chunks(chunks: list[Any]):
    """Accept either raw chunks or RetrievedChunk wrappers.

    Callers have one or the other depending on whether they're working
    from an index build or a retrieval result; making this tolerant avoids
    forcing every call site to unwrap.
    """
    for item in chunks:
        yield getattr(item, "chunk", item)


def _commit_paths(text: str) -> list[str]:
    marker = "Files changed:"
    if marker not in (text or ""):
        return []
    tail = text.split(marker, 1)[1].split("\n", 1)[0]
    return [p.strip() for p in tail.split(",") if p.strip() and not p.strip().startswith("(")]
