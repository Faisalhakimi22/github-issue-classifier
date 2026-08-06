"""Components: inferring them from paths, and measuring their health.

A "component" here is a directory-level grouping of a repository's files
(`ghic/service`, `src/auth`, `payments`). Inferred from the tree, not
configured, because a mapping an operator has to maintain is a mapping
that goes stale -- and a stale component map produces confidently wrong
attribution, which is worse than none.

## On health scores

The spec asks for percentages: "Authentication 92%". This module computes
them, and the docstring on `health_score` explains exactly what they are
and are not, because this project has twice refused to ship a number it
couldn't validate (see PRIORITY_CARD.md and SEVERITY_CARD.md: "a fabricated
number wearing a UI"). The same discipline applies here, so:

  - Every input is a **measured count** over indexed history -- issues
    touching a component, how many were regressions, how long they took to
    close. Those are facts, and they are reported alongside the score.
  - The composite score is a **heuristic index, not a probability**. It is
    not calibrated against any ground truth, because no ground truth for
    "component health" exists in this corpus. It is presented as an index
    with its formula documented, and never as a confidence or a
    probability.
  - Below `MIN_ISSUES_FOR_SCORE` observations the score is **not computed
    at all**. Three issues cannot distinguish a fragile component from an
    unlucky one, and a 67% derived from three data points is precisely the
    false precision this project exists not to ship.

If that distinction ever stops being maintained -- if the index starts
being consumed as though it were validated -- it should stop shipping, by
the same logic that kept the ML priority head out.
"""
from __future__ import annotations

import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .. import utils

logger = utils.get_logger(__name__)

# Directory names that are packaging, not components: a file in `src/` is
# not "part of the src component". Peeled off so `src/auth/oauth.py`
# becomes `auth` rather than `src`.
_PACKAGING_DIRS = frozenset({
    "src", "lib", "app", "apps", "packages", "pkg", "internal", "cmd",
    "source", "sources", "modules", "components",
})

# Below this many observed issues, no score is produced. See the module
# docstring -- this is the guard against false precision, not a tunable.
MIN_ISSUES_FOR_SCORE = 8

# Phrases in an issue or commit that mark it as a regression: something that
# used to work. High-precision by design; a missed regression is a smaller
# error than a mislabelled one.
_REGRESSION_MARKERS = (
    "regression", "regressed", "used to work", "worked before", "worked in",
    "since upgrading", "after upgrading", "after updating", "no longer works",
    "stopped working", "broke after", "broken after", "since the update",
    "since version", "after the release", "reverted",
)

_ISSUE_REFERENCE_RE = re.compile(r"#(\d{1,7})\b")


def component_for_path(path: str, *, depth: int = 2) -> str:
    """The component a file belongs to.

    Takes up to `depth` meaningful directory segments after peeling
    packaging directories, so `src/auth/oauth/token.py` -> `auth/oauth`
    and `ghic/service/app.py` -> `ghic/service`. A root-level file is its
    own component (`README.md` -> `README.md`), which is honest: it belongs
    to nothing else.
    """
    cleaned = (path or "").replace("\\", "/").strip("/")
    if not cleaned:
        return ""
    # Synthetic paths from Phase 2 (`commit:abc`, `issue:412`) are not files
    # and have no component.
    if ":" in cleaned.split("/")[0]:
        return ""
    parts = cleaned.split("/")
    if len(parts) == 1:
        return parts[0]
    directories = parts[:-1]
    while directories and directories[0].lower() in _PACKAGING_DIRS:
        directories = directories[1:]
    if not directories:
        return parts[-2] if len(parts) >= 2 else parts[0]
    return "/".join(directories[:depth])


def is_regression_text(text: str) -> bool:
    """Whether text describes something that used to work."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _REGRESSION_MARKERS)


def referenced_issue_numbers(text: str) -> list[int]:
    """Issue numbers mentioned in a commit message or PR body.

    This is the link that makes component health measurable at all: a
    commit says which files it touched *and* which issue it closed, so an
    issue can be attributed to a component from real evidence rather than
    from guessing at its title.
    """
    return [int(n) for n in _ISSUE_REFERENCE_RE.findall(text or "")][:10]


@dataclass
class ComponentStats:
    """Measured counts for one component. Every field is observed."""
    name: str
    issue_count: int = 0
    regression_count: int = 0
    commit_count: int = 0
    resolution_days: list[float] = field(default_factory=list)
    files: set[str] = field(default_factory=set)
    recent_issue_count: int = 0        # within RECENT_WINDOW_DAYS
    last_activity: float = 0.0

    @property
    def regression_rate(self) -> float:
        return self.regression_count / self.issue_count if self.issue_count else 0.0

    @property
    def median_resolution_days(self) -> float | None:
        return statistics.median(self.resolution_days) if self.resolution_days else None

    @property
    def has_enough_history(self) -> bool:
        return self.issue_count >= MIN_ISSUES_FOR_SCORE

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.name,
            "issue_count": self.issue_count,
            "regression_count": self.regression_count,
            "regression_rate": round(self.regression_rate, 3),
            "commit_count": self.commit_count,
            "median_resolution_days": (
                round(self.median_resolution_days, 1)
                if self.median_resolution_days is not None else None
            ),
            "recent_issue_count": self.recent_issue_count,
            "file_count": len(self.files),
            "last_activity": self.last_activity,
            "health_index": self.health_index,
            "health_band": self.health_band,
            "scored": self.has_enough_history,
        }

    @property
    def health_index(self) -> int | None:
        return health_score(self)

    @property
    def health_band(self) -> str:
        """A coarse band, available even when the index isn't.

        Three buckets is what this evidence supports. A component with
        four issues can still be described as "limited history" honestly,
        where a percentage would be a fabrication.
        """
        if not self.has_enough_history:
            return "insufficient history"
        score = self.health_index or 0
        if score >= 85:
            return "healthy"
        if score >= 65:
            return "watch"
        return "fragile"


RECENT_WINDOW_DAYS = 90


def health_score(stats: ComponentStats) -> int | None:
    """A 0-100 heuristic index, or None when there isn't enough history.

    **This is not a probability and is not calibrated.** It is a weighted
    combination of three measured rates, chosen to be explicable rather
    than optimal, and it exists so a maintainer can rank components
    against each other -- not so anyone can say "this component is 92%
    healthy" and mean something precise by it.

    The formula, stated so it can be argued with:

      start at 100
      minus  40 x regression_rate        (a component that keeps breaking
                                          the same way is the strongest
                                          signal of fragility here)
      minus  25 x recent_issue_pressure  (share of its issues filed in the
                                          last 90 days: getting worse now
                                          matters more than historic noise)
      minus  20 x slow_resolution        (median close time relative to a
                                          30-day reference, capped)

    Returns None below MIN_ISSUES_FOR_SCORE observations rather than
    computing a number from too little data.
    """
    if not stats.has_enough_history:
        return None

    penalty = 40.0 * min(1.0, stats.regression_rate)

    recent_pressure = stats.recent_issue_count / stats.issue_count if stats.issue_count else 0.0
    penalty += 25.0 * min(1.0, recent_pressure)

    median_days = stats.median_resolution_days
    if median_days is not None:
        penalty += 20.0 * min(1.0, median_days / 30.0)

    return max(0, min(100, int(round(100.0 - penalty))))


class ComponentAnalyzer:
    """Builds component statistics from indexed chunks.

    Takes chunks rather than fetching anything: the same inversion Phase 1
    used for `RepositoryIndexer`, so this is testable against a handful of
    dataclasses and reusable by anything that has retrieved evidence.
    """

    def __init__(self, *, recent_window_days: int = RECENT_WINDOW_DAYS) -> None:
        self.recent_window_days = recent_window_days

    def analyze(self, chunks: list[Any]) -> dict[str, ComponentStats]:
        """Chunks (code + commits + issues) -> per-component statistics.

        Attribution runs commit-first: a commit names both the files it
        touched and, usually, the issue it closed, which links an issue to
        a component through evidence rather than through a guess at its
        title.
        """
        stats: dict[str, ComponentStats] = {}
        now = time.time()
        cutoff = now - self.recent_window_days * 86400

        def bucket(name: str) -> ComponentStats:
            if name not in stats:
                stats[name] = ComponentStats(name=name)
            return stats[name]

        # Pass 1: files -> components.
        for chunk in chunks:
            if getattr(chunk, "source", "code") != "code":
                continue
            name = component_for_path(chunk.path)
            if name:
                bucket(name).files.add(chunk.path)

        # Pass 2: commits. Establishes component activity and the
        # issue -> component link.
        issue_to_components: dict[int, set[str]] = defaultdict(set)
        for chunk in chunks:
            if getattr(chunk, "source", "") != "commit":
                continue
            touched = _paths_in_commit_text(chunk.text)
            components = {component_for_path(p) for p in touched}
            components.discard("")
            if not components:
                continue
            is_regression = is_regression_text(f"{chunk.symbol} {chunk.text}")
            for name in components:
                entry = bucket(name)
                entry.commit_count += 1
                entry.last_activity = max(entry.last_activity, chunk.timestamp)
                if is_regression:
                    entry.regression_count += 1
            for number in referenced_issue_numbers(f"{chunk.symbol} {chunk.text}"):
                issue_to_components[number].update(components)

        # Pass 3: issues, attributed through the commits that closed them.
        for chunk in chunks:
            if getattr(chunk, "source", "") not in ("issue", "pull_request"):
                continue
            number = _number_from_reference(chunk.reference)
            components = issue_to_components.get(number, set()) if number else set()
            if not components:
                continue
            is_regression = is_regression_text(f"{chunk.symbol} {chunk.text}")
            for name in components:
                entry = bucket(name)
                entry.issue_count += 1
                if is_regression:
                    entry.regression_count += 1
                if chunk.timestamp and chunk.timestamp >= cutoff:
                    entry.recent_issue_count += 1
                entry.last_activity = max(entry.last_activity, chunk.timestamp)

        return stats

    def hotspots(self, stats: dict[str, ComponentStats], limit: int = 10) -> list[ComponentStats]:
        """Components most in need of attention.

        Ranked by measured pressure -- regressions first, then recent
        issue volume -- and restricted to components with enough history to
        say anything about. An "insufficient history" component is not a
        hotspot; it's an unknown.
        """
        scored = [s for s in stats.values() if s.has_enough_history]
        scored.sort(key=lambda s: (-s.regression_count, -s.recent_issue_count, -s.issue_count))
        return scored[:limit]


def _paths_in_commit_text(text: str) -> list[str]:
    """File paths out of a rendered commit chunk.

    The chunk renderer writes "Files changed: a/b.py, c/d.py" -- parsed
    back rather than re-running git, so this works identically against a
    persisted index where the checkout is long gone.
    """
    marker = "Files changed:"
    if marker not in (text or ""):
        return []
    tail = text.split(marker, 1)[1]
    tail = tail.split("\n", 1)[0]
    return [p.strip() for p in tail.split(",") if p.strip() and not p.strip().startswith("(")]


def _number_from_reference(reference: str) -> int | None:
    match = re.search(r"\d+", reference or "")
    return int(match.group()) if match else None


def summarize_repository(stats: dict[str, ComponentStats]) -> dict[str, Any]:
    """Repository-level rollup for the dashboard.

    `risk_index` follows the same rules as component health: measured
    inputs, documented formula, no claim to calibration, and withheld
    entirely when there isn't enough history to compute it from.
    """
    scored = [s for s in stats.values() if s.has_enough_history]
    total_issues = sum(s.issue_count for s in stats.values())
    total_regressions = sum(s.regression_count for s in stats.values())

    indices = [s.health_index for s in scored if s.health_index is not None]
    risk_index = int(round(100 - (sum(indices) / len(indices)))) if indices else None

    return {
        "components": len(stats),
        "components_with_enough_history": len(scored),
        "total_issues_attributed": total_issues,
        "total_regressions": total_regressions,
        "regression_rate": (
            round(total_regressions / total_issues, 3) if total_issues else 0.0
        ),
        "risk_index": risk_index,
        "risk_index_note": (
            "Heuristic index over measured history, not a calibrated probability. "
            "None when too few components have enough history to score."
        ),
        "fragile_components": [
            s.name for s in scored if s.health_band == "fragile"
        ][:10],
    }
