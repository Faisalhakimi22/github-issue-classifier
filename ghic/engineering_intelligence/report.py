"""Rendering the engineering analysis into the GitHub comment.

Two rules carried forward from the Phase 1 comment, because they are what
make the whole thing trustworthy:

**Rendered in Python, never by the model.** Every reference here -- commit
SHA, issue number, file path, line span, release tag -- comes out of an
`Evidence` object built from a retrieved chunk. The LLM writes the prose
sections (summary, reasoning, next step); this writes the sections that
make factual claims about the repository. A model cannot fabricate a commit
hash into a section it doesn't author.

**Silence over filler.** A finding with no claims renders nothing. An
analysis with no findings adds no sections at all, and the comment
degrades exactly to the Phase 2 shape. Empty headings under confident
titles are how a tool looks thorough while saying nothing.

The `Observed` / `Inferred` prefixes are load-bearing, not decoration:
they're the promise in Feature 14 that a maintainer can tell what was
measured from what was reasoned, and they appear on every claim.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .evidence import Confidence, Finding

_MAX_TIMELINE_ROWS = 8
_MAX_EVIDENCE_PER_FINDING = 4


def _fmt_date(timestamp: float) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


def render_finding(finding: Finding | None, *, heading_level: str = "###") -> list[str]:
    """One finding as markdown lines, or [] when there's nothing to say."""
    if finding is None or finding.is_empty:
        return []

    lines = ["", "---", "", f"{heading_level} {finding.title}", ""]
    if finding.confidence is not Confidence.LOW or len(finding.claims) > 1:
        lines += [f"**Confidence: {finding.confidence.label}**", ""]
    if finding.summary:
        lines += [f"_{finding.summary}_", ""]

    for claim in finding.claims:
        lines.append(f"- {claim.render()}")

    evidence = finding.all_evidence[:_MAX_EVIDENCE_PER_FINDING]
    if evidence:
        lines += ["", "<details><summary>Evidence</summary>", ""]
        lines += [f"- {item.render()}" for item in evidence]
        lines += ["", "</details>"]
    return lines


def render_investigation_plan(plan: list[Any]) -> list[str]:
    """The ordered checklist. Numbered, because order is the point."""
    if not plan:
        return []
    lines = ["", "---", "", "### Investigation plan", ""]
    for index, claim in enumerate(plan, start=1):
        lines.append(f"{index}. {claim.statement}")
    return lines


def render_timeline(timeline: list[dict[str, Any]]) -> list[str]:
    """A dated table of retrieved artifacts leading to this issue.

    Rendered only when there is more than the issue itself to show -- a
    one-row timeline containing only "this issue was reported" is a heading
    with no information under it.
    """
    if len(timeline) < 2:
        return []

    icons = {
        "commit": "commit", "issue": "issue", "pull_request": "PR", "release": "release",
    }
    lines = ["", "---", "", "### Timeline", "", "| When | What | |", "|---|---|---|"]
    for event in timeline[-_MAX_TIMELINE_ROWS:]:
        if event["kind"] == "current_issue":
            # The anchor row names itself; it has no reference to print.
            what = "**this issue**"
        else:
            reference = event.get("reference", "")
            reference = (
                f"[{reference}]({event['url']})" if event.get("url") else f"`{reference}`"
            )
            what = f"{icons.get(event['kind'], event['kind'])} {reference}"
        label = (event.get("label") or "").replace("|", "\\|")[:70]
        lines.append(f"| {_fmt_date(event['timestamp'])} | {what} | {label} |")
    return lines


def render_components(components: list[str], analysis: Any) -> list[str]:
    if not components:
        return []
    return [
        "", "---", "", "### Affected components", "",
        " ".join(f"`{c}`" for c in components[:5]),
    ]


def render_engineering_sections(analysis: Any) -> list[str]:
    """Every Phase 3 section, in reading order, for one issue.

    Order is deliberate and matches how a maintainer triages: what is this
    about (components), what probably broke and why (root cause,
    regression), has this happened before (recurrence), how bad is it
    (impact), how did we get here (timeline), what do I do first
    (investigation plan).
    """
    if analysis is None or getattr(analysis, "is_empty", True):
        return []

    lines: list[str] = []
    lines += render_components(analysis.components, analysis)
    lines += render_finding(analysis.root_cause)
    lines += render_finding(analysis.regression)
    lines += render_finding(analysis.recurrence)
    lines += render_finding(analysis.impact)
    lines += render_timeline(analysis.timeline)
    lines += render_investigation_plan(analysis.investigation_plan)
    return lines


def render_attribution_note() -> list[str]:
    """The footnote that makes the Observed/Inferred labels legible.

    Worth the two lines: the distinction is useless if the reader doesn't
    know it's being made.
    """
    return [
        "",
        "> **Observed** statements come directly from indexed repository "
        "artifacts. **Inferred** statements are correlations GHIC drew "
        "between them — leads to check, not conclusions.",
    ]
