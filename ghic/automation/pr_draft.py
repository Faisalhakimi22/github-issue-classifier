"""Draft pull request descriptions, and weekly engineering digests.

Both are documents a maintainer edits, never anything GHIC submits. The PR
draft is text placed in a comment for someone to copy; nothing here opens a
pull request, and the GitHub App's permissions don't extend to it.

The draft is deliberately **incomplete in the places only a human can
fill**. "Suggested changes" names the files and the reasoning but leaves the
actual change description blank, and the testing checklist is a list of
things to verify rather than claims that they pass. A draft that looks
finished invites being submitted unread, which is the failure mode this
whole phase is built to avoid -- so the gaps are load-bearing, not laziness.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .. import utils
from ..engineering_intelligence.evidence import EvidenceKind

logger = utils.get_logger(__name__)


def build_pr_draft(
    repo: str,
    issue_number: int,
    title: str,
    analysis: Any,
    context: Any,
    *,
    test_plan: list[Any] | None = None,
) -> str:
    """A PR description a maintainer edits and submits themselves."""
    code = list(getattr(context, "code_chunks", []))
    history = list(getattr(context, "history_chunks", []))
    commits = [rc for rc in history if rc.chunk.kind == "commit"]
    issues = [rc for rc in history if rc.chunk.kind == "issue"]

    reference = f"#{issue_number}" if issue_number else "the linked issue"
    lines: list[str] = [
        f"## Fix: {title.strip()[:100]}",
        "",
        "### Summary",
        "",
        f"<!-- Describe the change. Drafted by GHIC from {reference}; edit before opening. -->",
        "",
        "### Problem",
        "",
        f"Reported in {reference}: {title.strip()[:200]}",
        "",
    ]

    if analysis is not None and getattr(analysis, "root_cause", None):
        top = analysis.root_cause.claims[0]
        lines += ["### Likely cause", "", f"{top.statement}", ""]
        lines += [
            "_GHIC inferred this from timing and file overlap — confirm before "
            "describing it as the cause._", "",
        ]

    if code:
        lines += ["### Suggested changes", ""]
        for retrieved in code[:4]:
            chunk = retrieved.chunk
            target = chunk.qualified_symbol or "module scope"
            lines.append(
                f"- `{chunk.path}` (lines {chunk.start_line}-{chunk.end_line}) — "
                f"`{target}` <!-- what changed and why -->"
            )
        lines.append("")

    lines += ["### Testing checklist", ""]
    if test_plan:
        for recommendation in test_plan:
            lines.append(f"- [ ] {recommendation.title} — {recommendation.detail}")
    else:
        lines.append("- [ ] Add a test that fails before this change and passes after")
    lines += [
        "- [ ] Existing test suite passes",
        "- [ ] Manually verified against the reproduction in the issue",
        "",
        "### Breaking changes",
        "",
        "<!-- None / describe them. GHIC cannot determine this from an issue. -->",
        "",
    ]

    if issues:
        lines += ["### Related issues", ""]
        lines += [
            f"- {rc.chunk.reference} — {rc.chunk.symbol.strip()[:80]}" for rc in issues[:4]
        ]
        lines.append("")

    if commits:
        lines += ["### Related commits", ""]
        lines += [
            f"- `{rc.chunk.reference}` — {rc.chunk.symbol.strip()[:80]}" for rc in commits[:4]
        ]
        lines.append("")

    lines += [
        f"Closes {reference}",
        "",
        "---",
        "",
        "> Draft prepared by GHIC from indexed repository evidence. Review and edit "
        "before opening — GHIC does not create or merge pull requests.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feature 9: weekly engineering digest
# ---------------------------------------------------------------------------
def build_weekly_digest(
    repo: str,
    component_stats: dict[str, Any],
    clusters: list[Any],
    *,
    generated_at: float | None = None,
    fmt: str = "markdown",
) -> str:
    """A repository summary from measured history.

    Every number here is a count over indexed artifacts. Where a section
    has no data it says so rather than printing a zero, because "0
    regressions" and "we have no regression data" look identical in a
    dashboard and mean opposite things.

    `fmt` covers markdown (GitHub, email) and plain text (Slack, Teams) --
    the two shapes those destinations actually accept.
    """
    from ..engineering_intelligence.components import summarize_repository

    summary = summarize_repository(component_stats)
    stamp = datetime.fromtimestamp(
        generated_at or datetime.now(tz=timezone.utc).timestamp(), tz=timezone.utc
    ).strftime("%Y-%m-%d")

    scored = [s for s in component_stats.values() if s.has_enough_history]
    fragile = sorted(scored, key=lambda s: (s.health_index or 100))[:5]
    busiest = sorted(
        component_stats.values(), key=lambda s: -s.commit_count
    )[:5]

    lines = [
        f"# Engineering digest — {repo}",
        f"_Week ending {stamp}_",
        "",
        "## Overview",
        "",
        f"- Components tracked: **{summary['components']}**",
        f"- Components with enough history to score: **"
        f"{summary['components_with_enough_history']}**",
        f"- Issues attributed to components: **{summary['total_issues_attributed']}**",
        f"- Regressions: **{summary['total_regressions']}** "
        f"({summary['regression_rate']:.0%} of attributed issues)",
    ]
    if summary["risk_index"] is not None:
        lines.append(f"- Repository risk index: **{summary['risk_index']}/100**")
        lines.append(f"  _{summary['risk_index_note']}_")
    else:
        lines.append(
            "- Repository risk index: **not computed** — too few components have "
            "enough history to score."
        )

    lines += ["", "## Most fragile components", ""]
    if fragile:
        lines.append("| Component | Health | Issues | Regressions | Median close |")
        lines.append("|---|---|---|---|---|")
        for stats in fragile:
            median = stats.median_resolution_days
            lines.append(
                f"| `{stats.name}` | {stats.health_index}/100 ({stats.health_band}) | "
                f"{stats.issue_count} | {stats.regression_count} | "
                f"{f'{median:.0f}d' if median is not None else '—'} |"
            )
    else:
        lines.append("_No component has enough history to score yet._")

    lines += ["", "## Most active components", ""]
    if any(s.commit_count for s in busiest):
        for stats in busiest:
            if stats.commit_count:
                lines.append(f"- `{stats.name}` — {stats.commit_count} commit(s)")
    else:
        lines.append("_No commit activity indexed._")

    lines += ["", "## Recurring problem areas", ""]
    if clusters:
        for cluster in clusters[:5]:
            marker = " **(regression family)**" if cluster.is_regression_family else ""
            lines.append(
                f"- `{cluster.component}` — {cluster.size} related issues"
                f"{marker}: {', '.join(cluster.references[:5])}"
            )
    else:
        lines.append("_No recurring clusters detected._")

    lines += [
        "",
        "---",
        "",
        "> Counts are over indexed repository history. Health and risk are heuristic "
        "indices with published formulas, not calibrated probabilities — see "
        "models/ENGINEERING_INTELLIGENCE_CARD.md.",
    ]

    markdown = "\n".join(lines)
    return markdown if fmt == "markdown" else _to_plain_text(markdown)


def _to_plain_text(markdown: str) -> str:
    """Strip markdown for Slack/Teams plain-text destinations.

    A real Slack integration would use Block Kit; this produces something
    readable when pasted, which is what a digest actually needs.
    """
    import re

    text = re.sub(r"^#+\s*", "", markdown, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_automation_sections(bundle: Any) -> list[str]:
    """The automation block for the GitHub comment.

    Collapsed behind `<details>` deliberately: this is a lot of text, and
    the analysis above it is what a maintainer reads first. Suggestions
    should be available on demand, not shouted.
    """
    if bundle is None or bundle.is_empty:
        return []

    lines = ["", "---", "", "### Suggested next steps", ""]

    if bundle.labels:
        labels = " ".join(f"`{r.title}`" for r in bundle.labels)
        lines += [f"**Suggested labels:** {labels}", ""]

    if bundle.checklist:
        lines += ["<details><summary>Triage checklist</summary>", ""]
        lines += [f"- [ ] **{r.title}** — {r.detail}" for r in bundle.checklist]
        lines += ["", "</details>", ""]

    if bundle.fix_plan:
        lines += ["<details><summary>Implementation plan</summary>", ""]
        for recommendation in bundle.fix_plan:
            lines += [f"- **{recommendation.title}** — {recommendation.detail}"]
        lines += ["", "</details>", ""]

    if bundle.test_plan:
        lines += ["<details><summary>Suggested tests</summary>", ""]
        for recommendation in bundle.test_plan:
            lines += [f"- **{recommendation.title}** — {recommendation.detail}"]
        lines += ["", "</details>", ""]

    if bundle.duplicate_workflow:
        lines += ["<details><summary>Related issue handling</summary>", ""]
        lines += [f"- **{r.title}** — {r.detail}" for r in bundle.duplicate_workflow]
        lines += ["", "</details>", ""]

    if bundle.assignees:
        lines += ["<details><summary>Possible reviewers</summary>", ""]
        lines += [f"- {r.title} — {r.detail}" for r in bundle.assignees]
        lines += ["", "</details>", ""]

    if bundle.pr_draft:
        lines += ["<details><summary>Draft PR description</summary>", "", "````markdown",
                  bundle.pr_draft, "````", "", "</details>", ""]

    lines += [
        "> Every suggestion above is advisory. GHIC does not apply labels, assign "
        "maintainers, close issues, open pull requests, or modify code.",
    ]
    return lines


def evidence_reference_kinds() -> set[str]:
    """Exposed for tests asserting the audit surface stays complete."""
    return {kind.value for kind in EvidenceKind}
