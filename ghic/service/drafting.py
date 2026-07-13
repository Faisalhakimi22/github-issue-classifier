"""LLM-assisted triage, deliberately scoped.

The calibrated classifier decides *whether* an issue looks actionable — that
decision is the thing the whole evaluation protocol makes trustworthy, and it
is never delegated to a generative model. What the LLM does here is narrower:
when a deterministic trigger says an issue is under-specified, it drafts the
"could you add more information?" comment a good triager would write, grounded
in the issue text and (when available) similar prior issues from the duplicate
index.

Degradation ladder, in order:
  1. Anthropic API key configured -> Claude drafts a tailored comment.
  2. No key / SDK missing / API error -> deterministic template listing the
     concrete missing elements. The feature never blocks or breaks triage.

The trigger (`needs_more_info`) is a pure function with tests — per the build
discipline, the trigger condition is the load-bearing part, not the prose.
"""
from __future__ import annotations

import os
from typing import Any

from .. import utils
from ..config import Config, get_config

logger = utils.get_logger(__name__)

DEFAULT_MODEL = "claude-opus-4-8"
_MIN_BODY_TOKENS = 40          # bodies shorter than this rarely contain a repro


# ---------------------------------------------------------------------------
# Trigger: is this issue under-specified? (deterministic, tested)
# ---------------------------------------------------------------------------
def missing_elements(title: str, body: str, cfg: Config | None = None) -> list[str]:
    """Concrete, checkable gaps in a report. Empty list = well-specified."""
    from ..features import _CODE_BLOCK_RE, _repro_keyword_hits, _token_count

    cfg = cfg or get_config(require_token=False)
    repro_keywords = cfg.features.section("text").get("repro_keywords", [])
    body = body or ""
    gaps: list[str] = []
    if _repro_keyword_hits(f"{title}\n{body}", repro_keywords) == 0:
        gaps.append("steps to reproduce")
    if not _CODE_BLOCK_RE.search(body):
        gaps.append("error output, stack trace, or a minimal code sample")
    if _token_count(body) < _MIN_BODY_TOKENS:
        gaps.append("a fuller description of the environment and expected vs. actual behavior")
    return gaps


def needs_more_info(title: str, body: str, cfg: Config | None = None) -> bool:
    """True when the issue is missing enough that a triager would ask.

    Requires at least two concrete gaps — a short-but-precise report with a
    stack trace should not be nagged.
    """
    return len(missing_elements(title, body, cfg)) >= 2


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------
def _template_draft(gaps: list[str]) -> str:
    lines = [
        "Thanks for the report! To help maintainers act on this issue, could",
        "you add:",
        "",
    ]
    lines += [f"- {gap}" for gap in gaps]
    lines += [
        "",
        "Well-specified reports are far more likely to be fixed — thank you!",
    ]
    return "\n".join(lines)


def _llm_draft(
    title: str,
    body: str,
    gaps: list[str],
    related: list[dict[str, Any]],
    model: str,
) -> str | None:
    """Draft with Claude; None on any failure (caller falls back to template)."""
    try:
        import anthropic
    except ImportError:
        logger.info("anthropic SDK not installed; using template draft")
        return None

    related_context = ""
    if related:
        items = "\n".join(
            f"- #{r['number']}: {r['title']}" for r in related
        )
        related_context = (
            f"\nSimilar prior issues in this repository (mention them if relevant):\n{items}\n"
        )
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=(
                "You draft short, friendly GitHub comments asking issue authors "
                "for missing information. You never judge whether the issue is "
                "valid or a bug — a separate system does that. Be specific to "
                "this issue's content, not generic. Maximum 120 words, GitHub "
                "markdown, no headings, no sign-off."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Issue title: {title}\n\nIssue body:\n{(body or '')[:3000]}\n\n"
                    f"Missing elements a triager needs: {', '.join(gaps)}.\n"
                    f"{related_context}\n"
                    "Draft the comment."
                ),
            }],
        )
        if response.stop_reason == "refusal":
            logger.warning("draft request refused; using template")
            return None
        text = next((b.text for b in response.content if b.type == "text"), "")
        return text.strip() or None
    except Exception as e:  # any API problem degrades to the template
        logger.warning("LLM draft failed (%s); using template", e)
        return None


def draft_missing_info(
    title: str,
    body: str,
    related: list[dict[str, Any]] | None = None,
    cfg: Config | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Full flow: trigger -> draft -> annotated result. None if well-specified."""
    gaps = missing_elements(title, body, cfg)
    if len(gaps) < 2:
        return None
    model = model or os.environ.get("GHIC_LLM_MODEL", DEFAULT_MODEL)
    text = _llm_draft(title, body, gaps, related or [], model)
    source = "llm" if text else "template"
    return {
        "missing": gaps,
        "draft": text or _template_draft(gaps),
        "source": source,
    }
