"""Make LLM output safe to post as GHIC's own GitHub comment.

The issue title and body are written by anyone who can open an issue, and
they reach the model verbatim. Prompt instructions reduce how often the
model follows text planted there, but they are not a control. This is: no
string from the model reaches GitHub until it has been through here.

What a planted instruction could otherwise make GHIC post, as itself, on a
maintainer's repository:
  - @mentions, which notify arbitrary users and teams;
  - links to anywhere (phishing with the bot's authority) and images
    (tracking pixels);
  - `#123` / `owner/repo#123` references, which add "mentioned" events to
    other issues, including in other repositories;
  - raw HTML, headings and extra sections, by smuggling newlines into a
    field that is rendered as one line or one bullet;
  - unbounded text.

Rules, applied to every string field:
  - newlines and repeated whitespace collapse to single spaces;
  - `<` is escaped, so no HTML renders;
  - markdown image syntax is reduced to a plain link;
  - a URL is kept only if it appears verbatim in the issue itself;
  - @mentions and issue references are broken with a zero-width space, so
    they display normally but GitHub does not link or notify them;
  - each field is cut to a fixed length.
"""
from __future__ import annotations

import re
from dataclasses import replace

from .models import IssueAnalysis

_ZWSP = "\u200b"

_URL_RE = re.compile(r"https?://[^\s<>()\[\]`'\"]+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<![\w`])@(?=[A-Za-z0-9])")
# Any "#" before a digit: bare #12 and cross-repo owner/repo#12 alike (the
# cross-repo form is the one that writes into other people's issues).
_ISSUE_REF_RE = re.compile(r"(?<!&)#(?=\d)")
_IMAGE_RE = re.compile(r"!\[")
_SPACE_RE = re.compile(r"\s+")

LIMITS = {
    "category": 40,
    "summary": 500,
    "recommended_action": 300,
    "item": 240,
    "label": 40,
}


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def clean_text(text: str, *, source_text: str, limit: int) -> str:
    """One model-written string, made safe to render inside a comment."""
    out = _SPACE_RE.sub(" ", str(text or "")).strip()
    out = out.replace("<", "&lt;")
    out = _IMAGE_RE.sub("[", out)

    def _keep_known_url(match: re.Match[str]) -> str:
        url = match.group(0).rstrip(".,;:!?")
        tail = match.group(0)[len(url):]
        return (url if url in source_text else "[link removed]") + tail

    out = _URL_RE.sub(_keep_known_url, out)
    out = _MENTION_RE.sub("@" + _ZWSP, out)
    out = _ISSUE_REF_RE.sub("#" + _ZWSP, out)
    return _clip(out, limit)


def clean_label(label: str, *, source_text: str) -> str:
    # Labels are rendered inside backticks; a backtick in one would close it.
    return clean_text(label.replace("`", "'"), source_text=source_text, limit=LIMITS["label"])


def sanitize_analysis(analysis: IssueAnalysis, *, source_text: str) -> IssueAnalysis:
    """Return a copy of `analysis` with every string field made safe.

    `source_text` is the issue's own title and body: the only place a URL
    in the output is allowed to have come from.
    """
    item = LIMITS["item"]

    def items(values: list[str]) -> list[str]:
        cleaned = (clean_text(v, source_text=source_text, limit=item) for v in values)
        return [v for v in cleaned if v]

    return replace(
        analysis,
        category=clean_text(analysis.category, source_text=source_text, limit=LIMITS["category"]),
        summary=clean_text(analysis.summary, source_text=source_text, limit=LIMITS["summary"]),
        recommended_action=clean_text(
            analysis.recommended_action, source_text=source_text,
            limit=LIMITS["recommended_action"],
        ),
        reasoning=items(analysis.reasoning),
        risk_reasons=items(analysis.risk_reasons),
        business_impact=items(analysis.business_impact),
        missing_information=items(analysis.missing_information),
        recommended_labels=[
            v for v in (clean_label(x, source_text=source_text) for x in analysis.recommended_labels) if v
        ],
    )
