"""Deterministic labeling of collected GitHub issues.

`label_issue(issue_dict, labeling_config)` returns either a class (0 or 1)
with the audit rule that fired, or None with a drop reason. Rules apply in
priority order; first match wins.

The audit counter is the defense of the labels — it shows, per rule, how many issues
landed where, so the labeling logic is defensible end-to-end.

The same function is reused at inference time by the webhook service:
the webhook builds an issue dict from the GitHub event payload and calls
label_issue to compute a training-time label for active-learning feedback.
This is why the module is pandas-free and depends only on stdlib + config.

CLI:
  python -m ghic.label                  # input: data/processed/combined.csv, output: labeled.csv
  python -m ghic.label --audit-only     # print the rule histogram, don't write CSV
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import utils
from .config import Config, LabelingConfig, get_config

logger = utils.get_logger(__name__)

# Issue bodies can be large (especially with code blocks / tracebacks). Default
# csv field size limit is too low for the JSON-encoded cross_referenced_prs
# column on issues with many long PR bodies attached.
csv.field_size_limit(2**31 - 1)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LabelResult:
    """Outcome of applying the rules to one issue.

    `label` is 1 (Actionable Bug), 0 (Non-Actionable), or None (drop).
    `rule` is the audit ID for whichever branch fired.
    """
    label: int | None
    rule: str


# Rule IDs as constants so tests assert against them without magic strings.
DROP_NO_AUTHOR = "drop_no_author"
DROP_BOT = "drop_bot"
DROP_LOCKED = "drop_locked"
DROP_QUESTION = "drop_question"
R1_MERGED_PR = "R1_merged_pr_link"
R1B_MERGED_PR_TEXTUAL = "R1b_merged_pr_textual_reference"
R2_BUG_LABEL_COMPLETED = "R2_bug_label_completed"
R3_NON_ACTIONABLE_LABEL = "R3_non_actionable_label"
R3B_STATE_NOT_PLANNED = "R3b_state_not_planned"
R4_DEFAULT = "R4_default_non_actionable"


# ---------------------------------------------------------------------------
# Core rule application
# ---------------------------------------------------------------------------
def label_issue(issue: dict[str, Any], labeling: LabelingConfig) -> LabelResult:
    """Apply rules in priority order; first match wins.

    Expects an issue dict in collect.normalize_issue() format. Required keys:
        number, author_login, locked, state_reason, labels_at_close,
        closed_by_merged_pr, cross_referenced_prs.
    """
    # --- Exclusions: dropped from the dataset entirely ---
    author = issue.get("author_login")
    if not author:
        return LabelResult(None, DROP_NO_AUTHOR)
    if author in labeling.bot_logins:
        return LabelResult(None, DROP_BOT)
    if labeling.drop_locked_issues and issue.get("locked"):
        return LabelResult(None, DROP_LOCKED)

    labels_lower = {
        (l or "").strip().lower() for l in (issue.get("labels_at_close") or [])
    }
    if labeling.drop_question_labeled_issues and labels_lower & labeling.question_labels:
        return LabelResult(None, DROP_QUESTION)

    # --- Class 1 rules ---
    # R1: GitHub explicitly recorded a merged PR as the closer.
    if issue.get("closed_by_merged_pr"):
        return LabelResult(1, R1_MERGED_PR)

    # R1b: a merged PR cross-referenced this issue with a closing keyword.
    issue_number = issue.get("number")
    if issue_number is not None:
        pattern = re.compile(labeling.pr_fix_regex(int(issue_number)))
        for pr in issue.get("cross_referenced_prs") or []:
            if pr.get("merged") and pattern.search(pr.get("body") or ""):
                return LabelResult(1, R1B_MERGED_PR_TEXTUAL)

    # R2: explicit bug label + closed-as-completed.
    if (labels_lower & labeling.bug_labels) and issue.get("state_reason") == "COMPLETED":
        return LabelResult(1, R2_BUG_LABEL_COMPLETED)

    # --- Class 0 rules ---
    # R3: explicit non-actionable label.
    if labels_lower & labeling.non_actionable_labels:
        return LabelResult(0, R3_NON_ACTIONABLE_LABEL)
    # R3b: GitHub's own machine-readable "not planned" signal.
    if issue.get("state_reason") == "NOT_PLANNED":
        return LabelResult(0, R3B_STATE_NOT_PLANNED)

    # R4: conservative default. Track its share carefully via the audit —
    # if it dominates the dataset, the labels are mostly noise.
    return LabelResult(0, R4_DEFAULT)


def label_dataset(
    issues: list[dict[str, Any]], cfg: Config | None = None,
) -> tuple[list[dict[str, Any]], Counter]:
    """Label every issue, return (kept_rows_with_label_appended, audit_counter).

    Dropped issues are excluded from the kept rows but counted in the audit.
    Every kept row has `label` (int) and `label_rule` (str) keys appended.
    """
    cfg = cfg or get_config(require_token=False)
    audit: Counter = Counter()
    kept: list[dict[str, Any]] = []
    for issue in issues:
        result = label_issue(issue, cfg.labeling)
        audit[result.rule] += 1
        if result.label is not None:
            row = dict(issue)
            row["label"] = result.label
            row["label_rule"] = result.rule
            kept.append(row)
    return kept, audit


# ---------------------------------------------------------------------------
# Audit reporting — the auditable rule histogram
# ---------------------------------------------------------------------------
_CLASS1_RULES = (R1_MERGED_PR, R1B_MERGED_PR_TEXTUAL, R2_BUG_LABEL_COMPLETED)
_CLASS0_RULES = (R3_NON_ACTIONABLE_LABEL, R3B_STATE_NOT_PLANNED, R4_DEFAULT)
_DROP_RULES = (DROP_NO_AUTHOR, DROP_BOT, DROP_LOCKED, DROP_QUESTION)


def format_audit(audit: Counter) -> str:
    """Render the audit as a stable, sorted text block."""
    total = sum(audit.values())
    lines = [f"Labeling audit (input={total}):"]
    for k in _DROP_RULES:
        lines.append(f"  DROP  {k:38s} {audit.get(k, 0):6d}")
    for k in _CLASS1_RULES:
        lines.append(f"  C1    {k:38s} {audit.get(k, 0):6d}")
    for k in _CLASS0_RULES:
        lines.append(f"  C0    {k:38s} {audit.get(k, 0):6d}")
    dropped = sum(audit.get(k, 0) for k in _DROP_RULES)
    c1 = sum(audit.get(k, 0) for k in _CLASS1_RULES)
    c0 = sum(audit.get(k, 0) for k in _CLASS0_RULES)
    kept = c1 + c0
    ratio = c1 / kept if kept else 0.0
    r4_share = audit.get(R4_DEFAULT, 0) / kept if kept else 0.0
    lines.append("  " + "-" * 52)
    lines.append(
        f"  dropped={dropped}, kept={kept} "
        f"(C1={c1}, C0={c0}, C1 ratio={ratio:.2%}, R4 share of kept={r4_share:.2%})"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV IO (stdlib only — keeps label.py pandas-free for webhook reuse)
# ---------------------------------------------------------------------------
_JSON_COLUMNS = ("labels_at_close", "cross_referenced_prs")
_BOOL_COLUMNS = ("locked", "closed_by_merged_pr")
_INT_COLUMNS = ("number", "author_public_repos", "author_followers")


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    """Convert CSV string fields back to Python types matching normalize_issue()."""
    out: dict[str, Any] = {k: (None if v == "" else v) for k, v in row.items()}
    for k in _JSON_COLUMNS:
        v = out.get(k)
        out[k] = json.loads(v) if v else []
    for k in _BOOL_COLUMNS:
        v = out.get(k)
        out[k] = v in ("True", "true", "1") if v is not None else False
    for k in _INT_COLUMNS:
        v = out.get(k)
        out[k] = int(v) if v not in (None, "None") else None
    return out


def _read_collected_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [_coerce_row(r) for r in csv.DictReader(f)]


def _write_labeled_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        logger.warning("No rows to write to %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized: list[dict[str, Any]] = []
    for row in rows:
        s: dict[str, Any] = {}
        for k, v in row.items():
            s[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
        serialized.append(s)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(serialized[0].keys()))
        writer.writeheader()
        writer.writerows(serialized)
    logger.info("wrote %s (%d rows)", path, len(serialized))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply labeling rules to collected issues.")
    parser.add_argument(
        "--input",
        type=Path,
        default=utils.DATA_PROCESSED / "combined.csv",
        help="Input CSV from collect.py (default: data/processed/combined.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=utils.DATA_PROCESSED / "labeled.csv",
        help="Output CSV with label + label_rule columns appended (default: data/processed/labeled.csv)",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Print the audit and exit without writing the output CSV.",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        logger.error("Input not found: %s. Run `python -m ghic.collect` first.", args.input)
        return 1

    logger.info("Reading %s", args.input)
    issues = _read_collected_csv(args.input)
    logger.info("Loaded %d issues", len(issues))

    kept, audit = label_dataset(issues)
    print(format_audit(audit))

    if args.audit_only:
        return 0

    _write_labeled_csv(args.output, kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
