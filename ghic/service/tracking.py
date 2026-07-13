"""Online outcome tracking: the bot grades its own predictions.

When an issue the service scored is later closed, the `issues.closed` webhook
carries the final labels and state_reason. We run the SAME deterministic
labeling rules used to build the training set (ghic.label.label_issue) on that
payload to derive an approximate ground truth, compare it to what the model
predicted at open time, and accumulate a live confusion matrix that /stats
exposes. This turns every installation into a continuous evaluation harness.

Honest approximation note: the plain webhook payload cannot see whether a
merged PR closed the issue (rules R1/R1b need timeline data), so bugs fixed
via PR without a bug label are under-counted as Class 1 — live recall is a
LOWER BOUND. state_reason and labels drive rules R2/R3/R3b/R4 faithfully.

Predictions and outcomes are appended to a JSONL file so restarts don't lose
the ledger; the file is replayed on startup.
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import utils

logger = utils.get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class PredictionTracker:
    def __init__(self, ledger_path: Path | None = None) -> None:
        self.ledger_path = ledger_path
        self._lock = threading.Lock()
        # (repo, number) -> predicted label at open time
        self.pending: dict[tuple[str, int], dict[str, Any]] = {}
        self.confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.actions = 0            # GitHub writes performed (audit records)
        self.label_events = 0       # maintainer label add/remove events observed
        # Dashboard analytics, all rebuilt from the ledger on restart.
        # (Older ledger lines lack timestamps/related counts; they are
        # counted where possible and skipped from date-keyed facets.)
        self.daily: Counter = Counter()          # YYYY-MM-DD -> predictions
        self.proba_hist = [0] * 10               # deciles of P(actionable)
        self.per_repo: dict[str, dict[str, float]] = {}
        self.with_related = 0                    # predictions w/ >=1 dup candidate
        self.resolutions = Counter()             # truth class at close
        self.label_counts: Counter = Counter()   # labels added by maintainers
        if ledger_path and ledger_path.exists():
            self._replay()

    # -- persistence ----------------------------------------------------------
    def _append(self, record: dict[str, Any]) -> None:
        if not self.ledger_path:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _replay(self) -> None:
        n = 0
        with self.ledger_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "prediction":
                    self._note_prediction(rec)
                elif rec.get("type") == "outcome":
                    self._note_outcome(rec)
                elif rec.get("type") == "action":
                    self.actions += 1
                elif rec.get("type") == "label_event":
                    self._note_label_event(rec)
                n += 1
        logger.info("replayed %d ledger records from %s", n, self.ledger_path)

    # -- recording ------------------------------------------------------------
    def _note_prediction(self, rec: dict[str, Any]) -> None:
        repo, proba = rec["repo"], float(rec["proba"])
        self.pending[(repo, int(rec["number"]))] = {
            "predicted": int(rec["predicted"]),
            "proba": proba,
        }
        if rec.get("at"):
            self.daily[rec["at"][:10]] += 1
        self.proba_hist[min(9, int(proba * 10))] += 1
        stats = self.per_repo.setdefault(
            repo, {"scored": 0, "positive": 0, "proba_sum": 0.0}
        )
        stats["scored"] += 1
        stats["positive"] += int(rec["predicted"])
        stats["proba_sum"] += proba
        if rec.get("related_count"):
            self.with_related += 1

    def _note_outcome(self, rec: dict[str, Any]) -> None:
        key = (rec["repo"], int(rec["number"]))
        pred = self.pending.pop(key, None)
        if pred is None:
            return                       # closed issue we never scored
        truth, predicted = int(rec["truth"]), pred["predicted"]
        cell = {(1, 1): "tp", (0, 1): "fp", (1, 0): "fn", (0, 0): "tn"}[(truth, predicted)]
        self.confusion[cell] += 1
        self.resolutions["actionable" if truth == 1 else "non_actionable"] += 1

    def _note_label_event(self, rec: dict[str, Any]) -> None:
        self.label_events += 1
        if rec.get("added") and rec.get("label"):
            self.label_counts[rec["label"]] += 1

    def record_prediction(self, repo: str, number: int, proba: float, predicted: int,
                          related_count: int = 0) -> None:
        rec = {"type": "prediction", "repo": repo, "number": number,
               "proba": round(proba, 4), "predicted": predicted,
               "related_count": related_count, "at": _utcnow()}
        with self._lock:
            self._note_prediction(rec)
            self._append(rec)

    def record_action(self, repo: str, number: int, action: str,
                      detail: str = "") -> None:
        """Audit trail: every write the bot performs on GitHub, who/what/when.

        `who` is implicit (the bot is the only writer); `when` is recorded at
        append time so the ledger line is the authoritative timestamp.
        """
        rec = {"type": "action", "repo": repo, "number": number,
               "action": action, "detail": detail, "at": _utcnow()}
        with self._lock:
            self.actions += 1
            self._append(rec)

    def record_label_event(self, repo: str, number: int, label: str, added: bool) -> None:
        """Maintainer labeling activity, observed live.

        This is deliberately collected: category labels applied while an
        issue is open are early ground truth for the category head, and
        duplicate labels with timestamps are exactly the pairwise signal the
        duplicate-detection card says is missing. Recording them costs one
        ledger line now and buys future evaluations real data.
        """
        rec = {"type": "label_event", "repo": repo, "number": number,
               "label": label, "added": added, "at": _utcnow()}
        with self._lock:
            self._note_label_event(rec)
            self._append(rec)

    def record_outcome(self, repo: str, number: int, truth: int) -> bool:
        """Returns True when the outcome matched a tracked prediction."""
        rec = {"type": "outcome", "repo": repo, "number": number, "truth": truth}
        with self._lock:
            known = (repo, number) in self.pending
            self._note_outcome(rec)
            if known:
                self._append(rec)
            return known

    # -- reporting --------------------------------------------------------------
    def analytics(self) -> dict[str, Any]:
        """The dashboard's six facets, computed from real ledger data only.

        Facets that need data the deploy hasn't produced yet report zeros —
        never placeholders or invented numbers.
        """
        scored = sum(r["scored"] for r in self.per_repo.values())
        days = sorted(self.daily)[-30:]
        return {
            "issue_trends": {
                "predictions_per_day": {d: self.daily[d] for d in days},
            },
            "duplicate_rate": {
                "predictions_with_related_candidates": self.with_related,
                "rate": round(self.with_related / scored, 4) if scored else None,
                "duplicate_labels_observed_live": sum(
                    n for lab, n in self.label_counts.items()
                    if "duplicate" in lab.lower()
                ),
            },
            "resolution_analytics": {
                "resolved_actionable": self.resolutions.get("actionable", 0),
                "resolved_non_actionable": self.resolutions.get("non_actionable", 0),
                "awaiting_outcome": len(self.pending),
            },
            "confidence_metrics": {
                "proba_histogram_deciles": list(self.proba_hist),
                "mean_proba": round(
                    sum(r["proba_sum"] for r in self.per_repo.values()) / scored, 4
                ) if scored else None,
            },
            "label_stats": {
                "events_observed": self.label_events,
                "top_labels_added": dict(self.label_counts.most_common(15)),
            },
            "component_analytics": {
                # per-repo is the component boundary this service actually has;
                # finer-grained component labels (comp:*, area:*) appear in
                # label_stats as maintainers apply them.
                repo: {
                    "scored": r["scored"],
                    "positive_rate": round(r["positive"] / r["scored"], 4)
                    if r["scored"] else None,
                    "mean_proba": round(r["proba_sum"] / r["scored"], 4)
                    if r["scored"] else None,
                }
                for repo, r in sorted(self.per_repo.items())
            },
        }

    def summary(self) -> dict[str, Any]:
        c = self.confusion
        resolved = sum(c.values())
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else None
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else None
        accuracy = (c["tp"] + c["tn"]) / resolved if resolved else None
        return {
            "awaiting_outcome": len(self.pending),
            "github_writes_audited": self.actions,
            "label_events_observed": self.label_events,
            "resolved": resolved,
            "confusion": dict(c),
            "live_precision": round(precision, 4) if precision is not None else None,
            "live_recall_lower_bound": round(recall, 4) if recall is not None else None,
            "live_accuracy": round(accuracy, 4) if accuracy is not None else None,
            "note": (
                "truth derived from close labels/state_reason via the training "
                "label rules; merged-PR links are invisible to webhooks, so "
                "recall is a lower bound"
            ),
        }
