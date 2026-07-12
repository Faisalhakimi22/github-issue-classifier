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
from pathlib import Path
from typing import Any

from .. import utils

logger = utils.get_logger(__name__)


class PredictionTracker:
    def __init__(self, ledger_path: Path | None = None) -> None:
        self.ledger_path = ledger_path
        self._lock = threading.Lock()
        # (repo, number) -> predicted label at open time
        self.pending: dict[tuple[str, int], dict[str, Any]] = {}
        self.confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
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
                n += 1
        logger.info("replayed %d ledger records from %s", n, self.ledger_path)

    # -- recording ------------------------------------------------------------
    def _note_prediction(self, rec: dict[str, Any]) -> None:
        self.pending[(rec["repo"], int(rec["number"]))] = {
            "predicted": int(rec["predicted"]),
            "proba": float(rec["proba"]),
        }

    def _note_outcome(self, rec: dict[str, Any]) -> None:
        key = (rec["repo"], int(rec["number"]))
        pred = self.pending.pop(key, None)
        if pred is None:
            return                       # closed issue we never scored
        truth, predicted = int(rec["truth"]), pred["predicted"]
        cell = {(1, 1): "tp", (0, 1): "fp", (1, 0): "fn", (0, 0): "tn"}[(truth, predicted)]
        self.confusion[cell] += 1

    def record_prediction(self, repo: str, number: int, proba: float, predicted: int) -> None:
        rec = {"type": "prediction", "repo": repo, "number": number,
               "proba": round(proba, 4), "predicted": predicted}
        with self._lock:
            self._note_prediction(rec)
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
    def summary(self) -> dict[str, Any]:
        c = self.confusion
        resolved = sum(c.values())
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else None
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else None
        accuracy = (c["tp"] + c["tn"]) / resolved if resolved else None
        return {
            "awaiting_outcome": len(self.pending),
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
