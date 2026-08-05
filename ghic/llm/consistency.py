"""Detects when the LLM's own judgment contradicts the ML verdict it was
given as evidence, so the comment can say so honestly instead of showing
two conclusions that don't agree.

Deterministic and testable, matching drafting.py's `needs_more_info()`
precedent in this codebase: the trigger condition is the load-bearing part,
not the prose around it.

Explicitly not attempted here: asking the LLM (the same call or a second
one) to "reconcile" a detected disagreement. A follow-up prompt doesn't
validate which side was right -- it just produces a differently-confident-
sounding answer, and the disagreement itself is the informative signal.
Surfacing it plainly is more honest than papering over it with a second
guess this project has no way to validate either.
"""
from __future__ import annotations

# Terms strong enough that, combined with the LLM's own high/critical
# priority AND severity, a "not actionable" ML verdict is worth flagging
# rather than trusting silently. Kept intentionally narrow -- this is a
# precision-first check (only flag when genuinely confident something's
# off), not an attempt to catch every possible disagreement.
_STRONG_ACTIONABLE_SIGNAL_TERMS = frozenset({
    "regression", "regressed",
    "crash", "crashes", "crashing", "crashed",
    "reproducible", "reproduce", "reproduces", "reproduced",
    "data loss", "corrupt", "corrupts", "corrupted", "corruption",
    "security", "vulnerability", "exploit",
    "unusable", "breaks", "broken",
})

# "high" and "critical" both count, not just a literal "high" match --
# critical is a strictly stronger actionable signal than high, so excluding
# it would make the check fire on the milder case and miss the more severe
# one. See models/LLM_ANALYSIS_CARD.md.
_HIGH_SIGNAL_LEVELS = frozenset({"high", "critical"})


def detect_disagreement(
    ml_predicted_label: int, priority: str, severity: str, reasoning: list[str],
) -> bool:
    """True when the LLM's priority/severity/reasoning strongly implies an
    actionable bug but the ML classifier said this issue is not one.

    Directional by design (matches the concrete failure mode this was built
    for): a "likely not actionable" ML verdict sitting next to
    high-confidence bug language is the contradiction that actually erodes
    trust in a triage comment. The reverse pairing -- ML says actionable,
    LLM's own severity read is mild -- isn't flagged; a calibrated
    probability landing just over the decision threshold on a real but
    minor issue is expected behavior, not a contradiction.
    """
    if ml_predicted_label != 0:
        return False
    if priority not in _HIGH_SIGNAL_LEVELS or severity not in _HIGH_SIGNAL_LEVELS:
        return False

    combined = " ".join(reasoning).lower()
    return any(term in combined for term in _STRONG_ACTIONABLE_SIGNAL_TERMS)
