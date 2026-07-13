# Card — severity prediction (not shipped)

**Status: correctly not shipped.** Companion to `PRIORITY_CARD.md`; the
evidence differs enough to warrant its own card.

## The ground-truth problem

No severity labels exist in the collected corpus: no `severity:*`, no
`sev1`/`sev2`, nothing equivalent, in any of the three repos (checked at
Phase 14 across all labels with ≥5 occurrences). The closest artifact is
vscode's `freeze-slow-crash-leak` (45 issues) — a *symptom* tag, not a
severity scale.

## The circularity trap (why a proxy is worse than none)

The tempting shortcut is keyword-derived severity: "crash"/"data loss" →
high, "cosmetic" → low. But those keywords are in the issue *text*, which
is the model's *input* — a classifier trained on keyword-derived labels
learns to detect its own labeling rule, and its reported accuracy is a
near-tautology. That is exactly the kind of impressive-looking number this
project refuses to publish. (The deterministic rule itself needs no model:
if keyword matching were the product feature, it should ship as a regex,
honestly labeled.)

## What would make this buildable

Ground truth that is *not* derived from the input text: severity fields
from linked incident trackers, or repos whose maintainers apply an explicit
severity scale. The champion protocol applies unchanged once such data
exists.
