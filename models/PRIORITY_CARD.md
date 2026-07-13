# Card — priority prediction (not shipped)

**Status: correctly not shipped.** This card documents why, so the gap is
a decision with evidence rather than an omission.

## The ground-truth problem

The corpus (6,175 closed issues from microsoft/vscode, facebook/react,
tensorflow/tensorflow, calendar 2024) contains **no priority labels**.
Checked exhaustively at Phase 14: no `P0`–`P4`, no `priority:*`; the
nearest signal is vscode's `important` — 31 issues, one repo, too few to
train or even validate against.

## Proxies considered and rejected

| candidate proxy | why it fails |
|---|---|
| time-to-close (fast close = high priority) | conflates priority with issue difficulty, backlog depth, and stale-bot closure; also already claimed by the effort experiment, where it measurably carries little open-time signal (see `EFFORT_CARD.md`) |
| maintainer response latency | first-response timestamps were not collected; and response speed tracks maintainer availability at least as much as priority |
| reaction counts (👍) | popularity, not priority — feature requests dominate; also grows after open time (leaky) |
| the model's own P(actionable) | that head already exists; relabeling its output "priority" would be renaming, not a new capability |

## What would make this buildable

Repos that actually use priority labels (many enterprise-internal repos
do; among large OSS repos, e.g. `flutter/flutter` uses `P0`–`P3`).
Collecting such repos gives real ground truth, and the existing champion
protocol applies unchanged. Until then, any "priority score" this product
displayed would be a fabricated number wearing a UI.
