# Privacy

This service is designed to store as close to nothing as possible.

**What it receives:** GitHub webhook payloads for issues on repositories
where the App is installed — issue title, body, author login, labels, and
close reason — plus, for enrichment, the author's public profile fields
(account creation date, public repo count, follower count) and the repo's
latest release date via the GitHub API.

**What it stores:** one line per prediction in a local ledger
(`data/predictions.jsonl`): repository name, issue number, predicted
probability, and — when the issue closes — the rule-derived outcome. No issue
text, no author data, and no tokens are ever persisted. Disable even this
with `GHIC_LEDGER=""` (metrics then live in memory only).

**What it sends:** only back to GitHub — an optional prediction comment
and/or label on the scored issue. Nothing is transmitted to any third party.

**Data deletion:** delete the ledger file; there is nothing else to delete.
Uninstalling the App stops all data flow immediately.
