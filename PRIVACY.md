# Privacy

This service is designed to store as close to nothing as possible.

**What it receives:** GitHub webhook payloads for issues on repositories
where the App is installed — issue title, body, author login, labels, and
close reason — plus, for enrichment, the author's public profile fields
(account creation date, public repo count, follower count) and the repo's
latest release date via the GitHub API.

**What it stores:** an append-only local ledger (`data/predictions.jsonl`)
with one line per event: predictions (repository name, issue number,
predicted probability, timestamp), rule-derived outcomes at close, an audit
record for each write the bot performs (action type + timestamp), and
maintainer label events (label name + timestamp — these are future training
ground truth). No issue text, no author data, and no tokens are ever
persisted. Disable even this with `GHIC_LEDGER=""` (metrics then live in
memory only).

**What it sends:** back to GitHub — an optional prediction comment, label,
and/or project placement on the scored issue. One optional exception: if the
operator enables LLM drafting (`GHIC_DRAFT_MISSING_INFO=true` **and**
configures `ANTHROPIC_API_KEY`), the title/body of under-specified issues
and the titles of similar prior issues are sent to the Anthropic API to
draft the "missing information" comment. This is off by default, and without
an API key the feature uses a local template and transmits nothing. No other
third-party transmission exists.

**Data deletion:** delete the ledger file; there is nothing else to delete.
Uninstalling the App stops all data flow immediately.
