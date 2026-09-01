# Data retention

The marketing site and the privacy policy both say the same specific thing:

> Prediction records are retained for **90 days**, then deleted.

Until `ghic/service/retention.py` existed, that was false. A retention claim
that does not hold is not a missing feature — it is a statement to customers
that is not true, which is a different kind of problem and a worse one.

## What is deleted

| Table | Deleted after 90 days | Why |
|---|---|---|
| `ghic_ledger` | yes | The prediction records the policy names. |
| `ghic_idempotency` | yes | Not promised, but `idempotency.py` documents that nothing prunes these and they grow one row per delivery forever. A key older than the window cannot suppress a real redelivery — GitHub gives up retrying long before then. |

## What is not, and why

- **`ghic_usage_events`** — billing records. A customer needs them to dispute
  a charge, and they say nothing about the content of an issue.
- **`ghic_repo_chunks` / `ghic_repository_state`** — derived from code the
  App can re-read at any time, and replaced wholesale on reindex. Deleting
  them by age would silently degrade retrieval rather than protect anything.
- **`ghic_github_installations` / `ghic_github_repositories`** — connection
  state. This is what makes authorization work; ageing it out would revoke
  a working installation on a timer.
- **`ghic_users` / `ghic_workspaces`** — accounts. Deleting an account is
  what uninstall and account deletion are for; see `UNINSTALL_CLEANUP.md`.

Deletion is by age and nothing else — no workspace filter, no repository
filter. A retention window that applies to some tenants and not others is
not a retention window.

## How it runs

`GET /internal/retention`, on a Vercel cron at 03:00 UTC daily
(`vercel.json`). Authenticated with `CRON_SECRET`, the bearer token Vercel
Cron sends.

**With no `CRON_SECRET` configured the endpoint returns 503 rather than
running.** An unauthenticated deletion sweep reachable from the internet is
worse than a retention promise that is late.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `CRON_SECRET` | unset | Required. Without it the endpoint refuses. |
| `GHIC_RETENTION_DAYS` | `90` | The window. `0` disables the sweep — correct for a local run with no database, not a supported production setting. |

The sweep is safe to run repeatedly and safe to run concurrently: a second
pass finds nothing left, and two overlapping passes delete disjoint sets
because each `DELETE` takes its own row locks.

## Consequence worth knowing

Deleting the ledger shrinks the online-evaluation history. Analytics over a
window longer than 90 days will thin out, and predictions whose outcome
arrives after the window will not find their pending record. That is the
trade the published policy commits to, not an oversight.
