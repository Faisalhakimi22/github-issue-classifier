# Card — asynchronous webhook processing + idempotency

**Task:** `/webhook` should validate and return HTTP 200 fast, do the slow
work (LLM analysis, GitHub comment) after responding, and never process the
same event twice into a duplicate comment.

## Why this isn't just `BackgroundTasks`

That's the obvious first idea, and it's simply correct on Docker/Fly — a
persistent process can keep running code after sending a response, and
FastAPI's `BackgroundTasks` does exactly that. It is **not** reliable on
this project's other deployment target, Vercel's Python runtime, for two
independently confirmed reasons (checked directly, not assumed, before
building anything):

1. Vercel's `waitUntil()` continuation primitive — the supported way to
   keep work running after a response on Vercel Functions — is documented
   for Node.js/Edge functions. Python is not listed as supported.
2. FastAPI's `BackgroundTasks` on Vercel's Python runtime is explicitly
   unreliable: the function instance can be frozen/terminated before the
   background task finishes, because there's no persistent process backing
   it the way Docker/Fly has.

The other obvious fallback — "come back and finish the job on a timer" via
Vercel Cron Jobs — doesn't work either on this project's plan: **Vercel
Cron Jobs are capped at once per day on the Hobby tier.** A triage bot that
comments up to 24 hours after an issue opens has lost the entire point of
being a *triage* bot.

## What ships instead: Upstash QStash

A real durable queue, built for exactly this shape of problem (serverless
webhook receiver → reliable deferred HTTP callback). `/webhook` validates
and idempotency-checks the delivery, publishes the job to QStash, and
returns — typically well under a second, all synchronous work being one
small HTTP POST. QStash calls back `/internal/process-issue` with the same
payload, signed with a JWT (`Upstash-Signature` header, HS256, verified via
PyJWT — already a project dependency, no new SDK needed) so the callback
can't be spoofed. That endpoint runs the exact same `_process_issue_job()`
function the synchronous path always used — same ML scoring, same LLM
analysis, same comment format, same everything; only *when* it runs
changed.

**Off by default.** `GHIC_USE_ASYNC_PROCESSING=false` until the operator
sets that flag and `QSTASH_TOKEN` / `GHIC_PUBLIC_BASE_URL` /
`QSTASH_CURRENT_SIGNING_KEY`. With it off, `/webhook` processes every issue
inline exactly as it did before this feature existed — this is purely
additive, not a rewrite of the default path. If QStash's publish call
itself fails (network issue, bad token), the handler logs a warning and
falls back to processing inline rather than dropping the issue.

## Idempotency

Keyed on GitHub's own `X-GitHub-Delivery` header, which GitHub reuses
verbatim when it retries a delivery that didn't get a timely 2xx. Checked
at the very top of `/webhook`, before any routing — so a retried delivery
of *any* event type (not just `issues.opened`) short-circuits to
`{"ok": true, "duplicate": true}` without reprocessing. This is independent
of the async-queue feature and active whenever a delivery ID is present,
regardless of whether `GHIC_USE_ASYNC_PROCESSING` is on.

Three backends (`ghic/service/idempotency.py`), same split as the
online-evaluation ledger: Postgres (atomic `INSERT ... ON CONFLICT DO
NOTHING RETURNING`, the one that matters under real concurrency — shares
`database_url` with the ledger, no separate connection string) for Vercel;
a lock-guarded JSON file for Docker/Fly; in-memory (this process's
lifetime only) when neither is configured. None of the three backends
prune old keys — a known, undramatic limitation, same growth
characteristic the ledger already has.

## What this doesn't change

- The ML classifier's decision, the LLM analysis logic, the comment
  format — none of it moved. `_process_issue_job()` is a rename/extraction
  of the code that was always there, not new logic.
- Every other webhook event (`edited`, `closed`, `labeled`/`unlabeled`)
  stays synchronous — they're all already fast (no LLM call in those
  paths), so there's nothing to defer.
