# Uninstall cleanup

What happens to stored data when a GitHub App installation goes away, and why
each record is on the side of the line it is on.

## The trigger

GitHub delivers `installation` with `action: deleted` when someone uninstalls
the App. That delivery is the source of truth. It arrives at `POST /webhook`,
is verified against `X-Hub-Signature-256`, deduplicated on `X-GitHub-Delivery`
via `ghic_idempotency`, and dispatched to `handle_installation_event`
(`ghic/service/github_connection.py`).

Two narrower events purge a single repository while leaving the installation
in place:

| Event | Effect |
|---|---|
| `installation` / `deleted` | purge the installation and everything it owns |
| `installation_repositories` / `removed` | purge only the named repositories |
| `repository` / `deleted` | purge only that repository |
| `repository` / `archived` | **no purge** — deactivate only |

Archiving is reversible and the code stays readable, so the index is kept
rather than paying to re-embed on unarchive. Deletion on GitHub is not
reversible and the index could never be refreshed or verified again.

## Deleted

Scoped to `installation_id = X`, and within that to each repository's
`(workspace_id, repo)` pair — the ownership key the foreign keys are actually
declared on.

| Table | Predicate | What it is |
|---|---|---|
| `ghic_repo_chunks` | `workspace_id = W AND repo = R` | indexed content and `vector(1536)` embeddings |
| `ghic_repository_state` | `workspace_id = W AND repo = R` | index state, `indexed_commit_sha`, background processing state |
| `ghic_ledger` | `workspace_id = W AND data->>'repo' = R` | predictions, analyses, actions, outcomes, label events |
| `ghic_ledger` | `workspace_id = W AND data->>'installation_id' = X` | installation-scoped rows naming no repository, such as authorization skips |
| `ghic_github_repositories` | `installation_id = X` | repository metadata |
| `ghic_github_installations` | `installation_id = X` | the installation record |

Analytics need no separate deletion. Every dashboard metric is derived from
`ghic_ledger` at query time in `product-data.mjs`; there is no materialised
analytics table. Removing the ledger rows removes the analytics.

Ledger attribution by repository name is unambiguous because
`ghic_github_repositories.repo_full_name` is a primary key — a repository
belongs to exactly one installation at a time.

## Retained

| Table | Why |
|---|---|
| `ghic_users` | Uninstalling an App is not account deletion. Account removal needs its own explicit flow. |
| `ghic_workspaces`, `ghic_workspace_members`, `ghic_workspace_settings` | Workspace-level and shared across installations. A workspace survives losing its last installation; it simply has nothing connected. |
| `ghic_org_settings` | Shared configuration, still used by anything else in the workspace. |
| `ghic_idempotency` | Global webhook delivery dedupe. **Deleting it would be actively harmful** — a redelivered GitHub event could recreate the data just removed. Retaining it is part of the resurrection defence below. |
| `ghic_github_installation_intents` | Pre-install handshake keyed by `state_hash`/`firebase_uid`/`workspace_id`, with no `installation_id` column. It cannot be attributed to an installation and expires on its own via `expires_at`. |
| `ghic_schema_migrations` | Schema bookkeeping. |
| `ghic_ledger` rows where `workspace_id IS NULL` | Belong to no installation, so no installation's removal may claim them. See the tenancy notes in the Hub repo. |
| Any repository row whose `workspace_id` is empty | Unattributable. Guessing an owner is how one workspace's data gets deleted by another's uninstall, so its content is left and a warning is logged. |

## Ordering

Nothing in this schema cascades, so every deletion is explicit and ordered
children-first, inside one transaction:

1. `SELECT ... FOR UPDATE` the installation, then its repositories
2. `ghic_repo_chunks`
3. `ghic_repository_state`
4. `ghic_ledger` (per repository)
5. `ghic_github_repositories`
6. `ghic_ledger` (installation-scoped remainder)
7. `ghic_github_installations`

The installation is locked before its repositories, matching the order
`authorization_lease` takes them, so the two cannot deadlock.

## Why background jobs cannot recreate the data

Three layers, two of which already existed:

1. **The lease fails closed.** `authorization_lease` joins installations to
   repositories `FOR UPDATE`. With the rows deleted the join returns nothing
   and it yields `authorized: False, reason: "repository_not_connected"`. Every
   queued QStash job dies there. Deleting the rows *is* the kill switch — there
   is no separate cancellation channel to keep in sync.
2. **The foreign key enforces on INSERT.** `ghic_repo_chunks (workspace_id,
   repo)` references `ghic_github_repositories`. The constraint is `NOT VALID`,
   which skips the scan of pre-existing rows but still enforces every new write.
   A job that somehow bypassed the lease could not insert chunks once the parent
   row is gone.
3. **The opening `FOR UPDATE` closes the check-to-write window.** A job already
   holding the lease finishes before the purge proceeds, rather than having rows
   deleted underneath it mid-index.

## Failure handling

`handle_installation_event` never raises; the rest of it is bookkeeping where a
failed reconciliation should not turn into a webhook error. A failed *purge* is
different: acknowledging it as done would leave behind exactly the data that
uninstalling was supposed to remove. So the purge path returns
`{"ok": False, "connection_sync": "purge_failed", "retryable": True}` and
`/webhook` answers 500 so GitHub retries.

That retry only works because the webhook **releases the delivery key first**.
Deliveries are marked consumed *before* being handled — correct for issue
events, where a retry must not post a second comment — which means a retried
delivery would otherwise be discarded as a duplicate and the purge would never
happen. `IdempotencyStore.release` exists for this one case and is safe here
only because the purge is idempotent. A successful purge keeps its key.

## Idempotency

Every statement is a `DELETE ... WHERE`, so a second pass removes nothing and
still reports success. Safe when records are already gone, safe on redelivery,
and safe to run by hand. Partial cleanup self-heals: a purge that failed
halfway rolled back entirely, and one that ran against already-deleted rows
reports zero counts.

## What this does not cover

Orphaned content — chunks or state whose repository row no longer exists —
is **not** reachable by this path, because the purge keys off
`ghic_github_installations`. That is deliberate: it will not guess an owner.
Clearing such residue is a manual, itemised operation
(`scripts/reset_orphaned_content.py`, with `scripts/verify_orphan_reset_targets.py`
to prove the targets first), never something a webhook can do. That script
refuses to run at all while any installation row exists, because the orphan
assumption would no longer hold.
