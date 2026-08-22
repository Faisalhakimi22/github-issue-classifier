"""Prove the reset can only touch the intended rows, before it runs.

Read-only. Every check is phrased as "how many rows are NOT a target" so a
non-zero answer blocks the reset instead of quietly widening it.
"""
import json
import os
import sys
from contextlib import closing
from urllib.parse import unquote, urlparse

import pg8000.native

REPO = "Faisalhakimi22/github-issue-classifier"
WS = "ghic-default-workspace"

p = urlparse(os.environ["DATABASE_URL_UNPOOLED"])
failures = []


def check(label, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + label + ((" -- " + detail) if detail else ""))
    if not ok:
        failures.append(label)


with closing(pg8000.native.Connection(
    user=unquote(p.username), password=unquote(p.password or ""), host=p.hostname,
    port=p.port or 5432, database=unquote(p.path.lstrip("/")), ssl_context=True,
    application_name="ghic-reset-target-verification",
    startup_params={"options": "-c default_transaction_read_only=on"},
)) as c:
    assert c.run("SELECT current_setting('transaction_read_only')")[0][0] == "on"

    # 1. Nothing is connected, so the orphan assumption holds.
    check("zero installations exist",
          c.run("SELECT count(*) FROM ghic_github_installations")[0][0] == 0)
    check("zero repositories exist",
          c.run("SELECT count(*) FROM ghic_github_repositories")[0][0] == 0)

    # 2. Every chunk is the target repo's. Any other chunk blocks the reset.
    total_chunks = c.run("SELECT count(*) FROM ghic_repo_chunks")[0][0]
    off_target_chunks = c.run(
        "SELECT count(*) FROM ghic_repo_chunks "
        "WHERE repo IS DISTINCT FROM :repo OR workspace_id IS DISTINCT FROM :ws",
        repo=REPO, ws=WS)[0][0]
    check("all chunks belong to the target repo", off_target_chunks == 0,
          f"{off_target_chunks} off-target of {total_chunks}")
    check("chunk count is exactly 1891", total_chunks == 1891, str(total_chunks))
    distinct_repos = [r[0] for r in c.run(
        "SELECT DISTINCT repo FROM ghic_repo_chunks")]
    check("only one distinct repo in chunks", distinct_repos == [REPO],
          str(distinct_repos))

    # 3. Same for repository state.
    total_state = c.run("SELECT count(*) FROM ghic_repository_state")[0][0]
    off_target_state = c.run(
        "SELECT count(*) FROM ghic_repository_state "
        "WHERE repo IS DISTINCT FROM :repo OR workspace_id IS DISTINCT FROM :ws",
        repo=REPO, ws=WS)[0][0]
    check("all repository_state rows are the target repo", off_target_state == 0,
          f"{off_target_state} off-target of {total_state}")
    check("repository_state count is exactly 1", total_state == 1, str(total_state))

    # 4. Every ledger row is unattributed. An attributed row would belong to a
    #    real workspace and must not be swept up by --include-ledger.
    total_ledger = c.run("SELECT count(*) FROM ghic_ledger")[0][0]
    attributed = c.run(
        "SELECT count(*) FROM ghic_ledger WHERE workspace_id IS NOT NULL")[0][0]
    check("no ledger row has a workspace", attributed == 0, str(attributed))
    check("ledger count is exactly 33", total_ledger == 33, str(total_ledger))
    print("      ledger breakdown:", json.dumps([
        dict(zip(("repo", "n"), (r[0], r[1]))) for r in c.run(
            "SELECT data->>'repo', count(*) FROM ghic_ledger GROUP BY 1 ORDER BY 2 DESC")
    ]))

    # 5. Protected tables, recorded so the after-state can be compared.
    protected = {}
    for table in ("ghic_users", "ghic_workspaces", "ghic_workspace_members",
                  "ghic_workspace_settings", "ghic_org_settings",
                  "ghic_schema_migrations", "ghic_idempotency",
                  "ghic_github_installation_intents"):
        protected[table] = c.run(f'SELECT count(*) FROM "{table}"')[0][0]
    print("      protected baseline:", json.dumps(protected))

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED -- do not run the reset: {failures}")
    sys.exit(1)
print("VERIFIED: the reset can only touch the intended rows.")
