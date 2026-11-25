"""Run one repository-index job outside the web service.

    python -m ghic.index_worker --repo owner/name --installation-id 123

This is what `.github/workflows/index-on-demand.yml` runs. It exists because
Vercel's Python runtime has no `git`: the service dispatches the workflow
(GitHubActionsIndexQueue), and the runner, which has git, clones, indexes
and writes to the same Postgres the service reads.

It goes through ghic/service/index_job.py, the same authorization path as
the QStash callback. A workflow input is a request, never a grant: an
installation that was removed, suspended or never connected is refused here
however the dispatch was made.

Output is deliberately terse. If the workflow runs in a public repository
its logs are public, so nothing about the customer's code (paths, languages,
file counts) is printed; only the outcome and a refusal reason.
"""
from __future__ import annotations

import argparse
import sys

from . import utils

logger = utils.get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one GHIC repository-index job.")
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--installation-id", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--correlation-id", default="")
    parser.add_argument(
        "--verbose", action="store_true",
        help="keep INFO logs (they name files in the customer's repository)",
    )
    args = parser.parse_args(argv)

    from .repository_intelligence import build_service
    from .service.github_app import GitHubAppClient
    from .service.github_connection import authorization_lease, authorize_issue
    from .service.index_job import run_index_job
    from .service.settings import load_settings

    settings = load_settings()
    if not settings.database_url:
        print("::error::no database configured (GHIC_DATABASE_URL / DATABASE_URL)",
              file=sys.stderr)
        return 2
    if not settings.can_call_github:
        print("::error::GitHub App credentials missing (GHIC_APP_ID, private key)",
              file=sys.stderr)
        return 2

    service = build_service(database_url=settings.database_url)
    if service is None:
        print("::error::repository intelligence is not available with this configuration",
              file=sys.stderr)
        return 2

    if not args.verbose:
        _quiet_ghic_loggers()

    gh = GitHubAppClient(
        settings.app_id,
        settings.private_key_pem,
        base_url=settings.api_base_url,
        timeout=settings.request_timeout,
    )
    try:
        result = run_index_job(
            {
                "repo": args.repo,
                "installation_id": args.installation_id,
                "force": args.force,
                "correlation_id": args.correlation_id,
            },
            database_url=settings.database_url,
            repo_intelligence=service,
            gh=gh,
            authorization_gate=authorize_issue,
            authorization_lease=authorization_lease,
        )
    except Exception as e:  # unexpected: fail the run so it shows red
        logger.exception("index job crashed")
        print(f"::error::index job crashed: {type(e).__name__}", file=sys.stderr)
        return 1

    if result.get("skipped"):
        # A refusal is a correct outcome, not a failure to retry.
        print(f"refused: {result.get('reason')}")
        return 0
    if not result.get("indexed"):
        print("::error::indexing did not produce an index")
        return 1
    print("indexed")
    return 0


def _quiet_ghic_loggers() -> None:
    """Drop every ghic logger to WARNING. The indexer's INFO lines name
    paths, languages and frameworks from the repository being indexed,
    which do not belong in a CI log."""
    import logging

    for name in list(logging.root.manager.loggerDict):
        if name == "ghic" or name.startswith("ghic."):
            logging.getLogger(name).setLevel(logging.WARNING)


if __name__ == "__main__":
    raise SystemExit(main())
