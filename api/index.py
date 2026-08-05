"""Vercel Python entrypoint: exposes the FastAPI app as the ASGI handler.

Vercel's Python runtime looks for a top-level `app` in api/index.py. The
`ghic` package isn't pip-installed in this deployment (see requirements.txt
at the repo root — pyproject.toml is hidden from Vercel via .vercelignore so
requirements.txt wins dependency detection), so it's imported straight from
source; the project root needs to be on sys.path first.

Settings come from GHIC_* environment variables configured in the Vercel
project (see docs/DEPLOYMENT.md, "Deploying on Vercel"). GHIC_DATABASE_URL /
DATABASE_URL / POSTGRES_URL, when set, switch the online-evaluation ledger to
Postgres — required here, since Vercel functions have no persistent disk.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghic.service.app import create_app  # noqa: E402

app = create_app()
