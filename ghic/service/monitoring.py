"""Error reporting. Off unless SENTRY_DSN is set.

Unhandled exceptions in a webhook or queue callback otherwise vanish into
Vercel's function logs, which nobody reads until a customer complains. This
sends them to Sentry instead.

What never leaves: request bodies (issue text and webhook payloads are
customer content), headers (signatures, tokens), cookies, user identity,
and local variables in stack frames (they hold issue bodies and tokens).
Sentry then sees an exception type, its message, a stack trace and the
route. That is enough to find the bug; anything more is customer data in a
third party's database, and would have to be listed in PRIVACY.md.
"""
from __future__ import annotations

import os
from typing import Any

from .. import utils

logger = utils.get_logger(__name__)

_initialized = False

# Request fields Sentry's integrations may attach. Dropped wholesale rather
# than scrubbed key by key: a scrub list is only as good as the last header
# someone remembered to add to it.
_DROP_REQUEST_FIELDS = ("data", "headers", "cookies", "query_string", "env")


def scrub_event(event: dict[str, Any], hint: Any = None) -> dict[str, Any]:
    """before_send hook: strip everything that could carry customer data."""
    request = event.get("request")
    if isinstance(request, dict):
        for key in _DROP_REQUEST_FIELDS:
            request.pop(key, None)
    event.pop("user", None)
    event.pop("extra", None)
    for exception in (event.get("exception") or {}).get("values") or []:
        for frame in ((exception.get("stacktrace") or {}).get("frames") or []):
            frame.pop("vars", None)
    # Breadcrumbs include outbound HTTP calls (GitHub, LLM providers) with
    # full URLs, and log lines that name repositories.
    event.pop("breadcrumbs", None)
    return event


def init_monitoring(version: str = "") -> bool:
    """Initialise Sentry once per process. Returns True when active.

    Never raises: a monitoring misconfiguration must not stop the service
    from booting.
    """
    global _initialized
    if _initialized:
        return True
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        logger.warning("SENTRY_DSN is set but sentry-sdk is not installed; errors are not reported")
        return False
    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get("VERCEL_ENV") or os.environ.get("GHIC_ENV") or "production",
            release=os.environ.get("VERCEL_GIT_COMMIT_SHA") or version or None,
            send_default_pii=False,
            max_request_body_size="never",
            include_local_variables=False,
            max_breadcrumbs=0,
            traces_sample_rate=0.0,   # errors only; no performance data
            before_send=scrub_event,
        )
    except Exception as e:
        logger.warning("could not initialise Sentry: %s", e)
        return False
    _initialized = True
    logger.info("error reporting enabled")
    return True
