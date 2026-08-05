"""Upstash QStash: the durable queue that makes async webhook processing
correct on Vercel's Python runtime, where neither FastAPI's BackgroundTasks
nor Vercel's own `waitUntil()` continuation is reliably available (waitUntil
is documented for Node.js/Edge functions, not Python; BackgroundTasks
depends on the process staying alive after the response, which a Vercel
Python function doesn't guarantee).

Flow: /webhook publishes a job here and returns immediately; QStash calls
back a separate endpoint (/internal/process-issue) with the same payload,
signed so we can verify it actually came from QStash. See
docs/DEPLOYMENT.md's async-processing section for setup and
models/ASYNC_PROCESSING_CARD.md for why this exists instead of a simpler
mechanism.
"""
from __future__ import annotations

from typing import Any

import requests

from .. import utils

logger = utils.get_logger(__name__)

_PUBLISH_TIMEOUT_S = 10.0


class QStashError(Exception):
    """Publishing to QStash failed. Callers should catch this and fall back
    to synchronous processing rather than dropping the issue entirely."""


def publish(
    destination_url: str,
    payload: dict[str, Any],
    token: str,
    *,
    region: str = "us-east-1",
    timeout: float = _PUBLISH_TIMEOUT_S,
) -> str:
    """Enqueue `payload` for delivery to `destination_url`. Returns QStash's
    messageId. Raises QStashError on any failure -- the caller decides
    whether to fall back to processing synchronously."""
    publish_url = f"https://qstash-{region}.upstash.io/v2/publish/{destination_url}"
    try:
        resp = requests.post(
            publish_url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise QStashError(f"could not reach QStash: {e}") from e

    if not resp.ok:
        raise QStashError(f"QStash publish returned HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    message_id = data.get("messageId")
    if not message_id:
        raise QStashError(f"QStash publish response had no messageId: {data}")
    logger.info("published job to qstash: messageId=%s destination=%s", message_id, destination_url)
    return message_id


def verify_signature(
    signature_header: str | None,
    body: bytes,
    signing_keys: list[str],
    expected_url: str,
) -> bool:
    """Validate the `Upstash-Signature` header QStash attaches to its
    callback. Tries each configured signing key in order (current, then
    next) -- Upstash rotates these, and a request signed under the
    previous key during a rotation window must still verify.
    """
    import base64
    import hashlib

    import jwt

    if not signature_header or not signing_keys:
        return False

    body_hash = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("=")

    for key in signing_keys:
        if not key:
            continue
        try:
            claims = jwt.decode(
                signature_header, key, algorithms=["HS256"],
                options={"require": ["exp", "iss", "sub"]},
            )
        except jwt.InvalidTokenError:
            continue
        if claims.get("iss") != "Upstash":
            continue
        if claims.get("sub") != expected_url:
            continue
        claimed_hash = str(claims.get("body", "")).rstrip("=")
        if claimed_hash != body_hash:
            continue
        return True
    return False
