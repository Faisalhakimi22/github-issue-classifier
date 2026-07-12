"""Project utilities: paths, logging, JSON IO, content-addressed cache, retries.

Imported by every other ghic/ module. Holds no project-specific logic — anything
referencing labels, features, or the GitHub schema lives elsewhere so this
module can be reused unchanged by the webhook service.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, TypeVar


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Project root is the parent of the package directory housing this file. Override
# via GHIC_PROJECT_ROOT when the package is vendored into another service
# (e.g. the webhook) that does not share this layout.
def _resolve_project_root() -> Path:
    override = os.environ.get("GHIC_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = _resolve_project_root()
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
LOG_DIR: Path = PROJECT_ROOT / "data" / "logs"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Idempotent: repeated calls reuse the same handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_LOG_FMT))
    logger.addHandler(console)
    return logger


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------
def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Content-addressed cache
# ---------------------------------------------------------------------------
# Keys are arbitrary strings (typically GraphQL query + serialized variables).
# Hashing them to a stable filename means repeated identical fetches reuse the
# disk copy without re-hitting the GitHub API.
def cache_key(*parts: Any) -> str:
    """Stable SHA256 hex of the JSON-serialized parts."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(namespace: str, key: str) -> Path:
    return DATA_RAW / namespace / f"{key}.json"


def cache_get(namespace: str, key: str) -> Any | None:
    p = cache_path(namespace, key)
    return read_json(p) if p.exists() else None


def cache_put(namespace: str, key: str, value: Any) -> None:
    write_json(cache_path(namespace, key), value)


# ---------------------------------------------------------------------------
# Retry with exponential backoff + jitter
# ---------------------------------------------------------------------------
F = TypeVar("F", bound=Callable[..., Any])


def retry_with_backoff(
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    logger: logging.Logger | None = None,
) -> Callable[[F], F]:
    """Decorator: retry `fn` with exponential backoff + jitter.

    Re-raises the final exception if all attempts fail. Logs a warning between
    attempts so a stalled collection run is visible in stdout.
    """
    log = logger or get_logger("retry")

    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    delay = min(base_delay * 2 ** (attempt - 1), max_delay)
                    delay += random.uniform(0, delay * 0.25)
                    log.warning(
                        "%s attempt %d/%d failed: %s — retrying in %.1fs",
                        fn.__name__, attempt, max_attempts, e, delay,
                    )
                    time.sleep(delay)

        return wrapper  # type: ignore[return-value]

    return deco
