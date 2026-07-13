"""Load-test the running service — real numbers, no projections.

Fires concurrent POST /api/predict requests at a live instance and reports
client-observed latency percentiles and throughput. The scoring path is
CPU-bound sklearn inside a single uvicorn worker, so rising p95 under
concurrency is expected — the point is to measure it, not assume it.

Usage:
  python scripts/loadtest.py [--url http://127.0.0.1:8000] [--token SECRET]
                             [--requests 200] [--concurrency 8]
Requires the service running, e.g.:
  GHIC_ALLOW_UNSIGNED=true python -m ghic.service.app
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

BODY_BUG = {
    "title": "Crash when opening the settings panel",
    "body": ("Steps to reproduce:\n1. open settings\n2. crash\n\n"
             "```\nTraceback (most recent call last): ...\n```\n"
             "Expected: no crash. Actual: crashes every time on v2.3 / Windows 11."),
}
BODY_VAGUE = {"title": "please add dark mode", "body": "would be nice, thanks"}


async def _worker(client, url: str, headers: dict, n: int,
                  latencies: list[float], errors: list[str]) -> None:
    for i in range(n):
        payload = BODY_BUG if i % 2 == 0 else BODY_VAGUE
        started = time.perf_counter()
        try:
            resp = await client.post(f"{url}/api/predict", json=payload,
                                     headers=headers, timeout=60)
            elapsed = (time.perf_counter() - started) * 1000
            if resp.status_code == 200:
                latencies.append(elapsed)
            else:
                errors.append(f"HTTP {resp.status_code}")
        except Exception as e:
            errors.append(type(e).__name__)


async def run(url: str, token: str, total: int, concurrency: int) -> dict:
    import httpx

    headers = {"X-GHIC-Token": token} if token else {}
    latencies: list[float] = []
    errors: list[str] = []
    per_worker = max(1, total // concurrency)

    async with httpx.AsyncClient() as client:
        health = await client.get(f"{url}/healthz", timeout=10)
        health.raise_for_status()
        model = health.json().get("model")

        started = time.perf_counter()
        await asyncio.gather(*[
            _worker(client, url, headers, per_worker, latencies, errors)
            for _ in range(concurrency)
        ])
        wall = time.perf_counter() - started

    latencies.sort()

    def pct(p: float) -> float:
        return round(latencies[min(len(latencies) - 1, int(len(latencies) * p))], 1)

    return {
        "model": model,
        "requests_ok": len(latencies),
        "errors": len(errors),
        "concurrency": concurrency,
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(len(latencies) / wall, 2) if wall else None,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1),
            "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
            "max": round(latencies[-1], 1),
        } if latencies else {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", default="reports/loadtest.json")
    args = parser.parse_args()

    result = asyncio.run(run(args.url, args.token, args.requests, args.concurrency))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
