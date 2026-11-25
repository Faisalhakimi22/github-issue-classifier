"""Error reporting must be inert without a DSN and must never ship
customer content when it is on."""
from __future__ import annotations

from ghic.service import monitoring


def test_without_dsn_monitoring_stays_off(monkeypatch) -> None:
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    monkeypatch.setattr(monitoring, "_initialized", False)
    assert monitoring.init_monitoring("1.0") is False


def test_scrub_drops_bodies_headers_identity_and_frame_locals() -> None:
    event = {
        "request": {
            "url": "https://api/webhook",
            "method": "POST",
            "data": {"issue": {"body": "customer secret text"}},
            "headers": {"X-Hub-Signature-256": "sha256=abc", "Authorization": "Bearer t"},
            "cookies": {"s": "1"},
            "query_string": "token=x",
            "env": {"REMOTE_ADDR": "1.2.3.4"},
        },
        "user": {"ip_address": "1.2.3.4"},
        "extra": {"payload": "x"},
        "breadcrumbs": {"values": [{"data": {"url": "https://api.github.com/repos/c/a"}}]},
        "exception": {"values": [{
            "type": "ValueError",
            "stacktrace": {"frames": [{"function": "f", "vars": {"body": "secret"}}]},
        }]},
    }
    out = monitoring.scrub_event(event)

    assert out["request"] == {"url": "https://api/webhook", "method": "POST"}
    assert "user" not in out and "extra" not in out and "breadcrumbs" not in out
    frame = out["exception"]["values"][0]["stacktrace"]["frames"][0]
    assert frame == {"function": "f"}
    assert out["exception"]["values"][0]["type"] == "ValueError"


def test_scrub_tolerates_minimal_events() -> None:
    assert monitoring.scrub_event({"message": "x"}) == {"message": "x"}
