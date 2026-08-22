from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

from ghic.service.tracking import PredictionTracker


class RecordingBackend:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(dict(record))

    def replay(self):
        return iter(())


def test_concurrent_events_keep_their_explicit_workspace() -> None:
    backend = RecordingBackend()
    tracker = PredictionTracker(backend=backend)
    workers = 12
    barrier = Barrier(workers)

    def emit(workspace_id: str, sequence: int) -> None:
        repo = f"{workspace_id}/repo"
        barrier.wait()
        tracker.record_prediction(
            repo, sequence, 0.9, 1, workspace_id=workspace_id
        )
        tracker.record_analysis(
            repo, sequence, {"title": "issue"}, workspace_id=workspace_id
        )
        tracker.record_processing_failure(
            repo, sequence, RuntimeError("test"), workspace_id=workspace_id
        )
        tracker.record_action(
            repo, sequence, "comment", workspace_id=workspace_id
        )
        tracker.record_authorization_skip(
            repo, sequence, sequence, "revoked_installation",
            workspace_id=workspace_id,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(emit, "workspace-a" if i % 2 == 0 else "workspace-b", i)
            for i in range(workers)
        ]
        for future in futures:
            future.result()

    assert len(backend.records) == workers * 5
    for record in backend.records:
        assert record["workspace_id"] in {"workspace-a", "workspace-b"}
        assert record["repo"].startswith(f"{record['workspace_id']}/")


def test_workspace_attribution_is_not_tracker_process_state() -> None:
    tracker = PredictionTracker()

    assert not hasattr(tracker, "workspace_id")
    assert not hasattr(tracker, "set_workspace_id")
