"""Tests for the retraining pipeline's versioning plumbing (ghic.retrain).

The full pipeline is exercised by an actual documented run (see
models/REGISTRY.md); these tests cover the registry/snapshot mechanics so a
refactor can't silently stop recording history.
"""
from __future__ import annotations

from ghic import retrain


class TestRegistry:
    def test_header_written_once_rows_append(self, tmp_path, monkeypatch):
        monkeypatch.setattr(retrain, "REGISTRY_PATH", tmp_path / "REGISTRY.md")
        monkeypatch.setattr(retrain, "MODELS_DIR", tmp_path)  # no champion.joblib
        champion = {"winner": "rf_balanced", "test_calibrated": {"pr_auc": 0.7351}}
        category = {"test": {"macro_f1": 0.4701}}

        retrain.append_registry("2026-07-14T00-00-00Z", champion, category)
        retrain.append_registry("2026-07-14T01-00-00Z", champion, None)

        text = (tmp_path / "REGISTRY.md").read_text(encoding="utf-8")
        assert text.count("# Model registry") == 1
        assert "| 2026-07-14T00-00-00Z | `rf_balanced` | 0.7351 | 0.4701 | `missing` |" in text
        assert "| 2026-07-14T01-00-00Z | `rf_balanced` | 0.7351 | n/a | `missing` |" in text

    def test_sha256_is_content_addressed(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"model-bytes")
        b.write_bytes(b"model-bytes")
        assert retrain._sha256(a) == retrain._sha256(b)
        b.write_bytes(b"other")
        assert retrain._sha256(a) != retrain._sha256(b)


class TestSnapshot:
    def test_snapshot_copies_present_files_only(self, tmp_path, monkeypatch):
        models = tmp_path / "models"
        reports = tmp_path / "reports"
        models.mkdir()
        reports.mkdir()
        (models / "MODEL_CARD.md").write_text("card", encoding="utf-8")
        (reports / "champion.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(retrain, "MODELS_DIR", models)
        monkeypatch.setattr(retrain, "RUNS_DIR", reports / "runs")
        monkeypatch.setattr(retrain.evaluate, "REPORTS_DIR", reports)

        run_dir = retrain.snapshot_run("2026-07-14T00-00-00Z")
        assert (run_dir / "MODEL_CARD.md").read_text(encoding="utf-8") == "card"
        assert (run_dir / "champion.json").exists()
        assert not (run_dir / "CATEGORY_CARD.md").exists()   # absent input, absent output
