"""End-to-end tests for the unified `ghic` CLI.

Passthrough subcommands are tested by delegation (the target module's main
is already covered by its own tests); predict/explain run end-to-end against
the real trained model when one is present.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghic import cli

MODEL_PRESENT = (Path(__file__).resolve().parents[1] / "models" / "champion.joblib").exists() or (
    Path(__file__).resolve().parents[1] / "models" / "rf_balanced.joblib"
).exists()


class TestDelegation:
    def test_train_passes_args_through(self, monkeypatch):
        seen = {}
        monkeypatch.setattr("ghic.train.main", lambda argv: seen.update(argv=argv) or 0)
        assert cli.main(["train", "--no-plots", "--champion"]) == 0
        assert seen["argv"] == ["--no-plots", "--champion"]

    def test_benchmark_delegates_to_backtest(self, monkeypatch):
        seen = {}
        monkeypatch.setattr("ghic.backtest.main", lambda argv: seen.update(argv=argv) or 0)
        assert cli.main(["benchmark", "--calibrate"]) == 0
        assert seen["argv"] == ["--calibrate"]

    def test_collect_and_label_delegate(self, monkeypatch):
        calls = []
        monkeypatch.setattr("ghic.collect.main", lambda argv: calls.append(("collect", argv)) or 0)
        monkeypatch.setattr("ghic.label.main", lambda argv: calls.append(("label", argv)) or 0)
        cli.main(["collect", "--measure-cost"])
        cli.main(["label", "--audit-only"])
        assert calls == [("collect", ["--measure-cost"]), ("label", ["--audit-only"])]

    def test_serve_delegates(self, monkeypatch):
        monkeypatch.setattr("ghic.service.app.main", lambda: 0)
        assert cli.main(["serve"]) == 0

    def test_unknown_command_errors(self):
        with pytest.raises(SystemExit):
            cli.main(["frobnicate"])


class TestDashboard:
    def test_dashboard_opens_service_url(self, monkeypatch, capsys):
        opened = []
        monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
        monkeypatch.setenv("GHIC_PORT", "9999")
        monkeypatch.delenv("PORT", raising=False)
        assert cli.main(["dashboard"]) == 0
        assert opened == ["http://127.0.0.1:9999/dashboard"]
        assert "9999/dashboard" in capsys.readouterr().out


@pytest.mark.skipif(not MODEL_PRESENT, reason="no trained model artifact present")
class TestPredictEndToEnd:
    def test_predict_prints_prediction_json(self, capsys):
        rc = cli.main(["predict", "--title", "Crash on startup",
                       "--body", "Steps to reproduce: 1. open 2. crash"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert 0.0 <= out["proba_actionable_bug"] <= 1.0
        assert out["predicted_class"] in ("actionable-bug", "non-actionable")
        assert "top_features" not in out

    def test_explain_includes_feature_contributions(self, capsys):
        rc = cli.main(["explain", "--title", "Crash on startup",
                       "--body", "Steps to reproduce: 1. open 2. crash"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert "top_features" in out

    def test_threshold_changes_decision(self, capsys):
        cli.main(["predict", "--title", "Crash", "--body", "crash", "--threshold", "0.0"])
        low = json.loads(capsys.readouterr().out)
        assert low["predicted_class"] == "actionable-bug"   # any proba >= 0.0
