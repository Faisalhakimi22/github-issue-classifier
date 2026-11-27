"""`ghic preview`: production's comment for any public issue, never posted.

No network in these tests: GitHub is a stub, the clone copies a small local
project, and the indexing and retrieval are real (local vectors, in-memory
state).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ghic.llm.models import IssueAnalysis, IssueContext
from ghic.llm.service import LLMService
from ghic.preview import PreviewError, PublicGitHub, main, parse_issue_ref, run_preview
from ghic.service.inference import Prediction
from ghic.service.settings import ServiceSettings

ISSUE = {
    "number": 7, "title": "Saving a profile without education crashes",
    "body": "Clicking save throws TypeError in save_profile when education is empty.",
    "created_at": "2026-11-20T10:00:00Z", "user": {"login": "alice"}, "labels": [],
}


# -- references ------------------------------------------------------------------------
@pytest.mark.parametrize("ref", [
    "https://github.com/o/r/issues/7", "github.com/o/r/issues/7", "o/r#7",
    "https://github.com/o/r/issues/7#issuecomment-1",
])
def test_issue_references_parse(ref: str) -> None:
    assert parse_issue_ref(ref) == ("o/r", 7)


@pytest.mark.parametrize("ref", ["https://github.com/o/r/pull/7", "o/r", "o/r#x", ""])
def test_bad_references_are_explained(ref: str) -> None:
    with pytest.raises(PreviewError):
        parse_issue_ref(ref)


# -- the read-only GitHub client ----------------------------------------------------
class _Resp:
    def __init__(self, status: int, data: Any = None) -> None:
        self.status_code, self._data = status, data

    def json(self) -> Any:
        return self._data


class _GetOnlySession:
    """Has no post/patch/put/delete: any write attempt raises AttributeError."""

    def __init__(self, responses: list[_Resp]) -> None:
        self.responses = responses
        self.headers_seen: list[dict[str, str]] = []

    def get(self, url: str, headers: Any = None, params: Any = None, timeout: Any = None):
        self.headers_seen.append(dict(headers or {}))
        return self.responses.pop(0)


def test_the_preview_client_cannot_write() -> None:
    for name in ("post_comment", "add_labels", "add_issue_to_project", "find_analysis_comment"):
        assert not hasattr(PublicGitHub, name), name


@pytest.mark.parametrize("resp, message", [
    (_Resp(404), "public repositories only"),
    (_Resp(403), "GITHUB_TOKEN"),
    (_Resp(200, {**ISSUE, "pull_request": {}}), "pull request"),
])
def test_issue_errors_are_actionable(resp: _Resp, message: str) -> None:
    client = PublicGitHub(session=_GetOnlySession([resp]))
    with pytest.raises(PreviewError, match=message):
        client.issue("o/r", 7)


def test_an_expired_token_is_dropped_not_fatal() -> None:
    session = _GetOnlySession([_Resp(401), _Resp(200, ISSUE)])
    client = PublicGitHub(token="expired", session=session)
    assert client.issue("o/r", 7)["number"] == 7
    assert "Authorization" in session.headers_seen[0]
    assert "Authorization" not in session.headers_seen[1]


# -- the whole pipeline ----------------------------------------------------------------
class _GitHub:
    def issue(self, repo: str, number: int) -> dict[str, Any]:
        return ISSUE

    def get_user(self, login: str, installation_id: Any = None) -> dict[str, Any]:
        return {"created_at": "2020-01-01T00:00:00Z", "public_repos": 3, "followers": 1}

    def get_latest_release_date(self, full_name: str, installation_id: Any = None) -> None:
        return None

    def author_has_prior_issues(self, *a: Any, **k: Any) -> bool:
        return True


class _Predictor:
    model_name = "stub"

    @property
    def cfg(self) -> Any:
        from ghic.config import get_config

        return get_config(require_token=False)

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> Prediction:
        self.calls.append(kwargs)
        return Prediction(repo=kwargs["repo_full_name"], issue_number=kwargs["issue_number"],
                          proba=0.2, threshold=0.5, predicted_label=0, model_name="stub")


class _Provider:
    def __init__(self) -> None:
        self.contexts: list[IssueContext] = []

    def analyze(self, context: IssueContext) -> IssueAnalysis:
        self.contexts.append(context)
        return IssueAnalysis(
            category="bug", priority="medium", severity="medium", confidence=0.8,
            summary="Saving crashes; see https://evil.example/fix and ask @maintainer.",
            reasoning=["Stack trace names save_profile."], recommended_action="Guard the empty list.",
            risk_level="medium", risk_reasons=["Reproducible"],
        )


class _NoCacheLLM(LLMService):
    def _cache_get(self, key: str) -> None:  # type: ignore[override]
        return None

    def _cache_put(self, key: str, analysis: IssueAnalysis) -> None:
        return None


def _project(tmp_path: Path) -> Path:
    src = tmp_path / "project"
    (src / "app").mkdir(parents=True)
    (src / "app" / "profile.py").write_text(
        "def save_profile(profile):\n"
        "    # crashes when education is empty\n"
        "    return [e.title for e in profile.education]\n",
        encoding="utf-8",
    )
    (src / "README.md").write_text("# Demo\nA profile app.\n", encoding="utf-8")
    return src


def _copy_clone(source: Path):
    def clone(repo: str, dest: Path) -> str:
        shutil.copytree(source, dest)
        return "abc12345"
    return clone


def _settings(**overrides: Any) -> ServiceSettings:
    values: dict[str, Any] = {"model_path": Path("unused.joblib"), "webhook_secret": "x"}
    values.update(overrides)
    return ServiceSettings(**values)


def test_full_preview_matches_production_and_never_touches_the_database(tmp_path, monkeypatch):
    # If the preview reached for Postgres, this URL would make it fail loudly.
    monkeypatch.setenv("DATABASE_URL", "postgresql://must-not-be-used.invalid/db")
    monkeypatch.setenv("GHIC_DATABASE_URL", "postgresql://must-not-be-used.invalid/db")
    provider, predictor = _Provider(), _Predictor()

    result = run_preview(
        "o/r#7", github=_GitHub(), settings=_settings(), predictor=predictor,
        llm_service=_NoCacheLLM(provider), clone=_copy_clone(_project(tmp_path)),
    )

    assert "files indexed at abc12345" in result.evidence_note
    assert "app/profile.py" in result.relevant_files
    assert result.comment is not None and "### Repository Evidence" in result.comment
    # Production's rules: uncalibrated repo, so no score reached the model or the comment.
    assert provider.contexts[0].score_calibrated is False
    assert result.score_shown is False
    # The sanitizer ran: the planted link and mention never reach the comment.
    assert "evil.example" not in result.comment and "@maintainer" not in result.comment
    # The author history lookup reached the classifier, as in the webhook.
    assert predictor.calls[0]["author_has_prior_issues"] is True


def test_without_ai_or_evidence_the_preview_says_ghic_would_stay_silent():
    result = run_preview(
        "o/r#7", github=_GitHub(), settings=_settings(), predictor=_Predictor(),
        llm_service=None, evidence=False,
    )
    assert result.comment is None


def test_cli_explains_a_bad_reference(capsys):
    assert main(["not-an-issue"]) == 2
    assert "not an issue reference" in capsys.readouterr().err
