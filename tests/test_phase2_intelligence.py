"""Phase 2: Commit, Pull Request, and Historical Issue Intelligence.

Exercised against a real git repository built in tmp_path -- commit
parsing is exactly the kind of string handling that a mocked `git log`
would validate against the author's assumptions rather than against git.
(One bug found that way: `--name-only` appends each commit's file list
after the record separator, so with a trailing separator a commit's paths
land in the *next* record's SHA field.)
"""
from __future__ import annotations

import subprocess
import time

import pytest

from ghic.repository_intelligence import (
    RepositoryIntelligenceConfig,
    RepositoryIntelligenceService,
)
from ghic.repository_intelligence.models import (
    SOURCE_CODE,
    SOURCE_COMMIT,
    SOURCE_ISSUE,
    SOURCE_PULL_REQUEST,
    CodeChunk,
)
from ghic.repository_intelligence.sources import (
    CommitSource,
    IssueHistorySource,
    is_noise_commit,
    recency_weight,
    summarize_sources,
)

NOW = time.time()
DAY = 86400.0


@pytest.fixture
def git_repo(tmp_path):
    """A small repository with real history."""
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)

    def git(*args):
        subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)

    git("init", "-q", ".")
    git("config", "user.email", "dev@example.com")
    git("config", "user.name", "Dev")

    (root / "src" / "csv_import.py").write_text(
        "def import_csv(path):\n    return open(path).read()\n", encoding="utf-8"
    )
    git("add", "-A")
    git("commit", "-q", "-m", "Add CSV importer")

    (root / "src" / "csv_import.py").write_text(
        "def import_csv(path, encoding='utf-8'):\n"
        "    return open(path, encoding=encoding).read()\n",
        encoding="utf-8",
    )
    git("add", "-A")
    git("commit", "-q", "-m", "Fix UnicodeDecodeError when importing non-UTF-8 CSV files")

    git("commit", "-q", "--allow-empty", "-m", "Merge branch 'main' into dev")
    git("commit", "-q", "--allow-empty", "-m", "Bump version to 2.1.0")
    return root


@pytest.fixture
def issue_items():
    return [
        {
            "number": 412, "title": "CSV import crashes on latin-1 files",
            "state": "closed", "state_reason": "completed",
            "closed_at": "2026-05-01T00:00:00Z",
            "body": "Importing a latin-1 CSV raises UnicodeDecodeError.",
            "resolution": "Added an encoding parameter to import_csv.",
            "html_url": "https://github.com/acme/proj/issues/412",
            "labels": [{"name": "bug"}, {"name": "csv"}],
        },
        {"number": 500, "title": "Add dark mode", "state": "open"},
        {
            "number": 511, "title": "Detect CSV encoding before decoding",
            "state": "closed", "pull_request": {"url": "x"},
            "closed_at": "2026-06-01T00:00:00Z", "body": "Sniff the encoding first.",
        },
        {
            "number": 600, "title": "CSV encoding request", "state": "closed",
            "state_reason": "not_planned", "closed_at": "2026-01-01T00:00:00Z",
            "body": "We declined this CSV encoding request.",
        },
    ]


# ---------------------------------------------------------------------------
# Commit Intelligence
# ---------------------------------------------------------------------------
class TestCommitSource:
    def test_indexes_commits_with_real_shas(self, git_repo):
        chunks = CommitSource().fetch("acme/proj", root=git_repo)
        assert chunks
        for chunk in chunks:
            assert chunk.source == SOURCE_COMMIT
            assert chunk.kind == "commit"
            # The regression this guards: a leaked file path parsed as a SHA.
            assert len(chunk.reference) == 8
            assert all(c in "0123456789abcdef" for c in chunk.reference), chunk.reference
            assert chunk.path.startswith("commit:")

    def test_subject_becomes_the_symbol(self, git_repo):
        subjects = {c.symbol for c in CommitSource().fetch("acme/proj", root=git_repo)}
        assert "Fix UnicodeDecodeError when importing non-UTF-8 CSV files" in subjects

    def test_touched_paths_are_included(self, git_repo):
        chunks = CommitSource().fetch("acme/proj", root=git_repo)
        blob = " ".join(c.text for c in chunks)
        assert "src/csv_import.py" in blob

    def test_merge_and_release_commits_are_skipped(self, git_repo):
        subjects = {c.symbol for c in CommitSource().fetch("acme/proj", root=git_repo)}
        assert not any(s.lower().startswith("merge branch") for s in subjects)
        assert not any(s.lower().startswith("bump version") for s in subjects)

    def test_timestamps_are_parsed(self, git_repo):
        assert all(c.timestamp > 0 for c in CommitSource().fetch("acme/proj", root=git_repo))

    def test_max_commits_is_respected(self, git_repo):
        chunks = CommitSource().fetch("acme/proj", root=git_repo, max_commits=1)
        assert len(chunks) <= 1

    def test_non_git_directory_returns_nothing(self, tmp_path):
        assert CommitSource().fetch("acme/proj", root=tmp_path) == []

    def test_root_is_required(self):
        with pytest.raises(ValueError):
            CommitSource().fetch("acme/proj")

    @pytest.mark.parametrize("subject,noise", [
        ("Merge branch 'main'", True),
        ("Merge pull request #12 from x/y", True),
        ("Bump version to 1.2.3", True),
        ("chore(release): 2.0.0", True),
        ("", True),
        ("Fix UnicodeDecodeError in the CSV importer", False),
        ("Add retry logic to the webhook handler", False),
    ])
    def test_noise_classification(self, subject, noise):
        assert is_noise_commit(subject) is noise


# ---------------------------------------------------------------------------
# Issue / PR Intelligence
# ---------------------------------------------------------------------------
class TestIssueHistorySource:
    def test_closed_issues_and_prs_are_indexed(self, issue_items):
        chunks = IssueHistorySource().fetch("acme/proj", items=issue_items)
        refs = {c.reference for c in chunks}
        assert "#412" in refs and "#511" in refs

    def test_open_issues_are_excluded(self, issue_items):
        """Open issues are the current backlog, not history -- the existing
        duplicate detector already covers them."""
        chunks = IssueHistorySource().fetch("acme/proj", items=issue_items)
        assert "#500" not in {c.reference for c in chunks}

    def test_pull_requests_get_their_own_source(self, issue_items):
        chunks = IssueHistorySource().fetch("acme/proj", items=issue_items)
        by_ref = {c.reference: c for c in chunks}
        assert by_ref["#511"].source == SOURCE_PULL_REQUEST
        assert by_ref["#412"].source == SOURCE_ISSUE

    def test_resolved_issue_is_marked_resolved(self, issue_items):
        chunks = IssueHistorySource().fetch("acme/proj", items=issue_items)
        text = {c.reference: c.text for c in chunks}["#412"]
        assert "closed as completed (resolved)" in text
        assert "Added an encoding parameter" in text

    def test_declined_issue_is_never_presented_as_a_resolution(self, issue_items):
        """A closed-as-not-planned issue records that nobody fixed it --
        useful context, but it must not read as a fix."""
        chunks = IssueHistorySource().fetch("acme/proj", items=issue_items)
        text = {c.reference: c.text for c in chunks}["#600"]
        assert "not a resolution" in text
        assert "resolved" not in text.replace("not a resolution", "")

    def test_labels_are_included(self, issue_items):
        chunks = IssueHistorySource().fetch("acme/proj", items=issue_items)
        assert "bug" in {c.reference: c.text for c in chunks}["#412"]

    def test_url_is_carried_for_linking(self, issue_items):
        chunks = IssueHistorySource().fetch("acme/proj", items=issue_items)
        assert {c.reference: c.url for c in chunks}["#412"].endswith("/issues/412")

    def test_items_without_a_title_or_number_are_skipped(self):
        chunks = IssueHistorySource().fetch("acme/proj", items=[
            {"number": 1, "state": "closed"}, {"title": "no number", "state": "closed"},
        ])
        assert chunks == []

    def test_empty_input_is_fine(self):
        assert IssueHistorySource().fetch("acme/proj", items=[]) == []

    def test_max_items_is_respected(self, issue_items):
        assert len(IssueHistorySource().fetch(
            "acme/proj", items=issue_items, max_items=1
        )) <= 1

    def test_from_github_swallows_client_failure(self):
        class Broken:
            def list_recent_closed_issues(self, *a, **k):
                raise RuntimeError("API down")

        assert IssueHistorySource.from_github(Broken(), "acme/proj", 1) == []


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
class TestRecencyWeight:
    def test_recent_beats_old(self):
        assert recency_weight(NOW - 30 * DAY) > recency_weight(NOW - 1000 * DAY)

    def test_unknown_timestamp_is_not_penalized(self):
        assert recency_weight(0) == 1.0

    def test_half_life_is_a_year(self):
        assert recency_weight(NOW - 365 * DAY) == pytest.approx(0.5, abs=0.02)

    def test_old_history_is_dampened_not_eliminated(self):
        """A five-year-old issue describing exactly this bug is still the
        right answer -- a prior, not a filter."""
        assert recency_weight(NOW - 5 * 365 * DAY) > 0.0


class TestSourceSummary:
    def test_counts_per_source(self):
        chunks = [
            CodeChunk(repo="a/b", path="x.py", language="Python", text="t",
                      start_line=1, end_line=1, source=SOURCE_CODE),
            CodeChunk(repo="a/b", path="commit:1", language="", text="t",
                      start_line=0, end_line=0, source=SOURCE_COMMIT),
            CodeChunk(repo="a/b", path="commit:2", language="", text="t",
                      start_line=0, end_line=0, source=SOURCE_COMMIT),
        ]
        assert summarize_sources(chunks) == {SOURCE_CODE: 1, SOURCE_COMMIT: 2}


# ---------------------------------------------------------------------------
# Backward compatibility of the extended chunk
# ---------------------------------------------------------------------------
class TestChunkCompatibility:
    def test_phase_one_chunks_default_to_code(self):
        chunk = CodeChunk(repo="a/b", path="x.py", language="Python", text="t",
                          start_line=1, end_line=2)
        assert chunk.source == SOURCE_CODE

    def test_a_phase_one_index_row_still_loads(self):
        """An index written before Phase 2 has no source/reference fields."""
        legacy = {
            "repo": "a/b", "path": "x.py", "language": "Python", "text": "t",
            "start_line": 1, "end_line": 2, "kind": "function", "symbol": "f",
            "parent_symbol": "",
        }
        restored = CodeChunk.from_dict(legacy)
        assert restored.source == SOURCE_CODE
        assert restored.reference == ""

    def test_round_trip_preserves_phase_two_fields(self):
        chunk = CodeChunk(
            repo="a/b", path="commit:abc", language="", text="t", start_line=0,
            end_line=0, kind="commit", symbol="Fix things", source=SOURCE_COMMIT,
            reference="abc12345", url="https://example.com", timestamp=NOW,
        )
        assert CodeChunk.from_dict(chunk.as_dict()) == chunk


# ---------------------------------------------------------------------------
# End-to-end through the service
# ---------------------------------------------------------------------------
class TestPhase2Integration:
    @pytest.fixture
    def service(self, tmp_path, issue_items):
        cfg = RepositoryIntelligenceConfig(
            cache_dir=tmp_path / "cache", index_commits=True, index_issue_history=True,
        )
        return RepositoryIntelligenceService(cfg, issue_history_items=issue_items)

    def test_retrieval_surfaces_the_fixing_commit(self, service, git_repo):
        service.index_local_path("acme/proj", git_repo)
        context = service.get_context(
            "acme/proj", "CSV import fails with UnicodeDecodeError",
            "Importing a latin-1 file crashes.",
        )
        commits = [rc for rc in context.history_chunks if rc.chunk.kind == "commit"]
        assert commits
        assert "UnicodeDecodeError" in commits[0].chunk.symbol

    def test_code_and_history_are_separated(self, service, git_repo):
        service.index_local_path("acme/proj", git_repo)
        context = service.get_context("acme/proj", "CSV import encoding", "")
        assert all(rc.chunk.source == SOURCE_CODE for rc in context.code_chunks)
        assert all(rc.chunk.source != SOURCE_CODE for rc in context.history_chunks)

    def test_relevant_files_never_includes_synthetic_paths(self, service, git_repo):
        """`commit:abc1234` is not a file and must never render as one."""
        service.index_local_path("acme/proj", git_repo)
        context = service.get_context("acme/proj", "CSV import encoding", "")
        assert all(not p.startswith(("commit:", "issue:", "pr:"))
                   for p in context.relevant_files)

    def test_history_slots_are_bounded(self, tmp_path, git_repo, issue_items):
        cfg = RepositoryIntelligenceConfig(
            cache_dir=tmp_path / "c", index_commits=True, index_issue_history=True,
            history_result_slots=1, min_similarity=0.0,
        )
        service = RepositoryIntelligenceService(cfg, issue_history_items=issue_items)
        service.index_local_path("acme/proj", git_repo)
        context = service.get_context("acme/proj", "CSV encoding", "")
        assert len(context.history_chunks) <= 1

    def test_history_does_not_crowd_out_code(self, tmp_path, git_repo):
        """Issue text matches issue text far more strongly than code does;
        without separate budgets, history sweeps every slot."""
        many = [
            {"number": n, "title": f"CSV import encoding problem {n}", "state": "closed",
             "state_reason": "completed", "closed_at": "2026-05-01T00:00:00Z",
             "body": "CSV import encoding UnicodeDecodeError latin-1 problem."}
            for n in range(50)
        ]
        cfg = RepositoryIntelligenceConfig(
            cache_dir=tmp_path / "c", index_commits=False, index_issue_history=True,
            min_similarity=0.0,
        )
        service = RepositoryIntelligenceService(cfg, issue_history_items=many)
        service.index_local_path("acme/proj", git_repo)
        context = service.get_context("acme/proj", "csv import encoding", "import_csv")
        assert context.code_chunks, "code was entirely crowded out by history"

    def test_disabled_flags_index_no_history(self, tmp_path, git_repo, issue_items):
        cfg = RepositoryIntelligenceConfig(
            cache_dir=tmp_path / "c", index_commits=False, index_issue_history=False,
        )
        service = RepositoryIntelligenceService(cfg, issue_history_items=issue_items)
        service.index_local_path("acme/proj", git_repo)
        context = service.get_context("acme/proj", "CSV encoding", "")
        assert context.history_chunks == []

    def test_history_failure_does_not_break_the_code_index(self, tmp_path, git_repo):
        class Exploding(list):
            def __iter__(self):
                raise RuntimeError("history source exploded")

        cfg = RepositoryIntelligenceConfig(
            cache_dir=tmp_path / "c", index_commits=False, index_issue_history=True,
        )
        service = RepositoryIntelligenceService(cfg, issue_history_items=Exploding([{}]))
        assert service.index_local_path("acme/proj", git_repo) is not None
        assert not service.get_context("acme/proj", "csv import", "").is_empty


# ---------------------------------------------------------------------------
# Rendering: prompt + comment
# ---------------------------------------------------------------------------
class TestPhase2Rendering:
    @pytest.fixture
    def context(self, tmp_path, git_repo, issue_items):
        cfg = RepositoryIntelligenceConfig(
            cache_dir=tmp_path / "c", index_commits=True, index_issue_history=True,
        )
        service = RepositoryIntelligenceService(cfg, issue_history_items=issue_items)
        service.index_local_path("acme/proj", git_repo)
        return service.get_context(
            "acme/proj", "CSV import fails with UnicodeDecodeError", "latin-1 crash"
        )

    def test_prompt_frames_history_as_possibilities(self, context):
        from ghic.llm.prompts import build_repository_section

        section = build_repository_section(context)
        assert "Related project history" in section
        assert "not confirmed matches" in section
        assert "never cite" in section

    def test_comment_lists_related_history(self, context):
        from ghic.service.inference import Prediction, format_llm_comment
        from test_service import make_llm_analysis

        pred = Prediction(repo="acme/proj", issue_number=1, proba=0.9, threshold=0.5,
                          predicted_label=1, model_name="stub")
        comment = format_llm_comment(pred, make_llm_analysis(), repository_context=context)
        assert "**Related history**" in comment
        assert "possible leads, not confirmed matches" in comment

    def test_comment_history_references_come_from_retrieved_chunks(self, context):
        """The anti-hallucination guarantee extended to Phase 2: every
        rendered reference is one the retriever returned."""
        from ghic.service.inference import Prediction, format_llm_comment
        from test_service import make_llm_analysis

        pred = Prediction(repo="acme/proj", issue_number=1, proba=0.9, threshold=0.5,
                          predicted_label=1, model_name="stub")
        comment = format_llm_comment(pred, make_llm_analysis(), repository_context=context)
        retrieved = {rc.chunk.reference for rc in context.history_chunks}
        for line in comment.split("\n"):
            if line.startswith("- commit ") or line.startswith("- issue ") \
                    or line.startswith("- PR "):
                assert any(ref in line for ref in retrieved), line
