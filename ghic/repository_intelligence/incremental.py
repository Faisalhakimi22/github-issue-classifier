"""Incremental re-indexing: only touch what changed.

A full re-index of a large repository costs a clone walk, thousands of
chunk operations, and (with a paid embedding provider) real money -- for a
push that changed two files. `git diff --name-status OLD NEW` gives the
exact change set, so a re-index becomes: delete the chunks for changed and
deleted paths, re-chunk and re-embed only the changed ones.

Three things make this safe rather than clever:

**It verifies the old commit is still reachable.** A shallow clone
(`--depth 1`) usually does *not* contain the previously indexed commit, so
the diff would fail or, worse, silently produce a wrong change set. When
the old SHA isn't in the local object database, this returns None and the
caller does a full rebuild. Correctness over cleverness: a wrong
incremental update leaves an index permanently describing code that no
longer exists, and nothing would ever detect it.

**It bounds itself.** Past a threshold of changed files a full rebuild is
both simpler and faster than thousands of individual deletes, so the
change set is checked against `max_incremental_files` before being used.

**Deletions are handled explicitly.** A removed file's chunks must go, or
the evidence section will keep citing a path that no longer exists -- the
exact hallucination-shaped failure this whole subsystem is built to avoid,
arriving through the back door of a stale index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .. import utils
from .cache import RepositoryCacheError, run_git

logger = utils.get_logger(__name__)

# Git status letters that mean "this file's content is now different".
_CHANGED_STATUSES = frozenset({"A", "M", "C", "R", "T"})
_DELETED_STATUSES = frozenset({"D"})


@dataclass(frozen=True)
class ChangeSet:
    """Files whose chunks must be rewritten (`changed`) or dropped
    (`deleted`). A rename appears as both: the old path is deleted and the
    new path is changed."""
    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.changed and not self.deleted

    @property
    def total(self) -> int:
        return len(self.changed) + len(self.deleted)

    @property
    def stale_paths(self) -> list[str]:
        """Every path whose existing chunks are now wrong -- both the
        changed (which get re-added) and the deleted (which don't)."""
        return list(dict.fromkeys([*self.changed, *self.deleted]))


def commit_exists(root: Path, sha: str, timeout: float = 30.0) -> bool:
    """Whether `sha` is present in the local object database."""
    if not sha:
        return False
    try:
        run_git(["cat-file", "-e", f"{sha}^{{commit}}"], root, timeout)
        return True
    except RepositoryCacheError:
        return False


def diff_change_set(
    root: Path, old_sha: str, new_sha: str, timeout: float = 60.0
) -> ChangeSet | None:
    """Files changed between two commits, or None if a diff isn't possible.

    None means "fall back to a full rebuild" -- returned when either commit
    is missing locally (the shallow-clone case) or git fails for any other
    reason. It is never an error the caller has to handle specially; a full
    rebuild is always a correct answer.
    """
    if not old_sha or not new_sha:
        return None
    if old_sha == new_sha:
        return ChangeSet()
    if not commit_exists(root, old_sha, timeout) or not commit_exists(root, new_sha, timeout):
        logger.info(
            "cannot diff %s..%s in %s (commit not present in a shallow clone); "
            "falling back to a full re-index", old_sha[:8], new_sha[:8], root.name,
        )
        return None

    try:
        output = run_git(
            ["diff", "--name-status", "--no-renames", old_sha, new_sha], root, timeout
        )
    except RepositoryCacheError as e:
        logger.info("git diff failed (%s); falling back to a full re-index", e)
        return None

    changed: list[str] = []
    deleted: list[str] = []
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0][:1].upper(), parts[-1]
        if status in _DELETED_STATUSES:
            deleted.append(path)
        elif status in _CHANGED_STATUSES:
            changed.append(path)
    return ChangeSet(changed=changed, deleted=deleted)


def should_use_incremental(change_set: ChangeSet | None, max_files: int) -> bool:
    """Whether an incremental update is worth it.

    An empty change set is *not* incremental-worthy: nothing to do at all
    is a separate, better outcome the caller short-circuits on. Above
    `max_files`, per-path deletes stop being cheaper than one bulk rebuild.
    """
    if change_set is None or change_set.is_empty:
        return False
    if change_set.total > max_files:
        logger.info(
            "%d changed files exceeds the incremental limit of %d; rebuilding in full",
            change_set.total, max_files,
        )
        return False
    return True
