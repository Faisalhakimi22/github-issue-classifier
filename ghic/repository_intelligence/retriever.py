"""Query construction and semantic retrieval.

Two things happen here, and the first matters more than the second.

**Query construction.** An issue body is not a search query. It contains a
greeting, a version table, a 60-line stack trace, and three paragraphs of
narrative, and embedding all of it dilutes the handful of tokens that
actually identify the relevant code. `build_query` keeps the title
(weighted, because issue titles are dense with the right nouns), the
identifier-like tokens anywhere in the body (`parse_csv`, `UnicodeDecodeError`,
`src/import/reader.py` -- the things that literally name the code), and the
first prose paragraph, then drops the rest.

**Retrieval.** A dot product against the index, a similarity floor, and
per-file diversification. The floor is the honesty mechanism: below
`min_similarity` the engine returns nothing rather than the least-bad match,
because "no directly related source files were confidently identified" is a
true and useful answer, and a wrong file path in a maintainer-facing comment
is worse than no file path at all.
"""
from __future__ import annotations

import re

from .. import utils
from .config import RepositoryIntelligenceConfig
from .embeddings import EmbeddingProvider
from .indexer import VectorStore
from .models import SOURCE_CODE, RetrievedChunk

logger = utils.get_logger(__name__)

# Tokens that look like code rather than prose: dotted paths, snake_case,
# camelCase, file paths, ALL_CAPS constants, and `backticked` spans.
_IDENTIFIER_RE = re.compile(
    r"`[^`\n]{2,60}`"                              # `inline code`
    r"|\b[\w./-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|cs|md|json|ya?ml|toml|cfg|ini)\b"
    r"|\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b"   # a.b.c
    r"|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"                          # snake_case
    r"|\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"                              # camelCase
    r"|\b[A-Z][A-Za-z0-9]*(?:Error|Exception|Warning)\b"           # ValueError
    r"|\b[A-Z][A-Z0-9_]{3,}\b"                                     # MAX_RETRIES
)

_MAX_QUERY_CHARS = 2_000
_TITLE_REPEATS = 3       # cheap term weighting: repeat, don't scale the vector


def build_query(
    title: str, body: str, *, category: str = "", predicted_label: int | None = None
) -> str:
    """Issue fields -> a retrieval query.

    `category` (from the LLM or the category head) is appended as a plain
    hint word; `predicted_label` is accepted for interface completeness and
    deliberately unused in the text -- an actionability probability says
    nothing about *which file* is relevant, and injecting "actionable bug"
    into every query would pull every result toward whatever code happens to
    mention bugs.
    """
    title = (title or "").strip()
    body = (body or "").strip()

    parts: list[str] = [title] * _TITLE_REPEATS if title else []

    identifiers = _IDENTIFIER_RE.findall(body)
    if identifiers:
        cleaned = [token.strip("`") for token in identifiers]
        parts.append(" ".join(dict.fromkeys(cleaned)))  # dedupe, keep order

    prose = _first_prose_paragraph(body)
    if prose:
        parts.append(prose)

    if category:
        parts.append(category)

    return " ".join(p for p in parts if p)[:_MAX_QUERY_CHARS]


def _first_prose_paragraph(body: str) -> str:
    """The first paragraph that isn't a code fence, a table, or boilerplate.

    Issue templates open with checklists and headings; the first real
    sentence is usually the one describing the actual problem.
    """
    in_fence = False
    collected: list[str] = []
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not stripped:
            if collected:
                break
            continue
        if stripped.startswith(("#", ">", "|", "-", "*", "[", "!")) and not collected:
            continue
        collected.append(stripped)
        if sum(len(c) for c in collected) > 400:
            break
    return " ".join(collected)[:400]


class SemanticRetriever:
    """Search one repository's index. Holds no state about which repo."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        cfg: RepositoryIntelligenceConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.cfg = cfg or RepositoryIntelligenceConfig()

    def retrieve(
        self,
        store: VectorStore,
        title: str,
        body: str,
        *,
        category: str = "",
        predicted_label: int | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Top-k chunks above the similarity floor, diversified across files.

        Returns [] rather than low-confidence guesses -- see the module
        docstring. Never raises: a retrieval failure degrades to "no
        evidence", which the comment states plainly.
        """
        top_k = top_k or self.cfg.top_k
        query = build_query(title, body, category=category, predicted_label=predicted_label)
        if not query.strip() or not store.size:
            return []

        try:
            vector = self.embedder.embed_query(query)
            # Over-fetch so diversification has something to choose from
            # after the floor removes weak matches.
            candidates = store.search(vector, top_k * 3)
        except Exception as e:  # embedding API down, malformed index, ...
            logger.warning("repository retrieval failed: %s", e)
            return []

        # The floor gates on the *true* similarity (is this a confident
        # match at all?); ordering then applies ranking priors (given two
        # confident matches, which one does a maintainer want first?).
        # Keeping those separate is why `RetrievedChunk.score` can stay a
        # plain cosine similarity while the order isn't strictly by it.
        above_floor = [rc for rc in candidates if rc.score >= self.cfg.min_similarity]
        above_floor.sort(key=lambda rc: -_ranking_score(rc))

        # Code and history answer different questions, so they get separate
        # budgets rather than competing for the same slots. Without this,
        # a repository with thousands of indexed issues drowns its own code
        # -- issue text matches issue text far more strongly than code
        # does, which is the docs-vs-code problem from Phase 1 in a
        # sharper form.
        history = [rc for rc in above_floor if rc.chunk.source != SOURCE_CODE]
        code = [rc for rc in above_floor if rc.chunk.source == SOURCE_CODE]
        if not history:
            return _diversify(code, top_k)

        history_slots = min(self.cfg.history_result_slots, top_k)
        selected_history = _diversify(history, history_slots)
        selected_code = _diversify(code, top_k - len(selected_history))
        return selected_code + selected_history


_PROSE_LANGUAGES = frozenset({"Markdown", "reStructuredText"})

_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|specs?|__tests__|e2e|fixtures)(/|$)"
    r"|(^|/)(test_|spec_)"
    r"|(_test|_spec|\.test|\.spec)\.[A-Za-z]+$"
)

# Multiplicative, not a hard filter. A test file is legitimate evidence --
# an existing failing test is often exactly what a maintainer wants to see
# -- but it should not outrank the implementation it exercises. Test names
# restate the feature in the same natural language the issue uses
# ("test_webhook_rejects_bad_signature"), so they match issue text more
# strongly than the terse implementation does; without this prior the
# evidence section reliably shows tests and hides the code to fix. A factor
# rather than a filter means a strongly-matching test still beats a weakly-
# matching implementation, which is the correct outcome when the issue is
# genuinely about the test suite.
_TEST_RANK_FACTOR = 0.75


def _ranking_score(rc: RetrievedChunk) -> float:
    score = rc.score
    if rc.chunk.source != SOURCE_CODE:
        # History decays gently with age: a resolution from last month is
        # more likely to still apply than one from four years ago. A
        # prior, not a filter -- an old issue describing exactly this bug
        # is still the right answer.
        from .sources import recency_weight

        return score * recency_weight(rc.chunk.timestamp)
    if _TEST_PATH_RE.search(rc.chunk.path.lower()):
        return score * _TEST_RANK_FACTOR
    return score


def _diversify(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Cap how much of the result set any one file -- or documentation as a
    whole -- can occupy.

    Two separate caps, both learned from watching real output:

    *Per file (2 chunks).* A 900-line module chunked into 30 pieces
    otherwise sweeps every slot and the evidence section shows one file
    thirty times. Two preserves "here are the two relevant functions in
    this file" while leaving room for the other files a fix would touch.

    *Prose share (a third of the slots).* Markdown competes unfairly with
    code under any lexical or embedding similarity: docs are written in the
    same natural language as the issue, while code is token-sparse, so an
    unconstrained ranking hands every slot to the README and the changelog.
    The first smoke test of this engine answered "webhook signature
    verification fails" with four documentation files and no source at all.
    Docs stay eligible -- they genuinely answer "how is this configured" --
    but they can't crowd out the code the maintainer actually has to edit.

    Both caps are hard: when they leave fewer than `top_k` results, the
    short list is returned as-is. An earlier version backfilled from the
    rejected chunks to always reach `top_k`, which silently defeated both
    caps -- a two-file repository still returned ten chunks from one file.
    Returning six good results is the honest answer to "the ten most
    relevant chunks"; padding with the very chunks just judged redundant is
    not.
    """
    prose_budget = max(1, top_k // 3)
    per_file: dict[str, int] = {}
    selected: list[RetrievedChunk] = []
    prose_used = 0

    for rc in chunks:
        if len(selected) >= top_k:
            break
        is_prose = rc.chunk.language in _PROSE_LANGUAGES
        if per_file.get(rc.chunk.path, 0) >= 2 or (is_prose and prose_used >= prose_budget):
            continue
        per_file[rc.chunk.path] = per_file.get(rc.chunk.path, 0) + 1
        prose_used += is_prose
        selected.append(rc)

    return selected
