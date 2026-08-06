# Card — Repository Intelligence Engine (RAG over the issue's own repository)

**Task:** retrieve the code an issue is actually about, feed it to the LLM
as evidence, and show the maintainer which files it came from — so the
analysis reasons from the repository, not just from the issue text.

Pipeline position:

```
issue → ML classification → repository retrieval → LLM reasoning → comment
                            ^^^^^^^^^^^^^^^^^^^^
```

The ML actionability decision is untouched. This layer runs after it and
feeds the same LLM layer that already existed (`ghic/llm/`).

## The one rule: evidence or silence

Every repository claim must come from a retrieved chunk. That is enforced
structurally, in two places, rather than by asking the model to behave:

1. **The prompt only ever contains retrieved chunks.**
   `prompts.build_repository_section()` renders the actual
   `RetrievedChunk` objects and nothing else — no file listing, no
   directory tree, no "here's what the repo looks like" summary the model
   could extrapolate from. It also states its own limits inline ("this is
   a partial view", "absence here doesn't mean absence in the repo"), so
   the model is told both not to invent files *and* not to claim a file is
   missing.
2. **The comment's evidence section is rendered in Python, not by the
   model.** `inference._repository_evidence_lines()` builds the
   "Relevant files" list directly from `RetrievedChunk` metadata. A path
   printed in a GHIC comment is a path that exists in the index at the
   indexed commit — by construction, not by the model's good behaviour.
   The LLM's prose can be wrong about many things; it cannot fabricate the
   file list, because it never writes it.

When retrieval finds nothing above the confidence floor, every layer says
so plainly: `RepositoryContext.is_empty`, the prompt's explicit "no
repository code was retrieved — do not speculate", and the comment's
"No directly related source files were confidently identified."

## What it does *not* claim

This is retrieval, not comprehension. The engine finds files whose text is
similar to the issue's; it does not know whether they are the *cause*. The
comment says exactly that ("a starting point for investigation, not a
diagnosis") and the similarity score is never presented as a probability
that a file is at fault — it isn't one, and there is no labelled
issue→file corpus here to calibrate one against. Same discipline as
PRIORITY_CARD.md and DUPLICATE_CARD.md: assistive, clearly bounded, never
dressed up as a validated prediction.

## Design decisions worth defending

**Chunking is structural, never fixed-size.** Python via the stdlib `ast`
module (exact); Markdown by heading, tracking fenced code blocks so a
`# comment` inside a bash fence isn't read as a heading; JS/TS/Go/Rust/
Java/C# via a declaration-regex + brace-depth scanner that skips string and
comment content. Fixed windows cut mid-function, producing chunks that
retrieve plausibly and read as nonsense — the failure mode is invisible in
metrics and obvious in output. The C-family scanner is explicitly a
heuristic, not a parser: tree-sitter would be better and is a compiled
per-grammar dependency this project's Vercel bundle (already ~295MB of a
500MB cap) cannot absorb. When it finds no declarations it degrades to
whole-file chunks, so a language it reads poorly is *less useful*, never
*wrong*.

**Short named declarations survive the noise filter.** `min_chunk_chars`
drops trivia, but applying it uniformly also deleted three-line methods
and one-sentence `## Installation` sections — chunks whose value is mostly
in their *name*, which is exactly what an issue reporter types. A chunk
with a symbol clears a much lower bar (`_is_substantive()`).

**The default embedder is lexical, and says so.** `HashingEmbeddingProvider`
is signed feature hashing over identifier-aware tokens (`parse_csv_file` →
`parse_csv_file`, `parse`, `csv`, `file`), L2-normalized, sublinear TF.
Deterministic, offline, no API key, no per-repo cost, no extra dependency.
It matches an issue naming `parse_csv` to the function `parse_csv`; it will
*not* match "the importer chokes on foreign characters" to `decode()` with
no shared vocabulary — a real embedding model would. That tradeoff is
deliberate: lexical retrieval that always works beats semantic retrieval
that needs a key most deployments won't set. Set
`GHIC_REPO_INTEL_EMBEDDING_PROVIDER=openai` for the semantic version;
nothing else changes, and the index auto-invalidates because the embedder
name is part of its cache key.

**Exact search, not approximate.** `NumpyVectorStore` is one dense
matrix-vector product. At repository scale (a few thousand to a few tens
of thousands of chunks — 20k × 512 float32 is ~40MB and low single-digit
milliseconds) FAISS's approximation buys nothing and costs recall; its
value starts at millions of vectors. `FaissVectorStore` exists behind the
same `VectorStore` interface and is selected automatically above 50k
chunks. Chroma/Pinecone/Qdrant/Milvus are one subclass each.

**Two ranking corrections, both from watching real output.** The first
smoke test of this engine answered "webhook signature verification fails"
with four documentation files and zero source. Docs match issue text
unfairly well — they're written in the same natural language, while code
is token-sparse. So: prose is capped at a third of the result slots, and
test files are demoted by a 0.75 factor (multiplicative, not a filter — a
strongly-matching test still beats a weakly-matching implementation, which
is right when the issue is genuinely about the test suite). Per-file cap of
2 stops one large module sweeping every slot. All caps are **hard**: an
earlier version backfilled to always return `top_k`, which silently
defeated every cap. Returning six good results is the honest answer to
"the ten most relevant chunks."

**The file cap degrades by relevance, not alphabetically.** A repository
over `max_files` is ranked by `_priority()` (source before docs, tests and
`_archive/` last, shallower paths first) before truncation. The first
version truncated a sorted walk, which on this very repository meant
indexing `_archive/` and a gitignored sibling project while never reaching
`ghic/`.

**Indexing reads git's file list, not the filesystem.** `git ls-files`
respects `.gitignore` for free. Found the hard way: the first smoke test
indexed a gitignored nested project and returned its files as "evidence"
for this repository. Falls back to a filesystem walk for a non-git
directory.

## Never blocks, never fails

`get_context()` only ever *reads* an existing index. It never clones and
never indexes — those take seconds to minutes against a ~10s webhook
budget. A repository with no index queues one through QStash (the same
queue the async processing feature already uses) and returns empty, so the
first issue on a new repository is analyzed from text alone and every issue
after it gets code evidence. That is the entire async story; there is no
path where a webhook waits on a clone.

Failure is absorbed at three levels: `RepositoryIntelligenceService.get_context()`
catches everything and returns an empty context; `_process_issue_job()`
wraps the call anyway (the contract is one implementation's promise, the
try/except is the webhook's guarantee); and the comment renders no
evidence section at all rather than an empty one. Git missing, clone
denied, index corrupt, embedding API down, repository empty — all produce
the same outcome: analysis continues exactly as it did before this module
existed.

## Caching and invalidation

| Cached | Keyed by | Invalidated when |
|---|---|---|
| Clone | repo full name | never (refreshed via `git fetch`, never re-cloned) |
| Vector index | (repo, commit SHA, embedder name) | the SHA changes, the embedder changes, or 7 days pass |
| "Latest index" pointer | repo full name | each successful index run |

The commit SHA in the key *is* the invalidation strategy: an unchanged
repo reuses its index, a changed one gets a new key and rebuilds. There is
no way to serve an index that doesn't match the code it describes. The
pointer file exists so the read path never has to run git to discover the
current SHA — retrieval is pure filesystem.

An installation token is injected into the clone URL for private repos and
is deliberately never persisted: `git remote set-url` rewrites the stored
remote to a tokenless URL immediately after cloning, and `_redact()` strips
credentials from any git error before it reaches a log.

## Deployment reality

This needs a **writable, persistent** cache directory to be useful. On
Vercel's ephemeral filesystem an index does not survive between
invocations, so the engine degrades to "never indexed" on every request —
correct behaviour, but pointless. Docker/Fly with a volume is the intended
home. `GHIC_USE_REPO_INTELLIGENCE` is off by default everywhere.

Deploys without QStash never auto-index; they use the CLI:

```bash
python -m ghic.repo_index --repo owner/name            # clone + index
python -m ghic.repo_index --repo owner/name --path .   # index a local tree
python -m ghic.repo_index --repo owner/name --query "csv import crashes"
python -m ghic.repo_index --list
```

The `--query` form is worth running before enabling the feature on a real
repository: retrieval quality varies a lot by codebase, and it shows
exactly what the maintainer-facing evidence section would contain.

## Extending it (later phases)

`VectorStore`, `EmbeddingProvider`, and `RepositoryIndexer` are the three
seams. Commit Intelligence, PR Intelligence, and Historical Issue Search
all need "chunk something, embed it, search it" — `RepositoryIndexer` is
deliberately handed a directory rather than doing its own cloning, so a
commit indexer that gets its content from the GitHub API reuses the same
chunker, embedder, and store without inheriting the webhook's assumptions.
`RepositoryContext` is the only type the LLM layer knows about, so a future
`CommitContext` slots in beside it without touching prompt or comment code.
