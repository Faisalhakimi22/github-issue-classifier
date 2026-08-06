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

## Production infrastructure (Phase 1.5)

Phase 1 stored everything on the local filesystem and inferred state from
it ("is there an index directory?"). That works on one box and fails
everywhere else. Phase 1.5 keeps every Phase 1 interface and adds the
infrastructure behind it — a service constructed the Phase 1 way still
works identically, which is enforced by
`test_phase_one_construction_still_works`.

### What's swappable, and what actually ships

| Seam | Interface | Ships | Not implemented |
|---|---|---|---|
| Vector store | `VectorStore` | `local` (numpy/FAISS), `postgres` (pgvector) | Qdrant, Pinecone, Milvus, Chroma |
| State store | `RepositoryStateStore` | Postgres, file, memory | — |
| Queue | `IndexQueue` | QStash, inline (dev), none | Redis, Celery, RabbitMQ |
| Embeddings | `EmbeddingProvider` | hashing (offline), OpenAI-compatible | Voyage, Jina, Gemini, local BGE/E5 |

The "not implemented" column is deliberate and worth defending: each of
those is a client library plus a running service this test suite cannot
exercise. Shipping adapter code that looks plausible but has never
connected to the thing it claims to support is worse than shipping the
interface — it reads as done, and fails in someone's production. The
interfaces are three to five methods each; the contracts are documented
above; adding one is a small, testable piece of work for whoever has that
service to test against.

**Naming an unimplemented provider raises rather than falling back.**
`GHIC_VECTOR_PROVIDER=pinecone` fails loudly at startup. Silently
substituting a local index would deliver exactly the failure the operator
was configuring their way out of.

### Persistence: pgvector on the Postgres that's already there

`PostgresVectorStore` reuses `DATABASE_URL` — the same database the ledger
and idempotency store use, and the same pure-Python `pg8000` driver, so
production persistence needs no new infrastructure and no compiled wheels
(which is what serverless bundle limits punish). Two search paths, chosen
at connect time: pgvector's `<=>` cosine operator with an IVFFlat index
when the extension is available, and in-Python scoring over one
repository's rows when it isn't. The fallback exists so the feature works
on a managed Postgres that won't grant `CREATE EXTENSION`, not because
it's a good idea at scale — it's logged as such.

Vectors are scoped by a `repo` column, so one table serves thousands of
repositories without their indexes interfering, and per-path deletes make
incremental updates possible.

### Incremental indexing

`git diff --name-status OLD NEW` gives the exact change set; a re-index
becomes "delete chunks for changed and deleted paths, re-embed only the
changed ones". Three guards make it safe rather than clever:

- **The old commit must be reachable.** A `--depth 1` clone usually does
  *not* contain the previously indexed commit, so the diff would fail or
  silently produce a wrong change set. `commit_exists()` checks; a miss
  returns None and the caller does a full rebuild. A wrong incremental
  update leaves an index permanently describing code that no longer
  exists, and nothing would ever detect it.
- **It's bounded.** Above `max_incremental_files` (200), per-path deletes
  stop being cheaper than one bulk rebuild.
- **Deletions are explicit.** A removed file's chunks must go, or the
  evidence section keeps citing a path that no longer exists — the exact
  hallucination-shaped failure this subsystem exists to prevent, arriving
  through a stale index.

Postgres-only, because a monolithic local index file can't do per-path
deletes without rewriting itself entirely, at which point it isn't
incremental.

### Lifecycle

`not_indexed → queued → indexing → ready`, plus `updating`, `failed`, and
`archived`. Two subtleties:

`UPDATING` is **searchable**: an incremental re-index leaves the previous
index in place, so an issue arriving mid-update gets slightly stale
evidence rather than none.

In-flight states suppress duplicate queueing — a busy repository queues one
index job, not one per issue — but only for `STALE_IN_FLIGHT_SECONDS` (1
hour). A worker that dies mid-index would otherwise wedge a repository in
`INDEXING` forever with nothing ever retrying it.

### The auto-disable rule

On an ephemeral filesystem (Vercel, Lambda, Cloud Run — detected from the
platform's own environment markers) with no persistent vector store
configured, `build_service()` returns **None** and logs why. Running anyway
would re-clone and re-embed on every cold start, discard the result minutes
later, cost real money against a paid embedding provider, and produce no
working feature — all while appearing enabled. Failing visibly at startup
beats degrading invisibly forever. `None` is handled everywhere exactly
like "feature off": analysis continues on issue text alone.

`/healthz` reports `state_durable` and `queue_durable` explicitly, because
"enabled but nothing persists" otherwise looks identical to a healthy
deploy until someone asks why no repository is ever ready.

### Secrets never enter the index

Two layers, because one isn't enough for a failure this bad — a secret
reaching the index reaches the embedding provider, the vector database, and
potentially a public GitHub comment:

1. **Path exclusion** (`is_sensitive_path`) drops `.env*`, `*.pem`, `*.key`,
   `id_rsa`, `credentials*`, `secrets*`, `.npmrc`, kubeconfig, terraform
   state, and service-account JSON — checked *before* the extension
   allowlist, so adding an extension later can't silently start indexing
   them.
2. **Content redaction** (`redact_secrets`) runs on every file before
   chunking, replacing GitHub/OpenAI/Anthropic/Groq/Slack/AWS/Google token
   shapes, JWTs, credentialed database URLs, PEM private-key blocks, and
   `password = "..."`-style assignments.

Pattern-based, not entropy-based, on purpose: an entropy heuristic over
source code flags hashes, UUIDs, and base64 fixtures constantly, and a
redactor that shreds ordinary code is one an operator switches off.

### Observability

Counters, timers, and gauges via `RepositoryIntelligenceMetrics`, exposed
at `/repositories/metrics`; per-repository status at `/repositories`.
Correlation IDs come from GitHub's own `X-GitHub-Delivery` where one exists
— inventing a second identifier when GitHub supplies a unique, user-visible
one just creates two things to correlate — and ride the queue payload so a
worker's logs join up with the webhook that triggered it minutes and a
process boundary earlier.

**Counters are per process.** On a horizontally-scaled deploy each
container reports its own and they reset on recycle. Durable
per-repository facts (chunk counts, index duration, retrieval hit rate)
live in the state store instead, which is why `RepositoryRecord` carries
them.

### Recommended production stack

| Component | Choice | Why |
|---|---|---|
| Host | Fly.io / Railway / Render / K8s | A real filesystem and a long-lived process |
| Vectors | `GHIC_VECTOR_PROVIDER=postgres` + pgvector | Survives restarts, shared across workers |
| State | `auto` → Postgres | Same `DATABASE_URL`, no new infrastructure |
| Queue | QStash | Durable, already used for async issue processing |
| Embeddings | `openai` | Semantic matching; the lexical default can't paraphrase |

On Vercel specifically: keep `GHIC_USE_REPO_INTELLIGENCE=false` unless
`GHIC_VECTOR_PROVIDER=postgres` is set, and expect indexing to happen on
the QStash worker rather than in the request path either way.

### Scaling notes

Chunk volume is the axis that matters: ~1k chunks per moderate repository,
so 10k repositories is ~10M rows. That is comfortable for Postgres with an
IVFFlat index and a `repo`-scoped query, and it is the point where a
dedicated vector database starts to earn its operational cost — which is
what the `VectorStore` seam is for. Indexing throughput is bounded by the
embedding provider, not by this code: batch size and concurrency are the
knobs (`GHIC_INDEX_BATCH_SIZE`), and the queue is what keeps that pressure
off the webhook path entirely.

Storage grows silently — a repository indexed once after an App install and
never queried again keeps its chunks forever — so `service.cleanup()` drops
indexes untouched for `GHIC_REPO_STALE_DAYS` (30). It keys on
`last_accessed_at` rather than `indexed_at` deliberately: a stable
repository that is still being searched shouldn't be evicted just because
its code hasn't changed.

## Extending it (later phases)

`VectorStore`, `EmbeddingProvider`, and `RepositoryIndexer` are the three
seams. Commit Intelligence, PR Intelligence, and Historical Issue Search
all need "chunk something, embed it, search it" — `RepositoryIndexer` is
deliberately handed a directory rather than doing its own cloning, so a
commit indexer that gets its content from the GitHub API reuses the same
chunker, embedder, and store without inheriting the webhook's assumptions.
`RepositoryContext` is the only type the LLM layer knows about, so a future
`CommitContext` slots in beside it without touching prompt or comment code.
