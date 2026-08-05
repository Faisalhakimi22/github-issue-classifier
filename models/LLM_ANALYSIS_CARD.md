# Card — LLM-assisted issue analysis (category / priority / severity / summary)

**Task:** turn the ML classifier's actionability probability into a
polished, human-readable analysis — category, priority, severity, a plain-
language summary and reasoning, missing-information suggestions, and label
suggestions — via an LLM, as a priority chain of two providers (below).

## Two providers, measured, not assumed

The original spec named OpenRouter's free `nvidia/nemotron-3-ultra-550b-a55b:free`
as the model. Measured directly against a real issue (`ghic/llm/openrouter.py`'s
docstring, same prompt both times):

| Provider / model | Measured latency | Output quality |
|---|---|---|
| OpenRouter `nvidia/nemotron-3-ultra-550b-a55b:free` | **~17.0s** | Good — correct classification, grounded reasoning |
| Groq `openai/gpt-oss-120b` | **~1.9s** | Comparable — correct classification, grounded reasoning, slightly terser |

17 seconds is well outside a synchronous webhook's realistic response
budget (GitHub's own delivery window is on the order of 10s). Rather than
drop the originally-specified model or block the webhook for that long,
this ships as a **priority chain** (`ghic/llm/fallback.py`): Groq is tried
first (fast enough to actually land inside the webhook response), and
OpenRouter/nemotron is the fallback if Groq fails for any reason —
network error, invalid key, rate limit. If a repo only configures one
provider's key, that's the one used, no chain involved.
`FallbackLLMProvider` is itself an `LLMProvider`, so `LLMService` and
everything upstream of it has no idea a fallback exists — same interface,
one call.

## This is not the same claim as PRIORITY_CARD.md / SEVERITY_CARD.md

Those two cards document a **deliberate refusal to ship** a priority or
severity *prediction*, because the training corpus has no ground truth for
either — any number displayed would be "a fabricated number wearing a UI."
That reasoning still holds, and this feature does not contradict it: it
never claims priority/severity here are statistically validated. They are
the LLM's judgment call — the same kind of read a maintainer forms skimming
an issue, not a number backed by an evaluation protocol. The comment and
the docs say so explicitly (see `format_llm_comment()`'s footer). If that
distinction stops being true — if this output starts getting presented or
consumed as if it were validated — it should stop shipping, by the same
logic that kept the ML version out.

## What's actually held to the existing bar

- **Actionability itself is untouched.** The ML probability is computed
  first and passed to the LLM as fixed evidence ("do not recompute");
  neither the prompt nor the schema give the model a field to report its
  own probability. See `ghic/llm/prompts.py`.
- **The webhook can never fail because of this feature.** Every failure
  mode (timeout, rate limit, malformed JSON, missing API key) is caught in
  `LLMService.analyze_issue()` and degrades to `None`, at which point the
  webhook posts the original ML-only comment (`format_comment()`)
  unchanged. See `tests/test_llm.py` and `tests/test_service.py::
  TestLLMAnalysisComment` for the failure-path tests.
- **Off by default.** `GHIC_USE_LLM_ANALYSIS=false` until the operator sets
  that flag *and* at least one of `GROQ_API_KEY` / `OPENROUTER_API_KEY` — a
  fresh deploy never starts making external LLM calls on its own, matching
  `GHIC_DRAFT_MISSING_INFO`'s precedent.
- **No raw model internals leak into the comment.** `numeric__*` / `text__*`
  feature names, TF-IDF terms, and importance values never appear in
  `format_llm_comment()`'s output — only in logs (`_handle_issue_opened`'s
  `logger.info` calls) and in `format_comment()`'s collapsed technical-
  details section, which this comment format doesn't use at all.

## Comment format and schema (v2)

The posted comment (`format_llm_comment()` in `ghic/service/inference.py`) was
rebuilt around one rule: a maintainer reading it should not be able to tell
there's a model behind it. Concretely:

- **`IssueAnalysis.reasoning` is `list[str]`, not a paragraph.** The prompt
  (`ghic/llm/prompts.py`) asks for 2–5 short factual bullets directly, and
  `parse_issue_analysis()` rejects a string in that field rather than
  splitting it after the fact — a model that's actually reasoning in bullets
  produces better bullets than a paragraph does when chopped up. Truncated to
  6 items defensively even though the prompt asks for at most 5.
- **`IssueAnalysis.recommended_action` is a new required field** — exactly
  one sentence naming the single next step a maintainer should take. It has
  its own section in the comment (`### Recommended Next Step`) rather than
  being folded into the summary or reasoning.
- **No confidence bar.** An earlier iteration rendered `pred.proba` behind a
  Unicode block-character progress bar (`confidence_bar()` in
  `ghic/service/explain.py`); that's still what `format_comment()` — the
  ML-only fallback — uses. `format_llm_comment()` deliberately does not: it
  prints the ML score as plain text (`| **ML Actionability Score** | 12% |`)
  and nothing else. A bar next to a number invites reading precision into a
  probability that doesn't have that precision.
- **`confidence` (the LLM's own confidence in its analysis) is captured in
  the schema and exposed in the API response, but never rendered in the
  comment.** Showing it next to the ML actionability score would read as two
  competing confidence numbers on the same message — exactly the ambiguity
  this redesign removes.
- **`recommended_labels` is capped at 5** and always rendered as individual
  backtick-wrapped tokens, never a comma-joined sentence.

## Consistency layer — `ghic/llm/consistency.py`

The LLM is given the ML verdict as fixed evidence and told not to
contradict it (see prompts.py above), but nothing stops its own
priority/severity/reasoning from *implying* a different conclusion than the
ML probability reached — e.g. ML says "not actionable" while the LLM's own
reasoning describes a reproducible crash with a stack trace. Shipping both
halves of that unreconciled onto the same comment is worse than either
alone: it reads as the tool contradicting itself.

`detect_disagreement()` is a deterministic, non-LLM check (same pattern as
`drafting.py`'s `needs_more_info()`): ML predicted "not actionable" **and**
the LLM's own priority and severity are both high/critical **and** its
reasoning text contains a strong actionable-signal term (crash, reproducible,
data loss, security, regression, etc.). All three must hold — it is a
precision-first check, tuned to only fire when genuinely confident something
is off, not to catch every soft disagreement.

**Deliberately not resolved with a second LLM call.** Asking the model (same
call or a follow-up) to "reconcile" a detected disagreement doesn't validate
which side was right — it just produces a differently-confident-sounding
answer, and this project has no ground truth to check that answer against.
When `detect_disagreement()` fires, the webhook logs a warning
(`llm/ml disagreement on {repo}#{number}`), the API response sets
`llm_ml_disagreement: true`, and the comment's Actionability line reads
`⚠️ Model Disagreement` instead of picking a side — every other section
(summary, reasoning, recommended action) still renders, since those stay
useful regardless of which verdict a maintainer ends up trusting.

Directional by design: only "ML says not-actionable, LLM signal says
otherwise" is flagged. The reverse (ML says actionable, LLM's own severity
read is mild) is not — a calibrated probability landing just over the
decision threshold on a real but minor issue is expected behavior, not a
contradiction. See `tests/test_llm.py::TestDetectDisagreement` and
`tests/test_service.py::TestLLMAnalysisComment` (the
`test_disagreement_*` / `test_webhook_end_to_end_surfaces_disagreement_*`
cases) for the covered scenarios.

## Cost / reliability notes

- Both default models are free tiers: no per-token cost, but free tiers
  carry tighter rate limits and can be deprioritized under load — part of
  why this ships as a two-provider chain rather than a single point of
  failure. `LLMService` caches identical (repo, title, body, rounded
  probability) requests for 24h to cut duplicate calls either way —
  best-effort: the cache is the project's existing content-addressed file
  cache (`data/raw/llm_analysis/`), which is not persistent on Vercel's
  serverless filesystem the way it is on Docker/Fly. A cache miss there
  just means every request hits the API fresh, not a broken feature.
- Retries are narrow by design: only transient failures (429, 5xx,
  connection errors) retry, capped at a couple of attempts with a short
  backoff. A timeout is never retried — the webhook has GitHub's ~10s
  delivery window to answer in, and a provider call already spent most of
  it if it timed out once. This is exactly why Groq is primary and
  nemotron is fallback rather than the reverse: a fallback that itself
  needs 17s doesn't help a request that's already blown its budget, but it
  costs nothing to have ready for the (fast) failure cases -- an invalid
  key, a network blip, Groq's own rate limit.

## Extending to another provider

`ghic/llm/provider.py`'s `LLMProvider` is the interface a new backend
(OpenAI, Anthropic, Gemini, Ollama) needs to implement — one method,
`analyze(context) -> IssueAnalysis`. `ghic/llm/_chat_completions.py`'s
`ChatCompletionsProvider` is a second, narrower base class for the common
case (any OpenAI-compatible `/chat/completions` API, which covers most
providers including both shipped ones) — a new provider on that shape is a
~15-line subclass setting `BASE_URL`/`DEFAULT_MODEL`/`PROVIDER_NAME` (see
`groq.py` for the minimal example, `openrouter.py` for one with an extra
request parameter). Nothing in `service.py`, `prompts.py`, or the webhook
handler branches on a provider name; `settings.llm_provider` is
informational/logging only — which provider(s) actually get constructed in
`app.py`'s `_build_llm_provider()` is decided by which API key(s) are set.
