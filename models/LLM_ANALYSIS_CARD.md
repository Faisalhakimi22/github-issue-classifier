# Card — LLM-assisted issue analysis (category / priority / severity / summary)

**Task:** turn the ML classifier's actionability probability into a
polished, human-readable analysis — category, priority, severity, a plain-
language summary and reasoning, missing-information suggestions, and label
suggestions — via an LLM (OpenRouter, default model
`nvidia/nemotron-3-ultra-550b-a55b:free`).

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
  both that flag and `OPENROUTER_API_KEY` — a fresh deploy never starts
  making external LLM calls on its own, matching `GHIC_DRAFT_MISSING_INFO`'s
  precedent.
- **No raw model internals leak into the comment.** `numeric__*` / `text__*`
  feature names, TF-IDF terms, and importance values never appear in
  `format_llm_comment()`'s output — only in logs (`_handle_issue_opened`'s
  `logger.info` calls) and in `format_comment()`'s collapsed technical-
  details section, which this comment format doesn't use at all.

## Cost / reliability notes

- The default model is a free OpenRouter tier: no per-token cost, but free
  tiers carry tighter rate limits and can be deprioritized under load.
  `LLMService` caches identical (repo, title, body, rounded probability)
  requests for 24h to cut duplicate calls — best-effort: the cache is the
  project's existing content-addressed file cache
  (`data/raw/llm_analysis/`), which is not persistent on Vercel's
  serverless filesystem the way it is on Docker/Fly. A cache miss there
  just means every request hits the API fresh, not a broken feature.
- Retries are narrow by design: only transient failures (429, 5xx,
  connection errors) retry, capped at a couple of attempts with a short
  backoff. A timeout is never retried — the webhook has GitHub's ~10s
  delivery window to answer in, and a provider call already spent most of
  it if it timed out once.

## Extending to another provider

`ghic/llm/provider.py`'s `LLMProvider` is the only interface a new backend
(OpenAI, Anthropic, Gemini, Ollama) needs to implement — one method,
`analyze(context) -> IssueAnalysis`. Nothing in `service.py`, `prompts.py`,
or the webhook handler branches on a provider name; `settings.llm_provider`
exists to select which `LLMProvider` gets constructed in `app.py`, not to
be checked anywhere else.
