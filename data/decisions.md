# Implementation Decisions

Running log of build choices not derivable from the code itself: model
selection, threshold tuning, baseline calibration, pipeline parallelization.
Useful as design rationale for any future maintainer.

## Models

### LLM / Judge: `gemini-3-flash-preview`

The spec named `gemini-3-flash-latest`. That model id returns 404 from the
Gemini API (`models/gemini-3-flash-latest is not found for API version
v1beta`). The published Gemini 3 Flash id (as of 2026-04) is
`gemini-3-flash-preview`. We use it for both the answer model
(`Settings.llm_model`) and the cheap judge model (`Settings.judge_model`).

Fallbacks if the preview tier is rate-limited or pulled:
- `gemini-flash-latest` — alias to the current stable flash family.
- `gemini-2.5-flash` — current stable flash, same SDK shape.

Either drop-in replacement is one config-line change.

### Embeddings: `gemini-embedding-001`

Matches the spec exactly. 3072-dim vectors. Cosine space configured on the
Chroma collection (`hnsw:space = cosine`).

## Retrieval Threshold

Spec sets `similarity_threshold = 0.55`. Phase 2 verification with three
calibration queries gave the following top-1 cosine similarities:

| Query                                            | Top-1 score | Behavior expected |
|--------------------------------------------------|-------------|-------------------|
| What is the gold card tax exemption?             | 0.718       | Answer            |
| How long is the Employment Gold Card valid?      | 0.731       | Answer            |
| Should I invest in Bitcoin?                      | 0.576       | Refuse off-topic  |

The Bitcoin query crosses the 0.55 threshold because several Digital Field
qualification chunks mention "blockchain" — the embedder treats *Bitcoin*
and *blockchain* as semantically close. The similarity threshold alone is
**not sufficient** to gate off-topic queries; the LLM-based input guard
(Phase 3) is the primary off-topic defense and the threshold serves only
as a low-confidence backstop for in-scope but poorly-covered questions.

This calibration is the reason the architecture pairs an input guard with
a similarity floor instead of trusting either alone.

Eval set will include both failure modes: (a) off-topic queries with semantic
neighbors in corpus, (b) on-topic queries with sparse coverage. Metrics
tracked separately.

## Pipeline Parallelization

The guarded chatbot pipeline has two stages that are independent of each
other on the input side:

- **Input guard** (`check_on_topic`) — LLM judge call, ~2.5 s.
- **Query embedding** (`embedder.embed`) — Gemini embedding call, ~0.3 s.

A naive sequential pipeline (input guard → embed → retrieve → generate →
output guard) wastes the embed latency entirely behind the input guard. The
chatbot launches both as `asyncio.create_task`s and joins via `asyncio.gather`,
so the second stage (vector store query) starts as soon as the slower of the
two finishes — i.e. as soon as the input guard returns.

If the input guard refuses, the embedding result is discarded. The cost of a
wasted Gemini embed call is roughly $0.00001 per refusal — well below the
latency we save on every accepted query (~2.5 s in the happy path).

This is the kind of thing a consulting deliverable should call out: real
production pipelines should not be linear chains of `await`. Independent
stages run concurrently; only stages that genuinely depend on prior outputs
block.

## Streaming + Accumulate

`GeminiClient.generate_stream` yields incremental `StreamChunk`s so Phase 5
can show tokens as they arrive. Phase 4 still uses streaming for the main
generation step but accumulates the full string before returning a
`ChatResponse` — same code path, no UI yet. Phase 5 will feed the chunks into
`st.write_stream`.

## Single Shared `GeminiClient`

`GuardedChatbot` constructs one `GeminiClient` and threads it through the
embedder, both guards, and the answer model. Token usage for a chat turn is
computed as the diff between `client.cumulative_usage` at chat entry and at
exit — single source of truth, no double-counting, no per-stage estimation.

## Inside-the-Pipeline Page = Task 3 Evidence

The Inside-the-Pipeline subpage is the single best evidence on the demo for
three Task 3 (Quality Oversight) standards: prompt versioning (the actual
`SYSTEM_PROMPT_GUARDED` string is rendered inline in the inspector for every
response, copyable verbatim), correlation IDs (every response carries a
UUID that all log lines for the request share — visible at the bottom of
each turn with a one-click copy button), and eval-suite hooks (the
threshold-coloured retrieval scores in the inspector use the same 0.55 trust
floor that gates production answers and that the eval set scores against).

## Naive Baseline Calibration

Initial Phase 4 testing revealed that even `SYSTEM_PROMPT_NAIVE` ("You are a
helpful chatbot for Talent Taiwan") induces refusal on obvious off-topic
queries (Bitcoin, weather) due to base model safety alignment. Decision:
do not weaken the prompt artificially. Instead, reframe the demo around three
failure modes where guards demonstrably add value beyond base alignment:
(1) domain-specific hallucination of regulatory specifics, (2) low-coverage
queries requiring escalation, (3) prompt injection and scope drift. The
Bitcoin-style off-topic case is retained in the demo for completeness but is
no longer the primary contrast point.

## Low-Confidence Surfacing

When retrieval's top similarity is below `settings.similarity_threshold`, the
chatbot returns `LOW_CONFIDENCE_RESPONSE` and surfaces a synthetic
`GuardDecision(decision=REFUSE_LOW_CONFIDENCE)` in the `input_guard` slot of
`ChatResponse` so the UI has a single field to inspect for any pre-generation
gate. The actual input-guard decision was `PASS`; the synthetic record's
`judge_model` is set to `"threshold:0.55"` so log readers can tell the
difference at a glance.
