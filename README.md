# vLLM Docs RAG: question answering with an evaluation gate in CI

A question-answering service over the [vLLM](https://github.com/vllm-project/vllm) v0.31.0 documentation.
Every pull request is evaluated against a hand-written golden set, and **a PR that lowers answer
quality is blocked from merging**.

> Problem: RAG systems are usually built once and never measured again, so quality drops quietly
> whenever the prompt, chunking or model changes. Here every change is measured before it ships.

## Results

| Metric | Value |
|---|---|
| Golden set | 60 questions: 15 factual, 15 how-to, 12 config-flag, 8 multi-hop, 10 unanswerable (`eval/golden.csv`) |
| Recall@5 / MRR@10 (dense, chunk 400) | 0.870 / 0.833 (multi-hop Recall@5: 0.688) |
| Faithfulness / Correctness (GPT-4.1 judge) | 0.957 / 0.800 |
| Refusal accuracy (unanswerable) / false refusals (answerable) | 1.00 / 0.14 |
| Latency p50 / p95 / p99 (end to end, uncached) | 1.43 s / 2.21 s / 3.89 s |
| Retrieval latency (warm) | ~4–50 ms |
| Cost per query (gpt-4o-mini generator) | $0.00025 · full 60-question eval incl. judge ≈ $0.19 |
| Load test throughput | _fill in from `make loadtest`_ |

## Architecture

```mermaid
flowchart LR
  subgraph OFFLINE["Offline · make index"]
    D["vLLM docs v0.31.0<br/>211 pages"] --> C["Header-aware chunker<br/>400 tok, 60 overlap"]
    C --> E["bge-small (dense) + BM25 (sparse)"]
    E --> Q[("Qdrant<br/>2,224 chunks")]
  end
  subgraph ONLINE["Online · REST/JSON"]
    U["User / chat page"] -- "POST /ask" --> A["FastAPI"]
    A -- "embed + search" --> Q
    A -- "top-5 excerpts + prompt" --> L["gpt-4o-mini"]
    L --> A
    A -- "answer, citations, cost,<br/>latency, config_hash" --> U
    A -. "background" .-> LOG["request logs"] -.-> DASH["Streamlit dashboard"]
    A -. "20% sample" .-> J["GPT-4.1 judge"]
    A -. "async export" .-> LF["Langfuse traces<br/>retrieve → generate spans"]
  end
  subgraph CI["CI · every pull request"]
    PR["PR"] --> EV["Eval on golden set"] --> G{"Gate vs baseline<br/>config/gate.yaml"}
    G -- "regression" --> X["❌ check fails, merge blocked"]
    G -- "ok" --> OK["✅ mergeable"]
  end
```

**Request path:** embed the question (same model as the index; the service refuses to start on a mismatch) →
top-20 dense search (optionally fused with BM25 by Reciprocal Rank Fusion) → refuse if nothing relevant →
top-5 excerpts wrapped in `[BEGIN DOCUMENT n]` delimiters → gpt-4o-mini answers with `[n]` citations →
citations checked against the retrieved set. If the LLM fails, the service returns the relevant doc sections instead of an error.

## Evaluation

| Layer | Metrics | How | When |
|---|---|---|---|
| Retrieval | Recall@1/5/10, MRR@10, Section@5, per slice | Programmatic, against `source_docs` | Every PR (free) |
| Generation | Faithfulness, correctness, citation validity | GPT-4.1 judge (stronger than the generator; human-agreement calibrated) | Every PR when keys are set |
| Refusals | Refusal accuracy, false-refusal rate | Programmatic, on the `unanswerable` slice | Every PR |
| Online | Thumbs up/down, sampled faithfulness | `/feedback`, background judge on 20% of traffic | Live |

Gate rules (`config/gate.yaml`): a metric fails if it drops more than its tolerance from the target
branch's baseline **or** falls below an absolute floor. Retrieval tolerances are tight because those
metrics are deterministic; judge tolerances are wider because judge scores are noisy.

## Tracing (Langfuse)

Set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` and every answer becomes a Langfuse trace:
`rag-answer` → `retrieve` (pages + scores) → `generate` (prompt, model, tokens, cost). Traces are tagged with
`config_hash`, prompt version and retrieval mode, so you can filter by version and compare before/after a change.
Scores attached to traces: `faithfulness` and `correctness` from eval runs (tagged `eval`), `faithfulness_online`
from the sampled live judge, and `user_feedback` from 👍/👎. The API's `request_id` is the trace id.
Export happens in a background thread, so tracing adds no request latency; without keys it is a no-op.

## Versioning

- All tunables are in `config/rag.yaml`; prompts are in `prompts/*.txt`.
- `config_hash` (config + prompt text) is returned with every answer and recorded in every log row and eval run.
- `index_hash` (corpus + chunking + embedding model + chunking code) names the Qdrant collection, so a
  chunking change builds a new index instead of mixing vector versions.

## Run it

```bash
make setup                 # virtualenv + dependencies
cp .env.example .env       # add OPENAI_API_KEY (and optional Langfuse keys)
make index                 # ~2 min on a laptop CPU
make eval-retrieval        # free retrieval eval
make eval                  # full eval with the LLM judge
make serve                 # API + chat page on http://localhost:8000
make dashboard             # monitoring on http://localhost:8501
make test                  # unit tests
```

## Stack and why

| Choice | Over | Because |
|---|---|---|
| Qdrant (embedded locally/CI, Cloud optional) | Chroma, pgvector | Payload filtering, dense + sparse vectors in one collection, and the same client with no server in CI |
| bge-small via fastembed (local CPU) | API embeddings | Free, deterministic in CI, no vendor lock-in |
| gpt-4o-mini generator, GPT-4.1 judge (same provider) | Two providers | One API key and budget; the stronger judge plus a human-agreement check bounds the self-preference risk |
| Chunk size 400 tokens | 512 | bge-small truncates at 512 tokens, and every chunk carries a page/section breadcrumb |
| Hosted LLM API | Self-hosted vLLM on a GPU | At ~1k queries/day the API costs a few dollars, a GPU costs hundreds a month |

## Repository layout

```
config/       rag.yaml (all tunables), gate.yaml (CI rules)
prompts/      versioned system prompts
data/         vLLM v0.31.0 docs snapshot (Apache-2.0, see data/SOURCE.md)
src/rag/      chunking, index, retrieve, pipeline, api, telemetry
src/rag/evaluation/  dataset, metrics, judge, run, compare (the gate), sweep (experiments)
eval/         golden.csv, smoke.csv, baseline.json, doc_map.md
dashboard/    Streamlit monitoring app
loadtest/     Locust load test
```
