# 🚀 Enterprise Agentic RAG Platform

> **Building in Public** — A production-grade AI system built from scratch, documented phase by phase with real engineering decisions, bottlenecks, and fixes.

Production-ready **Agentic Retrieval-Augmented Generation (RAG)** platform built on FastAPI. This project stitches together a modern stack designed for developers, data scientists, and operators who need a scalable system for ingesting documents, running RAG pipelines, orchestrating agentic workflows, and measuring performance with open benchmarks.

The most interesting part of this project? **The system decides how to answer — and sometimes it decides not to use AI at all.**

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI · Uvicorn · Pydantic v2 |
| **LLM** | Gemini 2.0 Flash (Multimodal) · Groq (Fallback) · LangChain · LangGraph |
| **Embeddings / Reranking** | `gemini-embedding-2-preview` · `ms-marco-MiniLM-L-6-v2` |
| **Vector Store** | Qdrant (dense + sparse hybrid) |
| **Evaluation** | RAGAS (reference-free metrics) |
| **Observability** | Langfuse (tracing · spans · prompt versioning) |
| **Cache** | Upstash Redis (serverless REST API) |
| **Database** | Neon PostgreSQL · SQLAlchemy async · Alembic |
| **Deployment** | Docker · GitHub Actions |

---

## 🗺 Build Roadmap

```
Phase 1 ✅  RAG Core         — PDF ingestion · vector search · FastAPI backend
Phase 2 ✅  Optimization     — Hybrid search · Cross-encoder reranking · RAGAS evaluation
Phase 3 ✅  Agentic Layer    — LangGraph agent · Intelligent routing · Conversation memory
Phase 4 ✅  Multimodal       — Images · Tables · Gemini 2.0 Flash
Phase 5 ✅  MLOps            — Langfuse observability · Token tracking · Prompt versioning · A/B testing · RAGAS pipeline
Phase 6 ✅  Deployment       — Docker · CI/CD · Live on cloud
```

---

## 🧠 Phase 1 — RAG Core

**Goal:** PDF ingestion pipeline, dense vector search, and a working FastAPI backend.

### What was built
- **PDF Parser** (`app/ingestion/parser.py`) — PyMuPDF (fitz) page-by-page extraction with graceful handling of image-only pages and corrupted pages. Each page is a dict with `page_number`, `text`, and `char_count`.
- **Chunker** (`app/ingestion/chunker.py`) — LangChain `RecursiveCharacterTextSplitter` splitting on `\n\n`, `\n`, `. `, ` ` in priority order, preserving page-number metadata per chunk.
- **Ingestion Pipeline** (`app/ingestion/pipeline.py`) — Orchestrates parse → chunk → embed → upsert. Blocking CPU calls (embedding, parsing) are dispatched via `asyncio.to_thread` to keep the FastAPI event loop free.
- **Qdrant Vector Store** (`app/vectorstore/qdrant_client.py`) — Dense cosine-distance search using `BAAI/bge-small-en-v1.5` (384-dim vectors). Deterministic content-based UUIDs derived from `source:page:chunk_index` SHA-256 hash to ensure idempotent upserts.
- **FastAPI Backend** (`app/api/`) — Versioned REST endpoints for document upload, querying, and health checks. Pydantic v2 schemas for validation.
- **Config** (`app/core/config.py`) — Pydantic `BaseSettings` with per-service env-prefix groups (`GROQ_`, `QDRANT_`, etc.) loaded from `.env` via `lru_cache`.

### 🐛 Issues Encountered & Resolutions

**Issue 1: Image-only PDF pages causing silent failures**
- **Where:** `app/ingestion/parser.py`
- **Cause:** PyMuPDF's `page.get_text()` returns an empty string (not an error) for image-only pages. Downstream, zero-character chunks were silently embedded as near-zero vectors, polluting the index.
- **Fix:** Added explicit `if not text.strip(): continue` guard in the chunker to skip empty pages. Parser now also wraps `page.get_text()` in a try/except to gracefully handle corrupted pages without failing the entire document.

**Issue 2: Duplicate chunks on repeated ingestion**
- **Where:** `app/vectorstore/qdrant_client.py` — `upsert_chunks()`
- **Cause:** Using random UUIDs meant that re-uploading the same PDF created duplicate points, causing retrieval to return the same content multiple times.
- **Fix:** Switched to deterministic UUIDs: `uuid.UUID(bytes=sha256(f"{source}:{page}:{idx}".encode())[:16])`. Repeated upserts now overwrite the existing point cleanly.

---

## ⚡ Phase 2 — Optimization

**Goal:** Replace pure dense search with a hybrid retrieval pipeline and add inline quality evaluation.

### What was built
- **Hybrid Search** (`app/vectorstore/qdrant_client.py`) — Two-leg search (dense cosine + sparse BM25) fused with **Reciprocal Rank Fusion (RRF)**. BM25 uses `rank_bm25` with stable positive-integer token hashes. Qdrant's named-vector API (`"dense"` + `"sparse"`) stores both vector types per point.
- **Cross-Encoder Reranking** (`app/rag/retrieval.py`) — `ms-marco-MiniLM-L-6-v2` cross-encoder scores (query, chunk) pairs jointly, capturing token-level interactions that bi-encoder search misses. Fetches `top_k × 4` candidates from hybrid search, reranks, returns best `top_k`. Loaded as a module-level singleton to pay the ~1s download cost only once.
- **RAGAS Evaluation** (`app/evaluation/ragas_evaluator.py`) — Inline, reference-free evaluation using `Faithfulness`, `ResponseRelevancy`, and `LLMContextPrecisionWithoutReference`. No ground-truth answers needed — runs on every live request when `evaluate=True`. Backed by `ChatGroq` + `LangchainLLMWrapper`.
- **Redis Cache** (`app/cache/redis_client.py`) — Upstash Redis over its serverless REST API (no TCP connection). MD5 cache key from `"{question}::{collection}"`. TTL defaults to 3600 s. Fully fault-tolerant — all cache errors log a warning and fall through to a live query.
- **Langfuse Tracing** (`app/observability/langfuse_tracer.py`) — Fault-tolerant `LangfuseTracer` wrapper. When credentials are absent or the server unreachable, every call returns a `_NoOpTrace`/`_NoOpSpan` — silent no-op, no errors. The RAG pipeline creates 4 spans per query: `hybrid-search`, `reranking`, `llm-generation`, and `ragas-evaluation` (as trace metadata).

### 🐛 Issues Encountered & Resolutions

**Issue 1: Dense-only schema breaking hybrid search**
- **Where:** `app/vectorstore/qdrant_client.py` — `ensure_collection()`
- **Cause:** The Phase 1 collection was created with a single unnamed dense vector config. Qdrant's named-vector and sparse-vector APIs require a different schema from the start — trying to add sparse vectors to an existing dense-only collection raised a schema conflict error.
- **Fix:** `ensure_collection()` now always drops and recreates the collection to enforce the `{"dense": VectorParams, "sparse": SparseVectorParams}` schema. This is safe because the ingestion pipeline re-populates on the next upload, and schema mismatches silently break searches.

**Issue 2: Qdrant `search()` method removed in v1.x**
- **Where:** `app/vectorstore/qdrant_client.py` — `search()` and `hybrid_search()`
- **Cause:** `qdrant_client` v1.x removed the `.search()` method. Calls to it raised `AttributeError` at runtime.
- **Fix:** Migrated all search calls to `query_points()`, the universal search API in v1.x. The `using` parameter selects which named vector index to query (`DENSE_VECTOR_NAME` or `SPARSE_VECTOR_NAME`).

**Issue 3: Cross-encoder reranker crashing server startup when HuggingFace is unreachable**
- **Where:** `app/rag/retrieval.py` — module-level model loading
- **Cause:** `CrossEncoder(_RERANKER_MODEL)` raised a network error when HuggingFace Hub was behind a firewall or hadn't been pre-cached, preventing the server from starting.
- **Fix:** Wrapped the model load in a `try/except`. On failure, `_reranker` is set to `None`. `_rerank()` detects `None` and falls back to returning the top-k results by hybrid-search RRF order. Server always starts; reranking degrades gracefully.

**Issue  4: `ResponseRelevancy` returning `None` / not scoring**
- **Where:** `app/evaluation/ragas_evaluator.py` — `_run_ragas()`
- **Cause:** `ResponseRelevancy` requires an embedding model to measure cosine similarity between question and generated answer statements. When no `embeddings=` argument was passed to `evaluate()`, it silently returned `null` for that metric.
- **Fix:** Built a `LangchainEmbeddingsWrapper` around `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` and passed it as `embeddings=evaluator_embeddings` to the `evaluate()` call, reusing the already-cached local model.

---

## 🤖 Phase 3 — Agentic Layer ✅

**Goal:** Replace the direct RAG pipeline call with a LangGraph state machine that routes queries intelligently and persists conversation memory.

### What was built
- **LangGraph Agent** (`app/agents/graph.py`) — `StateGraph` with 5 nodes compiled once at module import as a singleton `agent_graph`. Routes: `router → [rag_retrieval | llm] → llm → memory → eval → END`.
- **Router Node** (`app/agents/nodes/router.py`) — Groq zero-shot classifier using a terse, few-shot-guided prompt. Returns exactly one of `"rag"`, `"direct"`, or `"out_of_scope"`. Includes `_parse_route()` with partial-match fallback + default-to-RAG on ambiguity.
- **RAG Node** (`app/agents/nodes/rag_node.py`) — Calls `RetrievalService.retrieve()`, populates `state["retrieved_chunks"]` and `state["reranked_chunks"]`.
- **LLM Node** (`app/agents/nodes/llm_node.py`) — Handles all three routing paths: RAG-grounded generation, direct parametric answer, and canned refusal for `out_of_scope` (zero LLM token cost).
- **Memory Node** (`app/agents/nodes/memory_node.py`) — Persists each turn to PostgreSQL (`ConversationTurn` model) and loads the last 5 turns for the session as `HumanMessage`/`AIMessage` pairs, injecting them into `state["messages"]`. Saves *after* generation so that partial failures are not persisted.
- **Eval Node** (`app/agents/nodes/eval_node.py`) — Opt-in RAGAS evaluation gate. Skips automatically for `direct`/`out_of_scope` routes (no retrieved context), empty chunks, or when `state["evaluate"]` is `False`.
- **PostgreSQL + Alembic** (`app/db/session.py` · `migrations/`) — Async SQLAlchemy engine with `asyncpg`. Alembic migrations for `conversation_turns` table. `pool_pre_ping=True` handles Neon serverless PostgreSQL idle connection drops.
- **Langfuse Tracing** (`app/observability/langfuse_tracer.py`) — Fault-tolerant `LangfuseTracer` wrapper. When credentials are absent or the server unreachable, every call returns a `_NoOpTrace`/`_NoOpSpan` — silent no-op, no errors. The agent creates 3 spans per query: `cache-check`, `langgraph-run` (the full agent graph execution), and `cache-write`. RAGAS scores are surfaced as trace-level metadata on the `agent-chat` trace.

### 🐛 Issues Encountered & Resolutions

**Issue 1: LangGraph router LLM returning multi-word responses instead of a single token**
- **Where:** `app/agents/nodes/router.py` — `_parse_route()`
- **Cause:** Smaller LLaMA variants occasionally responded with `"rag (because the question references documents)"` or `"out_of_scope."` instead of a bare single word, causing an exact-match lookup in `_VALID_ROUTES` to fail and defaulting everything to `"rag"`.
- **Fix:** Added `_parse_route()` with two-stage extraction: (1) strip whitespace and non-word characters with regex, (2) substring scan across valid routes for partial matches. Also added `max_tokens=10` to the Groq call to aggressively limit response length.

**Issue 2: Alembic `asyncpg` refusing `DATABASE_URL` with sslmode query parameters**
- **Where:** `app/db/session.py` — `_make_async_url()`
- **Cause:** Neon's `DATABASE_URL` includes `?sslmode=require&channel_binding=require`. `asyncpg` does not accept any URL query parameters and raised: `connect() got an unexpected keyword argument 'sslmode'` and `database "neondb&channel_binding=require" does not exist` (the `&` fragment corrupted the database name).
- **Fix:** Built `_make_async_url()` which strips the entire query string with `re.sub(r"\?.*$", "", url)` and separately extracts `sslmode` via `_extract_sslmode()` before stripping. SSL is then re-applied through `connect_args={"ssl": True}` where needed. The scheme is also normalised from `postgres://` / `postgresql+psycopg2://` to `postgresql+asyncpg://`.

**Issue 3: Alembic `env.py` not detecting the `ConversationTurn` model for autogenerate**
- **Where:** `migrations/env.py`
- **Cause:** Alembic's `autogenerate` compares `target_metadata` against the current DB schema. Because the model import was missing in `env.py`, `target_metadata` was `None` and `alembic revision --autogenerate` produced an empty migration.
- **Fix:** Added explicit `from app.models.conversation import ConversationTurn` in `env.py` and referenced `Base.metadata` as `target_metadata`. Also configured Alembic with the `async` template and patched `run_migrations_online()` to use `asyncio.run()` with `AsyncEngine.begin()`.

**Issue 4: Memory node persisting empty answers from upstream failures**
- **Where:** `app/agents/nodes/memory_node.py`
- **Cause:** If `llm_node` set a non-empty `error` but wrote an empty `final_answer` (e.g., due to a Groq API timeout), `memory_node` would still write the empty-answer row to PostgreSQL, polluting session history.
- **Fix:** Added an upstream error guard: if `state["error"]` is set and `final_answer` is empty, the INSERT is skipped but the history SELECT still runs so the next turn gets proper context. The check is `if question and answer:` before `db.add(turn)`.

**Issue 5: Langfuse trace events not appearing in UI until server restart**
- **Where:** `app/rag/pipeline.py` — `RAGPipeline.query()`
- **Cause:** The Langfuse Python SDK batches events and flushes them on a timer. With `uvicorn --reload`, the process restarted before the flush timer fired, so traces were lost.
- **Fix:** Added an explicit `tracer.flush()` call immediately after `tracer.end_trace()` in `pipeline.py`. This ensures events are POSTed to Langfuse on every request, not just at process shutdown.

**Issue 6: RAGAS + LangGraph LangSmith 403 Crash (`IndexError: list index out of range`)**
- **Where:** `app/evaluation/ragas_evaluator.py`
- **Cause:** LangGraph injects LangSmith tracing callbacks into every node automatically. RAGAS inherits them, attempts to reach the LangSmith API, gets a `403 Forbidden` (since it's not configured), and crashes trying to parse the failed trace callbacks with an `IndexError`.
- **Fix:** Explicitly disabled LangSmith tracing directly in the Python module before imports. Adding it to `.env` doesn't work because LangChain reads tracing flags at import time. Added the following at the top of the file:
  ```python
  import os
  os.environ["LANGCHAIN_TRACING_V2"] = "false"
  ```

---

## 🖼 Phase 4 — Multimodal ✅

**Goal:** Extend ingestion to handle images and tables natively using Gemini multimodal models, and upgrade the LLM provider for unified cross-modal reasoning.

### What was built
- **Unified Gemini Architecture** (`app/llm/gemini_client.py` & `app/rag/embeddings.py`) — Shifted from heterogeneous local/API models to a unified Google-GenAI stack. Replaced `BGE-small-en` with `gemini-embedding-2-preview` (multimodal, 768-dim MRL), and Groq with `gemini-2.0-flash` for generation. Groq is retained purely as a text-only fallback.
- **Multimodal Embedding** (`app/ingestion/image_extractor.py`) — Extracts embedded PDF images via PyMuPDF (ignoring decorative elements < 100x100px) and embeds the PIL Image *directly* using Gemini. No intermediate captioning is done, reducing latency and preserving pixel detail. Image vectors exist in the same space as text queries.
- **Vision at Query Time ("Option B")** (`app/ingestion/pipeline.py` & `app/rag/retrieval.py`) — Instead of static captions, raw `image_base64` strings are preserved in the Qdrant payload `metadata`. At query time, `retrieve_with_vision()` separates text chunks and image bytes.
- **Single Multimodal RAG Payload** (`app/rag/pipeline.py`) — Passes both relevant text chunks and base64 images into a *single* `generate_multimodal_response` call. The Gemini LLM gets cross-attention over the whole context (prose + charts + question) simultaneously, yielding superior grounded answers.
- **Domain-Specific Reranker Fine-Tuning** (`notebooks/reranker_finetuning.ipynb`) — Fine-tuned the `ms-marco-MiniLM-L-6-v2` cross-encoder using synthetic pairs generated by Gemini 2.5 Flash. Implemented "hard negative" mining to force the model to learn strict semantic boundaries, evaluated using NDCG@3 and MRR.
- **Dynamic Reranker Service** (`app/rag/reranker.py`) — Abstracted reranker initialization. Auto-detects the presence of the local `models/finetuned-reranker` directory, seamlessly upgrading to the domain-tuned model, with an automatic fallback to the HuggingFace base model in production where the volume might not be mounted.

### Known Limitation: Multimodal Evaluation

RAGAS is a text-only evaluation framework. When retrieved 
context contains image chunks (empty text fields), RAGAS 
scores are skipped for those responses to avoid incorrect 
faithfulness scoring.

**Production solution (not implemented in this version):**
Two-tier evaluation — RAGAS for text/table chunks, 
Gemini vision self-evaluation for image chunks. 
Known tradeoff: self-serving bias when using the same 
model for generation and evaluation.

### Reranker Fine-tuning Results

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| MRR | 0.5167 | 0.5750 | +0.0583 (+11.3%) |
| NDCG@3 | 0.5524 | 0.6393 | +0.0869 (+15.7%) |

**Key finding:** Improvement came from hard negative training —
generating training pairs where negative chunks share vocabulary 
with the correct answer but don't answer the query. Easy random 
negatives produced zero improvement across 3 experiments.

### 🐛 Issues Encountered & Resolutions

**Issue 1: Cross-encoder dropping image chunks due to empty text**
- **Where:** `app/rag/retrieval.py` — `retrieve_with_vision()`
- **Cause:** Image points in Qdrant intentionally have `text=""` (BM25 sparse search relies on text, but images are retrieved via dense vector similarity). However, when the 2nd-stage cross-encoder attempted to score `(query, "")`, it generated near-zero scores, causing relevant images to be ranked out of the top-k and silently dropped before hitting the LLM.
- **Fix:** Built a bypass array in `retrieve_with_vision()`. Image candidates are separated before reranking. Text chunks are reranked normally. Then, image candidates are force-merged back into the final result set up to the `top_k` capacity, guaranteeing they make it to the LLM.

**Issue 2: Reranker model failed to learn from simple synthetic negatives**
- **Where:** `notebooks/reranker_finetuning.ipynb`
- **Cause:** When evaluating the fine-tuned model against a baseline, it initially scored identically. Investigation revealed that utilizing "easy random negatives" during synthetic data generation fell short of teaching the model to handle overlapping vocabulary or tricky domain distractors.
- **Fix:** Modified the Gemini data generation prompt to explicitly require "Hard Negatives"—chunks sharing similar keywords/syntax with the query but answering a fundamentally different question. This forced the cross-encoder's attention mechanism to perform conceptual intent matching over simple keyword matching.

---

## 📊 Phase 5 — MLOps ✅

**Goal:** Full observability pipeline, prompt versioning, data-driven A/B testing, and automated evaluation — all without blocking the serving path.

### What was built

- **Granular Langfuse Token Observability** (`app/agents/nodes/llm_node.py`) — Every RAG request now captures per-stage token metrics via Gemini's `response.usage_metadata`: `input_tokens`, `output_tokens`, `total_tokens`. Two proxy metrics are also recorded — `context_chars` (total characters across all retrieved chunks) and `question_chars` (user question length). The ratio `context_chars / question_chars > 10:1` flags oversized chunks. All data surfaces in Langfuse as span output on the `llm-generation` span.

- **Prompt Registry** (`app/core/prompt_registry.py`) — Central `PromptRegistry` class with a 5-minute TTL in-process cache, Langfuse as the source of truth (fetched by `label="production"`), and hardcoded fallback prompts for every key. The same `session_id` always resolves to the same variant (session-stable), so users never flip mid-conversation. Cache is explicitly invalidatable per prompt name.

- **A/B Testing Infrastructure** (`app/core/prompt_registry.py` · `app/agents/nodes/llm_node.py`) — `get_ab_variant()` MD5-hashes the `session_id` and maps the first hex digit to a 50/50 split between Langfuse prompt version A and B. The resolved `prompt_variant` (`"A"` or `"B"`), `prompt_version`, and `prompt_source` are written to every `llm-generation` Langfuse span for downstream analysis.

- **Batch Evaluation Script** (`scripts/eval/batch_eval.py`) — Standalone CLI that reads a JSON file of Q/A/context triples, scores them with RAGAS, writes timestamped results JSON, and prints aggregate means. Designed to run nightly or post-deploy without touching the serving path. Supports pluggable metric sets via `--metrics` flag.

- **A/B Analysis Script** (`scripts/ab_test_analysis.py`) — Fetches the N most recent Langfuse traces that carry both RAGAS scores and variant tags, buckets them by variant, and prints a side-by-side metric table with a winner per metric and an actionable recommendation. `--verbose` flag shows per-trace timestamp, variant, question snippet, and individual scores. `--since` filter passes `from_timestamp` directly to the Langfuse API so the page budget isn't wasted on old history.

- **Controlled A/B Query Runner** (`scripts/run_ab_test_queries.py`) — Sends each evaluation question through **both** Variant A and Variant B (paired, not alternating) to control for question difficulty. Uses brute-forced UUIDs that hash to the target variant via the same MD5 logic as the router. 10 domain-specific, document-bound questions across 5 types (factual, comparative, definitional, procedural, analytical). Paces requests at 30 s intervals to respect Groq's free-tier token-per-minute limit.

### A/B Test: Prompt v1 vs Prompt v2 — Final Results

Two RAG system prompts were tested over 20 controlled query pairs:

| Metric | Variant A (v1) | Variant B (v2) | Winner |
|---|---|---|---|
| `answer_relevancy` | 0.597 | 0.542 | **A** |
| `faithfulness` | 0.775 | 0.667 | **A** |
| `llm_context_precision_without_reference` | 0.584 | 0.652 | B |

**Conclusion:** Variant A (the original production prompt) retained its lead. Variant B's "echo phrasing" rule (Rule 2: *"directly answer in the first sentence using the question's own phrasing"*) caused the model to refuse more often on coverage-gap questions — each refusal scores `faithfulness=0` and `relevancy=0` in RAGAS, pulling the mean down significantly. Variant A's more permissive style produces partial hedged answers that RAGAS still scores positively. Variant B won when both prompts answered, but the higher refusal rate was decisive. **Variant A promoted to 100% traffic.**

### 🐛 Issues Encountered & Resolutions

**Issue 1: `answer_relevancy` scoring 0.17 despite correct answers**
- **Where:** `app/evaluation/ragas_evaluator.py`
- **Cause:** RAGAS `ResponseRelevancy` reverse-engineers the original question from the answer text. Both prompts instructed the model to cite sources inline as `[Context 1]`, `[Context 2]`, etc. The reverse-question LLM received noise tokens and generated questions like *"What does Context 1 say?"* — which has near-zero cosine similarity to the actual user question. The metric was measuring citation format, not answer quality.
- **Fix:** Added `_strip_citations()` — a regex pre-processor that removes `[Context N]` markers from the `response` field before building `SingleTurnSample`. Applied only to `ResponseRelevancy`'s input; faithfulness and context precision receive the original answer with citations intact.

**Issue 2: Groq rate limits corrupting RAGAS evaluations mid-run**
- **Where:** `app/evaluation/ragas_evaluator.py`
- **Cause:** RAGAS makes 5–8 sequential LLM calls per evaluation (reverse question generation, claim decomposition, etc.). Using `llama-3.3-70b-versatile` — the same model as the main RAG pipeline — exhausted Groq's free-tier tokens-per-minute quota mid-evaluation. Symptoms were: completely empty scores on some traces, partial scores (only 1 of 3 metrics) on others, and suspiciously low `answer_relevancy` values of 0.05–0.07 on back-to-back queries.
- **Fix:** Introduced a dedicated evaluation model constant `_RAGAS_EVAL_MODEL`, defaulting to `llama-3.1-8b-instant` — a separate Groq model with ~5× higher tokens-per-minute headroom. RAGAS only needs strong instruction-following capability for its prompts; a 70B model is overkill. Overridable via `RAGAS_EVAL_MODEL` env var without code changes. The main RAG pipeline continues using `llama-3.3-70b-versatile` (via `settings.groq.model`) unaffected.

**Issue 3: A/B analysis script pulling old/contaminated traces**
- **Where:** `scripts/ab_test_analysis.py`
- **Cause:** The script fetched the 100 most recent Langfuse traces and collected the first 20 valid ones — but with `limit=100` and no time filter, pre-fix traces (with the 0.17 scores) dominated the results. After a code fix, the next run would still show the old numbers because old traces ranked higher.
- **Fix 1:** Added `--since` flag that passes `from_timestamp` directly to `lf.client.trace.list()` — the Langfuse API filters server-side so the page budget is spent only on post-cutoff traces. Client-side guard retained as belt-and-suspenders.
- **Fix 2:** Added `--verbose` flag that prints each trace's timestamp, variant, and per-metric scores alongside the question text (extracted from `trace.input`), so contaminated traces are immediately visible.
- **Fix 3:** Metric key lookup now checks both `"response_relevancy"` and `"answer_relevancy"` (RAGAS renamed the field between versions) so no scores are silently missed.

**Issue 4: Half the test queries routed away from RAG (non_rag_routing)**
- **Where:** `scripts/run_ab_test_queries.py` — initial question set
- **Cause:** Questions like *"What is retrieval augmented generation?"* and *"What are the trade-offs between fine-tuning the retriever vs generator?"* were correctly classified by the router as general-knowledge questions — no RAG needed. They were answered by the direct LLM path without document retrieval, which means no RAGAS evaluation ran. 10 of 20 queries produced no eval scores.
- **Fix:** Rewrote all 10 questions to be explicitly document-bound using anchors like *"according to the study"*, *"the paper describes"*, *"in its results section"*. These phrasings signal to the router that the answer requires a specific document, guaranteeing RAG routing and evaluation coverage.

---

## 🚀 Phase 6 — Deployment ✅

**Goal:** Containerise the full stack, establish CI/CD, and deploy to cloud without heavy infrastructure complexity.

### What was built
- **Multi-stage Docker Build** (`docker/Dockerfile`) — Optimised container image using `python:3.12-slim`. Drops build-time bloat mapping down to an ultra-light <500MB artifact, bypassing heavy Torch/PyWin32 dependencies by leaning heavily into lightweight Gemini/Jina HTTP APIs.
- **Docker Compose Orchestration** (`docker/docker-compose.yml`) — Configured a local development footprint bundling the API service with a streamlined local Qdrant container for identical cloud-like dev parity.
- **Entrypoint Automation** (`docker/entrypoint.sh`) — Injected an idempotent schema deployment sequence (`alembic upgrade head`) to automatically manage table migrations immediately prior to the Uvicorn boot sequence on every spawn.
- **Streamlit UI Interface** (`streamlit_app/app.py`) — Implemented an interactive conversational frontend communicating flawlessly to the FastAPI container backend on port 8000.
- **GitHub Actions CI/CD Pipeline** (`.github/workflows/deploy.yml` & `pr_check.yml`) — Created sequential `lint -> test -> deploy` pipelines. Fully automated strict Ruff formatting gates, parameterized Pytest suites mapping dynamically to environment secrets, and deployed dual generic `curl -X POST` web hooks asynchronously refreshing Render instances for both Web & API apps on successful `main` pushes while blocking deployment triggers safely during generic Pull Requests.

### 🐛 Issues Encountered & Resolutions

**Issue 1: Docker Compose failing to boot due to Qdrant native health-checks**
- **Where:** `docker/docker-compose.yml`
- **Cause:** The Qdrant official internal Alpine images are extraordinarily barebones—lacking internal `curl`, `bash` or `wget` utilities entirely—causing Docker's health engine to eternally stall backend FastAPI initialization. 
- **Fix:** Completely eliminated the strict container `healthcheck` layer bounding conditional polling targeting `service_healthy`. The FastAPI interface now boots dynamically parallel alongside the ultra-fast Qdrant DB startup sequence.

**Issue 2: Pytest failing instantly in GitHub Actions workflows**
- **Where:** `.github/workflows/deploy.yml`
- **Cause:** Execution sequence threw `ModuleNotFoundError: No module named 'app'` recursively failing all 64 Pytests consistently. The isolated GitHub Actions linux runner environment natively could not trace the root directory path resolving internal Python imports without local systemic IDE-level scaffolding.
- **Fix:** Injected `env: PYTHONPATH: .` within the `run: pytest` YAML step layer. Overrode the system environment variable routing directing GitHub runners reliably to the project working directory context.

---

## 📁 Project Structure

```
enterprise-rag-platform/
├── app/
│   ├── agents/           # LangGraph graph + nodes (router, rag, llm, memory, eval)
│   ├── api/              # FastAPI route handlers + versioned endpoints
│   ├── cache/            # Upstash Redis cache service
│   ├── core/             # Config, exceptions, shared utilities
│   ├── db/               # Async SQLAlchemy engine + session factory
│   ├── evaluation/       # RAGAS evaluator (reference-free metrics)
│   ├── ingestion/        # PDF parser, chunker, ingestion pipeline
│   ├── llm/              # Groq client + prompt templates
│   ├── models/           # SQLAlchemy ORM models (ConversationTurn)
│   ├── observability/    # Langfuse tracer (fault-tolerant wrapper)
│   ├── rag/              # RAG pipeline, retrieval service, embeddings
│   └── vectorstore/      # Qdrant client (dense + hybrid search)
├── migrations/           # Alembic async migrations
├── tests/                # unit / integration / e2e
├── docker/               # Dockerfile + Compose configs
├── docs/                 # Architecture diagrams, API reference, runbooks
└── scripts/              # Seed data, eval scripts
```

---

## 🏗 Getting Started

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (for Qdrant locally)
- Accounts: [Groq](https://console.groq.com) · [Qdrant Cloud](https://cloud.qdrant.io) or local · [Neon](https://neon.tech) · [Upstash](https://upstash.com) · [Langfuse](https://cloud.langfuse.com)

### Setup

1. Clone and create a virtual environment:
   ```bash
   git clone https://github.com/your-username/enterprise-rag-platform.git
   cd enterprise-rag-platform
   python -m venv .venv
   .\.venv\Scripts\activate     # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Fill in GROQ_API_KEY, QDRANT_URL, DATABASE_URL, UPSTASH_REDIS_*, LANGFUSE_* etc.
   ```

4. Run database migrations:
   ```bash
   alembic upgrade head
   ```

5. Start the server:
   ```bash
   uvicorn app.main:app --reload
   ```

6. Open API docs: `http://localhost:8000/docs`

---

## 🧪 Testing

```bash
pytest                     # full suite
pytest tests/unit/         # unit tests only
pytest tests/integration/  # integration tests
```

---

## 📘 Documentation

- **OpenAPI Docs:** `http://localhost:8000/docs` (live server)
- **Architecture:** `docs/architecture/`
- **Runbooks:** `docs/runbooks/`

---

## 📄 License

[Add license information here]

---

*Follow along on LinkedIn as each phase ships — real engineering decisions, not just demos.*