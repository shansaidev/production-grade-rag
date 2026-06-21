# CLAUDE.md — Production RAG System

> This file is read automatically by Claude Code at session start.
> It gives the AI assistant full context about this project and
> the behavioral guidelines it must follow while working here.

---

## Behavioral Guidelines (Karpathy Principles)

> Source: Derived from Andrej Karpathy's observations on LLM coding pitfalls.
> These apply to every task in this project, regardless of size.
> **Tradeoff:** These bias toward caution over speed. For trivial one-liners, use judgment.

---

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing anything:
- State assumptions explicitly. If uncertain, **ask first**.
- If multiple interpretations exist, **present them** — don't pick silently.
- If a simpler approach exists, say so. **Push back** when warranted.
- If something is unclear, **stop. Name what's confusing. Ask.**

#### RAG-Specific Examples

**❌ Wrong — silent assumption on ambiguous request:**
```
User: "Update the retrieval to use better chunking"

Claude: [immediately rewrites SmartChunker with semantic chunking]
```

**✅ Correct — surface the interpretation before acting:**
```
User: "Update the retrieval to use better chunking"

Claude: "Better chunking" could mean several things here:

1. Smaller chunks (128-256 tokens) → higher precision but may lose sentence context
2. Semantic chunking (split on topic changes) → better boundaries but slower ingest
3. Parent-child chunking (small chunks indexed, parent served to LLM) → best precision

Our SmartChunker currently targets 256-512 tokens with structure-aware splitting
(tables kept whole, headings preserved). Context recall is 0.74, context_precision 0.71.

Which problem are you seeing — wrong chunks retrieved, or too much noise in context?
That determines which approach makes sense.
```

**❌ Wrong — assumes "faster search" means vector index tuning:**
```
User: "Make the search faster"

Claude: [immediately adds HNSW ef_search tuning and prefetch optimisation]
```

**✅ Correct — disambiguate first:**
```
"Faster" in this RAG system could mean:

1. Query latency (currently p95=2.8s miss path, p95=95ms cache hit)
   → Cache hit rate increase, or reduce LLM calls per query
   → Effort: 2-3 hours

2. Qdrant search latency (~80ms currently)
   → Tune ef_search, reduce prefetch from 50 to 30
   → Effort: 30 min. Risk: lower recall

3. Ingestion throughput (how fast new docs become searchable)
   → Parallelise Celery workers, batch embed calls
   → Effort: 4-6 hours

Current bottleneck from logs: generate node averages 1.9s.
Is it query latency you're seeing, or something else?
```

---

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: *"Would a senior engineer say this is overcomplicated?"* If yes, simplify.

#### RAG-Specific Examples

**❌ Wrong — over-engineered reranker abstraction nobody asked for:**
```python
# User asked: "add FlashRank reranker"
class RerankerFactory:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(reranker_cls):
            cls._registry[name] = reranker_cls
            return reranker_cls
        return decorator

    @classmethod
    def create(cls, config: RerankerConfig) -> BaseReranker:
        ...

@RerankerFactory.register("flashrank")
class FlashRankReranker(BaseReranker):
    ...

@RerankerFactory.register("cohere")  # nobody asked for this
class CohereReranker(BaseReranker):
    ...
```

**✅ Correct — exactly what was asked:**
```python
# User asked: "add FlashRank reranker"
from flashrank import Ranker, RerankRequest

_ranker = None

def get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _ranker

def rerank(query: str, chunks: list[dict], top_n: int = 20) -> list[dict]:
    passages = [{"id": i, "text": c["text"]} for i, c in enumerate(chunks)]
    results = get_ranker().rerank(RerankRequest(query=query, passages=passages))
    return [chunks[r["id"]] | {"ce_score": r["score"]} for r in results[:top_n]]
```

**Add the factory pattern later only if a second reranker is actually needed.**

**❌ Wrong — speculative validation features:**
```python
# User asked: "add the Auditor validator"
class AuditorValidator:
    def __init__(self, llm, cache=None, metrics_collector=None,
                 fallback_validator=None, retry_config=None):
        # ... 80 lines building infrastructure nobody asked for
```

**✅ Correct:**
```python
class Auditor:
    def __init__(self):
        self.chain = AUDITOR_PROMPT | get_llm()

    async def audit(self, response: str, chunks: list[dict]) -> dict:
        result = await self.chain.ainvoke({...})
        try:
            return json.loads(result.content)
        except json.JSONDecodeError:
            return {"passed": False, "score": 0.0, "reasoning": "Parse error"}
```

---

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, **mention it — don't delete it**.

When your changes create orphans:
- Remove imports/variables/functions that **your changes** made unused.
- Don't remove pre-existing dead code unless asked.

**The test:** Every changed line should trace directly to the user's request.

#### RAG-Specific Examples

**❌ Wrong — drive-by changes when asked to "fix the auditor JSON parsing":**
```python
- class Auditor:
-     def __init__(self):
-         self.llm = ChatOllama(model="llama3.2:8b", temperature=0)
-         self.chain = AUDITOR_PROMPT | self.llm
+ class Auditor:
+     """LLM-based auditor for grounding verification."""   # added docstring nobody asked for
+     def __init__(self, model: str = "llama3.2:8b"):      # added param nobody asked for
+         self.llm = ChatOllama(model=model, temperature=0)
+         self.chain = AUDITOR_PROMPT | self.llm
+         self._call_count = 0                              # added metric nobody asked for

    async def audit(self, response: str, chunks: list[dict]) -> dict:
-       result = await self.chain.ainvoke({...})
-       return json.loads(result.content)                   # the bug: no error handling
+       result = await self.chain.ainvoke({...})
+       self._call_count += 1                              # side effect nobody asked for
+       try:
+           return json.loads(result.content)
+       except json.JSONDecodeError:
+           return {"passed": False, "score": 0.0}
```

**✅ Correct — fix only the JSON parsing bug:**
```python
    async def audit(self, response: str, chunks: list[dict]) -> dict:
        result = await self.chain.ainvoke({...})
-       return json.loads(result.content)
+       try:
+           return json.loads(result.content)
+       except json.JSONDecodeError:
+           return {"passed": False, "score": 0.0, "reasoning": "Parse error"}
```

**❌ Wrong — reformatting and type-hinting while adding a log line:**
```python
- def rerank(query, chunks, top_n=20):
+ def rerank(query: str, chunks: list[dict], top_n: int = 20) -> list[dict]:
+     """Rerank chunks using FlashRank cross-encoder."""
+     log.info("reranking", query=query[:50], candidates=len(chunks))
      passages = [{'id': i, 'text': c['text']} for i, c in enumerate(chunks)]
```

**✅ Correct — add only the log line, match existing style:**
```python
  def rerank(query, chunks, top_n=20):
+     log.info('reranking', query=query[:50], candidates=len(chunks))
      passages = [{'id': i, 'text': c['text']} for i, c in enumerate(chunks)]
```

---

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write a test where auditor catches a hallucination, then make it pass"
- "Fix the retrieval bug" → "Write a query that reproduces wrong results, then fix it"
- "Improve faithfulness" → "Faithfulness score on golden dataset moves from X to ≥ Y"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently.
Weak criteria ("make it better") require constant clarification.

#### RAG-Specific Examples

**❌ Wrong — vague approach with no verification:**
```
User: "The RAG is hallucinating too much"

Claude: I'll fix the hallucination issue by:
1. Reviewing the validation layer
2. Improving the prompts
3. Testing the changes

[Makes several changes with no measurement]
```

**✅ Correct — define what "fixed" looks like:**
```
User: "The RAG is hallucinating too much"

Claude: To fix hallucination, I need a measurable target.
Current state: faithfulness = 0.68 on 30-pair golden dataset (from evaluations table).

Plan:
1. Reproduce: Run golden dataset, find the 5 lowest-faithfulness queries
   Verify: Can identify specific ungrounded claims in each

2. Hypothesis test A — tighten system prompt:
   Add "ONLY use the provided CONTEXT. Never add general knowledge."
   Verify: Re-run 5 failing queries, grounding_score improves

3. Hypothesis test B — reduce top_k_final from 8 to 5:
   Fewer chunks = less noise per context window
   Verify: context_precision improves, check faithfulness delta

4. Full re-evaluation:
   Verify: faithfulness ≥ 0.80 on full 30-pair dataset

I'll start with step 1. Run: python -m src.evaluation.runner --verbose
and share which queries are failing.
```

**❌ Wrong — implements MMR without verifying it helps:**
```
User: "Add MMR to reduce redundant chunks"

Claude: [implements 80-line MMR function and wires it in]
Done! MMR is added.
```

**✅ Correct — verify MMR actually improves diversity:**
```
User: "Add MMR to reduce redundant chunks"

Plan:
1. Baseline measurement: pick 5 queries where I suspect redundancy
   Find queries where top-5 chunks have cosine_sim > 0.90 to each other
   Verify: Can show 3+ near-duplicate chunks in current results

2. Implement MMR (λ=0.5, k=5)
   Verify: Same 5 queries now return chunks with max cosine_sim < 0.85

3. Quality check: does MMR hurt faithfulness?
   Verify: Run golden dataset before/after — faithfulness must not drop > 0.02

4. Wire into full_reranker.py
   Verify: Integration test passes, p95 latency increase < 20ms

Starting with step 1 — running diagnostic queries now.
```

---

## Project Overview

A **production-grade Retrieval-Augmented Generation (RAG) system** built in Python.
Two parallel architectures are implemented:

| Architecture | Doc | Vector Store | Status |
|---|---|---|---|
| **v1 — pgvector** | `PRODUCTION_RAG_SYSTEM.md` | PostgreSQL + pgvector | Reference impl |
| **v2 — Qdrant Hybrid** | `PRODUCTION_RAG_QDRANT_HYBRID.md` | PostgreSQL + Qdrant | Primary (build this) |

**Active implementation: v2 (Qdrant Hybrid)**

---

## Tech Stack (Non-Negotiable)

```
Language:      Python 3.12
LLM:           Ollama (local) → llama3.2:8b | llama3.2:70b
Embeddings:    nomic-embed-text (dense, 768-dim, local via Ollama)
               FastEmbed SPLADE (sparse, local)
Vector DB:     Qdrant (port 6333 HTTP, 6334 gRPC)
Relational:    PostgreSQL 16 (port 5432) — source of truth
Cache:         Redis Stack DB1 (port 6379) — semantic cache via redisvl
Broker:        Redis Stack DB0 (port 6379) — Celery task queue
Storage:       MinIO (port 9000) — raw document storage
API:           FastAPI + uvicorn (port 8000)
Agents:        LangGraph (state machine, NOT simple chains)
Evaluation:    RAGAS (faithfulness, relevancy, recall, precision)
Logging:       structlog (JSON structured, every node timed)
```

---

## Repository Structure

```
production-rag/
├── CLAUDE.md                    ← this file (AI context + behavioral guidelines)
├── SKILLS.md                    ← coding conventions for this project
├── README.md                    ← project overview for humans
├── CONTRIBUTING.md              ← how to contribute / extend
├── docker-compose.yml
├── .env.example
├── pyproject.toml
├── alembic.ini
│
├── alembic/versions/
│   └── 0001_initial_schema.py
│
├── src/
│   ├── api/
│   │   ├── main.py              ← FastAPI app factory
│   │   ├── routers/
│   │   │   ├── documents.py     ← POST /documents, GET /documents
│   │   │   ├── queries.py       ← POST /query/sync, POST /query (stream)
│   │   │   ├── evaluation.py    ← POST /eval/run, GET /eval/runs
│   │   │   └── qdrant.py        ← GET /qdrant/health, POST /qdrant/reconcile
│   │   └── schemas/
│   │       ├── document.py      ← Pydantic request/response models
│   │       └── query.py
│   │
│   ├── ingestion/
│   │   ├── pipeline.py          ← orchestrates full ingestion flow
│   │   ├── parsers/
│   │   │   ├── base.py
│   │   │   ├── pdf_parser.py    ← pdfplumber, table-safe
│   │   │   ├── docx_parser.py   ← python-docx, heading hierarchy
│   │   │   ├── html_parser.py   ← bs4, strips nav/footer
│   │   │   ├── code_parser.py   ← tree-sitter, AST-aware
│   │   │   └── file_router.py   ← MIME detection → correct parser
│   │   ├── chunking/
│   │   │   └── smart_chunker.py ← structure-aware, table-safe
│   │   ├── metadata/
│   │   │   ├── generator.py     ← LLM: summary + keywords + HyDE questions
│   │   │   └── keyword_extractor.py ← KeyBERT
│   │   └── embedding/
│   │       ├── dense_embedder.py    ← nomic-embed-text via Ollama
│   │       └── sparse_embedder.py  ← FastEmbed SPLADE
│   │
│   ├── retrieval/
│   │   ├── hybrid_searcher.py   ← Qdrant Query API (dense+sparse+RRF)
│   │   ├── pg_hydrator.py       ← batch SELECT WHERE chunk_id=ANY($1)
│   │   ├── reranker.py          ← FlashRank cross-encoder (Stage 1)
│   │   ├── rule_reranker.py     ← recency+authority+keyword boost (Stage 2)
│   │   ├── mmr.py               ← MMR diversity filter (Stage 3)
│   │   ├── full_reranker.py     ← orchestrates all 3 stages
│   │   ├── hyde.py              ← Hypothetical Document Embeddings
│   │   └── context_assembler.py ← lost-in-middle ordering
│   │
│   ├── reasoning/
│   │   ├── engine.py            ← LangGraph graph definition
│   │   ├── state.py             ← RAGState TypedDict
│   │   ├── nodes.py             ← all node functions
│   │   ├── planner.py           ← query decomposition
│   │   └── tools.py             ← LangGraph tool definitions
│   │
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── retriever_agent.py   ← Agent 1: retrieval specialist
│   │   ├── reasoner_agent.py    ← Agent 2: synthesis + reasoning
│   │   └── verifier_agent.py    ← Agent 3: fact-checking
│   │
│   ├── validation/
│   │   ├── gatekeeper.py        ← "Does this answer the question?"
│   │   ├── auditor.py           ← "Is every claim grounded?"
│   │   └── strategist.py        ← "Does this make domain sense?"
│   │
│   ├── cache/
│   │   └── semantic_cache.py    ← redisvl SemanticCache, two-level
│   │
│   ├── evaluation/
│   │   ├── runner.py            ← RAGAS evaluate()
│   │   ├── metrics.py           ← retrieval precision, recall, MRR, NDCG
│   │   └── qdrant_metrics.py    ← fusion_gain measurement
│   │
│   ├── stress_testing/
│   │   ├── red_team.py          ← RedTeamRunner
│   │   └── test_cases/
│   │       ├── prompt_injection.py
│   │       ├── info_evasion.py
│   │       └── bias_tests.py
│   │
│   ├── db/
│   │   ├── postgres/
│   │   │   ├── connection.py
│   │   │   ├── models.py        ← ORM models (8 tables)
│   │   │   └── repositories/
│   │   └── qdrant/
│   │       ├── client.py
│   │       ├── collection.py
│   │       ├── writer.py
│   │       ├── lifecycle.py
│   │       └── reconciler.py
│   │
│   ├── workers/
│   │   ├── celery_app.py
│   │   └── tasks.py
│   │
│   ├── scripts/
│   │   └── setup_qdrant.py
│   │
│   └── core/
│       ├── config.py            ← pydantic-settings (get_settings())
│       ├── logging.py           ← structlog + @timed_node decorator
│       └── llm_client.py        ← unified LLM interface
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── golden_dataset/
│       └── qa_pairs.json        ← 30 Q&A pairs
│
└── docs/
    ├── PRODUCTION_RAG_SYSTEM.md
    ├── PRODUCTION_RAG_QDRANT_HYBRID.md
    ├── RAG_THEORY_COMPLETE.md
    └── RAG_IMPLEMENTATION_GUIDE.md
```

---

## Key Design Decisions (Never Change Without Good Reason)

### 1. PostgreSQL is always the source of truth
Vectors live in Qdrant. Text and metadata live in PostgreSQL.
If Qdrant is lost, rebuild from PostgreSQL. Never the reverse.

### 2. Write PostgreSQL FIRST, Qdrant SECOND
Every mutation: PG commit → Qdrant upsert/delete → mark `qdrant_synced=TRUE`.
On Qdrant failure: mark `qdrant_synced=FALSE`, reconciler fixes nightly.

### 3. LangGraph is a state machine, not a chain
All agent communication via shared `RAGState` TypedDict.
No direct agent-to-agent calls. Every node: `async (state) -> partial_state`.

### 4. Never cache unvalidated responses
Cache only when: `validation_passed=True` AND `confidence >= 0.70` AND `cache_decision != BYPASS`.

### 5. Reranking is three stages, in order
`Qdrant(N=50)` → `FlashRank(→20)` → `RuleBoost` → `MMR(→K=5)` → `LLM`.
N:K ratio must be ≥ 6:1. Enforced by startup assertion.

### 6. Tables are never split during chunking
Tables are always one chunk. Non-negotiable.

### 7. Soft delete before hard delete, always
Soft delete = mark deleted + remove Qdrant points + flush cache.
Hard delete = only after `deleted_at IS NOT NULL`.

---

## Environment Variables (.env)

```env
DATABASE_URL=postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb
REDIS_URL=redis://localhost:6379/0          # Celery (DB 0)
REDIS_CACHE_URL=redis://localhost:6379/1    # Semantic cache (DB 1)
MINIO_ENDPOINT=localhost:9000
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_COLLECTION=rag_chunks
QDRANT_USE_GRPC=true
QDRANT_PREFETCH_DENSE=50
QDRANT_PREFETCH_SPARSE=50
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:8b
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768
OLLAMA_BASE_URL=http://localhost:11434
CHUNK_SIZE_TOKENS=400
CHUNK_OVERLAP_TOKENS=50
TOP_K_FINAL=5
MMR_LAMBDA=0.5
RECENCY_WEIGHT=0.20
CACHE_SIMILARITY_THRESHOLD=0.92
CACHE_FACTUAL_TTL=86400
CACHE_TEMPORAL_TTL=3600
CACHE_MIN_CONFIDENCE=0.70
VALIDATION_ENABLED=true
MIN_GATEKEEPER_SCORE=0.70
MIN_AUDITOR_SCORE=0.75
```

---

## Running the System

```bash
docker compose up -d
python -m alembic upgrade head
python -m src.scripts.setup_qdrant
celery -A src.workers.celery_app worker --loglevel=info -Q ingestion -c 4 &
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
pytest tests/unit/ -v
python -m src.evaluation.runner
```

---

## API Quick Reference

```
POST  /api/v1/documents              Upload + ingest document
GET   /api/v1/documents              List documents
GET   /api/v1/jobs/{job_id}          Ingestion job status
POST  /api/v1/documents/{id}/versions  Version bump
PATCH /api/v1/documents/{id}         Metadata update (no re-embed)
DELETE /api/v1/documents/{id}        Soft delete
POST  /api/v1/query/sync             Synchronous query
POST  /api/v1/query                  Streaming query (SSE)
POST  /api/v1/eval/run               Trigger RAGAS evaluation
POST  /api/v1/qdrant/reconcile       Sync PG ↔ Qdrant
GET   /api/v1/health                 Full system health
```

---

## Quality Gates (Production Readiness)

| Metric | Target | How to Measure |
|---|---|---|
| RAGAS faithfulness | ≥ 0.80 | `python -m src.evaluation.runner` |
| RAGAS answer_relevancy | ≥ 0.70 | same |
| context_precision | ≥ 0.75 | same |
| Qdrant fusion_gain | ≥ 0.05 | `python -m src.evaluation.qdrant_metrics` |
| MRR (reranker) | ≥ 0.60 | `python -m src.evaluation.metrics` |
| Cache hit rate | ≥ 15% | `SELECT * FROM cache_hit_rate_daily;` |
| p95 latency (miss) | ≤ 3500ms | structlog + `queries` table |
| p95 latency (hit) | ≤ 120ms | structlog + `cache_metrics` |
| Red team pass rate | ≥ 80% | `python -m src.stress_testing.red_team` |
| CRITICAL injection blocked | 100% | same |

---

## PostgreSQL Tables (v2)

| Table | Purpose |
|---|---|
| `documents` | Every file: version, is_latest, department, checksum |
| `chunks` | Every chunk: text, heading, page, tsvector, qdrant_synced |
| `queries` | Every query: plan, chunk_ids, response, latency, tokens |
| `validations` | Per-query: gatekeeper/auditor/strategist scores |
| `evaluations` | RAGAS run results |
| `cache_metrics` | Cache hit/miss per query + latency |
| `deletion_audit` | GDPR trail |
| `qdrant_sync_log` | Daily reconciler run results |

---

## Common Tasks for Claude Code

### Adding a new document parser
1. Create `src/ingestion/parsers/{type}_parser.py`
2. Implement `parse(file_path, doc_id) -> ParsedDocument`
3. Register MIME type in `src/ingestion/parsers/file_router.py`
4. Add tests in `tests/unit/test_parser.py`

### Adding a new LangGraph node
1. Add `@timed_node("name") async def my_node(state: RAGState) -> RAGState` to `nodes.py`
2. Add `graph.add_node("name", nodes.my_node)` to `engine.py`
3. Add edge or conditional edge

### Changing the embedding model
1. Update `.env`: `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`
2. `POST /api/v1/admin/reindex` → wait → `GET /api/v1/admin/reindex/status`
3. Evaluate: `python -m src.evaluation.runner`
4. `POST /api/v1/admin/reindex/cutover`
5. `redis-cli -n 1 FLUSHDB` (flush stale cache)

---

## Documents to Reference

| What you want to know | Read |
|---|---|
| Why any architecture decision was made | `docs/PRODUCTION_RAG_QDRANT_HYBRID.md` §13 |
| Theory behind any algorithm | `docs/RAG_THEORY_COMPLETE.md` |
| Step-by-step build order | `docs/RAG_IMPLEMENTATION_GUIDE.md` |
| PostgreSQL schema (all 8 tables) | `docs/RAG_IMPLEMENTATION_GUIDE.md` §2 Phase 1 |
| Embedding lifecycle (CRUD) | `docs/PRODUCTION_RAG_QDRANT_HYBRID.md` §16 |
| Semantic cache design | `docs/PRODUCTION_RAG_QDRANT_HYBRID.md` §17 |
| Reranking pipeline (3 stages) | `docs/RAG_THEORY_COMPLETE.md` §16 |
| Red team test cases | `src/stress_testing/test_cases/` |
