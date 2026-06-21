# SKILLS.md — Coding Conventions & Patterns

> Project-specific coding standards for the Production RAG System.
> Every contributor (human or AI) must follow these conventions.
> Non-negotiable items are marked ⛔.

---

## Python Standards

### Version & Typing
```python
# ⛔ Python 3.12+ only. Use modern typing everywhere.

# ✅ Correct
async def embed_batch(texts: list[str]) -> list[list[float]]:
    ...

def classify(query: str, user_id: str | None) -> CacheDecision:
    ...

# ❌ Wrong — old-style typing
from typing import List, Optional
def embed_batch(texts: List[str]) -> List[List[float]]:
    ...
```

### Async First
```python
# ⛔ All I/O must be async. Never block the event loop.

# ✅ Correct
async def search(query: str) -> list[HydratedChunk]:
    results = await qdrant_client.query_points(...)
    rows = await pg_conn.fetch(...)
    return hydrate(results, rows)

# ❌ Wrong — synchronous I/O in async context
def search(query: str) -> list[HydratedChunk]:
    results = qdrant_client.search(...)   # blocks event loop
```

### Dataclasses Over Dicts
```python
# ✅ Use dataclasses or Pydantic models for structured data
@dataclass
class HydratedChunk:
    chunk_id: str
    text: str
    rrf_score: float
    page_number: int
    filename: str

# ❌ Don't pass raw dicts between functions
def process(chunk: dict) -> dict:   # What's in the dict? Nobody knows.
```

### Config Always via get_settings()
```python
# ⛔ Never hardcode values. Always use settings.

# ✅ Correct
from src.core.config import get_settings
settings = get_settings()
top_k = settings.top_k_final

# ❌ Wrong
TOP_K = 5   # hardcoded, can't be tuned without code change
```

### Singletons via @lru_cache
```python
# Heavy objects (models, DB clients) are instantiated once.
from functools import lru_cache

@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(host=settings.qdrant_host, ...)

@lru_cache
def _get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
```

---

## LangGraph Node Conventions

### Node Signature (⛔ Non-Negotiable)
```python
# Every node: async, takes RAGState, returns partial state dict
async def my_node(state: RAGState) -> RAGState:
    # Read from state
    query = state["query"]

    # Do work
    result = await do_something(query)

    # Return ONLY the fields this node modifies
    # (LangGraph merges this with existing state)
    return {"my_field": result}
```

### Always Use the Timing Decorator
```python
from src.core.logging import timed_node

@timed_node("retrieve")
async def retrieve(state: RAGState) -> RAGState:
    ...
# Automatically logs: {"event": "node_complete", "node": "retrieve", "duration_ms": 82.3}
```

### Conditional Edge Functions
```python
# Edge routing functions must be pure — no side effects, no I/O
def route_after_validation(state: RAGState) -> str:
    if state.get("validation_passed"):
        return "store_cache"
    if state.get("retry_count", 0) >= 2:
        return "give_up"
    return "replan"
# Returns a string key that matches the dict in add_conditional_edges()
```

---

## Database Conventions

### PostgreSQL: Always Use asyncpg for Performance Queries
```python
# ✅ Correct — asyncpg for batch hydration (hot path)
rows = await conn.fetch(
    "SELECT chunk_id, chunk_text, section_heading, page_number "
    "FROM chunks WHERE chunk_id = ANY($1::uuid[])",
    chunk_ids
)

# ✅ Correct — SQLAlchemy for mutations (lifecycle operations)
async with session.begin():
    await session.execute(
        text("UPDATE documents SET is_latest=FALSE WHERE doc_id=:id"),
        {"id": doc_id}
    )
```

### Qdrant: Always Use gRPC in Production
```python
# ✅ Correct — gRPC for speed
client = AsyncQdrantClient(
    host=settings.qdrant_host,
    grpc_port=settings.qdrant_grpc_port,
    prefer_grpc=True,
)

# ❌ Wrong for production — HTTP is slower for bulk ops
client = AsyncQdrantClient(host="localhost", port=6333)
```

### PostgreSQL Writes Before Qdrant Writes (⛔)
```python
# ⛔ Order is ALWAYS: PG commit → Qdrant upsert → mark synced
# Never write to Qdrant before the PG commit succeeds.

async def ingest_chunk(chunk, dense_vec, sparse_vec, payload):
    # 1. PostgreSQL FIRST (transactional)
    async with session.begin():
        await session.execute(INSERT_CHUNK_SQL, chunk_params)

    # 2. Qdrant SECOND (idempotent, retry-safe)
    try:
        await qdrant_writer.upsert_chunks([chunk_id], [dense_vec], [sparse_vec], [payload])
        await session.execute(UPDATE_SYNCED_SQL, {"id": chunk.chunk_id})
    except Exception as e:
        log.error("qdrant_write_failed", error=str(e))
        # Do NOT raise — reconciler will retry. PG write is safe.
```

---

## Ingestion Conventions

### Chunking Rules (⛔ Non-Negotiable)
```python
# 1. Tables: ALWAYS one chunk, NEVER split
if section.section_type == "table":
    return [make_single_chunk(section)]   # always length 1

# 2. Target token range: 256-512 tokens
assert 256 <= chunk.token_count <= 512 + 50   # 50 token tolerance

# 3. Every chunk gets its nearest heading as prefix
if heading:
    text = f"[{heading}]\n{chunk_text}"

# 4. Overlap: 50 tokens of previous chunk
OVERLAP_TOKENS = 50   # not configurable per-chunk, only per-corpus
```

### Metadata Generation: Always Batch LLM Calls
```python
# ✅ Correct — batch all chunks, one LLM call
summaries = await generate_summaries_batch(chunks, batch_size=10)

# ❌ Wrong — one LLM call per chunk (10x more expensive + slower)
for chunk in chunks:
    chunk.summary = await generate_summary(chunk)
```

---

## Retrieval Conventions

### N >> K Ratio (⛔ Enforced by Startup Assertion)
```python
# In Settings.check_nk_ratio():
assert self.top_k_final * 6 <= self.qdrant_prefetch_dense

# Default values — do not lower TOP_K_FINAL above 8
# QDRANT_PREFETCH_DENSE=50
# TOP_K_FINAL=5
```

### Reranking Pipeline Order (⛔)
```python
# Always in this order — do not skip stages
candidates_50 = await hybrid_searcher.search(query, limit=50)
reranked_20   = cross_encoder_rerank(query, candidates_50, top_n=20)
boosted_20    = rule_based_boost(reranked_20, query)
final_5       = mmr_select(query_vec, boosted_20, chunk_vecs, k=5)
```

### Context Assembly: Lost-in-Middle Ordering (⛔)
```python
# ⛔ ALWAYS put best chunk first, second-best chunk last
def assemble_context(chunks: list[HydratedChunk]) -> str:
    if len(chunks) <= 1:
        return chunks[0].text if chunks else ""
    ordered = [chunks[0]] + chunks[2:] + [chunks[1]]
    return "\n\n---\n\n".join(c.text for c in ordered)
```

---

## Semantic Cache Conventions

### Cache Decision Must Be Made Before Any LLM Work
```python
# ✅ Correct — check cache in the FIRST LangGraph node
async def check_cache_node(state: RAGState) -> RAGState:
    decision = cache.classify(state["query"], state.get("user_id"))
    if decision != CacheDecision.BYPASS:
        hit = await cache.get(state["query"])
        if hit:
            return {**state, "final_response": hit["answer"],
                    "cache_hit": True, "confidence": 0.95}
    return {**state, "cache_hit": False, "cache_decision": decision.value}
```

### Only Cache Validated Responses
```python
# ⛔ Three conditions must ALL be true before storing in cache
def should_cache(state: RAGState) -> bool:
    return (
        state.get("validation_passed") is True     # passed all validators
        and state.get("confidence", 0) >= 0.70     # above confidence threshold
        and state.get("cache_decision") != "bypass" # not personal/adversarial
    )
```

### Redis DB Isolation (⛔)
```python
# DB 0: Celery broker — NEVER use for cache
# DB 1: Semantic cache — NEVER use for Celery
REDIS_URL       = "redis://localhost:6379/0"   # Celery
REDIS_CACHE_URL = "redis://localhost:6379/1"   # SemanticCache
```

---

## Validation Conventions

### Three Validators, Always Run in Parallel
```python
# ⛔ Run Gatekeeper + Auditor in parallel (asyncio.gather)
# Strategist is optional — run only for domain-sensitive queries
gatekeeper_result, auditor_result = await asyncio.gather(
    gatekeeper.check(query=state["query"], response=state["draft_response"]),
    auditor.audit(response=state["draft_response"], chunks=state["retrieved_chunks"]),
)
```

### Validation Prompt Format (⛔ Must Return JSON)
```python
# Every validator prompt MUST end with:
# "Return ONLY valid JSON. No preamble. No markdown code blocks."
# Every validator MUST handle JSONDecodeError and return a safe default.

try:
    return json.loads(result.content)
except json.JSONDecodeError:
    return {"passed": False, "score": 0.0, "reasoning": "Parse error"}
```

---

## Structured Logging Conventions

### Every Log Line Must Have an Event Name
```python
import structlog
log = structlog.get_logger()

# ✅ Correct — machine-parseable event name + structured fields
log.info("cache_hit", level="semantic", similarity=0.94, latency_ms=7.2)
log.error("qdrant_write_failed", doc_id=doc_id, error=str(e))
log.info("validation_complete", gatekeeper=True, auditor=False, retry=1)

# ❌ Wrong — unstructured free text
log.info(f"Cache hit with similarity {similarity}")
log.error("Qdrant failed: " + str(e))
```

### Every LangGraph Node Must Log Duration
```python
# Use the @timed_node decorator — it handles this automatically
# If you can't use the decorator, log manually:
start = time.perf_counter()
result = await do_work()
log.info("node_complete", node="my_node",
         duration_ms=round((time.perf_counter()-start)*1000, 1))
```

---

## Error Handling Conventions

### Never Swallow Exceptions Silently in Core Logic
```python
# ✅ Log + re-raise OR log + return safe default (pick one, document which)
try:
    result = await qdrant.search(...)
except Exception as e:
    log.error("qdrant_search_failed", error=str(e), query=query[:50])
    raise   # Re-raise — let LangGraph handle via retry

# ✅ Safe default for non-critical paths
try:
    cache_result = await cache.get(query)
except Exception as e:
    log.warning("cache_get_failed", error=str(e))
    cache_result = None   # Continue without cache — not fatal
```

### Celery Tasks: Always Retry with Backoff
```python
@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,   # seconds, doubles each retry
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def ingest_document_task(self, doc_id: str, ...):
    ...
```

---

## Testing Conventions

### Unit Tests: Fast, No I/O, Deterministic
```python
# Unit tests must not hit Postgres, Qdrant, Redis, or Ollama
# Use pytest fixtures for all external dependencies

def test_smart_chunker_table_never_split():
    """A table ParsedSection must become exactly ONE chunk."""
    chunker = SmartChunker()
    doc = make_test_doc([ParsedSection("| A | B |\n|---|---|\n| 1 | 2 |", "table", None, 1)])
    chunks = chunker.chunk(doc)
    assert len([c for c in chunks if c.chunk_type == "table"]) == 1
```

### Integration Tests: Real Services, but Isolated Data
```python
# Integration tests use real services but with test-scoped data
# Always clean up: delete test documents after each test

@pytest.fixture(autouse=True)
async def cleanup_test_data(pg_conn, qdrant_client):
    yield
    await pg_conn.execute("DELETE FROM documents WHERE filename LIKE 'TEST_%'")
    await qdrant_client.delete(collection_name="rag_chunks",
                               points_selector=FilterSelector(...))
```

### Test Naming Convention
```python
# test_{what}_{condition}_{expected_result}
def test_auditor_hallucinated_claim_flagged(): ...
def test_chunker_table_input_single_chunk_output(): ...
def test_hybrid_search_department_filter_restricts_results(): ...
def test_soft_delete_document_invisible_to_search(): ...
```

---

## File Naming Conventions

```
src/ingestion/parsers/pdf_parser.py       # snake_case, descriptive
src/retrieval/hybrid_searcher.py          # noun + verb form
src/db/qdrant/lifecycle.py                # describes the concept
tests/unit/test_chunker.py                # test_ prefix always
tests/integration/test_hybrid_search.py
```

---

## What Claude Code Should Never Do

1. ⛔ Add a new dependency without checking `pyproject.toml` first
2. ⛔ Write synchronous database calls inside async functions
3. ⛔ Hardcode any value that's in `.env` or `config.py`
4. ⛔ Store a response in semantic cache without checking `validation_passed`
5. ⛔ Write to Qdrant before the PostgreSQL transaction commits
6. ⛔ Split a table across multiple chunks in SmartChunker
7. ⛔ Create a LangGraph node that communicates directly with another node (use state)
8. ⛔ Write a LangGraph node that is synchronous (all nodes must be `async def`)
9. ⛔ Skip the `@timed_node` decorator on new LangGraph nodes
10. ⛔ Return raw dicts from functions where a dataclass or Pydantic model exists
