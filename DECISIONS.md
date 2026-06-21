# Architecture Decision Records (ADRs)

> Every significant design decision is recorded here with context,
> the decision made, alternatives considered, and consequences.
> Read this before proposing architectural changes.

---

## ADR-001: PostgreSQL as Source of Truth, Qdrant as Vector Index

**Date**: Project inception  
**Status**: Accepted

**Context**: Need to store document text, metadata, and vector embeddings. Options: single vector DB (Qdrant only), single relational DB (pgvector), or hybrid.

**Decision**: PostgreSQL owns all relational data (text, metadata, audit trail). Qdrant owns vector representations. `chunk_id` (UUID) is the join key.

**Consequences**:
- ✅ SQL expressiveness for audit, lifecycle, joins, GDPR compliance
- ✅ If Qdrant is lost, full rebuild from PostgreSQL is possible
- ✅ Best-in-class vector performance from purpose-built engine
- ⚠️ Two systems to maintain and keep in sync
- ⚠️ Dual-write pattern required — reconciler handles drift

**Alternatives rejected**:
- Qdrant-only: no SQL joins, no ACID, no referential integrity, no audit trail
- pgvector-only: 5× lower ANN throughput, no sparse vectors, no quantization

---

## ADR-002: LangGraph Over Simple LangChain Chains

**Date**: Project inception  
**Status**: Accepted

**Context**: Need conditional routing (simple vs complex queries), retry loops (validation → replan → retry), human-in-the-loop checkpoints, and multi-agent coordination.

**Decision**: LangGraph with explicit `RAGState` TypedDict. All nodes are pure `async (state) -> partial_state` functions.

**Consequences**:
- ✅ Every state transition is visible and debuggable
- ✅ Conditional routing, loops, and checkpointing are native
- ✅ Human-in-the-loop is a first-class feature
- ⚠️ More setup than simple chains
- ⚠️ Steeper learning curve

**Alternatives rejected**:
- Plain LangChain chains: no branching, no retry loops, no shared state
- AutoGen: implicit message passing, harder to inspect
- CrewAI: role-based but less controllable for production validation

---

## ADR-003: Three-Stage Reranking Pipeline

**Date**: After Phase 12 implementation  
**Status**: Accepted

**Context**: Pure vector retrieval has ~75% recall but poor precision. Need to maximise both precision and response quality.

**Decision**: Three sequential stages after hybrid retrieval:
1. FlashRank cross-encoder (semantic precision)
2. Rule-based boost (recency + authority + keyword)
3. MMR diversity filter (prevent redundant context)

**N:K ratio enforced at 10:1 minimum (N=50, K=5).**

**Consequences**:
- ✅ Adds ~200ms latency (acceptable)
- ✅ context_precision improvement of ~0.05-0.15
- ✅ MMR prevents 5 chunks saying the same thing
- ⚠️ Requires dense embeddings at query time for MMR (already generated for HyDE)
- ⚠️ Rule-based weights need domain tuning

**Alternatives rejected**:
- LLM listwise reranking (RankGPT): 5-10× more expensive, marginal gain
- Single cross-encoder only: no diversity, no recency awareness
- ColBERT as primary retriever: complex index, slower ingestion

---

## ADR-004: Semantic Cache in Redis, Not Qdrant

**Date**: Semantic caching implementation  
**Status**: Accepted

**Context**: High query repetition rate expected. Need sub-100ms response for seen queries. Redis already in stack for Celery.

**Decision**: Redis Stack DB1 with redisvl SemanticCache (HNSW index on query vectors). DB0 reserved for Celery.

**Consequences**:
- ✅ Zero new infrastructure (Redis already deployed)
- ✅ In-memory → 5-8ms lookup vs 80ms Qdrant + 800ms LLM
- ✅ Native TTL auto-expires stale entries
- ✅ LRU eviction self-manages cache size
- ⚠️ Cache invalidation must be triggered on document changes
- ⚠️ Redis Stack image required (not plain Redis Alpine)

**Alternatives rejected**:
- Qdrant as cache: vector search overhead, no TTL, separation-of-concerns violation
- PostgreSQL cache: no in-memory guarantee, slower than Redis
- Application-level dict: not persistent, not distributed

---

## ADR-005: FlashRank Over ms-marco-MiniLM via sentence-transformers

**Date**: Reranker selection  
**Status**: Accepted

**Context**: Need fast local reranker. Options: full-precision sentence-transformers vs quantized FlashRank.

**Decision**: FlashRank with `ms-marco-MiniLM-L-12-v2` ONNX model.

**Consequences**:
- ✅ 5-10× faster than sentence-transformers CrossEncoder
- ✅ No GPU required (ONNX CPU runtime)
- ✅ int8 quantization with negligible quality loss
- ⚠️ Slightly lower quality than full-precision model
- ⚠️ Less model choice than sentence-transformers

**When to switch**: If quality is paramount → Cohere rerank API. If multilingual → BGE Reranker v2-m3.

---

## ADR-006: nomic-embed-text for Dense Embeddings

**Date**: Embedding model selection  
**Status**: Accepted

**Context**: Need local, private, high-quality dense embedding model.

**Decision**: `nomic-embed-text` via Ollama. 768-dim, Apache 2.0, 274MB.

**Consequences**:
- ✅ Completely local, no API cost, no data leaving machine
- ✅ Best quality-per-size among local models
- ✅ 768-dim is sufficient for production (smaller than OpenAI's 1536)
- ⚠️ Slightly lower quality than text-embedding-3-small
- ⚠️ Requires Ollama running

**When to switch**: If quality is inadequate after evaluation → text-embedding-3-small (requires re-index and API costs).

---

## ADR-007: Soft Delete Before Hard Delete, Always

**Date**: Lifecycle design  
**Status**: Accepted

**Context**: Need to handle document deletion safely without data loss.

**Decision**: Two-phase deletion. Soft delete = mark `deleted_at`, remove from Qdrant, flush cache. Hard delete = only allowed after soft delete, removes PG rows.

**Consequences**:
- ✅ 30-day recovery window for accidental deletions (POST /restore)
- ✅ GDPR compliance: hard delete removes all PII on request
- ✅ Audit trail preserved even after hard delete (deletion_audit table)
- ⚠️ Disk usage doesn't drop until hard delete

---

## ADR-008: gRPC for Qdrant in Production

**Date**: Qdrant client configuration  
**Status**: Accepted

**Context**: Qdrant supports both HTTP REST (port 6333) and gRPC (port 6334).

**Decision**: Use gRPC (`prefer_grpc=True`) for all production operations.

**Consequences**:
- ✅ ~30% faster for bulk upserts and batch queries
- ✅ Binary protocol reduces payload size
- ✅ Better streaming for large result sets
- ⚠️ gRPC port must be exposed in docker-compose.yml
- ⚠️ Debugging tools (curl, browser) only work with HTTP

**Use HTTP for**: health checks, Qdrant dashboard, debugging.

---

## ADR-009: Binary Quantization for Qdrant Dense Vectors

**Date**: Qdrant collection design  
**Status**: Accepted

**Context**: 768-dim float32 vectors consume 3KB per point. At 1M chunks = 3GB for vectors alone.

**Decision**: Binary quantization with `always_ram=True` and `rescore=True`.

**Consequences**:
- ✅ 32× memory reduction: 3KB → 96 bytes per vector
- ✅ 15-40× faster ANN search (popcount vs float multiply)
- ✅ On 64GB machine: can hold ~500M vectors in RAM
- ✅ `rescore=True` recovers recall loss to < 0.5%
- ⚠️ Requires Qdrant 1.7+ for binary quantization
- ⚠️ Slight quality degradation without rescore

---

## ADR-010: SPLADE via FastEmbed for Sparse Vectors

**Date**: Sparse embedding selection  
**Status**: Accepted

**Context**: Need sparse embeddings for hybrid search. Options: BM25 (keyword matching) or SPLADE (contextual sparse).

**Decision**: SPLADE via FastEmbed (`prithivida/Splade_PP_en_v1`). Local, ONNX, no GPU.

**Consequences**:
- ✅ SPLADE expands query terms contextually (e.g., "parental leave" → also scores "maternity", "FMLA")
- ✅ Better recall than BM25 for domain-specific terms
- ✅ Runs locally via ONNX, first call downloads ~400MB model
- ✅ FastEmbed is Qdrant's own library — best integration
- ⚠️ ~3× slower than BM25 for sparse encoding
- ⚠️ First call takes time for model download

**When to switch**: If SPLADE quality doesn't improve fusion_gain > 0.05 → try BM25 via `rank_bm25` library.
