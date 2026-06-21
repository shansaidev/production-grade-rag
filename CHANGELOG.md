# Changelog

All notable changes to the Production RAG System are documented here.
Format: [Semantic Versioning](https://semver.org). Dates: YYYY-MM-DD.

---

## [Unreleased]

### In Progress
- CRAG (Corrective RAG) — web search fallback when retrieval confidence is low
- LangSmith tracing integration alongside structlog
- Streaming SSE responses for `/api/v1/query`

---

## [0.4.0] — 2026-06-27 — Reranking Pipeline

### Added
- Three-stage reranking pipeline: FlashRank → Rule-based boost → MMR diversity filter
- `src/retrieval/rule_reranker.py` — recency decay, authority weight, keyword boost
- `src/retrieval/mmr.py` — Maximal Marginal Relevance diversity filtering (λ=0.5 default)
- `src/retrieval/full_reranker.py` — orchestrates all three stages
- N:K ratio startup assertion (prefetch ≥ top_k × 6) in `Settings.check_nk_ratio()`
- NDCG@k and MRR metrics in `src/evaluation/metrics.py`
- `TOP_K_FINAL`, `MMR_LAMBDA`, `RECENCY_WEIGHT` env vars

### Changed
- `TOP_K_FINAL` default: 8 → 5 (enforces 10:1 N:K ratio with prefetch=50)
- `src/retrieval/reranker.py` now outputs 20 candidates (intermediate) not 8 (final)
- Phase 12 in implementation guide now covers full 3-stage pipeline

### Fixed
- Reranker returning too few candidates when N:K ratio was too tight

---

## [0.3.0] — 2026-06-27 — Semantic Caching

### Added
- `src/cache/semantic_cache.py` — two-level cache (exact hash L1 + HNSW vector L2)
- Redis Stack DB1 for semantic cache (separate from Celery on DB0)
- `check_cache` and `store_cache` LangGraph nodes
- Cache invalidation in `soft_delete()` and `version_bump()` lifecycle ops
- `cache_metrics` PostgreSQL table + `cache_hit_rate_daily` view
- `REDIS_CACHE_URL`, `CACHE_SIMILARITY_THRESHOLD`, `CACHE_FACTUAL_TTL`, `CACHE_TEMPORAL_TTL`, `CACHE_MIN_CONFIDENCE` env vars
- RedisInsight dashboard at port 8001
- `docs/PRODUCTION_RAG_QDRANT_HYBRID.md` §17 — full semantic cache design

### Changed
- `docker-compose.yml`: `redis:7-alpine` → `redis/redis-stack:latest`
- LangGraph graph: entry point is now `check_cache` (before `retrieve`)

### Security
- Personal queries (`"my pto"`, `"my leave"`) bypass cache via `CacheDecision.BYPASS`
- Low-confidence responses (< 0.70) never stored in cache
- Adversarial queries bypass cache

---

## [0.2.0] — 2026-06-26 — Embedding Lifecycle

### Added
- Full CRUD lifecycle for embeddings: soft delete, hard delete, version bump, partial update, metadata-only update, blue-green reindex
- `src/db/qdrant/lifecycle.py` — Qdrant-side delete, payload update, upsert
- `src/ingestion/lifecycle_coordinator.py` — coordinates PG + Qdrant + cache
- `deletion_audit`, `qdrant_sync_log`, `reindex_progress` PostgreSQL tables
- `POST /api/v1/documents/{id}/hard-delete` (admin only)
- `POST /api/v1/documents/{id}/restore`
- `POST /api/v1/admin/reindex` — blue-green collection swap
- `GET /api/v1/admin/reindex/status`
- `POST /api/v1/admin/reindex/cutover` and `/rollback`
- `docs/PRODUCTION_RAG_QDRANT_HYBRID.md` §16 — 6 lifecycle workflows

### Changed
- `chunks` table: added `qdrant_synced`, `qdrant_synced_at` columns
- `documents` table: added `deleted_at`, `deletion_reason` columns

---

## [0.1.0] — 2026-06-22 — Initial Production System

### Added
- Full ingestion pipeline: PDF/DOCX/HTML parsers, SmartChunker, MetadataGenerator
- Dual embedding: nomic-embed-text (dense) + FastEmbed SPLADE (sparse)
- Qdrant hybrid search with native RRF fusion
- PostgreSQL hydration (batch SELECT by chunk_id array)
- LangGraph reasoning engine: analyze → retrieve → generate → validate → format
- Three-agent system: Retriever + Reasoner + Verifier
- Three-layer validation: Gatekeeper + Auditor + Strategist
- RAGAS evaluation runner with 30-pair golden dataset
- Red team suite: 11 tests (injection + evasion + bias)
- FastAPI with async ingestion via Celery
- HyDE (Hypothetical Document Embeddings)
- Lost-in-middle context assembly (best chunk first + last)
- structlog JSON logging with `@timed_node` decorator
- Two architecture documents: v1 (pgvector) and v2 (Qdrant hybrid)
- Three research papers: HyDE, RRF, Lost-in-Middle

### Architecture
- PostgreSQL 16 as source of truth (text + metadata + audit)
- Qdrant for vector search (HNSW + binary quantization + SPLADE)
- Redis Stack for Celery (DB0) and semantic cache (DB1)
- MinIO for raw document storage

---

[Unreleased]: https://github.com/yourname/production-rag/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/yourname/production-rag/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yourname/production-rag/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yourname/production-rag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourname/production-rag/releases/tag/v0.1.0
