# Glossary — Production RAG System

> Quick reference for all terms, acronyms, and concepts used in this project.
> Theory section references point to `docs/RAG_THEORY_COMPLETE.md`.

---

## Core RAG Terms

| Term | Definition | Theory § |
|---|---|---|
| **RAG** | Retrieval-Augmented Generation. LLM answers grounded in retrieved documents rather than parametric knowledge. | §6 |
| **Chunk** | A semantically coherent unit of text extracted from a document. The atomic unit of retrieval. 256-512 tokens in this system. | §9 |
| **Embedding** | A dense vector (list of floats) representing the semantic meaning of text. Similar meaning = vectors close in space. | §3 |
| **Dense vector** | A high-dimensional float vector where most values are non-zero. Encodes semantic meaning. 768-dim in this system. | §14 |
| **Sparse vector** | A vector where most values are zero. Encodes keyword/term weights. Used for BM25/SPLADE. | §13 |
| **Context window** | Maximum tokens a model can process. Everything the LLM "sees" must fit here. | §5 |
| **Parametric knowledge** | Information baked into LLM weights during training. Can hallucinate. Cannot be updated without retraining. | §5 |
| **Contextual knowledge** | Information provided at inference time via RAG. Accurate to source, updatable. | §5 |
| **Hallucination** | LLM generating plausible-sounding but unsupported or incorrect information. | §25 |

---

## Retrieval Terms

| Term | Definition | Theory § |
|---|---|---|
| **ANN** | Approximate Nearest Neighbor. Fast vector search that finds "close enough" neighbors. | §11 |
| **HNSW** | Hierarchical Navigable Small World. The ANN algorithm used by Qdrant and pgvector. | §11 |
| **IVF** | Inverted File Index. Alternative ANN algorithm. Requires rebuild on new inserts. | §11 |
| **BM25** | Best Match 25. Classic keyword ranking algorithm with TF saturation and length normalization. | §13 |
| **SPLADE** | Sparse Lexical and Dense Expansion. Neural sparse embeddings that expand terms contextually. Better than BM25. | §13 |
| **Hybrid search** | Combining dense (semantic) + sparse (keyword) search with RRF fusion. | §15 |
| **RRF** | Reciprocal Rank Fusion. Rank-based score fusion: `1/(k + rank)`. k=60 is standard. | §15 |
| **fusion_gain** | `hybrid_recall - max(dense_recall, sparse_recall)`. Measures sparse contribution. Target: > 0.05 | §27 |
| **HyDE** | Hypothetical Document Embeddings. Generate hypothetical answer → embed it → search with that vector. | §17 |
| **Reverse HyDE** | Generate hypothetical questions per chunk at index time. Improves recall at zero query-time cost. | §17 |
| **Prefetch** | Qdrant concept: retrieve large N candidates before fusion. N=50 in this system. | §11 |
| **Pre-filter** | Apply metadata filter BEFORE ANN search (not after). Critical for recall on filtered queries. | §18 |

---

## Reranking Terms

| Term | Definition | Theory § |
|---|---|---|
| **Biencoder** | Encode query and document separately → compare vectors. Used for first-stage retrieval. Fast but lossy. | §16 |
| **Cross-encoder** | Encode query + document together → single relevance score. More accurate, slower. Used for reranking. | §16 |
| **Two-stage pipeline** | Stage 1: biencoder retrieves N candidates. Stage 2: cross-encoder reranks to K. | §16 |
| **N >> K ratio** | Reranker candidate count (N) must far exceed final count (K). Default 10:1 (N=50, K=5). | §16 |
| **FlashRank** | Quantized ONNX cross-encoder. Fastest local reranker. ms-marco-MiniLM-L-12-v2. | §16 |
| **ColBERT** | Late interaction model. MaxSim: per-token query-document matching. Highest recall. | §16 |
| **MMR** | Maximal Marginal Relevance. Selects diverse + relevant chunks. λ balances relevance vs diversity. | §16 |
| **Rule-based reranking** | Deterministic score multipliers: recency decay, authority weight, keyword boost. Sub-millisecond. | §16 |
| **Lost-in-the-middle** | LLMs attend more to context beginning and end. Best chunk → position 0, 2nd-best → position N-1. | §5 |
| **NDCG@k** | Normalised Discounted Cumulative Gain. Measures ranking quality. Target > 0.80. | §16 |
| **MRR** | Mean Reciprocal Rank. Measures rank of first relevant result. Target > 0.60 after reranker. | §16 |

---

## Agent & Reasoning Terms

| Term | Definition | Theory § |
|---|---|---|
| **LangGraph** | Stateful agent graph framework. Nodes = functions, edges = transitions, state = TypedDict. | §23 |
| **RAGState** | The TypedDict that flows through all LangGraph nodes. Single source of truth for a query. | §23 |
| **ReAct** | Reason + Act. Agent loop: Thought → Action → Observation → repeat. | §22 |
| **Multi-hop reasoning** | Query requires multiple retrieval rounds, each informed by the previous. | §21 |
| **Conditional Router** | LangGraph conditional edge function. Returns string key to select next node. | §23 |
| **Checkpointing** | LangGraph saves state after each node. Enables resume, debugging, human-in-the-loop. | §23 |
| **Tool calling** | Structured LLM output: LLM requests a function call → system executes → returns result to LLM. | §24 |

---

## Validation Terms

| Term | Definition | Theory § |
|---|---|---|
| **Gatekeeper** | Validator 1. Checks: "Does this response address the question?" Score threshold: 0.70. | §26 |
| **Auditor** | Validator 2. Checks: "Is every claim grounded in retrieved context?" Score threshold: 0.75. | §26 |
| **Strategist** | Validator 3. Checks: "Does this make domain sense?" Domain-specific rules. | §26 |
| **Grounding score** | Auditor output. Fraction of claims directly supported by context. 0.0-1.0. | §26 |
| **Ungrounded claim** | A claim in the response not traceable to any retrieved chunk. Potential hallucination. | §26 |
| **Intrinsic hallucination** | Model contradicts the retrieved context. Type 1. | §25 |
| **Extrinsic hallucination** | Model adds information not in context. Type 2. Most common. | §25 |
| **Faithfulness** | RAGAS metric. `supported_claims / total_claims`. Most important RAG metric. Target > 0.80. | §27 |

---

## Lifecycle Terms

| Term | Definition |
|---|---|
| **Soft delete** | Mark `deleted_at` in PG + remove Qdrant points + flush cache. Document immediately invisible to search. Text kept for audit. |
| **Hard delete** | Remove all PG rows + MinIO file. Only allowed after soft delete. Satisfies GDPR erasure. |
| **Version bump** | Retire old document embeddings, ingest new file as version+1. Old version invisible to search. |
| **Partial update** | Re-embed only chunks on specific pages. Rest of document unchanged. |
| **Metadata update** | Change department/access_level with no re-embedding. PG UPDATE + Qdrant set_payload(). |
| **Blue-green reindex** | Create new Qdrant collection with new model. Serve queries from old collection until cutover. |
| **Cutover** | Atomic switch from old Qdrant collection to new. Config change only. |
| **Reconciler** | Nightly job that finds PG chunks with `qdrant_synced=FALSE` and re-uploads to Qdrant. |
| **qdrant_synced** | Boolean column on chunks table. FALSE = needs to be uploaded to Qdrant. |

---

## Caching Terms

| Term | Definition | Theory § |
|---|---|---|
| **Semantic cache** | Cache keyed by query meaning (not exact string). Uses vector similarity for lookup. | §32 |
| **Exact cache (L1)** | SHA256 hash of normalized query → O(1) lookup. Catches identical queries. | §32 |
| **Semantic cache (L2)** | HNSW vector search on Redis. Catches paraphrases with cosine_sim > threshold. | §32 |
| **Similarity threshold** | Minimum cosine similarity for a cache hit. Default 0.92. Lower = more hits, more false positives. | §32 |
| **TTL** | Time-to-live. Factual: 86400s (24h). Temporal: 3600s (1h). Personal: 0 (bypass). | §32 |
| **CacheDecision.BYPASS** | Do not check or store in cache. Used for personal, adversarial, or low-confidence queries. | §32 |
| **Cache invalidation** | Remove stale cache entries. Triggered by: doc soft-delete, version bump, TTL expiry. | §32 |
| **LRU eviction** | Least Recently Used. Redis auto-evicts old entries when maxmemory reached. | §32 |
| **Hit rate** | Fraction of queries served from cache. Target: > 15%. Measured via `cache_hit_rate_daily` view. | §32 |

---

## Evaluation Terms

| Term | Definition | Theory § |
|---|---|---|
| **Golden dataset** | Hand-crafted Q&A pairs with ground truth answers. Used to measure system quality. 30 pairs in this system. | §27 |
| **RAGAS** | RAG Assessment framework. Measures faithfulness, answer_relevancy, context_recall, context_precision. | §27 |
| **Faithfulness** | RAGAS. supported_claims / total_claims. Target > 0.80. | §27 |
| **Answer relevancy** | RAGAS. Does the answer address the question? Target > 0.70. | §27 |
| **Context recall** | RAGAS. Did we retrieve all relevant chunks? Target > 0.75. | §27 |
| **Context precision** | RAGAS. Were retrieved chunks relevant? Target > 0.75. | §27 |
| **Retrieval precision@k** | Fraction of top-k retrieved chunks that are relevant. | §27 |
| **Retrieval recall@k** | Fraction of all relevant chunks that were retrieved in top-k. | §27 |
| **fusion_gain** | hybrid_recall - max(dense_recall, sparse_recall). Sparse contribution measure. Target > 0.05. | §27 |
| **p50 / p95** | 50th and 95th percentile latency. p95 = latency that 95% of queries are faster than. | §31 |

---

## Infrastructure Terms

| Term | Definition |
|---|---|
| **chunk_id** | UUID primary key of a chunk. Same value in PostgreSQL `chunks.chunk_id` and Qdrant point ID. The join key. |
| **doc_id** | UUID primary key of a document. Stored in PG `documents` table and Qdrant payload. |
| **is_latest** | Boolean on documents table and Qdrant payload. FALSE = superseded or deleted. Search always filters is_latest=TRUE. |
| **Celery** | Async task queue. Ingestion runs as Celery tasks on Redis DB 0. |
| **Flower** | Celery monitoring dashboard. http://localhost:5555 |
| **RedisInsight** | Redis GUI dashboard. http://localhost:8001. Shows cache keys, memory, search indexes. |
| **MinIO** | S3-compatible object storage for raw uploaded files. http://localhost:9000 |
| **Alembic** | PostgreSQL schema migration tool. Run: `python -m alembic upgrade head` |
| **asyncpg** | Async PostgreSQL driver. Used for hot-path queries (hydration). Fastest available. |
| **redisvl** | Redis Vector Library. Provides `SemanticCache` with HNSW indexing on Redis. |
| **fastembed** | Qdrant's fast local embedding library. Used for SPLADE sparse embeddings. ONNX-optimized. |
