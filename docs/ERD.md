# Entity Relationship Diagram — Production RAG System

## Schema Overview

```
8 tables · 3 foreign key relationships · 1 view
PostgreSQL 16 · uuid-ossp extension · pg_trgm extension
```

---

## ER Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CORE DATA TABLES                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────┐          ┌──────────────────────────────────┐
│          documents           │          │              chunks               │
├──────────────────────────────┤          ├──────────────────────────────────┤
│ PK  doc_id         UUID      │1        ∞│ PK  chunk_id       UUID          │
│     filename       TEXT      │──────────│ FK  doc_id         UUID    ──┐   │
│     minio_path     TEXT      │          │     chunk_index    INT         │   │
│     doc_type       TEXT      │          │     chunk_text     TEXT        │   │
│     department     TEXT      │          │     chunk_type     TEXT        │   │
│     version        INT       │          │     section_heading TEXT       │   │
│     is_latest      BOOL      │          │     page_number    INT         │   │
│     access_level   TEXT      │          │     token_count    INT         │   │
│     created_at     TIMESTAMPTZ│         │     summary        TEXT        │   │
│     updated_at     TIMESTAMPTZ│         │     keywords       TEXT[]      │   │
│     deleted_at     TIMESTAMPTZ│         │     hypothetical_qs TEXT[]     │   │
│     deletion_reason TEXT     │          │     tsv            TSVECTOR    │   │
│     checksum       TEXT      │          │     qdrant_synced  BOOL        │   │
│     ingestion_status TEXT    │          │     qdrant_synced_at TIMESTAMPTZ│  │
│     chunk_count    INT       │          │     created_at     TIMESTAMPTZ │   │
└──────────────────────────────┘          └──────────────────────────────────┘
         │                                          │
         │  ON DELETE CASCADE                       │
         │  (delete doc → all chunks deleted)       │
         └──────────────────────────────────────────┘

         NOTE: chunk_id = Qdrant point ID (same UUID in both systems)


┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE TABLES                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────┐
│           queries            │
├──────────────────────────────┤
│ PK  query_id       UUID      │
│     user_id        TEXT      │
│     raw_query      TEXT      │
│     query_intent   TEXT      │  ← factual | analytical | comparative
│     complexity     TEXT      │  ← simple | complex
│     execution_plan JSONB     │  ← LangGraph routing decisions
│     retrieved_chunk_ids UUID[]│ ← parallel array: Qdrant result IDs
│     qdrant_scores  NUMERIC[] │  ← parallel array: RRF scores
│     final_response TEXT      │
│     response_time_ms INT     │
│     token_count_in  INT      │
│     token_count_out INT      │
│     validation_passed BOOL   │
│     retry_count    INT       │
│     created_at     TIMESTAMPTZ│
└──────────────────────────────┘
        │1                │1
        │                 │
        │∞                │∞
        ▼                 ▼
┌───────────────────┐  ┌──────────────────────┐
│    validations    │  │    cache_metrics      │
├───────────────────┤  ├──────────────────────┤
│ PK validation_id  │  │ PK metric_id  UUID    │
│ FK query_id  UUID │  │ FK query_id   UUID    │
│    validator_type │  │    cache_hit  BOOL    │
│      CHECK IN:    │  │    cache_level TEXT   │
│      gatekeeper   │  │      exact            │
│      auditor      │  │      semantic         │
│      strategist   │  │      miss             │
│    passed  BOOL   │  │    similarity NUMERIC │
│    score   NUM    │  │    latency_ms INT     │
│    reasoning TEXT │  │    created_at TS      │
│    latency_ms INT │  └──────────────────────┘
│    created_at TS  │           │
└───────────────────┘           │
                                │ feeds
                                ▼
                    ┌──────────────────────────┐
                    │   cache_hit_rate_daily   │
                    │         (VIEW)            │
                    ├──────────────────────────┤
                    │  date                     │
                    │  total_queries            │
                    │  cache_hits               │
                    │  hit_rate_pct             │
                    │  avg_hit_latency_ms       │
                    │  avg_miss_latency_ms      │
                    └──────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    STANDALONE OPERATIONAL TABLES                              │
│               (no foreign keys — record system-level events)                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐  ┌───────────────────────┐  ┌───────────────────┐
│       evaluations        │  │    deletion_audit      │  │  qdrant_sync_log  │
├──────────────────────────┤  ├───────────────────────┤  ├───────────────────┤
│ PK eval_id       UUID    │  │ PK audit_id    UUID    │  │ PK sync_id  UUID  │
│    run_name      TEXT    │  │    doc_id      UUID    │  │    run_at   TS     │
│    faithfulness  NUM     │  │    deleted_by  TEXT    │  │    chunks_checked  │
│    answer_relevancy NUM  │  │    reason      TEXT    │  │    chunks_missing  │
│    context_recall   NUM  │  │    deleted_at  TS      │  │    chunks_resynced│
│    context_precision NUM │  │    hard_       TS      │  │    status   TEXT  │
│    hybrid_recall_at_k NUM│  │    deleted_at          │  │    error_details  │
│    fusion_gain   NUM     │  └───────────────────────┘  └───────────────────┘
│    avg_latency_ms INT    │
│    p95_latency_ms INT    │  NOTE: doc_id has NO foreign key
│    sample_count   INT    │  intentionally — audit row must survive
│    created_at     TS     │  after the documents row is hard-deleted
└──────────────────────────┘
```

---

## Relationships

| From | Cardinality | To | Constraint | Meaning |
|---|---|---|---|---|
| `documents` | 1 → ∞ | `chunks` | `ON DELETE CASCADE` | One document has many chunks. Delete the document → all its chunks delete automatically. |
| `queries` | 1 → ∞ | `validations` | FK, no cascade | One query has up to 3 validation rows (gatekeeper, auditor, strategist). |
| `queries` | 1 → 1 | `cache_metrics` | FK, no cascade | Every query produces one cache_metrics row (hit or miss). |
| `evaluations` | standalone | — | none | RAGAS run results. Not tied to individual queries. |
| `deletion_audit` | standalone | — | none intentionally | Audit trail must survive after `documents` row is hard-deleted. |
| `qdrant_sync_log` | standalone | — | none | Reconciler run logs. System-level event, not per-document. |

---

## Crow's Foot Notation Key

```
||      exactly one (1)
|o      zero or one (0..1)
o{      zero or many (0..∞)
|{      one or many (1..∞)

documents ||--|{ chunks        → one document has one or many chunks
queries   ||--o{ validations   → one query has zero or many validations
queries   ||--o{ cache_metrics → one query has zero or many cache_metrics rows
```

---

## Table Reference

### `documents` — source of truth for every uploaded file

| Column | Type | Constraints | Purpose |
|---|---|---|---|
| `doc_id` | UUID | PK, default uuid | Primary key. Referenced by chunks. |
| `filename` | TEXT | NOT NULL | Original filename as uploaded. |
| `minio_path` | TEXT | NOT NULL | Object path in MinIO (`rag-documents/uuid/file.pdf`). |
| `doc_type` | TEXT | NOT NULL | `pdf` \| `docx` \| `html` \| `code` |
| `department` | TEXT | nullable | Used as Qdrant payload pre-filter. e.g. `hr`, `engineering`. |
| `version` | INT | NOT NULL, default 1 | Increments on each version bump. |
| `is_latest` | BOOL | NOT NULL, default TRUE | FALSE = superseded or soft-deleted. All queries filter `is_latest=TRUE`. |
| `access_level` | TEXT | NOT NULL, default `internal` | `public` \| `internal` \| `restricted` |
| `created_at` | TIMESTAMPTZ | NOT NULL, default NOW() | Upload timestamp. |
| `updated_at` | TIMESTAMPTZ | NOT NULL, default NOW() | Last metadata change. |
| `deleted_at` | TIMESTAMPTZ | nullable | NULL=active. Set=soft-deleted. Invisible to search immediately. |
| `deletion_reason` | TEXT | nullable | Why deleted. Stored for audit. |
| `checksum` | TEXT | NOT NULL, UNIQUE(checksum,version) | SHA256 of file. Prevents duplicate uploads. |
| `ingestion_status` | TEXT | NOT NULL, default `pending` | `pending` → `processing` → `done` \| `failed` |
| `chunk_count` | INT | nullable | Filled after ingestion completes. |

**Indexes:**
```sql
idx_documents_latest  ON documents(is_latest, doc_type, department)
idx_documents_status  ON documents(ingestion_status)
idx_documents_dept    ON documents(department) WHERE deleted_at IS NULL
```

---

### `chunks` — every text chunk, join key to Qdrant

| Column | Type | Constraints | Purpose |
|---|---|---|---|
| `chunk_id` | UUID | PK, default uuid | Also the Qdrant point ID. Same UUID in both systems. |
| `doc_id` | UUID | FK → documents, CASCADE | Which document this chunk came from. |
| `chunk_index` | INT | NOT NULL | Position within the document (0-based). |
| `chunk_text` | TEXT | NOT NULL | The actual text. Source of truth. Used for hydration. |
| `chunk_type` | TEXT | NOT NULL | `paragraph` \| `table` \| `code` \| `heading` |
| `section_heading` | TEXT | nullable | Nearest ancestor heading. Prepended to chunk text for context. |
| `page_number` | INT | nullable | Page in source document. Shown in citations. |
| `token_count` | INT | NOT NULL | tiktoken cl100k_base count. Always 256–512 tokens. |
| `summary` | TEXT | nullable | LLM-generated summary of the chunk. Boosts FTS quality. |
| `keywords` | TEXT[] | nullable | LLM-extracted keywords. Boosts FTS quality. |
| `hypothetical_qs` | TEXT[] | nullable | HyDE questions this chunk answers. For reverse-HyDE retrieval. |
| `tsv` | TSVECTOR | GENERATED STORED | Auto-computed from chunk_text + summary + keywords. FTS fallback. |
| `qdrant_synced` | BOOL | NOT NULL, default FALSE | FALSE = not yet in Qdrant. Reconciler fixes these. |
| `qdrant_synced_at` | TIMESTAMPTZ | nullable | When Qdrant write succeeded. |
| `created_at` | TIMESTAMPTZ | NOT NULL, default NOW() | Chunk creation timestamp. |

**Indexes:**
```sql
idx_chunks_doc        ON chunks(doc_id)
idx_chunks_fts        ON chunks USING GIN(tsv)         -- full-text search
idx_chunks_unsynced   ON chunks(qdrant_synced) WHERE qdrant_synced = FALSE
```

---

### `queries` — audit trail for every user query

| Column | Type | Purpose |
|---|---|---|
| `query_id` | UUID PK | Primary key. Referenced by validations and cache_metrics. |
| `user_id` | TEXT | Optional. For user-scoped query history. |
| `raw_query` | TEXT | Exact query string as submitted. |
| `query_intent` | TEXT | `factual` \| `analytical` \| `comparative` \| `procedural` — from query analyzer. |
| `complexity` | TEXT | `simple` \| `complex` — determines LangGraph routing. |
| `execution_plan` | JSONB | Full LangGraph decision log: intent, route taken, nodes executed. |
| `retrieved_chunk_ids` | UUID[] | Parallel array. `[0]` = highest-ranked chunk. Matches `qdrant_scores[i]`. |
| `qdrant_scores` | NUMERIC[] | Parallel array. RRF scores for each retrieved chunk. |
| `final_response` | TEXT | What was returned to the user. |
| `response_time_ms` | INT | End-to-end wall clock (cache check + retrieval + generation + validation). |
| `token_count_in` | INT | Tokens in the prompt sent to LLM. |
| `token_count_out` | INT | Tokens in the LLM response. |
| `validation_passed` | BOOL | Summary verdict. Detail is in `validations` table. |
| `retry_count` | INT | How many times LangGraph replanned (max 2). |
| `created_at` | TIMESTAMPTZ | Query timestamp. |

**Indexes:**
```sql
idx_queries_time  ON queries(created_at DESC)
idx_queries_user  ON queries(user_id) WHERE user_id IS NOT NULL
```

---

### `validations` — per-validator scores for each query

| Column | Type | Purpose |
|---|---|---|
| `validation_id` | UUID PK | Primary key. |
| `query_id` | UUID FK | Links to `queries`. One query → up to 3 rows. |
| `validator_type` | TEXT | `gatekeeper` \| `auditor` \| `strategist` (enforced by CHECK). |
| `passed` | BOOL | TRUE = validator approved the response. |
| `score` | NUMERIC(4,3) | 0.000–1.000. Threshold: gatekeeper ≥ 0.70, auditor ≥ 0.75. |
| `reasoning` | TEXT | LLM explanation for its score. Useful for debugging. |
| `latency_ms` | INT | How long this validator took (runs in parallel with others). |
| `created_at` | TIMESTAMPTZ | When validation ran. |

---

### `evaluations` — RAGAS evaluation run results

| Column | Type | Purpose |
|---|---|---|
| `eval_id` | UUID PK | Primary key. |
| `run_name` | TEXT | e.g. `baseline-v0.4`, `after-reranker`. |
| `faithfulness` | NUMERIC(4,3) | Target ≥ 0.80. `supported_claims / total_claims`. |
| `answer_relevancy` | NUMERIC(4,3) | Target ≥ 0.70. Does the answer address the question? |
| `context_recall` | NUMERIC(4,3) | Target ≥ 0.75. Were all relevant chunks retrieved? |
| `context_precision` | NUMERIC(4,3) | Target ≥ 0.75. Were retrieved chunks actually useful? |
| `hybrid_recall_at_k` | NUMERIC(4,3) | Recall of hybrid search (dense+sparse+RRF) at top-k. |
| `fusion_gain` | NUMERIC(4,3) | Target ≥ 0.05. `hybrid_recall - max(dense_recall, sparse_recall)`. |
| `avg_latency_ms` | INT | Mean end-to-end latency across evaluation queries. |
| `p95_latency_ms` | INT | 95th percentile latency. Target ≤ 3500ms. |
| `sample_count` | INT | Number of Q&A pairs in this evaluation run. |
| `created_at` | TIMESTAMPTZ | When the evaluation ran. |

---

### `cache_metrics` — semantic cache hit/miss tracking

| Column | Type | Purpose |
|---|---|---|
| `metric_id` | UUID PK | Primary key. |
| `query_id` | UUID FK | Links to `queries`. |
| `cache_hit` | BOOL | TRUE = served from cache, FALSE = full pipeline ran. |
| `cache_level` | TEXT | `exact` (SHA256 match) \| `semantic` (HNSW match) \| `miss` |
| `similarity` | NUMERIC(5,4) | Cosine similarity for semantic hits. e.g. `0.9421` |
| `latency_ms` | INT | Response time. Cache hits: ~75ms. Cache misses: ~1700ms. |
| `created_at` | TIMESTAMPTZ | Query timestamp. |

**Indexes:**
```sql
idx_cache_metrics_date  ON cache_metrics(created_at DESC)
idx_cache_metrics_hit   ON cache_metrics(cache_hit)
```

---

### `deletion_audit` — GDPR permanent erasure record

| Column | Type | Purpose |
|---|---|---|
| `audit_id` | UUID PK | Primary key. |
| `doc_id` | UUID | Document that was deleted. No FK — must survive after doc is gone. |
| `deleted_by` | TEXT | Who requested the deletion. `admin@company.com` |
| `reason` | TEXT | Why. e.g. `GDPR Art.17 erasure request`, `superseded by v2` |
| `deleted_at` | TIMESTAMPTZ | When soft delete happened. |
| `hard_deleted_at` | TIMESTAMPTZ | When all data was permanently removed. GDPR proof timestamp. |

---

### `qdrant_sync_log` — reconciler run history

| Column | Type | Purpose |
|---|---|---|
| `sync_id` | UUID PK | Primary key. |
| `run_at` | TIMESTAMPTZ | When the reconciler ran. |
| `chunks_checked` | INT | Total chunks examined in PostgreSQL. |
| `chunks_missing` | INT | Chunks in PG with `qdrant_synced=FALSE`. Target: 0. |
| `chunks_resynced` | INT | Chunks successfully re-uploaded to Qdrant in this run. |
| `status` | TEXT | `ok` \| `partial` \| `failed` |
| `error_details` | TEXT | Error message if `status=failed`. |

---

## View: `cache_hit_rate_daily`

Built from `cache_metrics`. Query this daily to monitor semantic cache health:

```sql
SELECT * FROM cache_hit_rate_daily LIMIT 7;
```

| Column | Type | Description |
|---|---|---|
| `date` | DATE | Calendar date. |
| `total_queries` | BIGINT | Total queries that day. |
| `cache_hits` | BIGINT | Queries served from cache. |
| `hit_rate_pct` | NUMERIC | `cache_hits / total_queries × 100`. Target ≥ 15%. |
| `avg_hit_latency_ms` | NUMERIC | Average latency for cache hits. Target ≤ 120ms. |
| `avg_miss_latency_ms` | NUMERIC | Average latency for cache misses. Target ≤ 3500ms. |

---

## Useful Queries

```sql
-- How many documents and chunks are indexed?
SELECT COUNT(*) AS docs   FROM documents  WHERE is_latest = TRUE;
SELECT COUNT(*) AS chunks FROM chunks     WHERE doc_id IN (
    SELECT doc_id FROM documents WHERE is_latest = TRUE
);

-- Which chunks are not in Qdrant yet?
SELECT COUNT(*) FROM chunks WHERE qdrant_synced = FALSE;

-- Cache performance this week
SELECT * FROM cache_hit_rate_daily LIMIT 7;

-- Latest RAGAS scores
SELECT run_name, faithfulness, context_precision, fusion_gain, created_at
FROM evaluations
ORDER BY created_at DESC LIMIT 5;

-- Most retried queries (high retry_count = validation struggling)
SELECT raw_query, retry_count, validation_passed, response_time_ms
FROM queries
WHERE retry_count > 0
ORDER BY created_at DESC LIMIT 20;

-- Which validators are failing most?
SELECT validator_type,
       COUNT(*) FILTER (WHERE NOT passed) AS failures,
       COUNT(*) AS total,
       ROUND(AVG(score), 3) AS avg_score
FROM validations
GROUP BY validator_type
ORDER BY failures DESC;

-- Documents pending ingestion
SELECT filename, ingestion_status, created_at
FROM documents
WHERE ingestion_status != 'done'
ORDER BY created_at DESC;
```