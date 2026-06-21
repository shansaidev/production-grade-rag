-- =============================================================================
-- 01_schema.sql — Production RAG System — PostgreSQL Schema
-- =============================================================================
-- This file runs AUTOMATICALLY when the postgres container starts for the
-- first time. It only runs once — if the data volume already exists, it skips.
--
-- How it works:
--   postgres:16-alpine runs every *.sql file in /docker-entrypoint-initdb.d/
--   in alphabetical order, once, on first container creation only.
--
-- To force a re-run: docker compose down -v && docker compose up -d
-- =============================================================================

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- uuid_generate_v4()
CREATE EXTENSION IF NOT EXISTS "pg_trgm";     -- trigram index for text search

-- ── documents ─────────────────────────────────────────────────────────────────
-- Every uploaded file. Source of truth for metadata and versions.
-- Vectors are NOT stored here — they live in Qdrant.
-- is_latest=FALSE means the document is superseded or soft-deleted.
CREATE TABLE documents (
    doc_id           UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename         TEXT        NOT NULL,
    minio_path       TEXT        NOT NULL,
    doc_type         TEXT        NOT NULL,        -- pdf | docx | html | code
    department       TEXT,
    version          INTEGER     NOT NULL DEFAULT 1,
    is_latest        BOOLEAN     NOT NULL DEFAULT TRUE,
    access_level     TEXT        NOT NULL DEFAULT 'internal',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at       TIMESTAMPTZ,                 -- NULL=active, set=soft-deleted
    deletion_reason  TEXT,
    checksum         TEXT        NOT NULL,        -- SHA256 of file, for dedup
    ingestion_status TEXT        NOT NULL DEFAULT 'pending',
    chunk_count      INTEGER,
    UNIQUE(checksum, version)
);

CREATE INDEX idx_documents_latest ON documents(is_latest, doc_type, department);
CREATE INDEX idx_documents_status ON documents(ingestion_status);
CREATE INDEX idx_documents_dept   ON documents(department) WHERE deleted_at IS NULL;

-- ── chunks ────────────────────────────────────────────────────────────────────
-- Every text chunk extracted from a document.
-- chunk_id is the Qdrant point ID — same UUID in both PostgreSQL and Qdrant.
-- No vector column here. chunk_text is the source of truth for text.

CREATE TABLE chunks (
    chunk_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    doc_id           UUID NOT NULL
        REFERENCES documents(doc_id) ON DELETE CASCADE,

    chunk_index      INTEGER NOT NULL,
    chunk_text       TEXT NOT NULL,

    chunk_type       TEXT NOT NULL,   -- paragraph | table | code | heading
    section_heading  TEXT,
    page_number      INTEGER,

    token_count      INTEGER NOT NULL,

    summary          TEXT,
    keywords         TEXT[],
    hypothetical_qs  TEXT[],

    -- Full-text search column (manual maintenance or app-side update)
    tsv              TSVECTOR,

    qdrant_synced    BOOLEAN NOT NULL DEFAULT FALSE,
    qdrant_synced_at TIMESTAMPTZ,

    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chunks_doc ON chunks(doc_id);

-- Full-text search index (now valid)
CREATE INDEX idx_chunks_fts ON chunks USING GIN(tsv);

CREATE INDEX idx_chunks_unsynced ON chunks(qdrant_synced)    WHERE qdrant_synced = FALSE;

-- tsv enables full-text search fallback when Qdrant is unavailable.
-- CREATE TABLE chunks (
--     chunk_id         UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
--     doc_id           UUID        NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
--     chunk_index      INTEGER     NOT NULL,
--     chunk_text       TEXT        NOT NULL,
--     chunk_type       TEXT        NOT NULL,        -- paragraph | table | code | heading
--     section_heading  TEXT,
--     page_number      INTEGER,
--     token_count      INTEGER     NOT NULL,
--     summary          TEXT,
--     keywords         TEXT[],
--     hypothetical_qs  TEXT[],
--     tsv TSVECTOR DEFAULT ''::tsvector,
--     -- tsv TSVECTOR GENERATED ALWAYS AS (
--     --     to_tsvector(
--     --         'english'::regconfig,
--     --         coalesce(chunk_text, '') || ' ' ||
--     --         coalesce(summary, '') || ' ' ||
--     --         coalesce(array_to_string(keywords, ' '), '')
--     --     )
--     -- ) STORED,
--     qdrant_synced    BOOLEAN     NOT NULL DEFAULT FALSE,
--     qdrant_synced_at TIMESTAMPTZ,
--     created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
-- );

-- CREATE INDEX idx_chunks_doc      ON chunks(doc_id);
-- -- CREATE INDEX idx_chunks_fts      ON chunks USING GIN(tsv);
-- CREATE INDEX idx_chunks_unsynced ON chunks(qdrant_synced) WHERE qdrant_synced = FALSE;



-- ── queries ───────────────────────────────────────────────────────────────────
-- Audit trail for every query. retrieved_chunk_ids and qdrant_scores
-- are parallel arrays (index 0 of both = top-ranked chunk).
CREATE TABLE queries (
    query_id             UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id              TEXT,
    raw_query            TEXT        NOT NULL,
    query_intent         TEXT,                    -- factual | analytical | comparative
    complexity           TEXT,                    -- simple | complex
    execution_plan       JSONB,
    retrieved_chunk_ids  UUID[],
    qdrant_scores        NUMERIC[],
    final_response       TEXT,
    response_time_ms     INTEGER,
    token_count_in       INTEGER,
    token_count_out      INTEGER,
    validation_passed    BOOLEAN,
    retry_count          INTEGER     DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_queries_time ON queries(created_at DESC);
CREATE INDEX idx_queries_user ON queries(user_id) WHERE user_id IS NOT NULL;

-- ── validations ───────────────────────────────────────────────────────────────
-- One row per validator per query (gatekeeper, auditor, strategist).
CREATE TABLE validations (
    validation_id  UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id       UUID        NOT NULL REFERENCES queries(query_id),
    validator_type TEXT        NOT NULL
                               CHECK (validator_type IN ('gatekeeper','auditor','strategist')),
    passed         BOOLEAN     NOT NULL,
    score          NUMERIC(4,3) CHECK (score BETWEEN 0 AND 1),
    reasoning      TEXT,
    latency_ms     INTEGER,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_validations_query ON validations(query_id);

-- ── evaluations ───────────────────────────────────────────────────────────────
-- One row per RAGAS evaluation run.
-- Query: SELECT * FROM evaluations ORDER BY created_at DESC LIMIT 10;
CREATE TABLE evaluations (
    eval_id            UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_name           TEXT        NOT NULL,
    faithfulness       NUMERIC(4,3),
    answer_relevancy   NUMERIC(4,3),
    context_recall     NUMERIC(4,3),
    context_precision  NUMERIC(4,3),
    hybrid_recall_at_k NUMERIC(4,3),
    fusion_gain        NUMERIC(4,3),
    avg_latency_ms     INTEGER,
    p95_latency_ms     INTEGER,
    sample_count       INTEGER,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── cache_metrics ─────────────────────────────────────────────────────────────
-- One row per query — tracks semantic cache hits and misses.
-- Query: SELECT * FROM cache_hit_rate_daily LIMIT 7;
CREATE TABLE cache_metrics (
    metric_id   UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id    UUID        REFERENCES queries(query_id),
    cache_hit   BOOLEAN     NOT NULL,
    cache_level TEXT,                            -- exact | semantic | miss
    similarity  NUMERIC(5,4),                   -- cosine sim for semantic hits
    latency_ms  INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cache_metrics_date ON cache_metrics(created_at DESC);
CREATE INDEX idx_cache_metrics_hit  ON cache_metrics(cache_hit);

-- ── deletion_audit ────────────────────────────────────────────────────────────
-- Permanent record of all deletions. Never purged (7-year retention).
-- hard_deleted_at is the GDPR proof-of-erasure timestamp.
CREATE TABLE deletion_audit (
    audit_id        UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id          UUID        NOT NULL,
    deleted_by      TEXT,
    reason          TEXT,
    deleted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hard_deleted_at TIMESTAMPTZ
);

CREATE INDEX idx_deletion_audit_doc ON deletion_audit(doc_id);

-- ── qdrant_sync_log ───────────────────────────────────────────────────────────
-- One row per reconciler run. Target: chunks_missing = 0.
-- Query: SELECT * FROM qdrant_sync_log ORDER BY run_at DESC LIMIT 10;
CREATE TABLE qdrant_sync_log (
    sync_id         UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    chunks_checked  INTEGER,
    chunks_missing  INTEGER,
    chunks_resynced INTEGER,
    status          TEXT,
    error_details   TEXT
);

-- ── Views ─────────────────────────────────────────────────────────────────────
-- Daily cache hit rate dashboard
-- Query daily: SELECT * FROM cache_hit_rate_daily LIMIT 7;
CREATE VIEW cache_hit_rate_daily AS
SELECT
    DATE(created_at)                                                          AS date,
    COUNT(*)                                                                  AS total_queries,
    COUNT(*) FILTER (WHERE cache_hit)                                         AS cache_hits,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE cache_hit) / NULLIF(COUNT(*), 0), 2
    )                                                                         AS hit_rate_pct,
    ROUND(AVG(latency_ms) FILTER (WHERE cache_hit), 1)                       AS avg_hit_latency_ms,
    ROUND(AVG(latency_ms) FILTER (WHERE NOT cache_hit), 1)                   AS avg_miss_latency_ms
FROM cache_metrics
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- =============================================================================
-- Done. Tables created:
--   documents, chunks, queries, validations, evaluations,
--   cache_metrics, deletion_audit, qdrant_sync_log
-- Views created:
--   cache_hit_rate_daily
-- =============================================================================