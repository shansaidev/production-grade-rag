# Monitoring Guide

What to watch, alert thresholds, and where to find each metric.

---

## The Five Signals That Matter

In priority order — if you can only watch five things:

| # | Signal | Target | Source |
|---|---|---|---|
| 1 | RAGAS faithfulness | ≥ 0.80 | Weekly eval run |
| 2 | p95 query latency (miss path) | ≤ 3500ms | `queries` table |
| 3 | Qdrant `qdrant_synced=FALSE` count | = 0 | `chunks` table |
| 4 | Red team CRITICAL pass rate | 100% | Weekly red team |
| 5 | Cache hit rate | ≥ 15% | `cache_hit_rate_daily` view |

---

## PostgreSQL Queries for Key Metrics

```sql
-- Daily latency p50 / p95
SELECT
    DATE(created_at) AS date,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY response_time_ms) AS p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) AS p95_ms,
    COUNT(*) AS query_count,
    AVG(CASE WHEN validation_passed THEN 1.0 ELSE 0 END) AS validation_pass_rate
FROM queries
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Cache hit rate (daily view)
SELECT * FROM cache_hit_rate_daily LIMIT 7;

-- Unsynced chunks (should always be 0)
SELECT COUNT(*) AS unsynced FROM chunks WHERE qdrant_synced = FALSE;

-- Ingestion health (failed jobs in last 24h)
SELECT ingestion_status, COUNT(*) 
FROM documents 
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY ingestion_status;

-- Validation failure rate by type
SELECT validator_type, 
       AVG(CASE WHEN passed THEN 1.0 ELSE 0 END) AS pass_rate,
       COUNT(*) AS total
FROM validations
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY validator_type;

-- Most common failing queries (low confidence)
SELECT raw_query, confidence, retry_count, created_at
FROM queries
WHERE confidence < 0.70
ORDER BY created_at DESC
LIMIT 20;
```

---

## Alert Thresholds

| Metric | Warning | Critical | Action |
|---|---|---|---|
| p95 latency | > 4000ms | > 6000ms | Check LLM latency, reduce top_k |
| p95 cache hit latency | > 200ms | > 500ms | Check Redis memory, HNSW index |
| Validation pass rate | < 85% | < 70% | Check auditor prompt, retrieval quality |
| `qdrant_synced=FALSE` | > 0 for 1h | > 100 | Run reconciler manually |
| Ingestion failure rate | > 5% | > 20% | Check Celery logs, file parsing errors |
| Cache hit rate (daily) | < 10% | < 5% | Check threshold, check Redis connectivity |
| RAGAS faithfulness (weekly) | < 0.77 | < 0.70 | Immediate investigation |
| Redis memory usage | > 70% | > 85% | Increase maxmemory, check key TTLs |

---

## Dashboards

### Qdrant Dashboard (localhost:6333/dashboard)
- Collection health (green/yellow/red)
- Points count (should match PostgreSQL chunks count)
- Segments count (high count = needs optimisation)
- Indexed vectors count

### RedisInsight (localhost:8001)
- DB1 key count (semantic cache entries)
- DB1 memory usage
- Hit/miss ratio (approximate from MONITOR)
- Search index status (llmcache_idx)

### Flower (localhost:5555)
- Active tasks
- Task success/failure rate
- Worker status
- Queue depth (ingestion queue)

---

## Log Analysis

All logs are JSON (structlog). Key event names to search:

```bash
# Tail application logs
journalctl -u rag-api -f | python3 -m json.tool

# High-latency queries
grep '"event": "node_complete"' app.log | \
  python3 -c "import sys,json; [print(json.loads(l)['node'], json.loads(l)['duration_ms']) for l in sys.stdin if json.loads(l).get('duration_ms',0) > 1000]"

# Cache misses (to understand what's not being cached)
grep '"event": "cache_miss"' app.log | tail -50

# Validation failures
grep '"event": "validation_complete"' app.log | \
  python3 -c "import sys,json; [print(json.loads(l)) for l in sys.stdin if not json.loads(l).get('auditor')]"

# Qdrant sync failures
grep '"event": "qdrant_write_failed"' app.log | tail -20
```

### Key Event Names

| Event | Meaning |
|---|---|
| `node_complete` | LangGraph node finished; has `node`, `duration_ms` |
| `cache_hit` | Semantic cache hit; has `level` (exact/semantic), `similarity` |
| `cache_stored` | New entry cached; has `ttl`, `query_preview` |
| `cache_invalidated` | Cache purged for a document; has `doc_id`, `entries_removed` |
| `qdrant_upsert` | Chunk uploaded to Qdrant; has `count`, `status` |
| `qdrant_write_failed` | Qdrant write failed (PG already committed); has `error` |
| `soft_delete` | Document soft deleted; has `doc_id`, `reason` |
| `validation_complete` | All validators run; has `gatekeeper`, `auditor`, `passed` |
| `cache_invalidated` | Cache cleared for doc change; has `doc_id`, `entries_removed` |
| `reconcile_check` | Reconciler ran; has `unsynced_pg`, `missing_qdrant` |

---

## Weekly Ops Checklist

```bash
# Every Monday morning
python -m src.evaluation.runner          # ← RAGAS scores
python -m src.stress_testing.red_team   # ← security tests
python -m src.evaluation.qdrant_metrics # ← fusion_gain

# Check in PostgreSQL
psql $DATABASE_URL -c "SELECT * FROM cache_hit_rate_daily LIMIT 7;"
psql $DATABASE_URL -c "SELECT COUNT(*) FROM chunks WHERE qdrant_synced=FALSE;"
psql $DATABASE_URL -c "SELECT run_name, faithfulness, created_at FROM evaluations ORDER BY created_at DESC LIMIT 5;"

# Check Qdrant health
curl http://localhost:6333/collections/rag_chunks | python3 -m json.tool
```
