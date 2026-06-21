# Deployment Guide

Local → Staging → Production runbook for the Production RAG System.

---

## Local Development (Current Setup)

Everything runs on your Windows 11 machine via Docker Compose.

```bash
# Start all services
docker compose up -d

# Verify
docker compose ps          # all healthy
curl localhost:6333/healthz # Qdrant: {"title":"qdrant"}
curl localhost:8001         # RedisInsight dashboard
curl localhost:5555         # Flower (Celery monitor)
curl localhost:9001         # MinIO console

# Run API
uvicorn src.api.main:app --reload --port 8000

# Run workers
celery -A src.workers.celery_app worker --loglevel=info -Q ingestion -c 4
```

### Services & Ports

| Service | Port | Purpose | Dashboard |
|---|---|---|---|
| FastAPI | 8000 | Main API | localhost:8000/docs |
| PostgreSQL | 5432 | Relational DB | via psql |
| Qdrant HTTP | 6333 | Vector DB | localhost:6333/dashboard |
| Qdrant gRPC | 6334 | Vector DB (fast) | — |
| Redis | 6379 | Celery (DB0) + Cache (DB1) | localhost:8001 |
| RedisInsight | 8001 | Redis dashboard | localhost:8001 |
| MinIO API | 9000 | Object storage | — |
| MinIO Console | 9001 | Storage dashboard | localhost:9001 |
| Flower | 5555 | Celery monitor | localhost:5555 |

---

## Environment Promotion Checklist

### Before Moving to Any New Environment

```bash
# 1. All tests pass
pytest tests/unit/ -v                    # unit: no external deps
pytest tests/integration/ -v --timeout=120  # integration: full stack

# 2. Evaluation scores meet thresholds
python -m src.evaluation.runner
# faithfulness ≥ 0.80, answer_relevancy ≥ 0.70

# 3. Red team passes
python -m src.stress_testing.red_team
# 100% CRITICAL pass rate, ≥ 80% overall

# 4. No hardcoded secrets
grep -r "password\|api_key\|secret" src/ --include="*.py" | grep -v ".env\|settings\|test"

# 5. Production gate test
pytest tests/integration/test_production_gate.py -v -s
```

---

## Staging Environment

> Staging mirrors production configuration but uses a separate dataset.

### Infrastructure Changes from Local

| Component | Local | Staging |
|---|---|---|
| PostgreSQL | Docker | Managed (RDS / Cloud SQL) |
| Qdrant | Docker | Qdrant Cloud or self-hosted VM |
| Redis | Docker (redis-stack) | Managed Redis with Search module |
| MinIO | Docker | S3 (AWS) or GCS |
| LLM | Ollama (local) | OpenAI API or hosted Ollama |
| Embeddings | Ollama (local) | OpenAI API or hosted Ollama |

### Key `.env` Changes for Staging

```env
# Switch LLM from local Ollama to API
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

# Switch embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# External services
DATABASE_URL=postgresql+asyncpg://user:pass@rds-endpoint:5432/ragdb
REDIS_URL=redis://redis-endpoint:6379/0
REDIS_CACHE_URL=redis://redis-endpoint:6379/1
QDRANT_HOST=qdrant-endpoint
MINIO_ENDPOINT=s3.amazonaws.com  # or use boto3 directly
```

### After Changing Embedding Model in Staging

Changing `EMBEDDING_MODEL` requires a full re-index:

```bash
# 1. Trigger blue-green reindex
curl -X POST https://staging.yourdomain.com/api/v1/admin/reindex \
  -d '{"new_model": "text-embedding-3-small", "dimensions": 1536}'

# 2. Monitor
curl https://staging.yourdomain.com/api/v1/admin/reindex/status

# 3. Evaluate new model BEFORE cutover
python -m src.evaluation.runner --base-url https://staging.yourdomain.com

# 4. Cutover only if evaluation passes
curl -X POST https://staging.yourdomain.com/api/v1/admin/reindex/cutover

# 5. Flush stale cache
redis-cli -h redis-endpoint -n 1 FLUSHDB
```

---

## Production Environment

### Additional Production Concerns

**Qdrant persistence:**
```yaml
# Ensure Qdrant data is persisted to a mounted volume
services:
  qdrant:
    volumes:
      - /mnt/persistent-disk/qdrant:/qdrant/storage
```

**PostgreSQL backups:**
```bash
# Daily backup (add to cron)
pg_dump $DATABASE_URL | gzip > backup_$(date +%Y%m%d).sql.gz
```

**Celery with multiple workers:**
```bash
# Scale ingestion workers horizontally
celery -A src.workers.celery_app worker \
  --loglevel=info \
  -Q ingestion \
  -c 8 \                  # 8 concurrent tasks per worker
  --max-tasks-per-child=100  # restart after 100 tasks (prevent memory leaks)
```

**FastAPI with Gunicorn (multi-process):**
```bash
gunicorn src.api.main:app \
  -w 4 \                  # 4 worker processes
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Production Scaling Signals

| Signal | Threshold | Action |
|---|---|---|
| Qdrant query p95 > 200ms | Sustained 5 min | Add Qdrant nodes (distributed mode) |
| Celery queue depth > 1000 | Sustained 10 min | Scale worker pods horizontally |
| PG connection pool exhausted | Any | Add PgBouncer |
| API p95 > 5s | Sustained 5 min | Add Gunicorn workers, load balancer |
| Redis cache memory > 80% | Sustained | Increase `maxmemory` or add node |

---

## Rollback Procedure

### API Rollback (code change)
```bash
# Deploy previous image / git checkout previous tag
git checkout v0.3.0
docker compose up -d --build
```

### Qdrant Collection Rollback (embedding model change)
```bash
# If cutover was already done and new model is worse
# Rollback: re-index with old model, cutover back
curl -X POST /api/v1/admin/reindex \
  -d '{"new_model": "nomic-embed-text", "dimensions": 768}'
# Wait for completion, evaluate, then cutover
```

### Database Rollback (schema change)
```bash
# Alembic rollback to previous migration
python -m alembic downgrade -1
```

### Emergency: Complete Qdrant Rebuild
```bash
# If Qdrant data is corrupted/lost
# 1. Recreate collection (loses all vectors)
python -m src.scripts.setup_qdrant --recreate

# 2. Run reconciler (re-embeds all chunks from PostgreSQL)
curl -X POST /api/v1/qdrant/reconcile
# This re-uploads all chunks — may take hours for large corpora
```
