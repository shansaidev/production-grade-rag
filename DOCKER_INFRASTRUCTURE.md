# Docker Infrastructure Guide
## Windows 11 — CMD & PowerShell

> Single `docker-compose.yml`. No layering. Every command for start, verify, inspect, stop.

---

## Before Your First `docker compose up`

Two files must exist in your project root before Docker can build:

```powershell
# Check they exist
dir pyproject.toml
dir Dockerfile
dir docker-compose.yml
```

If `pyproject.toml` or `Dockerfile` is missing, Docker cannot build the Celery worker
and you will see errors like:
```
failed to calculate checksum ... "/alembic.ini": not found
```

The fix: make sure your project root has `pyproject.toml` and `Dockerfile` before running
`docker compose up`. Alembic files are NOT needed inside the container — migrations run
from your terminal on Windows, not inside Docker.

---

## Services at a Glance

| Container | Image | Purpose | Ports |
|---|---|---|---|
| `rag_postgres` | postgres:16-alpine | Text, metadata, audit trail | 5432 |
| `rag_qdrant` | qdrant/qdrant:latest | Vector search (dense + sparse + RRF) | 6333, 6334 |
| `rag_redis` | redis/redis-stack:latest | Celery queue + semantic cache | 6379, 8001 |
| `rag_minio` | minio/minio:latest | Raw file storage | 9000, 9001 |
| `rag_celery_worker` | built from Dockerfile | Async document ingestion | — |
| `rag_flower` | mher/flower:2.0.1 | Celery task monitor | 5555 |

---

## Starting

### Start all services

```powershell
docker compose up -d
```

### Start and rebuild (after code changes)

```powershell
docker compose up -d --build
```

### Start with logs visible (not detached)

Press `Ctrl+C` to stop. Containers stop too.

```powershell
docker compose up --build
```

### Start infrastructure only (no celery/flower)

Useful if you haven't written `src/workers/celery_app.py` yet:

```powershell
docker compose up -d postgres qdrant redis minio
```

### Start a single service

```powershell
docker compose up -d postgres
docker compose up -d qdrant
docker compose up -d redis
docker compose up -d minio
```

---

## Verifying — After Every Start

### Check all containers are running

```powershell
docker compose ps
```

Every service should show `running` or `healthy`. Wait 20 seconds if any show `starting`.

### Check Qdrant

```powershell
# CMD
curl http://localhost:6333/healthz

# PowerShell
Invoke-WebRequest -Uri "http://localhost:6333/healthz" -UseBasicParsing
```

Expected: `{"title":"qdrant","version":"..."}`

### Check Redis

```powershell
docker exec rag_redis redis-cli ping
```

Expected: `PONG`

### Check PostgreSQL

```powershell
docker exec rag_postgres pg_isready -U raguser -d ragdb
```

Expected: `localhost:5432 - accepting connections`

### Check MinIO

```powershell
# CMD
curl http://localhost:9000/minio/health/live

# PowerShell
Invoke-WebRequest -Uri "http://localhost:9000/minio/health/live" -UseBasicParsing
```

Expected: HTTP 200

### Check Celery worker

```powershell
docker compose logs --tail=20 celery_worker
```

Look for: `celery@... ready.` — means worker started and connected to Redis.

### All checks in one block (PowerShell)

```powershell
Write-Host "`n--- Container status ---" -ForegroundColor Cyan
docker compose ps

Write-Host "`n--- Qdrant ---" -ForegroundColor Cyan
try { (Invoke-WebRequest -Uri "http://localhost:6333/healthz" -UseBasicParsing -TimeoutSec 3).StatusCode } catch { "FAIL" }

Write-Host "`n--- Redis ---" -ForegroundColor Cyan
docker exec rag_redis redis-cli ping

Write-Host "`n--- PostgreSQL ---" -ForegroundColor Cyan
docker exec rag_postgres pg_isready -U raguser -d ragdb

Write-Host "`n--- MinIO ---" -ForegroundColor Cyan
try { (Invoke-WebRequest -Uri "http://localhost:9000/minio/health/live" -UseBasicParsing -TimeoutSec 3).StatusCode } catch { "FAIL" }
```

---

## Dashboards (open in browser)

| Dashboard | URL | Login |
|---|---|---|
| Qdrant | http://localhost:6333/dashboard | — |
| RedisInsight | http://localhost:8001 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |
| Flower (Celery) | http://localhost:5555 | — |
| FastAPI docs | http://localhost:8000/docs | — (start FastAPI separately) |

---

## First-Time Setup (run once after first `up`)

```powershell
# 1. Create all 8 PostgreSQL tables
uv run python -m alembic upgrade head

# 2. Create Qdrant collection with named vectors + payload indexes
uv run python -m src.scripts.setup_qdrant
```

Verify tables were created:

```powershell
docker exec rag_postgres psql -U raguser -d ragdb -c "\dt"
```

Expected — 8 tables listed:
```
cache_metrics, chunks, deletion_audit, documents,
evaluations, qdrant_sync_log, queries, validations
```

Verify Qdrant collection — open http://localhost:6333/dashboard → `rag_chunks` collection visible.

---

## Starting FastAPI (separate terminal)

FastAPI runs on your Windows machine, not inside Docker:

```powershell
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Test it:

```powershell
# CMD
curl http://localhost:8000/health

# PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
```

---

## Viewing Logs

### Follow all services

```powershell
docker compose logs -f
```

Press `Ctrl+C` to stop following. Containers keep running.

### Follow one service

```powershell
docker compose logs -f postgres
docker compose logs -f qdrant
docker compose logs -f redis
docker compose logs -f minio
docker compose logs -f celery_worker
docker compose logs -f flower
```

### Last N lines only (no follow)

```powershell
docker compose logs --tail=50 celery_worker
docker compose logs --tail=100 qdrant
```

### Logs from the last 30 minutes

```powershell
docker compose logs --since=30m celery_worker
docker compose logs --since=2h postgres
```

---

## Inspecting Containers

### Open a shell inside a container

```powershell
# PostgreSQL interactive SQL shell
docker exec -it rag_postgres psql -U raguser -d ragdb

# Redis CLI
docker exec -it rag_redis redis-cli

# Qdrant shell
docker exec -it rag_qdrant sh

# MinIO shell
docker exec -it rag_minio sh
```

### Run a SQL query directly

```powershell
docker exec rag_postgres psql -U raguser -d ragdb -c "SELECT COUNT(*) FROM documents;"
docker exec rag_postgres psql -U raguser -d ragdb -c "SELECT COUNT(*) FROM chunks;"
docker exec rag_postgres psql -U raguser -d ragdb -c "SELECT COUNT(*) FROM chunks WHERE qdrant_synced = FALSE;"
docker exec rag_postgres psql -U raguser -d ragdb -c "SELECT * FROM cache_hit_rate_daily LIMIT 7;"
```

### Inspect Redis databases

```powershell
# Celery queue depth (DB 0)
docker exec rag_redis redis-cli -n 0 LLEN celery

# Semantic cache key count (DB 1)
docker exec rag_redis redis-cli -n 1 DBSIZE

# Redis memory usage
docker exec rag_redis redis-cli INFO memory
```

### Flush semantic cache only (does NOT touch Celery)

```powershell
docker exec rag_redis redis-cli -n 1 FLUSHDB
```

### Qdrant collection info

```powershell
# CMD
curl http://localhost:6333/collections/rag_chunks

# PowerShell
Invoke-WebRequest -Uri "http://localhost:6333/collections/rag_chunks" -UseBasicParsing | Select-Object -ExpandProperty Content
```

### Live resource usage (CPU, memory, network)

```powershell
# Live (press Ctrl+C to exit)
docker stats

# One-time snapshot
docker stats --no-stream
```

---

## Restarting Services

### Restart one service

```powershell
docker compose restart postgres
docker compose restart qdrant
docker compose restart redis
docker compose restart minio
docker compose restart celery_worker
```

### Rebuild and restart celery worker (after src/ code changes)

```powershell
docker compose up -d --build celery_worker
```

### Stop one service, then start it again

```powershell
docker compose stop redis
docker compose start redis
```

---

## Stopping

### Stop — keep all data (use this daily)

```powershell
docker compose down
```

Containers are removed. Named volumes (`postgres_data`, `qdrant_data`, `redis_data`, `minio_data`) are kept.
Next `docker compose up -d` restores everything exactly as it was.

### Stop — DELETE all data (nuclear option)

```powershell
docker compose down -v
```

The `-v` flag removes all named volumes. Every PostgreSQL row, Qdrant vector, Redis entry,
and MinIO file is permanently deleted. Use only when you want a completely clean slate.

### Stop one service

```powershell
docker compose stop celery_worker
docker compose stop flower
```

---

## Daily Workflow

```powershell
# Morning — start
docker compose up -d
# (new terminal) uv run uvicorn src.api.main:app --reload --port 8000

# Check everything is healthy
docker compose ps

# Evening — stop (keeps data)
docker compose down
```

---

## After Changing Files

| What changed | Command |
|---|---|
| `.env` values | `docker compose down` then `docker compose up -d` |
| `docker-compose.yml` | `docker compose up -d` (detects changes automatically) |
| `Dockerfile` | `docker compose up -d --build` |
| `src/` code (celery worker) | `docker compose up -d --build celery_worker` |
| `src/` code (FastAPI) | Auto-reloads — nothing needed (uvicorn `--reload` handles it) |
| `alembic/` migrations | `uv run python -m alembic upgrade head` — no Docker restart needed |

---

## Troubleshooting

### Container shows `exited` in `docker compose ps`

```powershell
# See what went wrong
docker compose logs celery_worker
docker compose logs postgres
```

**Common causes:**

`rag_postgres exited` → port 5432 already in use:
```powershell
netstat -ano | findstr :5432
# Kill the process using that port (replace 1234 with actual PID)
taskkill /PID 1234 /F
```

`rag_qdrant exited` → port 6333 in use:
```powershell
netstat -ano | findstr :6333
```

`rag_celery_worker exited` → build failed or `src/` missing:
```powershell
docker compose logs celery_worker
# Look for: ModuleNotFoundError or ImportError
```

`rag_redis exited` → another Redis already running on 6379:
```powershell
netstat -ano | findstr :6379
```

### Dockerfile build fails — "not found" errors

```
ERROR: failed to calculate checksum ... "/alembic.ini": not found
ERROR: failed to calculate checksum ... "/alembic/": not found
```

**Cause:** The Dockerfile was trying to `COPY alembic/ ./alembic/` but that directory
doesn't exist yet. The fixed `Dockerfile` in this project does NOT copy alembic — migrations
run from your Windows terminal, not inside Docker.

**Fix:** Make sure you are using the corrected `Dockerfile` (the one that only copies `src/`).

### Cannot connect from FastAPI to postgres/qdrant/redis

Your `.env` must use `localhost` (not service names like `postgres`):

```env
# Correct — FastAPI runs on Windows, connects via localhost
DATABASE_URL=postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb
QDRANT_HOST=localhost
REDIS_URL=redis://localhost:6379/0

# Wrong — service names only work INSIDE the Docker network
DATABASE_URL=postgresql+asyncpg://raguser:ragpassword@postgres:5432/ragdb
```

### Qdrant shows 0 points after ingestion

```powershell
# Check chunks in PG but not synced to Qdrant
docker exec rag_postgres psql -U raguser -d ragdb -c "SELECT COUNT(*) FROM chunks WHERE qdrant_synced = FALSE;"
```

If count > 0, run the reconciler:
```powershell
curl -X POST http://localhost:8000/api/v1/qdrant/reconcile
```

### Semantic cache not working

Verify the Redis image is `redis-stack` not plain `redis`:
```powershell
docker inspect rag_redis --format "{{.Config.Image}}"
```

Must show `redis/redis-stack`. If it shows `redis:alpine`, update `docker-compose.yml` and recreate:
```powershell
docker compose down
docker compose up -d
```

### Celery worker not processing jobs

```powershell
# Check worker is alive
docker compose logs --tail=30 celery_worker

# Check Redis queue
docker exec rag_redis redis-cli -n 0 LLEN celery

# Restart the worker
docker compose restart celery_worker
```

### Full reset (start completely fresh)

```powershell
docker compose down -v
docker compose up -d --build
uv run python -m alembic upgrade head
uv run python -m src.scripts.setup_qdrant
```

---

## Quick Reference

```
START
  All:          docker compose up -d
  Rebuild:      docker compose up -d --build
  Infra only:   docker compose up -d postgres qdrant redis minio
  One service:  docker compose up -d <name>

CHECK
  Status:       docker compose ps
  Qdrant:       curl http://localhost:6333/healthz
  Redis:        docker exec rag_redis redis-cli ping
  PostgreSQL:   docker exec rag_postgres pg_isready -U raguser -d ragdb
  MinIO:        curl http://localhost:9000/minio/health/live
  Resources:    docker stats --no-stream

LOGS
  All:          docker compose logs -f
  One service:  docker compose logs -f <name>
  Last 50:      docker compose logs --tail=50 <name>

SHELLS
  PostgreSQL:   docker exec -it rag_postgres psql -U raguser -d ragdb
  Redis:        docker exec -it rag_redis redis-cli
  Any:          docker exec -it <container> sh

RESTART
  One service:  docker compose restart <name>
  Rebuild one:  docker compose up -d --build <name>

STOP
  Keep data:    docker compose down
  Delete data:  docker compose down -v
```
