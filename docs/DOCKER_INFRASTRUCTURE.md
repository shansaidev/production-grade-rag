# Docker Infrastructure Guide
## Manual Commands for Windows 11 — CMD & PowerShell

> Every command you need to start, verify, inspect, and stop the RAG system infrastructure.
> All commands work in both **CMD** and **PowerShell** unless marked otherwise.

---

## Project Services at a Glance

| Container | Image | Purpose | Port(s) |
|---|---|---|---|
| `rag_postgres` | postgres:16-alpine | Source of truth — text, metadata, audit | 5432 |
| `rag_qdrant` | qdrant/qdrant:latest | Vector search — dense + sparse + RRF | 6333 (HTTP), 6334 (gRPC) |
| `rag_redis` | redis/redis-stack:latest | Celery queue (DB0) + Semantic cache (DB1) | 6379, 8001 (dashboard) |
| `rag_minio` | minio/minio:latest | Raw file storage (PDF, DOCX, HTML) | 9000 (API), 9001 (console) |
| `rag_celery_worker` | built from Dockerfile | Async document ingestion pipeline | — |
| `rag_flower` | mher/flower:2.0.1 | Celery task monitor | 5555 |

---

## Step 1 — Start the Infrastructure

### Start everything (recommended)

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

What the flags do:
- `-f docker-compose.yml` — loads the base file (postgres, qdrant, redis, minio)
- `-f docker-compose.dev.yml` — loads the dev override (celery, flower, exposed ports)
- `up -d` — starts containers in **detached** mode (runs in background, terminal stays free)
- `--build` — rebuilds the Dockerfile if source code changed (safe to always include)

### Start infrastructure only (no celery/flower)

Useful when you just need the databases and haven't built the Dockerfile yet:

```powershell
docker compose -f docker-compose.yml up -d
```

### Start a single service

```powershell
docker compose -f docker-compose.yml up -d postgres
docker compose -f docker-compose.yml up -d qdrant
docker compose -f docker-compose.yml up -d redis
docker compose -f docker-compose.yml up -d minio
```

### Start with logs visible (not detached)

Starts and streams all logs to your terminal. Press `Ctrl+C` to stop:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

---

## Step 2 — Verify All Services Are Running

### Check container status

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml ps
```

Expected output — every service should show `running` or `healthy`:

```
NAME                 IMAGE                        STATUS
rag_postgres         postgres:16-alpine           running (healthy)
rag_qdrant           qdrant/qdrant:latest         running (healthy)
rag_redis            redis/redis-stack:latest     running (healthy)
rag_minio            minio/minio:latest           running (healthy)
rag_celery_worker    production-rag-celery_...    running
rag_flower           mher/flower:2.0.1            running
```

If a service shows `starting` — wait 10–20 seconds and run the command again.
If a service shows `exited` — it crashed. See the [Troubleshooting](#troubleshooting) section.

### Verify Qdrant

```powershell
# PowerShell
Invoke-WebRequest -Uri "http://localhost:6333/healthz" -UseBasicParsing

# CMD
curl http://localhost:6333/healthz
```

Expected: `{"title":"qdrant","version":"..."}`

Open the dashboard: http://localhost:6333/dashboard

### Verify Redis

```powershell
docker exec rag_redis redis-cli ping
```

Expected: `PONG`

Open RedisInsight dashboard: http://localhost:8001

### Verify PostgreSQL

```powershell
docker exec rag_postgres pg_isready -U raguser -d ragdb
```

Expected: `localhost:5432 - accepting connections`

### Verify MinIO

```powershell
# PowerShell
Invoke-WebRequest -Uri "http://localhost:9000/minio/health/live" -UseBasicParsing

# CMD
curl http://localhost:9000/minio/health/live
```

Expected: HTTP 200 response

Open MinIO Console: http://localhost:9001
Login: `minioadmin` / `minioadmin123`

### Verify Flower (Celery monitor)

Open: http://localhost:5555

You should see the Flower dashboard with workers listed.

### Verify Celery Worker

```powershell
docker exec rag_celery_worker uv run celery -A src.workers.celery_app inspect ping
```

Expected: Response from worker with `pong`

### One-command full check (PowerShell)

```powershell
# Run all health checks in sequence
Write-Host "=== Container Status ===" -ForegroundColor Cyan
docker compose -f docker-compose.yml -f docker-compose.dev.yml ps

Write-Host "`n=== Qdrant ===" -ForegroundColor Cyan
try { Invoke-WebRequest -Uri "http://localhost:6333/healthz" -UseBasicParsing -TimeoutSec 3 | Select-Object StatusCode }
catch { Write-Host "FAIL - not responding" -ForegroundColor Red }

Write-Host "`n=== Redis ===" -ForegroundColor Cyan
docker exec rag_redis redis-cli ping

Write-Host "`n=== PostgreSQL ===" -ForegroundColor Cyan
docker exec rag_postgres pg_isready -U raguser -d ragdb

Write-Host "`n=== MinIO ===" -ForegroundColor Cyan
try { Invoke-WebRequest -Uri "http://localhost:9000/minio/health/live" -UseBasicParsing -TimeoutSec 3 | Select-Object StatusCode }
catch { Write-Host "FAIL - not responding" -ForegroundColor Red }
```

---

## Step 3 — Initialize the Database (First Time Only)

Run these once after first `up`. Not needed on subsequent starts:

```powershell
# Create all 8 PostgreSQL tables
uv run python -m alembic upgrade head

# Create Qdrant collection with named vectors + payload indexes
uv run python -m src.scripts.setup_qdrant
```

Verify tables were created:

```powershell
docker exec rag_postgres psql -U raguser -d ragdb -c "\dt"
```

Expected output:
```
 Schema | Name              | Type  | Owner
--------+-------------------+-------+---------
 public | cache_metrics     | table | raguser
 public | chunks            | table | raguser
 public | deletion_audit    | table | raguser
 public | documents         | table | raguser
 public | evaluations       | table | raguser
 public | qdrant_sync_log   | table | raguser
 public | queries           | table | raguser
 public | validations       | table | raguser
```

Verify Qdrant collection was created:

Open http://localhost:6333/dashboard → you should see `rag_chunks` collection listed.

---

## Step 4 — Start the FastAPI Application

FastAPI runs **outside Docker** in development (enables hot reload):

```powershell
# New terminal — keep Docker running in background
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Verify API is running:

```powershell
# PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing

# CMD
curl http://localhost:8000/health
```

Expected:
```json
{"status":"healthy","services":{"postgres":"green","qdrant":"green","redis":"green"}}
```

Open API docs: http://localhost:8000/docs

---

## Viewing Logs

### All services together

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f
```

Press `Ctrl+C` to stop tailing (containers keep running).

### Single service

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f postgres
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f qdrant
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f redis
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f minio
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f celery_worker
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f flower
```

### Last N lines (no follow)

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs --tail=50 celery_worker
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs --tail=100 qdrant
```

### Logs since a specific time

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs --since=30m celery_worker
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs --since=2h postgres
```

---

## Inspecting Services

### Open a shell inside a container

```powershell
# PostgreSQL interactive shell (psql)
docker exec -it rag_postgres psql -U raguser -d ragdb

# Redis CLI
docker exec -it rag_redis redis-cli

# Qdrant shell (Alpine sh)
docker exec -it rag_qdrant sh

# MinIO shell
docker exec -it rag_minio sh
```

### Run a single SQL query

```powershell
docker exec rag_postgres psql -U raguser -d ragdb -c "SELECT COUNT(*) FROM documents;"
docker exec rag_postgres psql -U raguser -d ragdb -c "SELECT COUNT(*) FROM chunks;"
docker exec rag_postgres psql -U raguser -d ragdb -c "SELECT COUNT(*) FROM chunks WHERE qdrant_synced = FALSE;"
docker exec rag_postgres psql -U raguser -d ragdb -c "SELECT * FROM cache_hit_rate_daily LIMIT 7;"
```

### Inspect Redis

```powershell
# Check Celery queue depth (DB 0)
docker exec rag_redis redis-cli -n 0 LLEN celery

# Check semantic cache key count (DB 1)
docker exec rag_redis redis-cli -n 1 DBSIZE

# Check Redis memory usage
docker exec rag_redis redis-cli INFO memory

# Flush semantic cache (DB 1 only — does NOT touch Celery DB 0)
docker exec rag_redis redis-cli -n 1 FLUSHDB
```

### Inspect Qdrant collection

```powershell
# PowerShell — collection stats
Invoke-WebRequest -Uri "http://localhost:6333/collections/rag_chunks" -UseBasicParsing | Select-Object -ExpandProperty Content

# CMD
curl http://localhost:6333/collections/rag_chunks
```

### Check container resource usage (live)

```powershell
docker stats
```

Shows live CPU %, memory usage, and network I/O for all running containers.
Press `Ctrl+C` to exit.

### Check container resource usage (one snapshot)

```powershell
docker stats --no-stream
```

### Inspect container configuration

```powershell
docker inspect rag_postgres
docker inspect rag_qdrant
docker inspect rag_redis
```

---

## Restarting Services

### Restart a single service (keeps data)

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart postgres
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart qdrant
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart redis
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart minio
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart celery_worker
```

### Stop a single service without removing it

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml stop redis
```

### Start a stopped service

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml start redis
```

---

## Stopping the Infrastructure

### Stop everything — keeps all data (recommended)

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

Containers are removed but named volumes (`postgres_data`, `qdrant_data`, etc.) are preserved.
Next `up` command restores everything exactly as you left it.

### Stop everything — DELETE all data

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v
```

The `-v` flag removes all named volumes. **Use this only when you want a completely clean slate.**
All PostgreSQL tables, Qdrant vectors, Redis cache, and MinIO files are permanently deleted.

### Stop a single service

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml stop celery_worker
docker compose -f docker-compose.yml -f docker-compose.dev.yml stop flower
```

---

## Daily Workflow

### Morning — start work

```powershell
# Terminal 1 — start infrastructure
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Wait ~15 seconds, then verify
docker compose -f docker-compose.yml -f docker-compose.dev.yml ps

# Terminal 2 — start FastAPI
uv run uvicorn src.api.main:app --reload --port 8000
```

### Evening — stop work

```powershell
# Stop FastAPI: Ctrl+C in Terminal 2

# Stop infrastructure (data preserved)
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

---

## Common Operations

### After changing docker-compose.yml

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

Docker Compose detects changed configuration and recreates only the affected containers.

### After changing Dockerfile or src/ code (for celery_worker)

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build celery_worker
```

The `--build` flag rebuilds the image before starting. Other services are unaffected.

### After changing .env

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

Environment variable changes require container recreation (not just restart).

### After changing alembic migrations

```powershell
uv run python -m alembic upgrade head
```

No Docker restart needed — alembic connects to the running postgres container.

### Upload a test document and query it

```powershell
# PowerShell — upload a document
$form = @{ file = Get-Item "tests\fixtures\sample_hr_policy.pdf"; department = "hr" }
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/documents" -Method POST -Form $form

# CMD — upload a document
curl -X POST http://localhost:8000/api/v1/documents -F "file=@tests/fixtures/sample_hr_policy.pdf" -F "department=hr"

# Query
curl -X POST http://localhost:8000/api/v1/query/sync -H "Content-Type: application/json" -d "{\"query\": \"What is the parental leave policy?\"}"
```

---

## Troubleshooting

### Container shows "exited" in `docker compose ps`

```powershell
# See the error that caused it to exit
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs celery_worker
```

Most common causes:
- **postgres exited** — port 5432 already in use. Check: `netstat -ano | findstr 5432`
- **qdrant exited** — port 6333 already in use. Check: `netstat -ano | findstr 6333`
- **celery_worker exited** — Dockerfile build failed, or .env missing variables
- **redis exited** — port 6379 conflict with another Redis instance

### Port already in use

```powershell
# Find what is using port 5432
netstat -ano | findstr :5432

# Find the process name from the PID (replace 1234 with actual PID)
tasklist | findstr 1234

# Kill the process (replace 1234 with actual PID)
taskkill /PID 1234 /F
```

### Cannot connect from Python to postgres / qdrant / redis

Verify your `.env` uses `localhost` (not the service name):

```env
# Correct for FastAPI running OUTSIDE Docker:
DATABASE_URL=postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb
QDRANT_HOST=localhost
REDIS_URL=redis://localhost:6379/0

# Wrong (service names only resolve INSIDE Docker network):
DATABASE_URL=postgresql+asyncpg://raguser:ragpassword@postgres:5432/ragdb
```

### Qdrant dashboard shows 0 points after ingestion

```powershell
# Check if chunks exist in PostgreSQL but not Qdrant
docker exec rag_postgres psql -U raguser -d ragdb -c "SELECT COUNT(*) FROM chunks WHERE qdrant_synced = FALSE;"
```

If count > 0, run the reconciler:

```powershell
curl -X POST http://localhost:8000/api/v1/qdrant/reconcile
```

### Redis cache not working (all misses)

Verify you are using `redis/redis-stack` not `redis:alpine`:

```powershell
docker inspect rag_redis | findstr Image
```

Must show `redis/redis-stack`. If it shows `redis:alpine`, update docker-compose.yml and run:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Celery worker not picking up jobs

```powershell
# Check worker is connected to broker
docker exec rag_celery_worker uv run celery -A src.workers.celery_app inspect active

# Check queue depth
docker exec rag_redis redis-cli -n 0 LLEN celery

# Restart the worker
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart celery_worker
```

### Complete reset (nuclear option)

```powershell
# Stop everything and delete ALL data
docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v

# Start fresh
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

# Re-initialize
uv run python -m alembic upgrade head
uv run python -m src.scripts.setup_qdrant
```

---

## Quick Reference Card

```
START
  All services:      docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
  Infra only:        docker compose -f docker-compose.yml up -d
  One service:       docker compose -f docker-compose.yml up -d <name>

VERIFY
  Status:            docker compose -f docker-compose.yml -f docker-compose.dev.yml ps
  Qdrant:            curl http://localhost:6333/healthz
  Redis:             docker exec rag_redis redis-cli ping
  PostgreSQL:        docker exec rag_postgres pg_isready -U raguser -d ragdb
  MinIO:             curl http://localhost:9000/minio/health/live
  Resources:         docker stats --no-stream

DASHBOARDS
  Qdrant:            http://localhost:6333/dashboard
  RedisInsight:      http://localhost:8001
  MinIO Console:     http://localhost:9001   (admin / minioadmin123)
  Flower:            http://localhost:5555
  FastAPI docs:      http://localhost:8000/docs

LOGS
  All:               docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f
  One service:       docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f <name>
  Last 50 lines:     docker compose -f docker-compose.yml -f docker-compose.dev.yml logs --tail=50 <name>

SHELLS
  PostgreSQL:        docker exec -it rag_postgres psql -U raguser -d ragdb
  Redis:             docker exec -it rag_redis redis-cli
  Any container:     docker exec -it <container> sh

RESTART
  One service:       docker compose -f docker-compose.yml -f docker-compose.dev.yml restart <name>
  Rebuild + restart: docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build <name>

STOP
  Keep data:         docker compose -f docker-compose.yml -f docker-compose.dev.yml down
  Delete all data:   docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v
```
