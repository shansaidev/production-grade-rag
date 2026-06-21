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