# API Reference

Base URL: `http://localhost:8000`  
All requests: `Content-Type: application/json` unless uploading files.  
All responses include: `{data, metadata: {request_id, duration_ms}, error?}`

---

## Documents

### Upload Document
```
POST /api/v1/documents
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | ✅ | PDF, DOCX, HTML, or code file |
| `department` | string | ❌ | e.g. `"hr"`, `"engineering"` (default: `"general"`) |
| `access_level` | string | ❌ | `"public"` / `"internal"` / `"restricted"` (default: `"internal"`) |

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@policy.pdf" \
  -F "department=hr" \
  -F "access_level=internal"
```

```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "status": "queued",
  "filename": "policy.pdf"
}
```

### Poll Ingestion Job
```
GET /api/v1/jobs/{job_id}
```

```json
{
  "job_id": "7c9e6679...",
  "status": "done",
  "doc_id": "550e8400...",
  "chunks_created": 47,
  "duration_ms": 8432
}
```

Statuses: `queued` → `processing` → `done` | `failed`

### List Documents
```
GET /api/v1/documents?department=hr&page=1&page_size=20
```

```json
{
  "documents": [
    {
      "doc_id": "550e8400...",
      "filename": "HR_Policy_2024.pdf",
      "department": "hr",
      "version": 1,
      "chunk_count": 47,
      "created_at": "2026-06-21T09:00:00Z"
    }
  ],
  "total": 142,
  "page": 1
}
```

### Get Document
```
GET /api/v1/documents/{doc_id}
```

### Version Bump (replace with new file)
```
POST /api/v1/documents/{doc_id}/versions
Content-Type: multipart/form-data
```

| Field | Type | Description |
|---|---|---|
| `file` | file | New version of the document |

Returns same shape as upload. Old version becomes invisible to search immediately.

### Metadata Update (no re-embedding)
```
PATCH /api/v1/documents/{doc_id}
```

```json
{ "department": "legal", "access_level": "restricted" }
```

Updates PostgreSQL + Qdrant payload. No embedding operations. Instant.

### Soft Delete
```
DELETE /api/v1/documents/{doc_id}
```

```json
{ "reason": "Document superseded", "deleted_by": "admin@company.com" }
```

Document immediately invisible to search. Text preserved for 30-day recovery window.

### Restore (undo soft delete)
```
POST /api/v1/documents/{doc_id}/restore
```

Re-queues embedding. Document searchable again after ingestion job completes.

### Hard Delete (admin only)
```
POST /api/v1/documents/{doc_id}/hard-delete
```

Requires: document must be soft-deleted first. Removes all PG rows + MinIO file.
Satisfies GDPR right-to-erasure requests.

---

## Queries

### Synchronous Query
```
POST /api/v1/query/sync
```

```json
{
  "query": "What is the parental leave policy for California employees?",
  "department": "hr",
  "top_k": 5,
  "user_id": "user-123"
}
```

```json
{
  "query_id": "a3f8b2c1...",
  "answer": "California employees are entitled to 12 weeks of parental leave under CFRA...",
  "confidence": 0.91,
  "sources": [
    {
      "chunk_id": "b4e5f6...",
      "document": "HR_Policy_2024.pdf",
      "section": "California Leave Policies",
      "page": 12,
      "rrf_score": 0.94,
      "excerpt": "California employees are entitled to..."
    }
  ],
  "validation": {
    "gatekeeper": true,
    "auditor": true,
    "strategist": true
  },
  "cache": {
    "hit": false,
    "level": null
  },
  "metadata": {
    "response_time_ms": 1876,
    "tokens_used": 2341,
    "retrieval_strategy": "hybrid_rrf",
    "retry_count": 0
  }
}
```

**Cache hit response** (75ms instead of 1876ms):
```json
{
  "cache": { "hit": true, "level": "semantic", "similarity": 0.94 },
  "metadata": { "response_time_ms": 75 }
}
```

### Streaming Query (SSE)
```
POST /api/v1/query
Accept: text/event-stream
```

Same request body as sync. Response streams token by token:
```
data: {"token": "California"}
data: {"token": " employees"}
data: {"token": " are"}
...
data: {"done": true, "sources": [...], "confidence": 0.91}
```

### Get Past Query
```
GET /api/v1/queries/{query_id}
```

Returns full query record including retrieved chunk IDs and validation scores.

---

## Evaluation

### Trigger RAGAS Evaluation
```
POST /api/v1/eval/run
```

```json
{
  "run_name": "baseline-v0.4",
  "golden_dataset_path": "tests/golden_dataset/qa_pairs.json"
}
```

```json
{
  "eval_id": "e1a2b3...",
  "status": "running",
  "estimated_duration_s": 120
}
```

### Get Evaluation Results
```
GET /api/v1/eval/runs/{eval_id}
```

```json
{
  "eval_id": "e1a2b3...",
  "run_name": "baseline-v0.4",
  "faithfulness": 0.83,
  "answer_relevancy": 0.76,
  "context_recall": 0.79,
  "context_precision": 0.81,
  "fusion_gain": 0.08,
  "avg_latency_ms": 1654,
  "p95_latency_ms": 2980,
  "sample_count": 30,
  "created_at": "2026-06-26T14:00:00Z"
}
```

### List Evaluation Runs
```
GET /api/v1/eval/runs?limit=10
```

---

## Qdrant Operations

### Health Check
```
GET /api/v1/qdrant/health
```

```json
{
  "status": "green",
  "collection": "rag_chunks",
  "points_count": 14782,
  "indexed_vectors_count": 14782,
  "segments_count": 3
}
```

### Trigger Reconciliation
```
POST /api/v1/qdrant/reconcile
```

Finds PostgreSQL chunks with `qdrant_synced=FALSE` and re-uploads to Qdrant.

```json
{
  "chunks_checked": 14782,
  "chunks_missing": 3,
  "chunks_resynced": 3,
  "status": "ok"
}
```

---

## Admin (Reindex)

### Start Blue-Green Reindex
```
POST /api/v1/admin/reindex
```

```json
{ "new_model": "text-embedding-3-large", "dimensions": 3072 }
```

Creates new Qdrant collection. Old collection serves live traffic throughout.

### Reindex Status
```
GET /api/v1/admin/reindex/status
```

```json
{
  "reindexed": 12000,
  "remaining": 2782,
  "total": 14782,
  "pct_complete": 81.18
}
```

### Cutover (activate new collection)
```
POST /api/v1/admin/reindex/cutover
```

Atomic switch. All subsequent queries use the new collection.

### Rollback (drop new collection)
```
POST /api/v1/admin/reindex/rollback
```

---

## System Health
```
GET /api/v1/health
```

```json
{
  "status": "healthy",
  "services": {
    "postgres": "green",
    "qdrant": "green",
    "redis_celery": "green",
    "redis_cache": "green",
    "minio": "green",
    "ollama": "green"
  },
  "version": "0.4.0"
}
```

---

## Error Responses

All errors follow:
```json
{
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "Document 550e8400... does not exist or has been deleted",
    "request_id": "04A0:6873E..."
  }
}
```

| HTTP Status | Code | When |
|---|---|---|
| 400 | `INVALID_REQUEST` | Missing required field, bad file type |
| 404 | `DOCUMENT_NOT_FOUND` | doc_id doesn't exist |
| 409 | `ALREADY_EXISTS` | Duplicate checksum upload |
| 422 | `VALIDATION_ERROR` | Pydantic schema mismatch |
| 500 | `INTERNAL_ERROR` | Unexpected server error (check logs) |
| 503 | `SERVICE_UNAVAILABLE` | Qdrant or PostgreSQL down |
