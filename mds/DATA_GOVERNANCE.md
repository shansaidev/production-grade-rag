# Data Governance

GDPR compliance, data retention, PII handling, and deletion procedures.

---

## Data Classification

| Data Type | Location | Classification | Retention |
|---|---|---|---|
| Raw uploaded documents | MinIO | As classified by uploader | Until hard delete |
| Chunk text (parsed) | PostgreSQL `chunks` | As classified by uploader | Until hard delete |
| Vector embeddings | Qdrant | Derived, non-personal | Until document hard delete |
| User queries | PostgreSQL `queries` | May contain PII | 90 days default |
| LLM responses | PostgreSQL `queries` | May reference PII | 90 days default |
| Validation scores | PostgreSQL `validations` | Metadata only | 90 days |
| Evaluation results | PostgreSQL `evaluations` | Aggregate metrics | Indefinite |
| Deletion audit trail | PostgreSQL `deletion_audit` | Compliance record | 7 years |
| Cache entries | Redis DB1 | As source document | 24h TTL (factual) |

---

## Document Lifecycle & Retention

```
Upload → [Active: searchable, retrievable]
           │
           ├── Version bump: old version → Superseded (invisible to search, text kept)
           │
           └── Soft delete → [Deleted: invisible to search, text kept 30 days]
                               │
                               └── Hard delete → [Purged: all data removed]
```

### Retention Periods

| State | Text in PostgreSQL | Vectors in Qdrant | Cache in Redis |
|---|---|---|---|
| Active | ✅ | ✅ | ✅ (TTL) |
| Superseded | ✅ (audit) | ❌ removed | ❌ invalidated |
| Soft deleted | ✅ (30-day hold) | ❌ removed immediately | ❌ invalidated immediately |
| Hard deleted | ❌ removed | ❌ (already removed) | ❌ (already removed) |

**Why soft delete removes from Qdrant immediately:** A soft-deleted document must stop being retrievable the moment the delete is requested. Waiting for a batch job creates a window where deleted content can appear in search results.

---

## GDPR Compliance

### Right to Erasure (Article 17)

When a data subject requests erasure of their data:

```bash
# Step 1: Identify documents containing their data
# (document uploaded by or about the subject)
GET /api/v1/documents?user_id=subject-123

# Step 2: Soft delete each document
DELETE /api/v1/documents/{doc_id}
# Body: {"reason": "GDPR erasure request Art.17", "deleted_by": "dpo@company.com"}

# Step 3: Allow 30-day review period, then hard delete
POST /api/v1/documents/{doc_id}/hard-delete

# Step 4: Anonymise query audit trail
# Queries containing their data are anonymised (not deleted — needed for audit)
UPDATE queries 
SET raw_query = '[REDACTED - GDPR Art.17]', 
    final_response = '[REDACTED]'
WHERE user_id = 'subject-123';

# Step 5: Record in deletion_audit
# This happens automatically via the hard-delete endpoint

# Step 6: Confirm to the data subject
# deletion_audit.hard_deleted_at is the proof of erasure
```

### Right to Access (Article 15)

```bash
# Export all data associated with a user
SELECT q.query_id, q.raw_query, q.created_at, q.final_response
FROM queries q
WHERE q.user_id = 'subject-123'
ORDER BY q.created_at;
```

### Data Processing Records (Article 30)

This system processes:
- **Purpose:** Question answering grounded in organisational knowledge
- **Legal basis:** Legitimate interest (internal operational use)
- **Categories:** Business documents (HR policies, procedures, technical docs)
- **Recipients:** Internal employees only
- **Transfers:** No cross-border transfer (data stays on-premises or in agreed cloud region)
- **Retention:** Document text 30-day soft-delete window + 7-year deletion audit

---

## PII Handling in Queries

User queries are stored in the `queries` table for audit trail and evaluation purposes.
Queries may contain PII (names, employee IDs, personal situations).

**Auto-anonymisation after 90 days:**
```sql
-- Run as a scheduled Celery beat task (daily)
UPDATE queries
SET raw_query = '[REDACTED AFTER 90 DAYS]',
    final_response = '[REDACTED AFTER 90 DAYS]'
WHERE created_at < NOW() - INTERVAL '90 days'
  AND raw_query != '[REDACTED AFTER 90 DAYS]';
```

**Immediate anonymisation on request:**
```python
await session.execute(
    text("UPDATE queries SET raw_query='[REDACTED]', final_response='[REDACTED]' WHERE user_id=:uid"),
    {"uid": user_id}
)
```

---

## Access Control

### Document Access Levels

| `access_level` | Who can retrieve |
|---|---|
| `public` | All authenticated users |
| `internal` | Employees of the organisation |
| `restricted` | Specific department only (matched against user's department claim) |

Access level is stored in both PostgreSQL (`documents.access_level`) and Qdrant payload
(for pre-filtering before vector search).

### Department Scoping

Queries are automatically scoped to the user's authorised departments.
A user in `department=hr` cannot retrieve chunks from `department=legal` even if the
query text is identical — the Qdrant payload pre-filter enforces this before ANN search.

---

## Audit Trail

The `deletion_audit` table provides a permanent record of all deletions:

```sql
SELECT 
    doc_id,
    deleted_by,
    reason,
    deleted_at,
    hard_deleted_at,
    CASE 
        WHEN hard_deleted_at IS NOT NULL THEN 'Fully purged'
        WHEN deleted_at IS NOT NULL THEN 'Soft deleted (recoverable)'
    END AS status
FROM deletion_audit
ORDER BY deleted_at DESC;
```

This table is **never purged** — it is the compliance record.
Minimum retention: 7 years (standard compliance requirement).
