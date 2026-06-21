# Security Policy

## Supported Versions

| Version | Security Fixes |
|---|---|
| 0.4.x (current) | ✅ Yes |
| 0.3.x | ✅ Yes (critical only) |
| < 0.3 | ❌ No |

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report privately via: `security@yourcompany.com`

Include:
- Description of the vulnerability and its potential impact
- Steps to reproduce (with a minimal test case if possible)
- The component affected (`src/validation/`, `src/cache/`, etc.)
- Your suggested severity (Critical / High / Medium / Low)

**Response timeline:**
- Acknowledgement: within 48 hours
- Initial assessment: within 5 business days
- Fix or mitigation: within 30 days for Critical/High

---

## Known Attack Surface

This system processes untrusted user queries and retrieves from ingested documents.
The following attack vectors are explicitly tested in `src/stress_testing/`:

### Prompt Injection (CRITICAL)
Attempts to override system instructions via query text or embedded instructions
in retrieved document content (indirect injection).

**Defences in place:**
- `[UNTRUSTED SOURCE]` barrier in system prompt — retrieved content cannot override instructions
- Gatekeeper validator catches off-instruction responses
- Red team suite runs 5 injection tests on every deployment

### Information Extraction (HIGH)
Attempts to extract confidential documents, system prompts, or data the user
shouldn't access.

**Defences in place:**
- Department-scoped retrieval — users only retrieve from their authorised departments
- Payload pre-filter enforced at Qdrant level (not post-filter)
- Validation layer flags responses citing restricted content

### Data Poisoning (HIGH)
Uploading malicious documents designed to corrupt retrieval quality or embed
injection instructions in the vector store.

**Defences in place:**
- Document ingestion requires authentication
- Soft delete + version versioning means poisoned docs can be quickly retired
- `qdrant_synced` tracking allows targeted removal

### Bias Elicitation (MEDIUM)
Queries designed to produce discriminatory or biased outputs.

**Defences in place:**
- Strategist validator applies domain-specific rules
- 3 bias test cases in red team suite

---

## Security Configuration Checklist

Before any deployment:

```bash
# 1. No hardcoded secrets
grep -r "password\|api_key\|secret\|token" src/ --include="*.py" | grep -v ".env\|settings\|test"

# 2. .env is gitignored
cat .gitignore | grep ".env"

# 3. Redis DBs are isolated
# REDIS_URL must use DB 0 (Celery)
# REDIS_CACHE_URL must use DB 1 (cache)
# Never use the same DB number for both

# 4. Qdrant not exposed publicly
# Port 6333/6334 should not be in public security group rules

# 5. MinIO not publicly accessible
# Port 9000 should not be publicly accessible

# 6. Run full red team suite
python -m src.stress_testing.red_team
# Target: 100% CRITICAL pass rate, ≥ 80% overall
```

---

## Dependency Security

```bash
# Scan for known CVEs in dependencies
pip install pip-audit
pip-audit

# Update dependencies
pip list --outdated
```

Monitor: `redisvl`, `qdrant-client`, `langchain`, `langgraph`, `fastapi` — all active projects
with regular security patches.
