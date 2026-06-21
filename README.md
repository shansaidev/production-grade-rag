# Production RAG System

A production-grade Retrieval-Augmented Generation system with hybrid vector search, multi-agent reasoning, semantic caching, and full evaluation.

## Architecture

**Two implementations — same API:**

| | v1 — pgvector | v2 — Qdrant Hybrid |
|---|---|---|
| Vector search | PostgreSQL + pgvector | Qdrant (HNSW + quantization) |
| Keyword search | PostgreSQL tsvector | Qdrant sparse (SPLADE) |
| Hybrid fusion | Manual RRF SQL | Qdrant native RRF |
| Recommendation | Simple stack | **Production default** |

## Quick Start

```bash
git clone https://github.com/yourname/production-rag
cd production-rag
cp .env.example .env

# Install Ollama models (first time only)
ollama pull nomic-embed-text
ollama pull llama3.2:8b

# Start all services
docker compose up -d

# First-time setup
uv sync
python -m alembic upgrade head
python -m src.scripts.setup_qdrant

# Start application
celery -A src.workers.celery_app worker -Q ingestion -c 4 &
uvicorn src.api.main:app --reload --port 8000
```

Open:
- API docs: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard
- RedisInsight: http://localhost:8001
- Celery Flower: http://localhost:5555

## Uploading Documents

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@your_document.pdf" \
  -F "department=engineering"
```

## Querying

```bash
curl -X POST http://localhost:8000/api/v1/query/sync \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the parental leave policy?"}'
```

Response includes `answer`, `sources` (with page citations), `confidence`, and `validation` status.

## Key Features

- **Structure-aware chunking** — tables never split, headings preserved
- **Dual embedding** — dense (nomic-embed-text) + sparse (SPLADE) vectors
- **Hybrid retrieval** — Qdrant native RRF fusion
- **3-stage reranking** — FlashRank → Rule-based → MMR diversity
- **HyDE** — hypothetical document embeddings for recall improvement
- **Lost-in-middle fix** — context assembly with best chunk first+last
- **3-agent reasoning** — Retriever + Reasoner + Verifier via LangGraph
- **3-layer validation** — Gatekeeper + Auditor + Strategist
- **Semantic caching** — Redis Stack (redisvl HNSW, 24h TTL, doc-level invalidation)
- **Full lifecycle** — soft/hard delete, version bump, blue-green reindex
- **RAGAS evaluation** — faithfulness, relevancy, recall, precision
- **Red teaming** — prompt injection, info evasion, bias tests (11 tests)
- **Structured observability** — structlog JSON, p95 latency per node

## Running Tests

```bash
pytest tests/unit/ -v                    # fast, no services needed
pytest tests/integration/ -v             # requires all services running
pytest tests/integration/test_production_gate.py -v -s  # production gate
```

## Evaluation

```bash
python -m src.evaluation.runner          # RAGAS on golden dataset
python -m src.stress_testing.red_team    # red team test suite
```

## Documentation

| Document | Purpose |
|---|---|
| `docs/PRODUCTION_RAG_QDRANT_HYBRID.md` | Full v2 architecture (17 sections, C4 diagrams) |
| `docs/PRODUCTION_RAG_SYSTEM.md` | Full v1 architecture (15 sections) |
| `docs/RAG_THEORY_COMPLETE.md` | Theory for all 33 RAG topics |
| `docs/RAG_IMPLEMENTATION_GUIDE.md` | Step-by-step build guide with tests |
| `CLAUDE.md` | AI assistant context (read by Claude Code) |
| `SKILLS.md` | Coding conventions for this project |

## System Requirements

- Python 3.12+
- Docker Desktop with WSL2 (Windows) or Docker Engine (Linux/Mac)
- 16 GB RAM minimum, 32 GB recommended, 64 GB for local 70B LLM
- 20 GB disk space (Docker images + Ollama models)
