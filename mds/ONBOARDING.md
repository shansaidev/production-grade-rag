# Onboarding Guide

New developer setup and first-day orientation for the Production RAG System.

---

## Day 1: Get Running (2–3 hours)

### Prerequisites

```powershell
# Windows 11 — install these first
choco install git python312 docker-desktop -y

# Then restart, enable WSL2
wsl --install
wsl --set-default-version 2
# Restart again after this
```

### Clone and Setup

```bash
git clone https://github.com/yourname/production-rag
cd production-rag

# Python environment
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# No changes needed for local development
```

### Start Infrastructure

```bash
docker compose up -d

# Wait ~30 seconds, then verify all healthy
docker compose ps
curl http://localhost:6333/healthz    # → {"title":"qdrant"}
curl http://localhost:8001            # → RedisInsight loads
```

### First-time DB Setup

```bash
python -m alembic upgrade head        # creates 8 tables in PostgreSQL
python -m src.scripts.setup_qdrant    # creates rag_chunks collection in Qdrant
```

### Install Ollama Models

```bash
# Download from https://ollama.com/download/windows, then:
ollama pull nomic-embed-text   # 274 MB — dense embeddings
ollama pull llama3.2:8b        # 4.7 GB — LLM (takes a few minutes)
ollama list                    # verify both appear
```

### Start the Application

```bash
# Terminal 1: Celery worker (document ingestion)
celery -A src.workers.celery_app worker --loglevel=info -Q ingestion -c 4

# Terminal 2: FastAPI
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Smoke Test

```bash
# Health check
curl http://localhost:8000/health

# Upload a test document
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@tests/fixtures/sample.pdf" \
  -F "department=engineering"
# → returns {doc_id, job_id}

# Wait ~10 seconds, then query
curl -X POST http://localhost:8000/api/v1/query/sync \
  -H "Content-Type: application/json" \
  -d '{"query": "What topics does this document cover?"}'
# → returns answer with sources
```

You're running. 🎉

---

## Day 1: Understand the Architecture (1 hour)

Read these in order — each is short and focused:

1. **`CLAUDE.md` — Behavioral Guidelines** (top section only, ~10 min)
   The 4 Karpathy principles with RAG-specific examples. These govern how you write code here.

2. **`README.md`** (~5 min)
   Feature list and quick start. You've done this.

3. **`docs/DECISIONS.md`** (~20 min)
   10 Architecture Decision Records. Read ADR-001 (PostgreSQL + Qdrant split) and ADR-003 (3-stage reranking) first — they explain the two most consequential choices.

4. **`docs/GLOSSARY.md`** (~5 min skim)
   Don't read top-to-bottom. Bookmark it. Come back when you encounter an unfamiliar term.

---

## Week 1: Go Deeper

### Run the Full Evaluation Suite

```bash
# Get baseline scores for this system
python -m src.evaluation.runner         # RAGAS (takes ~5 min)
python -m src.evaluation.qdrant_metrics # fusion_gain
python -m src.stress_testing.red_team  # security tests
```

Record your scores. You now have a baseline. Any change you make should be measured against it.

### Read the Theory (Pick What's Relevant to Your Work)

The theory doc (`docs/RAG_THEORY_COMPLETE.md`) has 33 sections. Don't read all of it.
Read sections relevant to the component you're working on:

| Your work | Read theory sections |
|---|---|
| Document parsing / chunking | §8, §9 |
| Embedding or vector search | §3, §10, §11, §14 |
| Hybrid search / RRF | §13, §15 |
| Reranking | §16 |
| LangGraph / agents | §22, §23 |
| Validation / hallucination | §25, §26 |
| Evaluation | §27 |
| Semantic cache | §32 |

### Make Your First Change

Good first tasks for onboarding:
1. **Add a test to the golden dataset** — open `tests/golden_dataset/qa_pairs.json`, add one Q&A pair from a real document, run `python -m src.evaluation.runner`, see your score
2. **Add logging to a node you care about** — find a LangGraph node in `src/reasoning/nodes.py`, add a `@timed_node` decorator if it's missing
3. **Read a parser and add a comment** explaining what a tricky line does — then submit as a PR

---

## Key Mental Models

### PostgreSQL is the source of truth

If Qdrant and PostgreSQL ever disagree, PostgreSQL wins. The reconciler fixes Qdrant.
Never make a decision based on what's in Qdrant if PostgreSQL says otherwise.

### LangGraph state flows forward

The `RAGState` TypedDict is the single record of everything that happened during a query.
Every node receives state and returns a partial update. Nothing communicates outside of state.
If you need a node to know something, add it to `RAGState` first.

### Measure before and after every change

The RAGAS metrics are your objective function. Every change should move at least one metric
without hurting others. If you can't measure it, you don't know if it helped.

### Cache only validated responses

The semantic cache is a trust amplifier. A cached response is returned without re-validation.
So we only cache responses that passed Gatekeeper + Auditor with confidence ≥ 0.70.
A cached hallucination is infinitely worse than a fresh hallucination.

---

## Who to Ask

| Topic | Where to look first |
|---|---|
| Architecture decisions | `docs/DECISIONS.md` |
| Specific error / failure | `docs/TROUBLESHOOTING.md` |
| Unfamiliar term | `docs/GLOSSARY.md` |
| API behaviour | `docs/API.md` |
| Evaluation metrics | `docs/EVALUATION.md` |
| Data deletion / GDPR | `docs/DATA_GOVERNANCE.md` |
| Anything else | Ask in team Slack / create a GitHub issue |
