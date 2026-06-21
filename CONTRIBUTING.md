# Contributing & Extending the Production RAG System

## Before You Start

1. Read `CLAUDE.md` — full project context
2. Read `SKILLS.md` — non-negotiable conventions
3. Understand the dual-write contract: **PostgreSQL FIRST, Qdrant SECOND, always**

---

## How to Add a New Document Parser

**When**: you have a new file type (Excel, PowerPoint, Notion export, etc.)

**Steps**:

```bash
# 1. Create the parser
touch src/ingestion/parsers/xlsx_parser.py
```

```python
# src/ingestion/parsers/xlsx_parser.py
from pathlib import Path
from src.ingestion.parsers.base import BaseParser, ParsedDocument, ParsedSection

class XLSXParser(BaseParser):
    def parse(self, file_path: Path, doc_id: str) -> ParsedDocument:
        doc = ParsedDocument(doc_id=doc_id, filename=file_path.name)
        # ... implementation
        # Tables in spreadsheets → section_type="table"
        # Sheet names → section_type="heading"
        return doc
```

```python
# src/ingestion/parsers/file_router.py — register it
MIME_MAP = {
    ...
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": XLSXParser,
    "application/vnd.ms-excel": XLSXParser,
}
```

```python
# tests/unit/test_parser.py — add tests
def test_xlsx_parser_sheet_names_become_headings(): ...
def test_xlsx_parser_data_rows_become_table_chunks(): ...
```

---

## How to Add a New LangGraph Node

**When**: you want a new step in the reasoning pipeline (e.g., a citation formatter, a context compressor)

```python
# src/reasoning/nodes.py — add the node function
from src.core.logging import timed_node

@timed_node("compress_context")
async def compress_context(state: RAGState) -> RAGState:
    """Remove redundant sentences from retrieved chunks before generation."""
    chunks = state["retrieved_chunks"]
    compressed = await run_compression(chunks, state["query"])
    return {"retrieved_chunks": compressed}
```

```python
# src/reasoning/engine.py — wire it in
graph.add_node("compress_context", nodes.compress_context)
# Add BEFORE generation
graph.add_edge("retrieve", "compress_context")
graph.add_edge("compress_context", "generate")
# Remove old edge: graph.add_edge("retrieve", "generate")
```

**Rules:**
- Node must be `async def`
- Node must return `dict` (partial state update)
- Node must use `@timed_node` decorator
- Node must NOT communicate directly with other nodes (use state)

---

## How to Add a New Validation Rule

**When**: domain-specific validation is needed (e.g., "response must include a disclaimer for medical queries")

```python
# src/validation/domain_validator.py
from langchain_core.prompts import ChatPromptTemplate
from src.core.llm_client import get_llm
import json

DOMAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a domain expert. Check if this response follows domain rules. Return JSON only."),
    ("human", "Query: {query}\nResponse: {response}\nReturn: {{\"passed\": bool, \"score\": 0-1, \"reasoning\": str}}")
])

class DomainValidator:
    def __init__(self):
        self.chain = DOMAIN_PROMPT | get_llm()

    async def validate(self, query: str, response: str) -> dict:
        result = await self.chain.ainvoke({"query": query, "response": response})
        try:
            return json.loads(result.content)
        except json.JSONDecodeError:
            return {"passed": False, "score": 0.0, "reasoning": "Parse error"}
```

Wire into `src/reasoning/nodes.py` → `validate` node alongside existing validators.

---

## How to Change the Embedding Model

**This requires a full corpus re-index. Plan accordingly.**

```bash
# 1. Start blue-green reindex with new model
curl -X POST http://localhost:8000/api/v1/admin/reindex \
  -H "Content-Type: application/json" \
  -d '{"new_model": "text-embedding-3-large", "dimensions": 3072}'

# 2. Monitor progress
curl http://localhost:8000/api/v1/admin/reindex/status
# {"done": 450, "remaining": 50, "total": 500, "pct_complete": 90.0}

# 3. Run evaluation on new model BEFORE cutover
python -m src.evaluation.runner --collection rag_chunks_v2

# 4. If new model is better → cutover
curl -X POST http://localhost:8000/api/v1/admin/reindex/cutover

# 5. If worse → rollback (drop the new collection, no impact on traffic)
curl -X POST http://localhost:8000/api/v1/admin/reindex/rollback

# 6. After cutover: flush semantic cache (old cached answers used old model embeddings)
redis-cli -n 1 FLUSHDB

# 7. Update .env
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072
```

---

## How to Add a New API Endpoint

```python
# src/api/routers/documents.py (or create new router file)
from fastapi import APIRouter, Depends
from src.api.schemas.document import MyRequestSchema, MyResponseSchema

router = APIRouter(prefix="/api/v1")

@router.post("/my-endpoint", response_model=MyResponseSchema)
async def my_endpoint(
    request: MyRequestSchema,
    session: AsyncSession = Depends(get_db_session),
) -> MyResponseSchema:
    """
    Brief description.
    
    Returns: MyResponseSchema with ...
    """
    result = await do_work(request, session)
    return MyResponseSchema(data=result, metadata={"request_id": str(uuid4())})
```

```python
# src/api/main.py — register the router
from src.api.routers.my_router import router as my_router
app.include_router(my_router)
```

---

## How to Add a Red Team Test Case

```python
# src/stress_testing/test_cases/my_category.py
MY_TESTS = [
    {
        "name": "descriptive_test_name",
        "query": "The adversarial query text",
        "should_not_contain": ["phrase1", "phrase2"],  # response must NOT contain these
        "severity": "CRITICAL",  # CRITICAL | HIGH | MEDIUM | LOW
        "category": "my_category",
        "rationale": "Why this is an attack and what it tests",
    },
]
```

```python
# src/stress_testing/red_team.py — register it
from src.stress_testing.test_cases.my_category import MY_TESTS

all_tests = (
    ...
    [("my_category", t) for t in MY_TESTS]
)
```

**Severity guide:**
- `CRITICAL`: system prompt leak, PII extraction, instruction override
- `HIGH`: confidential document access, role-switching attempts  
- `MEDIUM`: biased output, leading questions
- `LOW`: formatting manipulation, verbose injection

---

## How to Run the Full Evaluation Suite

```bash
# 1. Unit tests (no services needed, <30s)
pytest tests/unit/ -v

# 2. Integration tests (all services must be running, ~5 min)
pytest tests/integration/ -v --timeout=120

# 3. RAGAS evaluation on golden dataset
python -m src.evaluation.runner
# Outputs: faithfulness, answer_relevancy, context_recall, context_precision

# 4. Qdrant fusion_gain metric
python -m src.evaluation.qdrant_metrics

# 5. Red team suite
python -m src.stress_testing.red_team
# Target: ≥ 80% pass rate, 100% CRITICAL pass rate

# 6. Production gate (end-to-end)
pytest tests/integration/test_production_gate.py -v -s

# 7. Latency profiling
python -m src.evaluation.latency_profiler
# Target: p95 miss path < 3500ms, p95 hit path < 120ms
```

---

## Production Checklist Before Any Deployment

- [ ] All unit tests pass: `pytest tests/unit/ -v`
- [ ] All integration tests pass: `pytest tests/integration/ -v`
- [ ] RAGAS faithfulness ≥ 0.80
- [ ] RAGAS answer_relevancy ≥ 0.70
- [ ] Qdrant fusion_gain ≥ 0.05
- [ ] Red team: 100% CRITICAL pass, ≥ 80% overall
- [ ] p95 latency (miss path) ≤ 3500ms
- [ ] p95 latency (hit path) ≤ 120ms
- [ ] Qdrant reconciler: `chunks_unsynced = 0`
- [ ] No hardcoded secrets (grep -r "password\|api_key\|secret" src/ --include="*.py")
- [ ] All Celery tasks have `max_retries=3`
- [ ] `.env` is in `.gitignore`
