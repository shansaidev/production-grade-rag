# Tests

---

## Test Suites

### Unit Tests — `tests/unit/` (no external services)

```bash
pytest tests/unit/ -v
# Runs in < 30 seconds. No Docker needed.
```

| File | What It Covers |
|---|---|
| `test_parser.py` | PDF/DOCX/HTML parsers: table extraction, page tracking, no mid-sentence cuts |
| `test_chunker.py` | SmartChunker: table-safe, token limits, heading prefix, overlap |
| `test_validation.py` | Auditor/Gatekeeper: catches hallucinations, passes grounded responses |

### Integration Tests — `tests/integration/` (requires Docker services)

```bash
pytest tests/integration/ -v --timeout=120
# Requires: PostgreSQL, Qdrant, Redis, Ollama all running
```

| File | What It Covers |
|---|---|
| `test_hybrid_search.py` | Qdrant search returns ranked results, department filter works |
| `test_reasoning.py` | LangGraph end-to-end: simple query, complex query, validation failure |
| `test_lifecycle.py` | Soft delete invisible, metadata update filters work, reconciler |
| `test_production_gate.py` | 5 diverse query types on unseen document |
| `test_semantic_cache.py` | Paraphrase hits, personal query bypass, low-confidence not cached |

### Evaluation Tests — `src/evaluation/` (requires full stack + golden dataset)

```bash
python -m src.evaluation.runner           # RAGAS (5-10 min)
python -m src.evaluation.qdrant_metrics   # fusion_gain
python -m src.evaluation.metrics          # MRR + NDCG@5
```

### Security Tests — `src/stress_testing/`

```bash
python -m src.stress_testing.red_team
# 11 tests: 5 injection + 3 evasion + 3 bias
```

---

## Running Specific Tests

```bash
# Single test file
pytest tests/unit/test_chunker.py -v

# Single test function
pytest tests/unit/test_chunker.py::test_table_never_split -v

# All tests matching a keyword
pytest -k "cache" -v

# With coverage
pytest tests/unit/ --cov=src --cov-report=term-missing

# Stop on first failure
pytest tests/unit/ -x
```

---

## Writing New Tests

### Unit Test Template

```python
# tests/unit/test_my_component.py

def test_{what}_{condition}_{expected}():
    """One sentence: what is this testing and why it matters."""
    # Arrange — set up the minimal state needed
    chunker = SmartChunker()
    doc = make_test_doc([...])

    # Act — single action
    result = chunker.chunk(doc)

    # Assert — clear, specific assertion
    assert len([c for c in result if c.chunk_type == "table"]) == 1
    # Why specific: tells you exactly what failed if it breaks
```

### Integration Test Template

```python
# tests/integration/test_my_feature.py
import pytest

@pytest.fixture(autouse=True)
async def cleanup(pg_conn, qdrant_client):
    yield
    # Always clean up test data
    await pg_conn.execute("DELETE FROM documents WHERE filename LIKE 'TEST_%'")

async def test_my_feature(pg_conn, qdrant_client):
    """What this integration covers."""
    # Use TEST_ prefix on all test filenames
    result = await upload_and_search("TEST_sample.pdf", "test query")
    assert result is not None
```

---

## Fixtures

Common fixtures are in `tests/conftest.py`:

| Fixture | What It Provides |
|---|---|
| `pg_conn` | asyncpg connection to test PostgreSQL |
| `qdrant_client` | AsyncQdrantClient pointing to test Qdrant |
| `sample_pdf_path` | Path to `tests/fixtures/sample.pdf` |
| `golden_dataset` | Loaded `tests/golden_dataset/qa_pairs.json` |

---

## CI Gate

All PRs must pass:
```
pytest tests/unit/ -v            # required: all pass
pytest tests/integration/ -v     # required: all pass  
python -m src.evaluation.runner  # required: faithfulness ≥ 0.80
```
