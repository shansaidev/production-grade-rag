# Golden Dataset

Ground truth Q&A pairs used for RAGAS evaluation and ranking metrics.

**File:** `tests/golden_dataset/qa_pairs.json`  
**Current size:** 30 pairs (10 factual + 10 comparative + 10 procedural)  
**Owner:** Domain experts (not engineers)

---

## Schema

```json
{
  "id": "q001",
  "question": "How many weeks of parental leave do California employees receive?",
  "ground_truth": "California employees receive 12 weeks of parental leave under CFRA.",
  "ground_truth_chunk_keywords": ["California", "12 weeks", "CFRA", "parental leave"],
  "query_type": "factual",
  "relevant_chunk_ids": ["uuid-of-the-chunk-containing-the-answer"],
  "relevance_grades": {
    "uuid-chunk-1": 2,
    "uuid-chunk-2": 1,
    "uuid-chunk-3": 0
  },
  "document_source": "HR_Policy_2024.pdf",
  "notes": "Section 2.1 — California Leave Policies"
}
```

### Field Reference

| Field | Required | Description |
|---|---|---|
| `id` | ✅ | Unique ID: `q001`, `q002`, etc. |
| `question` | ✅ | As a real user would ask it (not "What does section 2.1 state?") |
| `ground_truth` | ✅ | The correct, complete answer |
| `ground_truth_chunk_keywords` | ✅ | Key terms that should appear in the relevant chunk |
| `query_type` | ✅ | `factual` / `comparative` / `procedural` |
| `relevant_chunk_ids` | ✅ | After ingestion, find the chunk_id that contains the answer |
| `relevance_grades` | Recommended | 2=highly relevant, 1=somewhat, 0=not. Needed for NDCG@5. |
| `document_source` | ✅ | Which document contains the answer |
| `notes` | ❌ | Optional: section reference, edge case notes |

---

## Query Type Distribution

| Type | Count | What it tests |
|---|---|---|
| `factual` | 10 | Direct lookup — one clear answer, one chunk |
| `comparative` | 10 | Multi-chunk synthesis — "how does X differ from Y" |
| `procedural` | 10 | Sequential steps — "how do I do X" |

**Add at least 1 unanswerable question** — where the answer genuinely isn't in any document.
The system should respond "I cannot find this in the provided documents" (tests hallucination resistance).

---

## How to Add a New Q&A Pair

### Step 1: Write the question
Write it the way a real user would type it in a chat interface.

✅ `"How much parental leave do I get as a California employee?"`  
❌ `"According to section 2.1, what is the CFRA entitlement?"`

### Step 2: Write the ground truth answer
Must come directly from the document. Paraphrase — don't copy verbatim.

### Step 3: Find the chunk_id
After ingesting the document:
```bash
curl -X POST http://localhost:8000/api/v1/query/sync \
  -H "Content-Type: application/json" \
  -d '{"query": "YOUR QUESTION HERE"}' | python3 -m json.tool
# Look at sources[0].chunk_id in the response
```

Verify the chunk actually contains the answer by reading the excerpt.

### Step 4: Add relevance grades (for NDCG)
```bash
# Get top 5 retrieved chunks
curl -X POST http://localhost:8000/api/v1/query/sync \
  -d '{"query": "YOUR QUESTION", "top_k": 5}'
```

For each chunk in `sources`, grade 0/1/2 and add to `relevance_grades`.

### Step 5: Validate your entry

```python
# Run quick validation
python3 -c "
import json, sys
with open('tests/golden_dataset/qa_pairs.json') as f:
    data = json.load(f)
required = {'id', 'question', 'ground_truth', 'query_type', 'relevant_chunk_ids', 'document_source'}
for item in data:
    missing = required - set(item.keys())
    if missing:
        print(f'MISSING in {item[\"id\"]}: {missing}')
    if item['query_type'] not in ['factual', 'comparative', 'procedural']:
        print(f'BAD TYPE in {item[\"id\"]}: {item[\"query_type\"]}')
print('Validation done')
"
```

---

## Updating chunk_ids After Re-Ingestion

When documents are re-ingested (version bump), chunk_ids change.
After re-ingestion, refresh `relevant_chunk_ids` for affected pairs:

```bash
python -m src.evaluation.refresh_chunk_ids \
  --golden tests/golden_dataset/qa_pairs.json \
  --doc-id YOUR_DOC_ID
```

This script re-queries each question and updates the chunk_ids automatically.

---

## Current Coverage

- Documents covered: HR_Policy_2024.pdf
- Departments: hr (all 30 current pairs)
- Edge cases: 1 unanswerable question (q030)

**To add coverage for a new department:** add 10 pairs from a document in that department,
maintaining the 10/10/10 factual/comparative/procedural split.
