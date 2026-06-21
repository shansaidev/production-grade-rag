# Evaluation Guide

How to measure RAG quality, build a golden dataset, interpret RAGAS scores,
and run the full evaluation suite.

---

## Quick Start

```bash
# Run RAGAS evaluation on default 30-pair golden dataset
python -m src.evaluation.runner

# Run with verbose output (shows per-query scores)
python -m src.evaluation.runner --verbose

# Run Qdrant-specific metrics (fusion_gain)
python -m src.evaluation.qdrant_metrics

# Run ranking metrics (MRR, NDCG@5)
python -m src.evaluation.metrics

# Run red team suite
python -m src.stress_testing.red_team

# Run all together
python -m src.evaluation.runner && \
python -m src.evaluation.qdrant_metrics && \
python -m src.evaluation.metrics && \
python -m src.stress_testing.red_team
```

---

## The Four RAGAS Metrics

### Faithfulness (most important — target ≥ 0.80)

**What it measures:** Are all claims in the answer directly supported by the retrieved context?

**Algorithm:**
1. LLM extracts all factual claims from the answer
2. For each claim, LLM checks if it appears in the context
3. `faithfulness = supported_claims / total_claims`

**Low faithfulness means:**
- System prompt too permissive (model adding general knowledge)
- Chunks too large (too much noise, model fills gaps)
- N:K ratio too tight (not enough filtering by reranker)

**How to improve:**
```
faithfulness < 0.70 → fix system prompt first (add "ONLY use provided CONTEXT")
faithfulness 0.70–0.79 → check chunk size (reduce CHUNK_SIZE_TOKENS to 300)
faithfulness 0.79–0.85 → check reranker (ensure N=50, K=5, 10:1 ratio)
faithfulness ≥ 0.85 → production-ready
```

### Answer Relevancy (target ≥ 0.70)

**What it measures:** Does the answer actually address what was asked?

**Algorithm:**
1. LLM generates N hypothetical questions the answer would address
2. Compute cosine similarity between those and the original query
3. `answer_relevancy = mean(similarity)`

**Low relevancy means:** Off-topic responses, Gatekeeper threshold too low.

### Context Recall (target ≥ 0.75)

**What it measures:** Did retrieval surface all the relevant information?

**Requires:** Ground truth answer in golden dataset.

**Algorithm:**
1. Extract claims from ground truth answer
2. Check each claim against retrieved chunks
3. `context_recall = attributable_claims / total_ground_truth_claims`

**Low recall means:** top_k too small, or relevant chunks not being retrieved.

### Context Precision (target ≥ 0.75)

**What it measures:** Were the retrieved chunks actually useful?

**Algorithm:** Checks whether useful chunks rank above useless ones.

**Low precision means:** Too many irrelevant chunks retrieved. Add or improve reranker.

---

## Qdrant-Specific Metrics

### Fusion Gain (target ≥ 0.05)

Measures how much the sparse (SPLADE) vectors contribute beyond dense-only retrieval.

```
fusion_gain = hybrid_recall@10 - max(dense_recall@10, sparse_recall@10)
```

If `fusion_gain < 0.03`: sparse vectors aren't helping.
- Check SPLADE tokenisation against your domain vocabulary
- Consider switching sparse model

If `fusion_gain > 0.10`: hybrid is working very well.

---

## Ranking Metrics

### MRR — Mean Reciprocal Rank (target ≥ 0.60)

How early does the first relevant chunk appear in the ranked list?

```
MRR = mean(1 / rank_of_first_relevant_chunk)
```

MRR = 1.0 → first result is always relevant  
MRR = 0.2 → first relevant result is typically at position 5

**Should improve 0.15–0.40 after adding the reranker.**

### NDCG@5 — Normalised Discounted Cumulative Gain

Are the most relevant chunks ranked highest?

**Requires graded relevance** in golden dataset (0 = irrelevant, 1 = relevant, 2 = highly relevant).

Target: NDCG@5 > 0.80

---

## Building the Golden Dataset

The golden dataset is ground truth for evaluation. Quality here determines the reliability of all metrics.

### File location
`tests/golden_dataset/qa_pairs.json`

### Schema

```json
[
  {
    "id": "q001",
    "question": "How many weeks of parental leave do California employees receive?",
    "ground_truth": "California employees receive 12 weeks of parental leave under CFRA.",
    "ground_truth_chunk_keywords": ["California", "12 weeks", "CFRA", "parental leave"],
    "query_type": "factual",
    "relevant_chunk_ids": ["uuid-of-the-chunk-that-contains-this-answer"],
    "relevance_grades": {
      "uuid-of-chunk-1": 2,
      "uuid-of-chunk-2": 1,
      "uuid-of-chunk-3": 0
    },
    "notes": "Covered in HR_Policy_2024.pdf section 2.1"
  }
]
```

### Query Types — Target Distribution

| Type | Count | Description | Example |
|---|---|---|---|
| `factual` | 10 | Direct lookup — one answer, one chunk | "What is the FMLA deadline?" |
| `comparative` | 10 | Multi-chunk synthesis | "How does CA leave differ from federal?" |
| `procedural` | 10 | Sequential steps | "How do I apply for parental leave?" |

### Rules for Good Q&A Pairs

1. **Write questions the way real users ask** — not "What does section 2.1 state?" but "How much parental leave do I get?"
2. **Ground truth must come from the document** — copy the exact relevant sentence, then paraphrase for the answer
3. **Include `relevant_chunk_ids`** — after ingesting documents, find which chunk_id contains the answer via:
   ```bash
   curl -X POST http://localhost:8000/api/v1/query/sync \
     -d '{"query": "your question"}' | jq '.sources[0].chunk_id'
   ```
4. **Add `relevance_grades` for NDCG** — rate the top 5 retrieved chunks: 2=highly relevant, 1=somewhat, 0=not
5. **Cover edge cases** — one question where the answer is NOT in the documents (tests "I don't know" response)

### How Many?

| Phase | Count | Purpose |
|---|---|---|
| Sunday (initial) | 10 | Quick sanity check. Enough for trend direction. |
| Thursday (full) | 30 | Statistically meaningful. Baseline for all future changes. |
| Production | 50+ | One month after launch — add real user queries that failed |

---

## Interpreting Results

### Evidence-Based Iteration (the only valid approach)

```
1. Run evaluation → record all 4 RAGAS scores + MRR + fusion_gain
2. Identify the LOWEST metric
3. Form ONE hypothesis about why it's low
4. Make ONE change
5. Re-run evaluation
6. Did the metric improve?
   YES → record what changed, move to next weakest metric
   NO  → revert the change, form a different hypothesis
```

**Never change two things at once** — you won't know which one helped.

### Score Interpretation Table

| Metric | < 0.60 | 0.60–0.74 | 0.75–0.84 | ≥ 0.85 |
|---|---|---|---|---|
| faithfulness | 🔴 Critical fix needed | 🟠 Significant work | 🟡 Acceptable | 🟢 Production ready |
| answer_relevancy | 🔴 Check gatekeeper | 🟠 Tighten prompts | 🟡 Acceptable | 🟢 Good |
| context_recall | 🔴 Check retrieval | 🟠 Increase top_k | 🟡 Acceptable | 🟢 Good |
| context_precision | 🔴 Add/fix reranker | 🟠 Tune reranker | 🟡 Acceptable | 🟢 Good |

### Storing Results

Every evaluation run is stored in the `evaluations` PostgreSQL table automatically.

Query historical results:
```sql
SELECT run_name, faithfulness, answer_relevancy, context_recall, 
       context_precision, fusion_gain, p95_latency_ms, created_at
FROM evaluations
ORDER BY created_at DESC
LIMIT 10;
```
