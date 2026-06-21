# Models Reference

LLM and embedding model options, comparison, and swap procedures.

---

## Current Configuration

```env
LLM_MODEL=llama3.2:8b           # generation + validation + planning
EMBEDDING_MODEL=nomic-embed-text # dense embeddings (768-dim)
SPARSE_MODEL=SPLADE_PP_en_v1     # sparse embeddings (via FastEmbed)
RERANKER_MODEL=ms-marco-MiniLM-L-12-v2  # cross-encoder reranker
```

---

## LLM Options

### Local (Ollama) — default

| Model | RAM | Quality | Speed | Best For |
|---|---|---|---|---|
| `llama3.2:3b` | ~3 GB | ★★★ | ★★★★★ | Planning, routing only |
| `llama3.2:8b` | ~6 GB | ★★★★ | ★★★★ | **Default — good balance** |
| `llama3.2:70b` | ~40 GB | ★★★★★ | ★★★ | Final generation (64GB machine can run this) |
| `mistral:7b` | ~5 GB | ★★★★ | ★★★★ | Alternative to llama3.2:8b |

**Your 64GB machine:** Run `llama3.2:8b` for all calls, or use `llama3.2:3b` for
planning/routing and `llama3.2:70b` for final generation. Both fit in RAM simultaneously.

### API Providers

| Provider | Model | Input $/1M | Output $/1M | Quality |
|---|---|---|---|---|
| OpenAI | `gpt-4o-mini` | $0.15 | $0.60 | ★★★★ |
| OpenAI | `gpt-4o` | $5.00 | $15.00 | ★★★★★ |
| Anthropic | `claude-haiku-4-5` | $0.25 | $1.25 | ★★★★ |
| Anthropic | `claude-sonnet-4-6` | $3.00 | $15.00 | ★★★★★ |

### How to Switch LLM

```env
# Switch to OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

# Switch to Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-haiku-4-5

# Back to local Ollama
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:8b
```

No code changes. Restart API after `.env` change.

---

## Embedding Models (Dense)

Changing the embedding model **requires a full corpus re-index**.
See `docs/DEPLOYMENT.md` for the blue-green reindex procedure.

| Model | Dims | Size | Quality | Privacy | Cost |
|---|---|---|---|---|---|
| `nomic-embed-text` (Ollama) | 768 | 274 MB | ★★★★ | ✅ Local | Free |
| `text-embedding-3-small` (OpenAI API) | 1536 | API | ★★★★★ | ❌ Cloud | $0.02/1M tokens |
| `text-embedding-3-large` (OpenAI API) | 3072 | API | ★★★★★ | ❌ Cloud | $0.13/1M tokens |
| `bge-large-en-v1.5` (Ollama) | 1024 | 1.3 GB | ★★★★ | ✅ Local | Free |
| `e5-large-v2` (Ollama) | 1024 | 1.3 GB | ★★★★ | ✅ Local | Free |

**Recommendation:** Stay on `nomic-embed-text` unless faithfulness < 0.75 after tuning.
Switching to `text-embedding-3-small` costs money but is the clearest quality upgrade.

### Embedding Model Swap Procedure

```bash
# 1. Update .env
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
LLM_PROVIDER=openai   # embeddings also come from OpenAI now

# 2. Start blue-green reindex
curl -X POST http://localhost:8000/api/v1/admin/reindex \
  -d '{"new_model": "text-embedding-3-small", "dimensions": 1536}'

# 3. Monitor (may take hours for large corpora)
watch -n 30 'curl -s http://localhost:8000/api/v1/admin/reindex/status | python3 -m json.tool'

# 4. Evaluate new model BEFORE switching live traffic
python -m src.evaluation.runner
# Compare faithfulness: new model vs old model

# 5. Only if new model is BETTER: cutover
curl -X POST http://localhost:8000/api/v1/admin/reindex/cutover

# 6. Flush semantic cache (old entries used old model embeddings)
redis-cli -n 1 FLUSHDB

# 7. If new model is WORSE: rollback
curl -X POST http://localhost:8000/api/v1/admin/reindex/rollback
# No impact on live traffic. Old collection still serving queries.
```

---

## Sparse Embedding Model (SPLADE)

| Model | Size | Quality | Use Case |
|---|---|---|---|
| `prithivida/Splade_PP_en_v1` | ~400 MB | ★★★★ | **Default — general English** |
| `naver/splade-cocondenser-ensembledistil` | ~400 MB | ★★★★★ | Best quality, slightly slower |
| BM25 (rank_bm25 library) | tiny | ★★★ | Fallback if SPLADE underperforms your domain |

**Check if SPLADE is helping:** `python -m src.evaluation.qdrant_metrics`
If `fusion_gain < 0.03`, SPLADE isn't contributing — check domain vocabulary coverage.

---

## Reranker Options

| Model | Speed | Quality | Privacy | When to Use |
|---|---|---|---|---|
| `ms-marco-MiniLM-L-12-v2` (FlashRank) | ★★★★★ | ★★★ | ✅ Local | **Default** |
| `ms-marco-MiniLM-L-12-v2` (sentence-transformers) | ★★★★ | ★★★★ | ✅ Local | If latency allows |
| `BAAI/bge-reranker-v2-m3` | ★★★ | ★★★★★ | ✅ Local | Multilingual content |
| Cohere `rerank-english-v3.0` | ★★★★ | ★★★★★ | ❌ Cloud | Best quality, cloud OK |

### Switch Reranker

```python
# src/retrieval/reranker.py
# Change model_name — no other code changes needed
_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")  # default

# For BGE multilingual
from sentence_transformers import CrossEncoder
_ranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# For Cohere
import cohere
co = cohere.Client(api_key=settings.cohere_api_key)
results = co.rerank(query=query, documents=[c["text"] for c in chunks], model="rerank-english-v3.0")
```

**Always measure impact:** Run RAGAS before and after. Check MRR delta.
