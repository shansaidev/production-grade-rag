# Retrieval Pipeline

Three-stage pipeline: Qdrant hybrid search → reranking → context assembly.

## Pipeline Order (Non-Negotiable)

```
Qdrant(N=50) → FlashRank(→20) → RuleBoost → MMR(→K=5) → LLM
```

Never skip stages. Never change the order.

## Files

| File | Stage | Purpose |
|---|---|---|
| `hybrid_searcher.py` | 0 — Retrieval | Qdrant dense+sparse+RRF, payload pre-filter |
| `pg_hydrator.py` | 0 — Hydration | Batch SELECT WHERE chunk_id = ANY($1::uuid[]) |
| `reranker.py` | 1 — Semantic | FlashRank cross-encoder: 50 → 20 candidates |
| `rule_reranker.py` | 2 — Deterministic | Recency + authority + keyword score multipliers |
| `mmr.py` | 3 — Diversity | MMR: removes near-duplicate chunks |
| `full_reranker.py` | Orchestrator | Calls stages 1→2→3 in sequence |
| `hyde.py` | Pre-retrieval | Hypothetical answer embedding for recall boost |
| `context_assembler.py` | Post-retrieval | Lost-in-middle ordering: best chunk first+last |

## Key Parameters (.env)

```
QDRANT_PREFETCH_DENSE=50   # N — candidates before reranking
TOP_K_FINAL=5              # K — chunks sent to LLM
MMR_LAMBDA=0.5             # 0=pure diversity, 1=pure relevance
RECENCY_WEIGHT=0.20        # 0=disabled, 0.5=strong recency boost
```

N:K ratio must be ≥ 6:1. Enforced by startup assertion in `Settings`.

## Measuring Pipeline Quality

```bash
python -m src.evaluation.qdrant_metrics   # fusion_gain (sparse contribution)
python -m src.evaluation.metrics          # MRR + NDCG@5 (ranking quality)
python -m src.evaluation.runner           # RAGAS context_precision
```
