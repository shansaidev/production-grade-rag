## What this PR does

<!-- One sentence. Be specific. -->

## Why

<!-- What problem does this solve, or what ADR does it implement? -->

---

## Type of change

- [ ] Bug fix
- [ ] New feature / component
- [ ] Architecture change (requires ADR update)
- [ ] Documentation only
- [ ] Evaluation / dataset improvement

---

## Checklist

### Code
- [ ] Follows conventions in `SKILLS.md` (async, typed, no hardcoded values)
- [ ] New LangGraph nodes use `@timed_node` decorator
- [ ] PostgreSQL writes happen before Qdrant writes (dual-write order)
- [ ] New parsers registered in `file_router.py`
- [ ] No secrets or API keys in code

### Tests
- [ ] Unit tests added/updated: `pytest tests/unit/ -v` passes
- [ ] Integration tests added/updated: `pytest tests/integration/ -v` passes
- [ ] If retrieval changed: RAGAS evaluation re-run and scores recorded below

### Documentation
- [ ] `CHANGELOG.md` updated (under `[Unreleased]`)
- [ ] If new architectural decision: `docs/DECISIONS.md` updated
- [ ] If API changed: `docs/API.md` updated
- [ ] If new config vars: `.env.example` updated

---

## Evaluation Scores (fill in if retrieval/generation changed)

| Metric | Before | After | Delta |
|---|---|---|---|
| faithfulness | | | |
| answer_relevancy | | | |
| context_precision | | | |
| context_recall | | | |
| p95 latency (ms) | | | |

Run with: `python -m src.evaluation.runner`

---

## Karpathy Checklist

- [ ] Only changed what was required (no drive-by refactors)
- [ ] No speculative features added
- [ ] Style matches existing code (didn't change quote style, add docstrings unprompted, etc.)
- [ ] If architectural change: presented tradeoffs before implementing

<!-- See CLAUDE.md §Surgical Changes for details -->
