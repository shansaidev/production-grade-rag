---
name: Bug report
about: Something isn't working correctly
labels: bug
---

## What happened

<!-- Describe the bug clearly and specifically. -->

## What you expected

## Steps to reproduce

```bash
# Minimal reproduction
```

## Environment

- Version / git commit:
- LLM_MODEL:
- EMBEDDING_MODEL:
- OS / Python version:

## Evaluation impact (if retrieval/quality related)

| Metric | Before bug | Current |
|---|---|---|
| faithfulness | | |
| answer_relevancy | | |
| p95 latency | | |

## Logs

```
# Paste relevant structlog JSON output
# grep '"event": "..."' app.log | tail -20
```

## Additional context

<!-- Relevant section from TROUBLESHOOTING.md you already checked: -->
