# Validation Layer

Three independent LLM validators that check the generated response before it is
returned to the user or stored in the semantic cache.

## Validators

| Validator | Question | Score Target | Fail Action |
|---|---|---|---|
| `gatekeeper.py` | Does the response address the question asked? | ≥ 0.70 | Replan |
| `auditor.py` | Is every claim grounded in the retrieved context? | ≥ 0.75 | Replan |
| `strategist.py` | Does the response make domain sense? | ≥ 0.65 | Flag |

## Why Three Separate LLM Calls?

Self-evaluation is biased — the same LLM that generated the response will tend to
validate it favourably. Three independent calls with different system prompts, each
checking a different dimension, catch more failure modes than one combined check.

## Execution Pattern

```python
# Always run gatekeeper + auditor in parallel
gatekeeper_result, auditor_result = await asyncio.gather(
    gatekeeper.check(query=state["query"], response=state["draft_response"]),
    auditor.audit(response=state["draft_response"], chunks=state["retrieved_chunks"]),
)
```

Run strategist only for domain-sensitive queries (medical, legal, financial).

## Adding a Validator

1. Create `{name}.py` with a class that has an `async def validate(...)` method
2. Method must return `{"passed": bool, "score": float, "reasoning": str}`
3. Must handle `json.JSONDecodeError` and return a safe default (fail closed)
4. Wire into `nodes.py` → `validate` node using `asyncio.gather`

## Critical Rules

- All validators must use `temperature=0` (deterministic, not creative)
- All validators must return JSON only (end system prompt with "Return ONLY valid JSON")
- Fail closed: on JSON parse error → return `{"passed": False, "score": 0.0}`
- A response that fails validation is NEVER stored in the semantic cache

## Testing Validators

```bash
# Test that auditor catches hallucinations
pytest tests/unit/test_validation.py::test_auditor_catches_hallucination -v

# Test that gatekeeper catches off-topic responses
pytest tests/unit/test_validation.py::test_gatekeeper_catches_off_topic -v

# Deliberately trigger in integration tests
pytest tests/integration/test_reasoning.py::test_validation_on_bad_retrieval -v
```
