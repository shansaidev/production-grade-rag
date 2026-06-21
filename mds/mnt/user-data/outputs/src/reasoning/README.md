# Reasoning Engine (LangGraph)

Stateful query processing graph: from user query to validated, cited response.

## Graph Structure

```
check_cache ──(hit)──→ format_response → END
     │
   (miss)
     ↓
analyze_query → retrieve → [conditional_route]
                                │          │
                           (simple)   (complex)
                                │          │
                            generate   multi_agent_dispatch
                                │          │
                                └────┬─────┘
                                  validate
                                     │          │
                                  (pass)     (fail, retry < 2)
                                     │          │
                               store_cache    retrieve  ← replan
                                     │
                             format_response → END
```

## Files

| File | Purpose |
|---|---|
| `state.py` | `RAGState` TypedDict — all fields shared across nodes |
| `engine.py` | Graph definition: nodes, edges, compile |
| `nodes.py` | All node functions (`async (state) -> RAGState`) |
| `planner.py` | Query decomposition into sub-queries |
| `tools.py` | LangChain tool definitions (search, calculator, etc.) |

## Adding a Node

```python
# 1. nodes.py
@timed_node("my_node")
async def my_node(state: RAGState) -> RAGState:
    result = await do_work(state["query"])
    return {"my_field": result}   # partial state update only

# 2. engine.py
graph.add_node("my_node", nodes.my_node)
graph.add_edge("previous_node", "my_node")
graph.add_edge("my_node", "next_node")
```

**Rules:**
- All nodes must be `async def`
- All nodes must use `@timed_node` (automatic duration_ms logging)
- Return only the fields this node modifies (partial state, not full state copy)
- No direct communication between nodes — use state

## Debugging

```bash
# Start LangGraph Studio for visual graph inspection
langgraph dev

# Or add state printing to a node
async def debug_node(state: RAGState) -> RAGState:
    import json
    print(json.dumps({k: str(v)[:100] for k, v in state.items()}, indent=2))
    return {}
```
