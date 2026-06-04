---
title: "API Reference — Ariadne"
description: "AriadneMemory API: remember, recall, forget, update, graph, add_edge, consolidate, stats. Complete method reference."
---


The high-level API implementing the Hermes agent memory protocol. Wraps `AriadneDB` and `Deduplicator` into a clean, unified interface.

## Constructor

```python
from arriadne import AriadneMemory, AriadneConfig

# Default configuration
mem = AriadneMemory()

# Custom database path and embedding dimension
mem = AriadneMemory(db_path="my_memory.db", embedding_dim=384)

# Full configuration
config = AriadneConfig(
    db_path="my_memory.db",
    embedding_dim=384,
    faiss_type="auto",
    dedup_threshold=0.8,
    dedup_num_perm=128,
)
mem = AriadneMemory(config=config)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `AriadneConfig \| None` | `None` | Full configuration object |
| `db_path` | `str \| Path \| None` | `None` | Database file path (overrides config) |
| `embedding_dim` | `int \| None` | `None` | Embedding dimension (overrides config) |

## Context Manager

```python
with AriadneMemory(db_path="memory.db") as mem:
    mem.remember("Hello world")
# Automatically closed on exit
```

---

## `remember()`

Store a new memory. Performs automatic deduplication and contradiction detection.

```python
result = mem.remember(
    content="User prefers dark mode",
    memory_type="semantic",
    importance=0.8,
    embedding=[0.1, 0.2, ...],  # optional
    entities=["user", "preferences"],
    metadata={"category": "ui"},
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | *required* | Text content to store |
| `memory_type` | `str` | `"semantic"` | Category: `semantic`, `episodic`, `procedural` |
| `importance` | `float` | `0.5` | Importance score (0.0–1.0) |
| `embedding` | `list[float] \| np.ndarray \| None` | `None` | Embedding vector |
| `entities` | `list[str] \| None` | `None` | Entity names to associate |
| `metadata` | `dict[str, Any] \| None` | `None` | JSON-serializable metadata |

### Returns

```python
{
    "memory_id": int | None,    # ID of created memory
    "status": str,               # "created", "duplicate", or "error"
    "duplicate_of": int | None,  # ID of duplicate (if status="duplicate")
    "contradictions": list,      # Contradiction details (if any found)
    "error": str | None,         # Error message (if status="error")
}
```

### Example

```python
# Store a fact
result = mem.remember(
    content="PostgreSQL supports JSONB columns",
    memory_type="semantic",
    importance=0.9,
    entities=["PostgreSQL"],
)
print(result)
# {'memory_id': 1, 'status': 'created'}

# Duplicate detection
result = mem.remember("PostgreSQL supports JSONB columns")
print(result)
# {'memory_id': None, 'status': 'duplicate', 'duplicate_of': 1}
```

---

## `recall()`

Search for memories matching a query. Uses hybrid search (vector + FTS5) when an embedding is provided, FTS5-only otherwise.

```python
results = mem.recall(
    query="database features",
    embedding=[0.1, 0.2, ...],  # optional
    k=10,
    type_filter="semantic",
    time_range=(start_timestamp, end_timestamp),
    importance_min=0.5,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *required* | Text query for FTS5 |
| `embedding` | `list[float] \| np.ndarray \| None` | `None` | Query embedding for vector search |
| `k` | `int` | `10` | Number of results to return |
| `type_filter` | `str \| None` | `None` | Filter by memory type |
| `time_range` | `tuple[float, float] \| None` | `None` | (start, end) Unix timestamps |
| `importance_min` | `float \| None` | `None` | Minimum importance threshold |

### Returns

`list[dict[str, Any]]` — Each dict contains:

```python
{
    "id": int,
    "content": str,
    "content_hash": str,
    "memory_type": str,
    "importance": float,
    "created_at": float,        # Unix timestamp
    "updated_at": float,
    "accessed_at": float,
    "access_count": int,
    "retention_strength": float,
    "is_deleted": bool,
    "metadata": dict | None,
    "score": float,             # RRF score or FTS rank
    "search_type": str,         # "hybrid", "vector", or "fts"
}
```

### Example

```python
# Search with embedding (hybrid)
results = mem.recall("dark mode preferences", k=5)
for r in results:
    print(f"[{r['search_type']}] score={r['score']:.4f} | {r['content'][:60]}")

# Filtered search
results = mem.recall(
    "deployment",
    k=10,
    type_filter="procedural",
    importance_min=0.7,
)
```

---

## `forget()`

Delete a memory by ID.

```python
forgotten = mem.forget(memory_id=42, hard=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_id` | `int` | *required* | Memory to delete |
| `hard` | `bool` | `False` | If `True`, permanently delete. Otherwise soft-delete. |

### Returns

`bool` — `True` if forgotten, `False` if not found.

### Example

```python
# Soft delete (can be recovered)
mem.forget(memory_id=42, hard=False)

# Hard delete (permanent)
mem.forget(memory_id=42, hard=True)
```

---

## `update()`

Update an existing memory's content, importance, embedding, or metadata.

```python
updated = mem.update(
    memory_id=42,
    content="Updated content here",
    importance=0.9,
    embedding=new_embedding,
    metadata={"updated": True},
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_id` | `int` | *required* | Memory to update |
| `content` | `str \| None` | `None` | New content |
| `importance` | `float \| None` | `None` | New importance score |
| `embedding` | `list[float] \| np.ndarray \| None` | `None` | New embedding vector |
| `metadata` | `dict[str, Any] \| None` | `None` | New metadata |

### Returns

`bool` — `True` if updated, `False` if not found.

### Example

```python
# Update importance
mem.update(memory_id=42, importance=0.95)

# Update content (also updates dedup index)
mem.update(memory_id=42, content="Corrected information")
```

---

## `graph()`

Traverse the knowledge graph from a starting entity.

```python
result = mem.graph(
    entity="Ariadne",
    edge_type="uses",
    hops=3,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity` | `str` | *required* | Starting entity name |
| `edge_type` | `str \| None` | `None` | Filter by edge type |
| `hops` | `int` | `1` | Maximum traversal depth |

### Returns

```python
{
    "nodes": list[str],    # Entity names reachable within hops
    "edges": [             # Connections between nodes
        {
            "source": str,
            "target": str,
            "type": str,
            "weight": float,
        }
    ]
}
```

### Example

```python
# Build a graph
mem.add_edge("Ariadne", "FAISS", edge_type="uses")
mem.add_edge("Ariadne", "SQLite", edge_type="uses")
mem.add_edge("FAISS", "vector", edge_type="related")

# Traverse
result = mem.graph("Ariadne", hops=2)
print(result["nodes"])
# ['Ariadne', 'FAISS', 'SQLite', 'vector']
```

---

## `add_edge()`

Add a directed edge between two entities.

```python
mem.add_edge(
    source="Ariadne",
    target="FAISS",
    edge_type="uses",
    weight=1.0,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | *required* | Source entity name |
| `target` | `str` | *required* | Target entity name |
| `edge_type` | `str` | `"related"` | Relationship type |
| `weight` | `float` | `1.0` | Edge weight (0.0–1.0) |

---

## `stats()`

Get comprehensive memory system statistics.

```python
stats = mem.stats()
print(stats)
```

### Returns

```python
{
    "total_memories": int,
    "active_memories": int,
    "deleted_memories": int,
    "by_type": {"semantic": 150, "episodic": 45, ...},
    "total_entities": int,
    "total_edges": int,
    "total_memory_links": int,
    "total_consolidations": int,
    "faiss_vectors": int,
    "faiss_type": str,            # "IndexFlatIP" or "IndexIVFFlat"
    "faiss_dimension": int,
    "avg_importance": float,
    "dedup_index_size": int,
    "db_size_bytes": int,
}
```

### Example

```python
stats = mem.stats()
print(f"Active memories: {stats['active_memories']}")
print(f"FAISS vectors: {stats['faiss_vectors']} ({stats['faiss_type']})")
print(f"Graph: {stats['total_entities']} entities, {stats['total_edges']} edges")
```

---

## `consolidate()`

Run memory consolidation — groups similar memories and creates merged summaries.

```python
groups_created = mem.consolidate()
```

### Returns

`int` — Number of consolidation groups created.

---

## `evict()`

Run priority-based memory eviction — soft-deletes the lowest-priority memories.

```python
evicted_count = mem.evict()
```

### Returns

`int` — Number of memories evicted.

---

## `close()`

Close the memory system, saving all state.

```python
mem.close()
```
