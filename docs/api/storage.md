---
title: "Storage Engine API — Ariadne"
description: "AriadneDB low-level API: direct access to SQLite storage, FAISS vector search, FTS5 keywords, and knowledge graph."
---


Low-level storage API providing direct access to SQLite, FAISS, and graph operations. Used internally by `AriadneMemory` — use this API for advanced operations.

## Constructor

```python
from arriadne import AriadneDB, AriadneConfig

config = AriadneConfig(db_path="memory.db", embedding_dim=384)
db = AriadneDB(config)
db.open()
```

## Context Manager

```python
with AriadneDB(AriadneConfig(db_path="memory.db")) as db:
    db.add_memory("Hello", memory_type="semantic")
# Automatically closed on exit
```

---

## `open()`

Open the database connection, create schema, and load FAISS index.

```python
db.open()
```

::: tip
Called automatically on first use if not called explicitly.
:::

## `close()`

Close the database connection and save the FAISS index to disk.

```python
db.close()
```

---

## `add_memory()`

Add a new memory to the database. Handles FAISS indexing and entity association.

```python
result = db.add_memory(
    content="User prefers dark mode",
    embedding=np.array([...], dtype=np.float32),
    memory_type="semantic",
    importance=0.8,
    entities=["user", "preferences"],
    metadata={"category": "ui"},
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | *required* | Text content |
| `embedding` | `np.ndarray \| None` | `None` | Embedding vector (auto L2-normalized) |
| `memory_type` | `str` | `"semantic"` | Memory category |
| `importance` | `float` | `0.5` | Importance score (0.0–1.0) |
| `entities` | `list[str] \| None` | `None` | Entity names to associate |
| `metadata` | `dict \| None` | `None` | JSON-serializable metadata |

### Returns

```python
{"memory_id": int, "status": "created" | "duplicate"}
```

### Behavior

1. Computes SHA-256 content hash for exact dedup
2. Checks for existing memory with same hash
3. L2-normalizes embedding if provided
4. Inserts into `memories` table
5. Adds to FAISS index
6. Auto-upgrades FAISS from FlatIP to IVFFlat if threshold exceeded
7. Associates entities via `memory_entities` table

---

## `get_memory()`

Retrieve a memory by ID. Automatically updates access tracking.

```python
memory = db.get_memory(memory_id=42)
```

### Returns

```python
{
    "id": int,
    "content": str,
    "content_hash": str,
    "memory_type": str,
    "importance": float,
    "created_at": float,
    "updated_at": float,
    "accessed_at": float,
    "access_count": int,
    "retention_strength": float,
    "is_deleted": bool,
    "metadata": dict | None,
}
```

::: warning
Each call to `get_memory()` increments `access_count` and updates `accessed_at`, affecting retention scoring.
:::

---

## `update_memory()`

Update an existing memory's fields.

```python
success = db.update_memory(
    memory_id=42,
    content="Updated content",
    importance=0.9,
    embedding=new_embedding,
    metadata={"updated": True},
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_id` | `int` | *required* | Memory to update |
| `content` | `str \| None` | `None` | New content (updates hash) |
| `importance` | `float \| None` | `None` | New importance |
| `embedding` | `np.ndarray \| None` | `None` | New embedding |
| `metadata` | `dict \| None` | `None` | New metadata |

### Returns

`bool` — `True` if updated, `False` if not found.

---

## `delete_memory()`

Delete a memory (soft or hard).

```python
success = db.delete_memory(memory_id=42, hard=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_id` | `int` | *required* | Memory to delete |
| `hard` | `bool` | `False` | `True` for permanent delete |

### Soft Delete

Sets `is_deleted = 1` and records `deleted_at`. Memory is excluded from search but data is preserved.

### Hard Delete

Permanently removes from `memories`, `memory_entities`, `memory_links`, and `access_log` tables.

---

## `vector_search()`

Search by vector similarity using FAISS.

```python
results = db.vector_search(query_embedding, k=10)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding` | `np.ndarray` | *required* | Query embedding (auto L2-normalized) |
| `k` | `int` | `10` | Number of results |

### Returns

`list[dict]` — Memory dicts with `score` (inner product similarity) and `search_type: "vector"`.

---

## `fts_search()`

Full-text keyword search using SQLite FTS5.

```python
results = db.fts_search("deploy production", k=10)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *required* | Search query |
| `k` | `int` | `10` | Number of results |

### Returns

`list[dict]` — Memory dicts with `score` (BM25 rank) and `search_type: "fts"`.

---

## `hybrid_search()`

Hybrid search combining vector and FTS with Reciprocal Rank Fusion.

```python
results = db.hybrid_search(
    query="deploy production",
    embedding=query_embedding,
    k=10,
    rrf_k=60,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *required* | FTS query text |
| `embedding` | `np.ndarray \| None` | `None` | Vector query (optional) |
| `k` | `int` | `10` | Number of results |
| `rrf_k` | `int` | `60` | RRF smoothing parameter |

### Returns

`list[dict]` — Memory dicts with `score` (RRF fused score) and `search_type: "hybrid"`.

### RRF Formula

$$\text{score}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

---

## `add_edge()`

Add a directed edge between two entities.

```python
db.add_edge("Ariadne", "FAISS", edge_type="uses", weight=1.0)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_entity` | `str` | *required* | Source entity name |
| `target_entity` | `str` | *required* | Target entity name |
| `edge_type` | `str` | `"related"` | Relationship type |
| `weight` | `float` | `1.0` | Edge weight |

---

## `traverse_graph()`

BFS traversal from an entity using recursive CTEs.

```python
result = db.traverse_graph("Ariadne", hops=3, edge_type="uses")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity_name` | `str` | *required* | Starting entity |
| `hops` | `int` | `1` | Max depth (capped at `max_graph_depth`) |
| `edge_type` | `str \| None` | `None` | Filter by edge type |

### Returns

```python
{
    "nodes": list[str],
    "edges": [{"source": str, "target": str, "type": str, "weight": float}]
}
```

---

## Retention & Priority Scoring

### `compute_retention_strength()`

Compute Ebbinghaus retention: `R = e^(-t/S)`.

```python
memory = db.get_memory(42)
R = db.compute_retention_strength(memory)
```

| Factor | Formula |
|--------|---------|
| `t` | `now - memory["accessed_at"]` |
| `S` | `retention_half_life × importance` |
| `R` | `exp(-t / S)` |

### `compute_priority_score()`

Weighted priority score.

```python
priority = db.compute_priority_score(memory)
```

| Component | Formula |
|-----------|---------|
| `importance` | Direct value (0.0–1.0) |
| `recency` | `1 / (1 + age_days)` |
| `access_norm` | `min(1, access_count / 100)` |
| `retention` | Ebbinghaus retention strength |
| **Priority** | `w_imp × importance + w_rec × recency + w_acc × access_norm + w_ret × retention` |

---

## `evict()`

Evict lowest-priority memories via soft delete.

```python
evicted = db.evict()
```

### Behavior

1. Count active memories
2. Compute budget: `max(1, int(total × eviction_budget))`
3. Score all active memories
4. Sort by priority (ascending)
5. Soft-delete the bottom `budget` memories

### Returns

`int` — Number of memories evicted.

---

## `consolidate()`

Group similar memories using Jaccard similarity and create consolidation records.

```python
groups = db.consolidate()
```

### Behavior

1. Load up to 5,000 active memories
2. Tokenize content into word sets
3. Compute Jaccard similarity between all pairs
4. Group memories with similarity ≥ `consolidation_threshold`
5. Create consolidation records (content joined by ` | `)

### Returns

`int` — Number of consolidation groups created.

---

## `stats()`

Get comprehensive database statistics.

```python
stats = db.stats()
```

### Returns

```python
{
    "total_memories": int,
    "active_memories": int,
    "deleted_memories": int,
    "by_type": dict[str, int],
    "total_entities": int,
    "total_edges": int,
    "total_memory_links": int,
    "total_consolidations": int,
    "faiss_vectors": int,
    "faiss_type": str,
    "faiss_dimension": int,
    "avg_importance": float,
    "db_size_bytes": int,
}
```
