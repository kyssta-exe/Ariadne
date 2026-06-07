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

Open the database connection, create the schema, and build the FAISS index from
the embeddings stored in the database.

```python
db.open()
```

::: tip
Called automatically on first use if not called explicitly.
:::

## `close()`

Close the database connection (with a final WAL checkpoint). There is no
separate FAISS file to save — embeddings live in the `.db` and the index is
rebuilt on the next `open()`.

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

1. Clamps `importance` into `[0, 1]`
2. Computes SHA-256 content hash for exact dedup
3. Checks for an existing (non-deleted) memory with the same hash
4. L2-normalizes the embedding if provided
5. Inserts into the `memories` table
6. Adds the vector to the FAISS index keyed by the new memory id (`IndexIDMap2`)
7. Upgrades FlatIP → IVFFlat if the count crossed the threshold
8. Associates entities via `memory_entities`

---

## `get_memory()`

Retrieve a memory by ID. This is a **pure read** — it does not change
`access_count`, `accessed_at`, or `retention_strength`. Record an access
explicitly with `touch_memory()` / `touch_memories()`.

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

---

## `touch_memory()` / `touch_memories()`

Record an access for one or many memories in a single transaction. Increments
`access_count`, refreshes `accessed_at`, grows `retention_strength` by
`retention_growth_factor` (capped), and logs to `access_log`.

```python
db.touch_memory(42)
db.touch_memories([1, 2, 3])
```

`AriadneMemory.recall()` calls `touch_memories()` on the results it returns, so
normal recall already records access — you only need these for manual control.

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
| `S` | `retention_half_life × importance × retention_strength` |
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

Group similar memories and merge each group into a single memory.

```python
groups = db.consolidate()
```

### Behavior

1. Load up to 5,000 active memories
2. Tokenize content into word sets and group by Jaccard similarity ≥ `consolidation_threshold` (min group `consolidation_min_group`)
3. For each group, create a merged memory: content joined by ` | `, importance = group max, mean-pooled embedding
4. Soft-delete the originals and link them to the merged memory (`memory_links`, `link_type='consolidated'`)
5. Record the group in the `consolidations` table

### Returns

`int` — Number of consolidation groups created.

---

## `prune_access_log()`

Keep only the most recent `keep_per_memory` access-log rows per memory (default
`max_access_log_per_memory`). Called automatically by `evict()`.

```python
deleted = db.prune_access_log(keep_per_memory=50)
```

Returns the number of rows deleted.

---

## `purge_deleted()`

Permanently remove soft-deleted memories (and their entity links, memory links,
access-log rows, and vectors). `older_than_seconds` keeps recent soft-deletes
recoverable; pass `0` to purge everything currently marked deleted.

```python
purged = db.purge_deleted(older_than_seconds=86400)
```

Returns the number of memories purged.

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
