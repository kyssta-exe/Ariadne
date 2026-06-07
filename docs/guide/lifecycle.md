---
title: "Memory Lifecycle — Ariadne"
description: "Ebbinghaus forgetting curve, priority-based retention, automatic consolidation, and eviction in Ariadne."
---


Ariadne models how human memory works: memories strengthen with use and fade without it. The lifecycle system manages retention, priority scoring, eviction, and consolidation.

## Ebbinghaus Forgetting Curve

Ariadne implements the **Ebbinghaus forgetting curve** to model memory decay:

$$R = e^{-t/S}$$

Where:
- **R** = Retention strength (0.0 to 1.0)
- **t** = Time since last access (seconds)
- **S** = Stability = `retention_half_life × importance × retention_strength`
  (`retention_strength` starts at 1.0 and grows on each access — see below)

### How It Works

```python
import math

# For a fresh memory with importance=0.8, default half_life=86400 (1 day),
# and retention_strength=1.0:
# S = 86400 * 0.8 * 1.0 = 69120 seconds

# Immediately after access (t=0):
R = math.exp(0)  # 1.0 — perfect retention

# After 1 hour (t=3600):
R = math.exp(-3600 / 69120)  # 0.949 — still strong

# After 1 day (t=86400):
R = math.exp(-86400 / 69120)  # 0.287 — significant decay

# After 1 week (t=604800):
R = math.exp(-604800 / 69120)  # 0.000125 — nearly forgotten
```

### Accessing Retention Scores

```python
from arriadne import AriadneMemory, AriadneDB, AriadneConfig

mem = AriadneMemory(db_path="memory.db")
db = mem._db  # Access the underlying storage

# Get a memory
memory = db.get_memory(memory_id=42)

# Compute its current retention strength
retention = db.compute_retention_strength(memory)
print(f"Retention: {retention:.4f}")  # e.g., 0.2873
```

## Stability Growth

Each time a memory is **accessed**, its stored `retention_strength` is multiplied
by `retention_growth_factor` (default **1.5**) and capped at
`retention_strength_cap` (default **100.0**). This feeds directly into the
stability term above, so frequently recalled memories decay more slowly — the
spacing effect.

```
Fresh:            retention_strength = 1.0
After 1st access: retention_strength = 1.5
After 2nd access: retention_strength = 2.25
After 3rd access: retention_strength = 3.375
After nth access: retention_strength = min(cap, 1.5^n)
```

### What counts as an access

Access tracking is explicit and batched — `get_memory()` is a **pure read** and
does not mutate anything. An access is recorded when:

- `recall()` surfaces a memory (it touches the returned top-k in one write), or
- you call `touch_memory(id)` / `touch_memories(ids)` directly.

A single `touch` increments `access_count`, refreshes `accessed_at`, grows
`retention_strength`, and logs one row to `access_log` (via a trigger).

```python
from arriadne import AriadneConfig

# Tune the growth curve
config = AriadneConfig(
    retention_growth_factor=1.5,   # multiplier per access (>= 1.0)
    retention_strength_cap=100.0,  # ceiling
)
```

## Priority Scoring

Ariadne computes a composite priority score using weighted components:

$$\text{Priority} = w_{imp} \cdot \text{importance} + w_{rec} \cdot \text{recency} + w_{acc} \cdot \text{access\_norm} + w_{ret} \cdot \text{retention}$$

### Default Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| `importance` | 0.4 | Static importance assigned at creation |
| `recency` | 0.3 | Time since creation, normalized |
| `access_count` | 0.2 | Number of times accessed (normalized to 100) |
| `retention` | 0.1 | Ebbinghaus retention strength |

### Computing Scores

```python
memory = db.get_memory(42)

# Compute priority
priority = db.compute_priority_score(memory)
print(f"Priority: {priority:.6f}")  # e.g., 0.432812
```

### Customizing Weights

```python
from arriadne import AriadneConfig

config = AriadneConfig(
    db_path="memory.db",
    priority_weights={
        "importance": 0.5,   # Higher weight on importance
        "recency": 0.2,      # Less weight on recency
        "access_count": 0.2,
        "retention": 0.1,
    },
)
```

## Eviction Policy

When memory needs to be reclaimed, Ariadne evicts the lowest-priority memories first.

### How Eviction Works

1. Count active memories: `total_active`
2. Compute eviction budget: `budget = max(1, int(total × eviction_budget))`
3. Score all active memories by priority
4. Sort by priority (ascending — lowest first)
5. **Soft-delete** the bottom `budget` memories

### Soft Delete vs Hard Delete

```python
# Soft delete (default) — marks as deleted, keeps data
mem.forget(memory_id=42, hard=False)

# Hard delete — permanently removes from database
mem.forget(memory_id=42, hard=True)
```

Soft-deleted memories:
- Are excluded from search results
- Keep their row (and embedding) in the database, so they can be recovered by
  setting `is_deleted = 0`
- Have their vector dropped from the live FAISS index, and are pruned entirely on
  the next open (the index is rebuilt from active memories only)
- Can be permanently removed with `purge_deleted()`

### Configuring Eviction

```python
config = AriadneConfig(
    eviction_budget=0.05,       # Evict 5% of memories per run
    retention_half_life=172800, # 2 days half-life (longer retention)
)

mem = AriadneMemory(config=config)

# Run eviction manually
evicted_count = mem.evict()
print(f"Evicted {evicted_count} memories")
```

### Running Eviction Automatically

```python
# Periodic maintenance
def maintenance():
    evicted = mem.evict()
    consolidated = mem.consolidate()
    print(f"Evicted {evicted}, consolidated {consolidated}")
```

## Consolidation

Consolidation groups similar memories and creates merged summaries, reducing redundancy while preserving information.

### How Consolidation Works

1. Load up to 5,000 active memories
2. Tokenize each memory's content into word sets
3. Compute **Jaccard similarity** between pairs
4. Group memories with similarity ≥ `consolidation_threshold` (default: 0.7),
   minimum group size `consolidation_min_group` (default: 2)
5. For each group, create one **merged memory**: content joined by ` | `,
   importance = the group's max, and a mean-pooled embedding (so it stays
   vector-searchable)
6. **Soft-delete the originals** and link them to the merged memory via
   `memory_links` (`link_type='consolidated'`); record the group in the
   `consolidations` table

### Jaccard Similarity

$$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

```python
# Example:
# Memory A: {"deploy", "production", "kubernetes"}
# Memory B: {"deploy", "production", "docker"}
# Intersection: {"deploy", "production"} → 2
# Union: {"deploy", "production", "kubernetes", "docker"} → 4
# Jaccard = 2/4 = 0.5 (below default threshold of 0.7)
```

### Running Consolidation

```python
groups_created = mem.consolidate()
print(f"Created {groups_created} consolidation groups")
```

### Configuring Consolidation

```python
config = AriadneConfig(
    consolidation_threshold=0.6,   # Lower = more aggressive grouping
    consolidation_min_group=3,     # Need 3+ memories to consolidate
)
```

## Lifecycle Management API

```python
# Get full stats
stats = mem.stats()
print(f"Active memories: {stats['active_memories']}")
print(f"Deleted memories: {stats['deleted_memories']}")
print(f"Consolidations: {stats['total_consolidations']}")

# One call runs the whole housekeeping cycle:
# consolidate -> evict -> prune access log -> purge soft-deleted
summary = mem.maintenance()
# {"consolidated": 2, "evicted": 5, "access_log_pruned": 40, "purged": 3}

# Or run the steps individually:
mem.consolidate()
mem.evict()
mem.prune_access_log()                         # bound access_log growth
mem.purge_deleted(older_than_seconds=86400)    # hard-remove old soft-deletes
```

