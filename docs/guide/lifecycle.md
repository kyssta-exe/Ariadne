# Memory Lifecycle

Ariadne models how human memory works: memories strengthen with use and fade without it. The lifecycle system manages retention, priority scoring, eviction, and consolidation.

## Ebbinghaus Forgetting Curve

Ariadne implements the **Ebbinghaus forgetting curve** to model memory decay:

$$R = e^{-t/S}$$

Where:
- **R** = Retention strength (0.0 to 1.0)
- **t** = Time since last access (seconds)
- **S** = Stability, derived from `retention_half_life × importance`

### How It Works

```python
import math

# For a memory with importance=0.8 and default half_life=86400 (1 day):
# S = 86400 * 0.8 = 69120 seconds

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

Each time a memory is accessed, its **retention strength grows by 1.5x**. This models the spacing effect — memories recalled at expanding intervals are retained longer.

```
Initial access:    S = 1.0
After 1st recall:  S = 1.5
After 2nd recall:  S = 2.25
After 3rd recall:  S = 3.375
After nth recall:  S = 1.0 × 1.5^n
```

Access tracking happens automatically via the `access_log` table:

```python
# Each recall() or get_memory() call:
# 1. Increments access_count
# 2. Updates accessed_at timestamp
# 3. Logs to access_log table
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
- Keep their FAISS vectors (for index consistency)
- Can be recovered by updating `is_deleted = 0`
- Never evict high-importance memories (> 0.9)

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
3. Compute **Jaccard similarity** between all pairs
4. Group memories with similarity ≥ `consolidation_threshold` (default: 0.7)
5. Create consolidation records with merged content and averaged importance
6. Minimum group size: `consolidation_min_group` (default: 2)

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

# Manual maintenance cycle
def full_maintenance():
    """Run complete lifecycle maintenance."""
    evicted = mem.evict()
    consolidated = mem.consolidate()
    stats = mem.stats()
    print(f"Maintained: evicted={evicted}, consolidated={consolidated}")
    print(f"Active: {stats['active_memories']}, total: {stats['total_memories']}")
```

