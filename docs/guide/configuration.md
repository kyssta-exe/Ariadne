---
title: "Configuration — Ariadne"
description: "Configure Ariadne: embedding dimensions, FAISS index type, dedup thresholds, retention settings, priority weights."
---


All settings live on `AriadneConfig`. Every field has a default — you can
construct `AriadneMemory()` with none of them — but `db_path` and
`embedding_dim` are the two you'll usually set.

## AriadneConfig

```python
from arriadne import AriadneConfig, AriadneMemory

config = AriadneConfig(
    db_path="memory.db",          # SQLite database path
    embedding_dim=384,            # Vector dimensions (must match your embedder)
    faiss_type="auto",            # "auto" | "flat_ip" | "ivf_flat"
    ivf_threshold=50_000,         # auto: switch Flat -> IVF at this many vectors
    ivf_nlist=128,                # IVF Voronoi cells (capped at sqrt(n))
    ivf_min_points=1_000,         # ivf_flat: min vectors before IVF is trained
    dedup_threshold=0.8,          # MinHash Jaccard threshold for near-dupes
    dedup_num_perm=128,           # MinHash permutations
    consolidation_threshold=0.7,  # Jaccard threshold for consolidation grouping
    consolidation_min_group=2,    # Min memories to form a consolidation group
    eviction_budget=0.1,          # Fraction of memories evicted per evict() run
    retention_half_life=86_400.0, # Ebbinghaus half-life in seconds (1 day)
    retention_growth_factor=1.5,  # Stability multiplier applied on each access
    retention_strength_cap=100.0, # Ceiling for accrued retention strength
    priority_weights={
        "importance": 0.4,
        "recency": 0.3,
        "access_count": 0.2,
        "retention": 0.1,
    },
    max_graph_depth=10,           # Hard cap on graph traversal hops
    max_access_log_per_memory=50, # Access-log rows kept per memory after pruning
    wal_autocheckpoint=1000,      # SQLite WAL autocheckpoint (pages)
)

mem = AriadneMemory(config=config)
```

## Defaults

| Setting | Default | Description |
|---------|---------|-------------|
| `db_path` | `"arriadne.db"` | SQLite database file path |
| `embedding_dim` | `384` | Vector dimension (matches `all-MiniLM-L6-v2`) |
| `faiss_type` | `"auto"` | `"auto"`, `"flat_ip"`, or `"ivf_flat"` |
| `ivf_threshold` | `50000` | `auto` mode: vectors before upgrading Flat → IVF |
| `ivf_nlist` | `128` | IVF cells; effective `nlist = min(ivf_nlist, √n)` |
| `ivf_min_points` | `1000` | `ivf_flat` mode: vectors before the IVF index trains |
| `dedup_threshold` | `0.8` | MinHash Jaccard threshold for near-duplicates |
| `dedup_num_perm` | `128` | MinHash permutations |
| `consolidation_threshold` | `0.7` | Jaccard threshold for consolidation grouping |
| `consolidation_min_group` | `2` | Minimum group size to consolidate |
| `eviction_budget` | `0.1` | **Fraction** of active memories evicted per run |
| `retention_half_life` | `86400.0` | Ebbinghaus half-life, seconds |
| `retention_growth_factor` | `1.5` | Stability multiplier per access (≥ 1.0) |
| `retention_strength_cap` | `100.0` | Max accrued retention strength |
| `max_graph_depth` | `10` | Maximum graph traversal hops |
| `max_access_log_per_memory` | `50` | Access-log rows kept per memory after pruning |
| `wal_autocheckpoint` | `1000` | SQLite WAL autocheckpoint interval (pages) |

`AriadneConfig` validates its inputs and raises `ValueError` for out-of-range
values (e.g. `embedding_dim < 1`, a `dedup_threshold` outside `[0, 1]`, an
`eviction_budget` outside `(0, 1]`, or an unknown `faiss_type`).

## Embedder

The embedding model is passed to `AriadneMemory`, not stored on the config —
when set, `remember()` and `recall()` embed text automatically:

```python
from arriadne import AriadneMemory
from arriadne.embeddings import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
mem = AriadneMemory(db_path="memory.db", embedding_dim=embedder.dim, embedder=embedder)
```

You can also pass any callable `str -> list[float]` (with a `.dim` attribute) or
a model-name string. Without an embedder, Ariadne is a keyword store unless you
pass vectors to `remember`/`recall` yourself. See [Embeddings](./embeddings).
