---
title: "Search & Retrieval — Ariadne"
description: "Hybrid search with FAISS vector similarity, FTS5 keyword matching, and Reciprocal Rank Fusion."
---


Ariadne's hybrid search combines **FAISS vector similarity**, **SQLite FTS5 keyword search**, and **Reciprocal Rank Fusion** for high-recall retrieval, entirely in-process.

## Search Pipeline

```
Query Text
    │
    ├────► FTS5 BM25 Keyword Search ──┐
    │                                  │
    │     Reciprocal Rank Fusion ──► Ranked Results
    │                                  │
    └────► FAISS Vector Search ────────┘
```

When a query vector is available, both search paths run and the rankings are
fused; otherwise only FTS5 keyword search is used. If you configured an
`embedder` on `AriadneMemory`, `recall()` embeds the query for you, so you get
hybrid search without passing a vector. `recall()` also records an access for the
memories it returns (see [Lifecycle](./lifecycle)).

## Vector Search (FAISS)

Vector search finds memories by semantic similarity using inner product distance on L2-normalized embeddings.

```python
import numpy as np
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

# Generate an embedding for the query
# (In production, use sentence-transformers)
query_embedding = np.random.randn(384).astype(np.float32)

results = mem.recall(
    query="deploy to production",
    embedding=query_embedding,
    k=5,
)
```

The FAISS index automatically selects the algorithm by vector count:
- **FlatIP** (exact) until the dataset reaches `ivf_threshold` (default 50,000)
- **IVFFlat** (approximate, faster at scale) beyond that

See [Architecture](./architecture#faiss-index-strategy) for the staged-upgrade
details.

## Full-Text Search (FTS5)

SQLite FTS5 provides BM25-ranked keyword search with stemming (Porter) and Unicode support.

```python
# Keyword-only search (no embedding needed)
results = mem.recall("database migration", k=10)
```

FTS5 queries are automatically escaped and quoted. `fts_search` runs them in two
phases: **AND first** (all terms must match — precise), falling back to **OR**
(any term — high recall) only when AND returns nothing. This avoids a single
shared stopword dragging in unrelated memories.

```python
from arriadne.storage import _fts_escape

_fts_escape("deploy production server", op="AND")
# '"deploy" AND "production" AND "server"'
_fts_escape("deploy production server", op="OR")
# '"deploy" OR "production" OR "server"'
```

## Hybrid Search with RRF

Reciprocal Rank Fusion (RRF) merges results from vector and keyword search using a rank-based scoring formula:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Where:
- `k = 60` (default smoothing parameter)
- `rank_r(d)` is the rank of document `d` in result list `r`
- Higher k reduces the impact of top-ranked items

### How RRF Works

```python
# Behind the scenes in hybrid_search():
#
# 1. Run FTS5 search, get ranked list
# 2. Run FAISS vector search, get ranked list
# 3. For each document:
#    rrf_score = 1/(60 + fts_rank) + 1/(60 + vector_rank)
# 4. Sort by rrf_score descending
```

Example:

| Doc | FTS Rank | Vector Rank | RRF Score |
|-----|----------|-------------|-----------|
| A | 1 | 3 | 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = **0.0323** |
| B | 5 | 1 | 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = **0.0318** |
| C | 2 | 8 | 1/(60+2) + 1/(60+8) = 0.0161 + 0.0147 = **0.0308** |

### Using Hybrid Search

```python
import numpy as np

# With both text query and embedding for hybrid search
query_emb = np.random.randn(384).astype(np.float32)
results = mem.recall(
    query="server configuration",
    embedding=query_emb,
    k=10,
)
for r in results:
    print(f"  [{r['search_type']}] score={r['score']:.4f} | {r['content'][:60]}")
```

When an embedding is provided, `recall()` calls `hybrid_search()` internally. Without an embedding, it falls back to `fts_search()`.

## Filtering Options

### Filter by Memory Type

```python
# Only semantic facts
results = mem.recall("Python", type_filter="semantic")

# Only episodic events
results = mem.recall("meeting", type_filter="episodic")

# Only procedures
results = mem.recall("deploy", type_filter="procedural")
```

### Filter by Importance

```python
# Only high-importance memories
results = mem.recall("critical config", importance_min=0.8)

# Only medium+ importance
results = mem.recall("setup", importance_min=0.5)
```

### Filter by Time Range

```python
import time

now = time.time()
one_day_ago = now - 86400
one_week_ago = now - 604800

# Last 24 hours only
results = mem.recall("deploy", time_range=(one_day_ago, now))

# Last week only
results = mem.recall("meeting", time_range=(one_week_ago, now))
```

### Combining Filters

```python
results = mem.recall(
    query="migration",
    k=10,
    type_filter="procedural",
    importance_min=0.7,
    time_range=(one_week_ago, now),
)
```

## Performance

Latency and recall depend on your hardware, embedding model, dimension, and
dataset size — so measure on your own data rather than trusting a table. The
[benchmarks guide](../benchmarks) has a ready-to-run harness. Architecturally,
vector search is a single BLAS matmul (FlatIP) or an inverted-file lookup (IVF),
keyword search uses FTS5's BM25 index, and everything runs in-process.

## Advanced: Direct Access to Search Engines

For advanced use cases, you can access the search engines directly through `AriadneDB`:

```python
import numpy as np
from arriadne import AriadneDB, AriadneConfig

config = AriadneConfig(db_path="memory.db")
db = AriadneDB(config)
db.open()

# Direct vector search
query_emb = np.random.randn(384).astype(np.float32)
vector_results = db.vector_search(query_emb, k=5)

# Direct FTS5 search
fts_results = db.fts_search("deploy production", k=5)

# Direct hybrid search with custom RRF k parameter
hybrid_results = db.hybrid_search(
    query="deploy production",
    embedding=query_emb,
    k=5,
    rrf_k=30,  # Lower k = more weight on top ranks
)

db.close()
```

