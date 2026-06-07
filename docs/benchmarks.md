---
title: "Benchmarks — Ariadne"
description: "How to benchmark Ariadne on your own hardware: vector, keyword, and hybrid search latency, plus a ready-to-run harness."
---


::: warning Measure on your own hardware
Search latency depends heavily on your CPU, the embedding model and dimension,
the dataset size, and the index type. Ariadne ships **no canned numbers** —
the harness below lets you measure on the box you actually deploy to. Treat any
figures you see quoted elsewhere as illustrative, not guaranteed.
:::

## What to measure

| Operation | What it exercises |
|-----------|-------------------|
| Vector search | FAISS `IndexFlatIP` (exact) or `IndexIVFFlat` (approximate) |
| Keyword search | SQLite FTS5 BM25 |
| Hybrid search | vector + FTS5, fused with Reciprocal Rank Fusion |
| Insert | content hash, dedup check, SQLite insert, FAISS add |
| Recall (end to end) | `AriadneMemory.recall()` incl. filtering + access recording |

## Harness

```bash
pip install "arriadne[embeddings]" numpy
```

```python
import time
import numpy as np
from arriadne import AriadneMemory, AriadneConfig

N = 10_000
DIM = 384

mem = AriadneMemory(config=AriadneConfig(db_path="bench.db", embedding_dim=DIM))

# --- Insert -------------------------------------------------------------
vecs = np.random.randn(N, DIM).astype("float32")
t0 = time.perf_counter()
for i, v in enumerate(vecs):
    mem.remember(f"memory {i}: content about topic {i % 100}", embedding=v)
insert_ms = (time.perf_counter() - t0) * 1000 / N
print(f"insert:  {insert_ms:.3f} ms/op")

# --- Search (warm up, then time) ---------------------------------------
queries = np.random.randn(1000, DIM).astype("float32")
for q in queries[:100]:                      # warm up
    mem.recall("topic", embedding=q, k=10)

def bench(fn):
    t = time.perf_counter()
    for q in queries:
        fn(q)
    return (time.perf_counter() - t) * 1000 / len(queries)

print(f"vector:  {bench(lambda q: mem._db.vector_search(q, k=10)):.3f} ms/query")
print(f"fts:     {bench(lambda q: mem._db.fts_search('topic', k=10)):.3f} ms/query")
print(f"hybrid:  {bench(lambda q: mem._db.hybrid_search('topic', embedding=q, k=10)):.3f} ms/query")
print(f"recall:  {bench(lambda q: mem.recall('topic', embedding=q, k=10)):.3f} ms/query")

mem.close()
```

### Methodology notes

- **Warm up** before timing (discard the first ~100 queries).
- **Average** over many queries (1,000+).
- Use `time.perf_counter()`.
- `recall()` records an access for the memories it returns (one batched write),
  so it is slightly heavier than the raw `vector_search` / `hybrid_search` calls
  — benchmark whichever matches your usage.
- Run **single-threaded** for clean numbers; Ariadne serializes operations under
  a lock, so concurrency adds throughput, not lower per-op latency.

## Index behaviour at scale

The FAISS index type changes with dataset size, which changes the latency
profile:

| Vectors | `auto` index | Notes |
|--------:|--------------|-------|
| `< ivf_threshold` (default 50K) | `IndexFlatIP` | exact; one BLAS matmul per query |
| `≥ ivf_threshold` | `IndexIVFFlat` | approximate; searches a subset of cells |

For IVF, effective `nlist = min(ivf_nlist, √n)`. Larger `nlist` (and a higher
search `nprobe`, if you tune it) trade recall for speed. Benchmark both regimes
if your dataset will cross the threshold.

## Why an in-process index is fast

- **FAISS** keeps vectors in optimized in-memory structures. `IndexFlatIP` is a
  single BLAS matrix multiply for the whole query; `IndexIVFFlat` partitions the
  space and only scans the nearest cells.
- **FTS5** uses an inverted BM25 index rather than scanning rows.
- **Everything is in-process** — no network hop to a vector DB or cloud API.

These are architectural properties, not a benchmark — always confirm with the
harness above on your own data and hardware.
