---
title: "Benchmarks — Ariadne"
description: "Real performance benchmarks on a shared 4-core 8GB VPS: sub-millisecond vector search, sub-3ms hybrid, 12× faster than sqlite-vec."
---

Real benchmarks from real hardware. No synthetic reports, no GPU acceleration, no cloud instances with 64GB RAM. These numbers are from the same kind of VPS you'd run in production.

## Hardware

| Component | Specification |
|-----------|--------------|
| CPU | 4 vCPU (shared), Intel Haswell |
| RAM | 8 GB DDR4 |
| Storage | NVMe SSD |
| OS | Ubuntu 24.04 LTS |
| Python | 3.11.15 |
| FAISS | 1.8.0 (CPU, no GPU) |
| NumPy | 1.24+ |

## Search Performance

**10K memories, 384-dim embeddings, single-threaded, FlatIP index:**

| Operation | p50 Latency | Engine |
|-----------|-------------|--------|
| Vector search (k=10) | **0.89ms** | FAISS IndexFlatIP |
| Keyword search (k=10) | **1.74ms** | SQLite FTS5 (BM25) |
| Hybrid search (k=10) | **2.46ms** | RRF (vector + FTS5) |
| Dedup check | **1.25ms** | MinHash LSH (datasketch) |
| Contradiction check | **0.07ms** | Pattern matching |

::: warning
These are **full-method** benchmarks — they measure the entire search pipeline including SQLite reads, result construction, and RRF fusion. Not just the raw FAISS `index.search()` call.

FTS5 and hybrid latency depends on **query selectivity**. Selective queries (matching ~10% of the database) run in 1-3ms. Broad queries (matching most of the database) are slower because FTS5 must rank more results. The numbers below reflect typical selective queries.
:::

**Scaling with dataset size (vector search, p50):**

| Memories | Latency | Index Type |
|:--------:|:-------:|:----------:|
| 100 | 0.15ms | FlatIP |
| 1,000 | 0.23ms | FlatIP |
| 5,000 | 0.47ms | FlatIP |
| 10,000 | 0.89ms | FlatIP |

For datasets under 50K, FAISS `IndexFlatIP` is fast enough. No need for IVFFlat clustering.

## FAISS vs sqlite-vec

Benchmarked on the same hardware, same 10K dataset, same 384-dim embeddings, same query set:

| Engine | Vector Search (10K) | Method |
|--------|:-------------------:|--------|
| **FAISS IndexFlatIP** | **0.87ms** | BLAS-optimized matrix multiply |
| sqlite-vec (HNSW) | 10.5ms | SQLite BLOB storage + C extension |
| **Speedup** | **12×** | |

**Why FAISS is faster:**

- **sqlite-vec** stores vectors as BLOBs in SQLite and runs distance computation through its C extension. Every search loads vectors, deserializes, and computes distances sequentially.
- **FAISS** keeps vectors in contiguous memory and uses BLAS-optimized matrix multiplication (`sgemm`) for the entire query in a single call.
- At 10K vectors the difference is 12×. At 100K+ with IVFFlat clustering, the gap grows further.

## Batch Search

Batch vector search sends multiple queries in a single FAISS call:

| Queries | Sequential | Batch | Speedup |
|:-------:|:----------:|:-----:|:-------:|
| 10 | 29.1ms | 10.8ms | **2.7×** |

## Deduplication

MinHash LSH with 128 permutations, 0.8 similarity threshold:

| Operation | Time |
|-----------|------|
| Index 10K documents | 15.4s (647 docs/sec) |
| Query (duplicate check) | 1.25ms |

Dedup indexing is slow because it builds the LSH index. Querying is fast because LSH does hash lookups, not pairwise comparison.

## Accuracy

Search accuracy depends entirely on embedding quality:

| Embedding Model | Recall@10 | Notes |
|-----------------|:---------:|-------|
| Random vectors | ~70% | Incidental similarity in 384D |
| all-MiniLM-L6-v2 | ~90%+ | Semantic embeddings |
| nomic-embed-text-v1.5 | ~92%+ | Best open-source option |

Ariadne does not bundle an embedding model — it accepts pre-computed vectors. Use whatever model fits your use case.

## Reproducing These Benchmarks

```bash
pip install arriadne numpy

python -c "
import time, numpy as np
from arriadne import AriadneDB, AriadneConfig

config = AriadneConfig(db_path='bench.db', embedding_dim=384, faiss_type='flat_ip')
db = AriadneDB(config=config)
db.open()

# Insert 10K memories with random embeddings
n = 10000
vecs = np.random.randn(n, 384).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

start = time.perf_counter()
for i in range(n):
    db.add_memory(content=f'Memory {i}: content about topic {i % 100}', embedding=vecs[i])
print(f'Insert 10K: {(time.perf_counter()-start)*1000:.0f}ms')

# Vector search benchmark
query = np.random.randn(384).astype(np.float32)
query /= np.linalg.norm(query)
times = []
for _ in range(1000):
    t0 = time.perf_counter()
    db.vector_search(query, k=10)
    times.append((time.perf_counter() - t0) * 1000)
print(f'Vector search: avg={np.mean(times):.3f}ms p50={np.percentile(times,50):.3f}ms')

# FTS search benchmark
times = []
for _ in range(1000):
    t0 = time.perf_counter()
    db.fts_search('memory topic', k=10)
    times.append((time.perf_counter() - t0) * 1000)
print(f'FTS search: avg={np.mean(times):.3f}ms p50={np.percentile(times,50):.3f}ms')

db.close()
"
```
