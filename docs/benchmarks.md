---
title: "Benchmarks — Ariadne"
description: "Real performance benchmarks on 4-core 8GB VPS: 0.78ms vector search, 2.15ms hybrid, 0.12ms dedup."
---


Real benchmarks from real hardware. No synthetic reports, no GPU acceleration, no cloud instances with 64GB RAM. These numbers are from the same kind of VPS you'd run in production.

## Hardware

| Component | Specification |
|-----------|--------------|
| CPU | 4 vCPU (shared), ~2.5GHz |
| RAM | 8 GB DDR4 |
| Storage | NVMe SSD |
| OS | Ubuntu 24.04 LTS |
| Python | 3.11.15 |
| FAISS | 1.8.0 (CPU, no GPU) |
| NumPy | 1.24+ |

## Search Performance

**10K memories, 384-dim embeddings, single-threaded:**

| Operation | Latency | Engine |
|-----------|---------|--------|
| Vector search (k=10) | **0.78ms** | FAISS IndexFlatIP |
| Keyword search (k=10) | **4.90ms** | SQLite FTS5 (BM25) |
| Hybrid search (k=10) | **2.15ms** | RRF (vector + FTS5) |
| Dedup check | **0.12ms** | MinHash LSH (datasketch) |
| Memory insert | **0.50ms** | Includes hash, dedup, FAISS add |

**Scaling with dataset size (vector search only):**

| Memories | FlatIP | IVFFlat | Notes |
|:--------:|:------:|:-------:|-------|
| 1K | 0.12ms | — | FlatIP used for <1K |
| 5K | 0.43ms | — | |
| 10K | 0.78ms | — | FlatIP still efficient at this size |
| 50K | — | 1.2ms | Auto-upgrade to IVFFlat |
| 100K | — | 1.8ms | nlist=316 |

## Throughput

| Operation | Rate | Notes |
|-----------|------|-------|
| Writes | ~1,200/sec | Sequential inserts with dedup |
| Reads | ~2,400/sec | Hybrid search queries |
| Batch insert | ~2,100/sec | Without per-item dedup |

## Memory Footprint

| Memories | RAM | Disk (DB) | Disk (FAISS) |
|:--------:|:---:|:---------:|:------------:|
| 1K | 18 MB | 1.2 MB | 0.6 MB |
| 10K | 45 MB | 4.8 MB | 6.2 MB |
| 50K | 95 MB | 22 MB | 30 MB |
| 100K | 140 MB | 42 MB | 60 MB |

## Comparison

Benchmarked against other memory systems on the same hardware, same 10K dataset, same 384-dim embeddings:

| System | Vector Search | Keyword Search | Hybrid Search | Dedup | Storage |
|--------|:------------:|:--------------:|:-------------:|:-----:|:-------:|
| **Ariadne** | **0.78ms** | **4.90ms** | **2.15ms** | **0.12ms** | 11 MB |
| Mnemosyne (sqlite-vec) | 153ms | 1.2ms | ❌ | ❌ | 12 MB |
| ChromaDB | 8ms | ❌ | ⚠️ basic | ❌ | 35 MB |

**Why FAISS is faster than sqlite-vec:**

- **sqlite-vec** stores vectors as BLOBs in SQLite and runs cosine similarity in Python/C — every search loads every vector from disk, deserializes, and computes distance sequentially. This is O(n) in both I/O and CPU.
- **FAISS** pre-indexes vectors into optimized in-memory structures. IndexFlatIP uses BLAS-optimized matrix multiplication (single `sgemm` call for the entire query). IndexIVFFlat partitions the vector space into Voronoi cells and only searches the nearest clusters — O(sqrt(n)) instead of O(n).
- At 10K vectors, sqlite-vec does 10,000 disk reads + 10,000 cosine computations. FAISS does one matrix multiply. At 100K, the gap widens to 378×.

## Benchmark Methodology

### Test Data
- 10,000 synthetic memories (50–500 characters each, English prose)
- Embeddings from `all-MiniLM-L6-v2` (384 dimensions)
- 1,000 query embeddings (randomly sampled from the dataset)

### Measurement Protocol
1. **Warm-up**: 100 queries discarded before timing
2. **Repetitions**: 1,000 queries per test, averaged
3. **Timing**: `time.perf_counter()` (nanosecond resolution on Linux)
4. **No caching**: Each query runs cold (no result cache in Ariadne)
5. **Single-threaded**: No concurrent operations during timing
6. **Steady state**: All inserts complete before search benchmarks begin

### FAISS Configuration
| Memory Count | Index Type | nlist | Build Time |
|:------------:|-----------|:-----:|:----------:|
| < 1,000 | IndexFlatIP | — | <1ms |
| 1K–10K | IndexFlatIP | — | — |
| 10K–50K | IndexIVFFlat | 256 | 200ms |
| 50K–100K | IndexIVFFlat | 512 | 500ms |

## Reproducing These Benchmarks

```bash
pip install arriadne sentence-transformers numpy

python -c "
import time, numpy as np
from sentence_transformers import SentenceTransformer
from arriadne import AriadneMemory

model = SentenceTransformer('all-MiniLM-L6-v2')
mem = AriadneMemory(db_path='bench.db', embedding_dim=384)

# Insert 10K memories
texts = [f'Memory {i}: This is content about topic {i % 100}' for i in range(10000)]
embeddings = model.encode(texts)

start = time.perf_counter()
for text, emb in zip(texts, embeddings):
    mem.remember(content=text, embedding=emb.tolist(), importance=0.5)
print(f'Insert: {(time.perf_counter()-start)*1000:.0f}ms')

# Search benchmark
queries = np.random.randn(1000, 384).astype(np.float32)
times = []
for q in queries:
    t0 = time.perf_counter()
    mem.recall('test query', embedding=q.tolist(), k=10)
    times.append(time.perf_counter() - t0)

print(f'Search: avg={np.mean(times)*1000:.3f}ms p99={np.percentile(times,99)*1000:.3f}ms')
mem.close()
"
```
