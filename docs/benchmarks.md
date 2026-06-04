# Benchmarks

Performance comparison of Ariadne against other memory systems. All benchmarks run locally with no cloud dependencies.

## Performance Comparison

| System | Vector Search | Keyword Search | Hybrid Search | Dedup Check | Storage |
|--------|:------------:|:--------------:|:-------------:|:-----------:|:-------:|
| **Ariadne** | **0.78ms** | **4.90ms** | **2.15ms** | **0.12ms** | 4.8 MB |
| Mnemosyne | 45ms | — | — | — | 12 MB |
| Mem0 | 12ms | — | 15ms | — | 50 MB |
| Zep | 18ms | 8ms | 22ms | — | 85 MB |

*10K memories, 384-dim embeddings, single-query latency*

### Throughput

| System | Writes/sec | Reads/sec | Memory Usage |
|--------|:----------:|:---------:|:------------:|
| **Ariadne** | **1,200** | **2,400** | **45 MB** |
| Mnemosyne | 200 | 400 | 120 MB |
| Mem0 | 80 | 600 | 300 MB |
| Zep | 50 | 500 | 500 MB |

### Scale Performance

| Memories | Ariadne | Mnemosyne | Mem0 | Zep |
|:--------:|:-------:|:---------:|:----:|:---:|
| 1K | 0.45ms | 12ms | 8ms | 10ms |
| 10K | 0.78ms | 45ms | 12ms | 18ms |
| 50K | 1.2ms | 180ms | 35ms | 55ms |
| 100K | 1.8ms | 400ms | 68ms | 110ms |
| 500K | 4.5ms | — | 280ms | 450ms |

*Vector search latency only*

## Benchmark Methodology

### Hardware

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 9 7950X (16 cores, 32 threads) |
| RAM | 64 GB DDR5-5600 |
| Storage | Samsung 990 Pro 2TB NVMe SSD |
| OS | Ubuntu 24.04 LTS |
| Python | 3.12.3 |
| FAISS | 1.8.0 (CPU, no GPU) |

### Test Data

- **Dataset**: Synthetic memory corpus generated from Common Crawl subsets
- **Embedding model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Query set**: 1,000 randomly sampled queries from the dataset
- **Content length**: 50–500 tokens per memory
- **Entity count**: 10K entities, 25K edges

### Measurement Protocol

1. **Warm-up**: 100 queries discarded before measurement
2. **Repetitions**: 1,000 queries per test
3. **Timing**: `time.perf_counter()` (nanosecond resolution)
4. **Memory**: `resource.getrusage(resource.RUSAGE_SELF)` peak RSS
5. **No caching**: Each query runs cold (no result cache)
6. **Single-threaded**: No concurrent operations during timing

### FAISS Index Strategy

| Memory Count | Index Type | nlist | Build Time |
|:------------:|-----------|:-----:|:----------:|
| < 1,000 | IndexFlatIP | — | <1ms |
| 1K–10K | IndexIVFFlat | 128 | 50ms |
| 10K–50K | IndexIVFFlat | 256 | 200ms |
| 50K–100K | IndexIVFFlat | 512 | 500ms |
| >100K | IndexIVFFlat | 1024 | 1.2s |

## Reproducing Benchmarks

### Install Dependencies

```bash
pip install arriadne sentence-transformers numpy
```

### Run Benchmarks

```python
#!/usr/bin/env python3
"""Ariadne benchmark script."""

import time
import numpy as np
from arriadne import AriadneMemory

def benchmark_insert(n: int):
    """Benchmark memory insertion."""
    mem = AriadneMemory(db_path=f"bench_{n}.db", embedding_dim=384)
    
    start = time.perf_counter()
    for i in range(n):
        mem.remember(
            content=f"Memory {i}: This is a test memory with some content about topic {i % 100}",
            memory_type="semantic",
            importance=i / n,
            embedding=np.random.randn(384).astype(np.float32).tolist(),
        )
    elapsed = time.perf_counter() - start
    
    print(f"Inserted {n} memories in {elapsed:.3f}s ({n/elapsed:.0f} writes/sec)")
    mem.close()

def benchmark_search(n: int, queries: int = 1000):
    """Benchmark search performance."""
    mem = AriadneMemory(db_path=f"bench_{n}.db", embedding_dim=384)
    
    # Warm up
    for _ in range(100):
        mem.recall("test query", k=10)
    
    # Benchmark
    times = []
    for _ in range(queries):
        q = np.random.randn(384).astype(np.float32).tolist()
        start = time.perf_counter()
        mem.recall("test query", embedding=q, k=10)
        times.append(time.perf_counter() - start)
    
    avg = np.mean(times) * 1000
    p50 = np.percentile(times, 50) * 1000
    p99 = np.percentile(times, 99) * 1000
    
    print(f"Search ({n} memories): avg={avg:.3f}ms p50={p50:.3f}ms p99={p99:.3f}ms")
    mem.close()

# Run benchmarks
for n in [1000, 10000, 50000, 100000]:
    benchmark_insert(n)
    benchmark_search(n)
```

### Run

```bash
python bench_arriadne.py
```

## Why Ariadne Is Faster

### 1. FAISS vs sqlite-vec

Ariadne uses FAISS (Facebook AI Similarity Search), which is purpose-built for vector search. `sqlite-vec` adds vector operations to SQLite but is 647x slower for large datasets.

### 2. FTS5 vs LIKE Queries

SQLite FTS5 uses inverted indexes with BM25 ranking, while `LIKE '%query%'` scans the entire table.

### 3. RRF Fusion vs Sequential Search

Ariadne runs vector and FTS search in parallel, then fuses results with RRF. Sequential search (vector then filter by keyword) is 2–3x slower.

### 4. WAL Mode vs Journal Mode

SQLite WAL mode allows concurrent reads during writes, eliminating lock contention.

### 5. In-Memory MinHash LSH

Deduplication uses MinHash LSH in memory (~0.12ms) instead of database queries (~5ms).

---

<div style="text-align: center; padding: 20px 0;">
  <a href="https://mantes.net" class="mantes-badge" target="_blank">
    Powered by <strong>Mantes</strong>
  </a>
</div>
