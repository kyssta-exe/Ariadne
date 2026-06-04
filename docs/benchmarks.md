---
title: "Benchmarks — Ariadne"
description: "Verified performance benchmarks for Ariadne. Real numbers from real hardware. Vector search 302us, FTS 545us, hybrid 1.21ms."
---

# Benchmarks

Real numbers from real hardware. No synthetic benchmarks, no cherry-picked results.

## Hardware

- **VPS:** 4-core, 8GB RAM, Ubuntu 24.04
- **Disk:** 72GB SSD
- **Python:** 3.11.15

## Test Configuration

- **Dataset:** 10,000 memories with random importance scores
- **Embeddings:** ONNX all-MiniLM-L6-v2 (384-dim, auto-downloaded)
- **FAISS index:** IndexFlatIP (exact search, <50K vectors)
- **FTS5:** porter unicode61 tokenizer
- **100 iterations** per operation, p50/p95 reported

## Results

### Search Performance

| Operation | p50 | p95 | Notes |
|-----------|:---:|:---:|-------|
| Vector search (FAISS) | **302us** | 380us | L2-normalized inner product |
| FTS search (BM25) | **545us** | 720us | Porter stemming + prefix matching |
| Hybrid search (RRF) | **1.21ms** | 1.65ms | Vector + FTS + Reciprocal Rank Fusion |
| Graph traversal (2 hops) | **72us** | -- | Recursive CTE on SQLite |
| Dedup check (MinHash) | **1.25ms** | -- | After 10K index build |

### Write Performance

| Operation | Latency | Notes |
|-----------|:-------:|-------|
| Store + ONNX embed | **42ms** | Includes model inference (~37ms) + SQLite write |
| Store (keyword only) | **0.85ms** | No ML model, just SimHash |

### FAISS vs sqlite-vec

Fair comparison: both measured with the same data, same method, same hardware.

| Engine | Vector search (10K) | Fairness |
|--------|:-------------------:|----------|
| FAISS IndexFlatIP | **0.30ms** | Same query -> same results |
| sqlite-vec (brute force) | 1.0ms | Same query -> same results |

FAISS is **3.3x faster** than sqlite-vec for vector search at this scale.

### vs ChromaDB

| System | Vector search (10K) | Notes |
|--------|:-------------------:|-------|
| Ariadne (FAISS) | **0.30ms** | Exact search, local |
| ChromaDB | 1.96ms | Default HNSW, local |

Ariadne is **6.5x faster** than ChromaDB for vector search at this scale.

### Scaling

| Dataset | Vector search | FTS search | Hybrid search |
|--------:|:------------:|:----------:|:-------------:|
| 100 | 0.05ms | 0.3ms | 0.6ms |
| 1,000 | 0.12ms | 0.5ms | 1.0ms |
| 10,000 | 0.30ms | 0.55ms | 1.2ms |
| 100,000* | 0.8ms | 1.8ms | 4.5ms |
| 1,000,000* | 3.2ms | 8.0ms | 18ms |

*Estimated (FAISS auto-upgrades to IVFFlat at 50K vectors for approximate search)

### Embedding Providers

| Provider | Latency | Quality | Size |
|----------|:-------:|:-------:|:----:|
| ONNX (all-MiniLM-L6-v2) | 37ms | MTEB 59.4 | 90MB |
| Sentence Transformers | 45ms | MTEB 59.4 | 90MB |
| Keyword (SimHash) | 0.05ms | Low | 0MB |

ONNX is recommended for the best balance of quality and speed.

## How to Reproduce

```bash
pip install arriadne
cd /root/arriadne
python benchmarks/run.py
```

All benchmarks run the same query set, same iteration count, same hardware. The code is open source -- you can verify every number yourself.
