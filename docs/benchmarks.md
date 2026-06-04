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
| Vector search (FAISS) | **0.83ms** | 0.95ms | L2-normalized inner product |
| FTS search (BM25) | **1.56ms** | 2.12ms | Porter stemming + prefix matching |
| Hybrid search (RRF) | **5.07ms** | 6.78ms | Vector + FTS + Reciprocal Rank Fusion |
| Graph traversal (2 hops) | **0.06ms** | — | Recursive CTE on SQLite |
| Dedup check (MinHash) | **1.25ms** | — | After 10K index build |

### Write Performance

| Operation | Latency | Notes |
|-----------|:-------:|-------|
| Store + ONNX embed | **42ms** | Includes model inference (~37ms) + SQLite write |
| Store (keyword only) | **0.85ms** | No ML model, just SimHash |

### FAISS vs sqlite-vec

Fair comparison: both measured with the same data, same method, same hardware.

| Engine | Vector search (10K) | Fairness |
|--------|:-------------------:|----------|
| FAISS IndexFlatIP | **0.83ms** | Same query → same results |
| sqlite-vec (brute force) | 10.5ms | Same query → same results |
| sqlite-vec (HNSW) | 7.8ms | Same query → same results |

FAISS is **12× faster** than sqlite-vec for vector search at this scale.

### Scaling

| Dataset | Vector search | FTS search | Hybrid search |
|--------:|:------------:|:----------:|:-------------:|
| 100 | 0.12ms | 0.8ms | 1.2ms |
| 1,000 | 0.25ms | 1.1ms | 1.8ms |
| 10,000 | 0.83ms | 1.6ms | 5.1ms |
| 100,000* | 2.1ms | 4.8ms | 12ms |
| 1,000,000* | 8.5ms | 18ms | 45ms |

*Estimated (FAISS auto-upgrades to IVFFlat at 50K vectors for approximate search)

### Embedding Providers

| Provider | Latency | Quality | Size |
|----------|:-------:|:-------:|:----:|
| ONNX (all-MiniLM-L6-v2) | 37ms | MTEB 59.4 | 90MB |
| Sentence Transformers | 45ms | MTEB 59.4 | 90MB |
| Keyword (SimHash) | 0.05ms | Low | 0MB |

ONNX is recommended for the best balance of quality and speed.

## Reproduce

```bash
pip install arriadne
cd /root/arriadne
python benchmarks/run.py
```
