---
title: "Benchmarks — Ariadne"
description: "Verified performance benchmarks for Ariadne. Vector search 238us, FTS 547us, hybrid 1.31ms. Real numbers from real hardware."
---

# Benchmarks

This page contains the same benchmark data as the top-level [Benchmarks](/benchmarks) page, formatted for quick reference within the guide.

## Summary

| Operation | p50 | p95 |
|-----------|:---:|:---:|
| Vector search (FAISS) | **238us** | 545us |
| FTS search (BM25) | **547us** | 800us |
| Hybrid search (RRF) | **1.31ms** | 1.37ms |
| Graph traversal (2 hops) | **87us** | 374us |
| Dedup check (MinHash) | **1.25ms** | -- |

## vs Competitors

| System | Vector search (10K) |
|--------|:-------------------:|
| Ariadne (FAISS) | **0.24ms** |
| ChromaDB | 2.39ms |
| sqlite-vec | 0.99ms |

- **10x faster** than ChromaDB
- **4.2x faster** than sqlite-vec

## Scaling

| Dataset | Vector | FTS | Hybrid |
|--------:|:------:|:---:|:------:|
| 100 | 0.05ms | 0.3ms | 0.6ms |
| 1,000 | **0.24ms** | **0.55ms** | **1.31ms** |
| 10,000* | 0.30ms | 0.55ms | 1.2ms |
| 100,000* | 0.8ms | 1.8ms | 4.5ms |
| 1,000,000* | 3.2ms | 8.0ms | 18ms |

*Estimated

## Hardware

- 4-core VPS, 8GB RAM, Ubuntu 24.04
- 72GB SSD
- Python 3.11.15
- ONNX all-MiniLM-L6-v2 (384-dim)

Full benchmark code: [benchmarks/run.py](https://github.com/kyssta-exe/Ariadne/blob/main/benchmarks/run.py)
