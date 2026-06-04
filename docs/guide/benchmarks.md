---
title: "Benchmarks — Ariadne"
description: "Verified performance benchmarks for Ariadne. Vector search 302us, FTS 545us, hybrid 1.21ms. Real numbers from real hardware."
---

# Benchmarks

This page contains the same benchmark data as the top-level [Benchmarks](/benchmarks) page, formatted for quick reference within the guide.

## Summary

| Operation | p50 | p95 |
|-----------|:---:|:---:|
| Vector search (FAISS) | **302us** | 380us |
| FTS search (BM25) | **545us** | 720us |
| Hybrid search (RRF) | **1.21ms** | 1.65ms |
| Graph traversal (2 hops) | **72us** | -- |
| Dedup check (MinHash) | **1.25ms** | -- |

## vs Competitors

| System | Vector search (10K) |
|--------|:-------------------:|
| Ariadne (FAISS) | **0.30ms** |
| ChromaDB | 1.96ms |
| sqlite-vec | 1.0ms |

- **6.5x faster** than ChromaDB
- **3.3x faster** than sqlite-vec

## Scaling

| Dataset | Vector | FTS | Hybrid |
|--------:|:------:|:---:|:------:|
| 100 | 0.05ms | 0.3ms | 0.6ms |
| 1,000 | 0.12ms | 0.5ms | 1.0ms |
| 10,000 | 0.30ms | 0.55ms | 1.2ms |
| 100,000* | 0.8ms | 1.8ms | 4.5ms |
| 1,000,000* | 3.2ms | 8.0ms | 18ms |

*Estimated

## Hardware

- 4-core VPS, 8GB RAM, Ubuntu 24.04
- 72GB SSD
- Python 3.11.15
- ONNX all-MiniLM-L6-v2 (384-dim)

Full benchmark code: [benchmarks/run.py](https://github.com/kyssta-exe/Ariadne/blob/main/benchmarks/run.py)
