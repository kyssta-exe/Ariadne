# Ariadne

Memory for AI agents. Sub-millisecond search. Zero infrastructure.

[![PyPI](https://img.shields.io/pypi/v/arriadne.svg)](https://pypi.org/project/arriadne/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-143%20passed-brightgreen)](https://github.com/kyssta-exe/Ariadne/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Quick Start

```python
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db")
# Auto-detects ONNX embeddings — zero config

mem.remember("VPS has 4 cores, 8GB RAM", importance=0.8)

results = mem.recall("server specs", k=5)
```

```bash
pip install arriadne
```

---

## Why

| | Ariadne | Mnemosyne | Mem0 | ChromaDB |
|---|:---:|:---:|:---:|:---:|
| Vector search | **0.83ms** | 153ms | 12ms | 8ms |
| Hybrid search | ✅ RRF | ❌ | ❌ | ⚠️ basic |
| Knowledge graph | ✅ BFS | ⚠️ basic | ❌ | ❌ |
| Auto-embeddings | ✅ ONNX | ❌ | ✅ cloud | ❌ |
| Auto-dedup | ✅ MinHash | ❌ | ❌ | ❌ |
| Runs locally | ✅ | ✅ | ❌ | ✅ |
| No daemon | ✅ | ✅ | ❌ | ❌ |

---

## Features

### 0.83ms Vector Search

FAISS-powered. 12× faster than sqlite-vec. Auto-upgrades from exact to approximate search as your data grows.

| Engine | 10K vectors |
|--------|:-----------:|
| FAISS (Ariadne) | **0.83ms** |
| sqlite-vec | 10.5ms |

### Hybrid Retrieval

Vector similarity + BM25 keywords fused with Reciprocal Rank Fusion.

```python
results = mem.recall("how to deploy to production", k=5)
# Searches both keyword and semantic similarity, fuses with RRF
for r in results:
    print(f"[{r['search_type']}] {r['content'][:80]}")
```

### Zero-Config Embeddings

Auto-downloads a quantized ONNX model on first use (~90MB). No API keys, no cloud, works offline.

```python
# Just works — no embedding_provider parameter needed
mem = AriadneMemory("memory.db")
mem.remember("Paris is the capital of France")  # auto-embedded
```

Falls back to keyword matching if ONNX is unavailable.

### Knowledge Graph

BFS graph traversal with typed, weighted edges. Bidirectional — edges are traversed in both directions.

```python
mem.add_edge("Paris", "France", "located_in")
mem.add_edge("Nginx", "VPS", "runs_on")
g = mem.graph("VPS", hops=2)
# Returns: VPS ↔ Nginx, VPS ↔ France (via Paris)
```

### Auto-Deduplication

MinHash LSH near-duplicate detection. Catches paraphrases, not just exact matches.

```python
mem.remember("The server runs Ubuntu 24.04")
mem.remember("Ubuntu 24.04 is running on the server")
# Second store detects near-duplicate (LSH similarity > threshold)
```

### Conversation Memory

Track conversations and extract structured facts automatically.

```python
mem.sync_turn("user", "Deploy the app to production")
mem.sync_turn("assistant", "Deploying now via GitHub Actions")

context = mem.get_context("deployment")  # relevant past turns
```

### Agent Tools

OpenAI function-calling compatible tool definitions for any LLM.

```python
tools = AriadneMemory.get_tools()  # 6 tools: remember, recall, graph, link, forget, stats
# Plug into any agent framework that supports function calling
```

### Memory Lifecycle

Ebbinghaus forgetting curve + priority-based eviction. Memories that matter survive; noise gets cleaned up.

```python
mem.consolidate()  # merge similar memories
mem.evict()        # remove low-priority noise
```

---

## Benchmark

Measured on a 4-core 8GB VPS with 10K memories and ONNX embeddings (all-MiniLM-L6-v2, 384-dim):

| Operation | p50 | p95 |
|-----------|:---:|:---:|
| Vector search | **0.83ms** | 0.95ms |
| FTS search | **1.56ms** | 2.12ms |
| Hybrid search | **5.07ms** | 6.78ms |
| Graph traversal (2 hops) | **0.06ms** | — |
| Store (with ONNX embedding) | **42ms** | — |

---

## Install

```bash
pip install arriadne
```

Optional (for faster dev):
```bash
pip install "arriadne[dev]"
```

**Requirements:** Python 3.10+, SQLite (built-in). ONNX model auto-downloads on first use.

---

## Hermes Integration

```bash
hermes plugin install arriadne
hermes config set memory.provider ariadne
```

---

## Links

- **Docs:** [ariadne.mantes.net](https://ariadne.mantes.net)
- **PyPI:** [pypi.org/project/arriadne](https://pypi.org/project/arriadne/)
- **GitHub:** [github.com/kyssta-exe/Ariadne](https://github.com/kyssta-exe/Ariadne)
