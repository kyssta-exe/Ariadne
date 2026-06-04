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

mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

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
| Vector search | **0.78ms** | 153ms | 12ms | 8ms |
| Hybrid search | ✅ RRF | ❌ | ❌ | ⚠️ basic |
| Knowledge graph | ✅ BFS | ⚠️ basic | ❌ | ❌ |
| Auto-dedup | ✅ MinHash | ❌ | ❌ | ❌ |
| Runs locally | ✅ | ✅ | ❌ | ✅ |
| No daemon | ✅ | ✅ | ❌ | ❌ |

---

## Features

### 0.78ms Vector Search

FAISS-powered. 196× faster than sqlite-vec. Auto-upgrades from exact to approximate search as your data grows.

| Engine | 10K vectors | 100K vectors |
|--------|:-----------:|:------------:|
| FAISS (Ariadne) | **0.78ms** | **1.8ms** |
| sqlite-vec | 153ms | 680ms |

### Hybrid Retrieval

Vector similarity + BM25 keywords + graph traversal, fused with Reciprocal Rank Fusion. 92% recall@10.

```python
results = mem.recall("how to deploy to production", k=5)
# Searches both "deploy" (keyword) and semantic similarity in parallel
```

### Knowledge Graph

Typed entities and relationships with multi-hop traversal via SQLite recursive CTEs:

```python
mem.add_edge("WebApp", "API", edge_type="depends_on")
mem.add_edge("API", "Database", edge_type="depends_on")
mem.graph("WebApp", hops=2)  # → [API, Database]
```

### Cognitive Retention

Ebbinghaus forgetting curve with stability growth on each access. Priority-weighted scoring from importance, recency, and access count. Memories strengthen with use, fade without it.

### Auto-Deduplication

MinHash LSH catches near-duplicates at 0.12ms before they enter the system.

---

## Performance

Benchmarked on a 4-core 8GB VPS, 10K memories, 384-dim embeddings:

| Operation | Latency |
|-----------|---------|
| Vector search (FAISS) | **0.78ms** |
| Keyword search (FTS5) | **4.90ms** |
| Hybrid search (RRF) | **2.15ms** |
| Dedup check (MinHash) | **0.12ms** |
| Memory insert | **0.50ms** |
| Graph traversal (3 hops) | **50ms** |

---

## Hermes Agent Integration

Ariadne works as a drop-in memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com/).

```bash
# Copy plugin
git clone https://github.com/kyssta-exe/Ariadne.git /tmp/ariadne-repo
cp -r /tmp/ariadne-repo/plugin ~/.hermes/plugins/ariadne

# Switch provider
hermes config set memory.provider ariadne
hermes restart
```

Full guide: [ariadne.mantes.net/guide/hermes](https://ariadne.mantes.net/guide/hermes)

---

## Configuration

```python
from arriadne import AriadneConfig, AriadneMemory

config = AriadneConfig(
    db_path="memory.db",
    embedding_dim=384,
    faiss_type="auto",          # auto | flat_ip | ivf_flat
    dedup_threshold=0.8,
    retention_half_life=86400,  # 1 day
)

mem = AriadneMemory(config=config)
```

---

## Documentation

**[ariadne.mantes.net](https://ariadne.mantes.net)**

- [Quick Start](https://ariadne.mantes.net/guide/quick-start)
- [Installation](https://ariadne.mantes.net/guide/installation)
- [Hermes Setup](https://ariadne.mantes.net/guide/hermes)
- [Search & Retrieval](https://ariadne.mantes.net/guide/search)
- [Knowledge Graph](https://ariadne.mantes.net/guide/graph)
- [API Reference](https://ariadne.mantes.net/api/)
- [Benchmarks](https://ariadne.mantes.net/benchmarks)

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <sub>Powered by <a href="https://mantes.net">Mantes</a></sub>
</p>
