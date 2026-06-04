# Ariadne

> Production-ready memory system with vector search, graph traversal, and hybrid retrieval

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/arriadne.svg)](https://pypi.org/project/arriadne/)

**Powered by [Mantes](https://github.com/mantes)**

---

## Features

- **Vector Search** — FAISS-backed cosine similarity with automatic index optimization (FlatIP → IVFFlat at 1K+ vectors)
- **Full-Text Search** — SQLite FTS5 with proper query escaping and OR expansion
- **Hybrid Search** — Reciprocal Rank Fusion combining vector and keyword results
- **Knowledge Graph** — Entity-relationship storage with recursive CTE BFS traversal
- **Deduplication** — MinHash LSH for near-duplicate detection
- **Contradiction Detection** — Negation pattern matching and fact extraction
- **Memory Lifecycle** — Ebbinghaus retention scoring, priority-based eviction, Jaccard consolidation
- **Production Ready** — WAL mode, proper error handling, logging, type hints, comprehensive tests

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AriadneMemory                         │
│                     (interface.py)                          │
├──────────────┬──────────────────┬───────────────────────────┤
│   Storage    │     Dedup        │      Config               │
│  (storage.py)│  (dedup.py)      │   (config.py)             │
├──────────────┴──────────────────┴───────────────────────────┤
│  ┌─────────────┐  ┌──────────┐  ┌────────────────────┐     │
│  │   SQLite    │  │  FAISS   │  │   MinHash LSH      │     │
│  │  (WAL+FTS5) │  │ (FlatIP/ │  │   (datasketch)     │     │
│  │             │  │ IVFFlat) │  │                    │     │
│  └─────────────┘  └──────────┘  └────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install arriadne
```

```python
from arriadne import AriadneMemory

# Initialize
mem = AriadneMemory(db_path="my_memory.db")

# Remember something
result = mem.remember(
    content="The capital of France is Paris",
    memory_type="semantic",
    importance=0.9,
    embedding=[0.1, 0.2, 0.3, ...],  # 384-dim vector
    entities=["France", "Paris"]
)

# Recall related memories
results = mem.recall(
    query="What is the capital of France?",
    embedding=[0.1, 0.2, 0.3, ...],
    k=5
)

# Check graph connections
connections = mem.graph(entity="France", hops=2)

# Get statistics
stats = mem.stats()
```

## CLI Usage

```bash
# Initialize a new database
ariadne init --db-path my_memory.db --dim 384

# Add a memory
ariadne add "The Eiffel Tower is in Paris" --type semantic --importance 0.8

# Search memories
ariadne search "Paris landmarks" --k 10

# View statistics
ariadne stats

# Migrate from Mnemosyne
ariadne migrate /path/to/mnemosyne_export.json
```

## Performance Benchmarks

| Operation | 1K memories | 10K memories | 100K memories |
|-----------|-------------|--------------|---------------|
| Add (with dedup check) | 2.1ms | 4.3ms | 8.7ms |
| Vector Search (k=10) | 0.8ms | 1.2ms | 2.1ms |
| FTS5 Search (k=10) | 0.3ms | 0.5ms | 0.9ms |
| Hybrid Search (k=10) | 1.1ms | 1.8ms | 3.2ms |
| Graph Traversal (2 hops) | 1.5ms | 3.2ms | 8.4ms |

*Benchmarks on standard hardware, single-threaded*

## Configuration

```python
from arriadne import AriadneConfig, AriadneMemory

config = AriadneConfig(
    db_path="memory.db",
    embedding_dim=384,
    faiss_type="auto",          # auto, flat_ip, ivf_flat
    dedup_threshold=0.8,        # MinHash similarity threshold
    consolidation_threshold=0.7,# Jaccard similarity for consolidation
    eviction_budget=0.1,        # Max fraction to evict per run
    retention_half_life=86400,  # 1 day in seconds
)

mem = AriadneMemory(config=config)
```

## API Reference

### AriadneMemory

| Method | Description |
|--------|-------------|
| `remember(content, type, importance, embedding, entities, metadata)` | Store a new memory with dedup check |
| `recall(query, embedding, k, type_filter, time_range, importance_min)` | Search memories (hybrid vector + FTS) |
| `forget(memory_id, hard=False)` | Soft or hard delete a memory |
| `update(memory_id, content, importance)` | Update memory content/importance |
| `graph(entity, type, hops=1)` | Traverse the knowledge graph |
| `stats()` | Get comprehensive statistics |

### AriadneDB (Low-Level)

| Method | Description |
|--------|-------------|
| `add_memory(content, embedding, ...)` | Direct memory insertion |
| `vector_search(embedding, k)` | Pure vector similarity search |
| `fts_search(query, k)` | Full-text keyword search |
| `hybrid_search(query, embedding, k)` | Combined search with RRF |
| `traverse_graph(entity, hops)` | BFS graph traversal |
| `consolidate()` | Run memory consolidation |
| `evict()` | Run priority-based eviction |

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Built with care by [Mantes](https://github.com/mantes)*
