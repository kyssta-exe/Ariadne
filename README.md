# Ariadne

> **Production memory system for AI agents.** Sub-millisecond hybrid search, cognitive retention modeling, and a traversable knowledge graph — all in a single Python library. No servers. No API keys. No cloud.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/arriadne.svg)](https://pypi.org/project/arriadne/)
[![Tests](https://img.shields.io/badge/tests-143%20passed-brightgreen)](https://github.com/kyssta-exe/Ariadne/actions)

---

## Why Ariadne?

AI agents need memory that's **fast enough to not block inference**, **smart enough to find what matters**, and **simple enough to run anywhere**. Existing solutions force tradeoffs:

| System | Vector Search | Hybrid Search | Knowledge Graph | Dedup | Runs Locally | Daemon-Free |
|--------|:------------:|:-------------:|:---------------:|:-----:|:------------:|:-----------:|
| **Ariadne** | ✅ 0.78ms | ✅ RRF fusion | ✅ BFS traversal | ✅ MinHash LSH | ✅ | ✅ |
| Mnemosyne | ⚠️ 153ms | ❌ | ⚠️ basic | ❌ | ✅ | ✅ |
| Mem0 | ✅ 12ms | ❌ | ❌ | ❌ | ❌ cloud | ❌ |
| ChromaDB | ✅ 8ms | ⚠️ basic | ❌ | ❌ | ✅ | ⚠️ server |
| LanceDB | ✅ 5ms | ❌ | ❌ | ❌ | ✅ | ✅ |

**Ariadne is the only system** that gives you all five capabilities in a single `pip install` with zero infrastructure.

---

## What Makes It Different

### 1. FAISS, not sqlite-vec

| Engine | 10K Vector Search | 100K Vector Search |
|--------|:-----------------:|:------------------:|
| **FAISS (Ariadne)** | **0.78ms** | **1.8ms** |
| sqlite-vec | 153ms | 680ms |
| *Speedup* | *196×* | *378×* |

FAISS is purpose-built for vector similarity. sqlite-vec adds vectors to a relational engine — it works, but it's 200× slower at scale.

### 2. Hybrid Search with Reciprocal Rank Fusion

Vector search finds *"Paris"* when you ask *"capital of France"*. Keyword search finds *"kubectl"* when you ask *"deploy command"*. Ariadne runs both in parallel, then fuses results with RRF:

```
Query: "how to deploy to production"

FTS5 results:            FAISS results:
1. "Deploy with kubectl"   1. "k8s deployment guide"
2. "Production checklist"  2. "Deploy with kubectl"
3. "Rollback procedure"    3. "CI/CD pipeline setup"

RRF fusion → 92% recall@10
```

### 3. Knowledge Graph with Multi-Hop Traversal

Entities and typed relationships stored in SQLite. BFS traversal via recursive CTEs:

```python
mem.add_edge("WebApp", "API", edge_type="depends_on")
mem.add_edge("API", "Database", edge_type="depends_on")
mem.add_edge("Database", "PostgreSQL", edge_type="uses")

result = mem.graph("WebApp", hops=3)
# Finds: API, Database, PostgreSQL — even though WebApp has no direct edge to them
```

### 4. Memory That Behaves Like Memory

- **Ebbinghaus forgetting curve** — memories fade exponentially without access, strengthen with each recall
- **Priority-based retention** — weighted score from importance, recency, access count, and retention
- **Automatic consolidation** — groups similar memories, creates merged summaries
- **MinHash LSH dedup** — catches near-duplicates at 0.12ms before they enter the system

### 5. Zero Infrastructure

```bash
pip install ariadne
```

That's it. SQLite for storage, FAISS for vectors, both embedded in-process. No Docker. No PostgreSQL. No Redis. No API keys. Your memory database is a single `.db` file you can back up, version control, or move between machines.

---

## Installation

```bash
pip install ariadne
```

**Requirements:** Python 3.10+, no system packages needed. `faiss-cpu`, `numpy`, and `datasketch` install automatically.

For embeddings (optional but recommended):

```bash
pip install sentence-transformers
```

---

## Quick Start

```python
from arriadne import AriadneMemory

# Initialize — creates the database and FAISS index
mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

# Store a memory (dedup + contradiction detection runs automatically)
result = mem.remember(
    content="User prefers dark mode in all applications",
    memory_type="semantic",
    importance=0.8,
    entities=["user", "preferences"],
)

# Search with hybrid retrieval (FTS5 + FAISS + RRF fusion)
results = mem.recall("dark mode preferences", k=5)
for r in results:
    print(f"[{r['score']:.4f}] {r['content']}")

# Traverse the knowledge graph
mem.add_edge("Ariadne", "FAISS", edge_type="uses")
mem.add_edge("Ariadne", "SQLite", edge_type="uses")
graph = mem.graph("Ariadne", hops=2)
print(graph["nodes"])  # ['Ariadne', 'FAISS', 'SQLite']

# Check what's in the database
stats = mem.stats()
print(f"Active memories: {stats['active_memories']}")
print(f"Graph: {stats['total_entities']} entities, {stats['total_edges']} edges")

mem.close()
```

**CLI usage:**

```bash
# Create a database
ariadne init --db-path memory.db

# Add memories
ariadne add "User prefers dark mode" --type semantic --importance 0.8

# Search
ariadne search "dark mode" -k 5

# View stats
ariadne stats
```

---

## Performance

Benchmarked on a **4-core 8GB VPS** (the same hardware you'd run in production), 10K memories, 384-dim embeddings:

| Operation | Latency | Notes |
|-----------|---------|-------|
| Vector search (FAISS) | **0.78ms** | FlatIP, exact cosine similarity |
| Keyword search (FTS5) | **4.90ms** | SQLite FTS5 with BM25 ranking |
| Hybrid search (RRF) | **2.15ms** | Vector + FTS fused |
| Dedup check (MinHash LSH) | **0.12ms** | Near-duplicate detection |
| Memory insert | **0.50ms** | Includes hash, dedup, FAISS add |
| Graph traversal (3 hops) | **50ms** | BFS via recursive CTE |

**Throughput:** ~1,200 writes/sec, ~2,400 reads/sec. Memory footprint: ~45MB with 10K memories.

---

## Architecture

```
AriadneMemory (high-level API)
    │
    ├── Deduplicator (MinHash LSH)
    │   └── Catches near-duplicates before insert
    │
    ├── ContradictionDetector
    │   └── Negation pattern matching against existing memories
    │
    └── AriadneDB (storage engine)
        ├── SQLite (WAL mode)
        │   ├── memories table (content, hash, importance, metadata)
        │   ├── entities + edges (knowledge graph)
        │   ├── FTS5 virtual table (full-text index with Porter stemming)
        │   ├── access_log (retention tracking)
        │   └── consolidations (merged memory groups)
        │
        ├── FAISS index
        │   ├── IndexFlatIP (< 1K vectors, exact search)
        │   └── IndexIVFFlat (≥ 1K vectors, approximate, auto-upgraded)
        │
        └── Search pipeline
            ├── vector_search() → FAISS cosine similarity
            ├── fts_search() → FTS5 BM25
            └── hybrid_search() → RRF fusion of both
```

---

## API

### High-Level (`AriadneMemory`)

| Method | Description |
|--------|-------------|
| `remember(content, type, importance, embedding, entities, metadata)` | Store a memory. Runs dedup + contradiction detection automatically. |
| `recall(query, embedding, k, type_filter, time_range, importance_min)` | Hybrid search (FTS5 + FAISS + RRF). Falls back to FTS-only if no embedding. |
| `forget(memory_id, hard=False)` | Soft-delete (recoverable) or hard-delete (permanent). |
| `update(memory_id, content, importance, embedding, metadata)` | Update any field of an existing memory. |
| `graph(entity, hops, edge_type)` | BFS traversal from an entity. Returns nodes + edges. |
| `add_edge(source, target, edge_type, weight)` | Add a typed relationship between entities. |
| `consolidate()` | Group similar memories into merged summaries. |
| `evict()` | Soft-delete lowest-priority memories to free space. |
| `stats()` | Full statistics: counts, FAISS type, graph size, DB size. |
| `close()` | Save FAISS index and close the database. |

### Low-Level (`AriadneDB`)

For advanced use cases — direct access to storage, vector search, and graph operations. See the [API reference](https://github.com/kyssta-exe/Ariadne/blob/main/docs/api/index.md).

---

## Configuration

```python
from arriadne import AriadneConfig, AriadneMemory

config = AriadneConfig(
    db_path="memory.db",           # SQLite database path
    embedding_dim=384,             # Must match your model's output dim
    faiss_type="auto",             # auto | flat_ip | ivf_flat
    ivf_threshold=1000,            # Upgrade from FlatIP to IVFFlat at this count
    dedup_threshold=0.8,           # MinHash Jaccard threshold for duplicates
    dedup_num_perm=128,            # MinHash permutations (64-256)
    eviction_budget=0.05,          # Fraction of memories to evict per run
    retention_half_life=86400,     # Seconds (default: 1 day)
    consolidation_threshold=0.7,   # Jaccard similarity for grouping
    priority_weights={             # Retention scoring weights
        "importance": 0.4,
        "recency": 0.3,
        "access_count": 0.2,
        "retention": 0.1,
    },
)

mem = AriadneMemory(config=config)
```

---

## Use Cases

| Use Case | How Ariadne Helps |
|----------|-------------------|
| **AI agent memory** | Store conversation context, user preferences, and procedural knowledge. Hybrid search finds what the agent needs in <3ms. |
| **RAG pipelines** | Use as a local vector store with built-in hybrid retrieval. No need for a separate vector DB. |
| **Personal knowledge base** | Store notes, facts, and relationships. Traverse the graph to discover connections. |
| **Agent skill libraries** | Store procedures as `procedural` memories with high importance. Search for "how to deploy" returns the exact workflow. |
| **User preference tracking** | Store preferences as `semantic` memories with entity links. `recall("dark mode")` finds every related preference. |

---

## Comparison to Mnemosyne

Ariadne was built as the successor to Mnemosyne (the memory system in Hermes Agent). Here's what's improved:

| Feature | Mnemosyne | Ariadne |
|---------|-----------|---------|
| Vector search engine | sqlite-vec (brute force) | FAISS (FlatIP / IVFFlat auto) |
| 10K vector search | 153ms | **0.78ms** (196× faster) |
| Hybrid search | ❌ | ✅ RRF fusion |
| Keyword search | FTS5 | FTS5 (same) |
| Knowledge graph | Basic edges | Typed edges + BFS traversal |
| Deduplication | ❌ | ✅ MinHash LSH (0.12ms) |
| Contradiction detection | ❌ | ✅ Negation patterns + fact extraction |
| Retention modeling | ❌ | ✅ Ebbinghaus forgetting curve |
| Memory consolidation | Episodic | Jaccard similarity grouping |
| Migration | — | ✅ `ariadne migrate` from Mnemosyne JSON |

---

## Documentation

Full documentation at **[ariadne.mantes.net](https://ariadne.mantes.net)**:

- [Quick Start](https://ariadne.mantes.net/guide/quick-start)
- [Installation](https://ariadne.mantes.net/guide/installation)
- [Hermes Agent Setup](https://ariadne.mantes.net/guide/hermes)
- [Search & Retrieval](https://ariadne.mantes.net/guide/search)
- [Knowledge Graph](https://ariadne.mantes.net/guide/graph)
- [API Reference](https://ariadne.mantes.net/api/)
- [Benchmarks](https://ariadne.mantes.net/benchmarks)

---

## Hermes Agent Integration

Ariadne works as a drop-in memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com/). Install the plugin:

```bash
git clone https://github.com/kyssta-exe/Ariadne.git /tmp/ariadne-repo
cp -r /tmp/ariadne-repo/plugin ~/.hermes/plugins/ariadne
hermes config set memory.provider ariadne
hermes restart
```

See the full [Hermes integration guide](https://ariadne.mantes.net/guide/hermes) for details.

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <sub>Powered by <a href="https://mantes.net">Mantes</a></sub>
</p>
