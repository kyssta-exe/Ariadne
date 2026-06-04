---
title: "Introduction — Ariadne"
description: "Ariadne is a local memory system for AI agents. Sub-millisecond hybrid search, cognitive retention, and knowledge graph traversal. Zero infrastructure."
---


**Ariadne** is a memory system for AI agents. Named after the Greek goddess who gave Theseus the thread to navigate the Labyrinth, Ariadne gives your AI agent the thread to navigate its own memories.

## Why Ariadne?

Current AI memory systems force you to choose between:

- **Fast but simple** — keyword search, no semantic understanding
- **Semantic but slow** — vector search that takes 100ms+ per query  
- **Feature-rich but heavy** — requires a running server, database, and API keys

Ariadne gives you **all three**: sub-millisecond search, semantic understanding, and a full-featured memory system that runs as a single Python library with zero external dependencies.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Ariadne                        │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  FAISS   │  │  FTS5    │  │  Knowledge   │  │
│  │  Vector  │  │  Keyword │  │  Graph       │  │
│  │  Search  │  │  Search  │  │  (BFS/DFS)   │  │
│  │  0.78ms  │  │  4.90ms  │  │  50ms        │  │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│       │              │               │           │
│       └──────┬───────┴───────────────┘           │
│              │                                   │
│     ┌────────▼─────────┐                         │
│     │ Reciprocal Rank  │                         │
│     │    Fusion        │                         │
│     │    2.15ms        │                         │
│     └────────┬─────────┘                         │
│              │                                   │
│  ┌───────────▼────────────────────────────────┐  │
│  │           SQLite (WAL mode)                │  │
│  │  memories │ entities │ edges │ FTS5 index  │  │
│  └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Performance

| Operation | Latency | Scale |
|-----------|---------|-------|
| Vector search (FAISS) | **0.78ms** | 10K memories |
| Keyword search (FTS5) | **4.90ms** | 10K memories |
| Hybrid search (RRF) | **2.15ms** | 10K memories |
| Dedup check (MinHash LSH) | **0.12ms** | 10K documents |
| Graph traversal (3 hops) | **50ms** | 10K nodes |

## What Makes It Different

1. **SQLite + FAISS** — Not sqlite-vec (200× slower). Not PostgreSQL (requires a server). Just SQLite for metadata/FTS5/graph, FAISS for vectors. Both embedded in-process.

2. **Ebbinghaus Forgetting Curve** — Memories strengthen with each access and fade without it. Stability grows exponentially with reinforcement.

3. **Priority-Based Retention** — When the memory budget is hit, low-priority memories get soft-deleted. High-importance memories are protected.

4. **Auto-Deduplication** — MinHash LSH catches near-duplicates at 0.12ms before they enter the system. No more redundant memories.

5. **Knowledge Graph** — Entities and typed relationships with multi-hop BFS traversal. Find connections that vector search alone would miss.

---

## Frequently Asked Questions

### What is Ariadne?
Ariadne is a local memory system for AI agents. It stores, searches, and manages memories using FAISS vector search, SQLite FTS5 keywords, a knowledge graph, and cognitive retention modeling. No cloud, no API keys, no daemon.

### How fast is Ariadne?
Ariadne's FAISS vector search returns results in **0.78ms** across 10,000 memories. Hybrid search (vector + keyword + graph) completes in **2.15ms**. This is 196x faster than sqlite-vec.

### Does Ariadne work with Hermes Agent?
Yes. Ariadne ships as a drop-in plugin for Hermes Agent. Install the plugin, set `memory.provider: ariadne`, restart. All existing tool names and conversations work unchanged.

### What makes Ariadne different from ChromaDB or LanceDB?
Ariadne combines vector search, keyword search, a knowledge graph, and deduplication in a single library. ChromaDB and LanceDB are vector-only stores.

### Can I migrate from Mnemosyne?
Yes. Run `ariadne migrate --from mnemosyne`. All memories, graph edges, and metadata transfer automatically.

