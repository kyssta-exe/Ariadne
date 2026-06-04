# Introduction

**Ariadne** is a next-generation memory system for AI agents. Named after the Greek goddess who gave Theseus the thread to navigate the Labyrinth, Ariadne gives your AI agent the thread to navigate its own memories.

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
| Dedup check (MinHash) | **0.12ms** | 10K documents |
| Graph traversal (3 hops) | **50ms** | 10K nodes |

## What Makes It Different

1. **SQLite + FAISS** — Not sqlite-vec (647x slower). Not PostgreSQL (requires a server). Just SQLite for metadata/FTS5/graph, FAISS for vectors.

2. **Ebbinghaus Forgetting Curve** — Memories strengthen with each access and fade without it. Stability grows exponentially with reinforcement.

3. **Priority-Based Retention** — When memory budget is hit, low-priority memories get soft-deleted, not important ones.

4. **Auto-Deduplication** — MinHash LSH catches near-duplicates before they enter the system.

5. **Knowledge Graph** — Entities and relationships with multi-hop BFS traversal. Find connections vector search misses.

---
<br>

<div style="text-align: center; padding: 20px 0;">
  <a href="https://mantes.net" class="mantes-badge" target="_blank">
    Powered by <strong>Mantes</strong>
  </a>
</div>
