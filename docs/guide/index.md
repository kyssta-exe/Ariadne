---
title: "Introduction — Ariadne"
description: "Ariadne is a local memory system for AI agents: hybrid search, cognitive retention, and knowledge graph traversal. Zero infrastructure."
---


**Ariadne** is a memory system for AI agents. Named after the Greek goddess who gave Theseus the thread to navigate the Labyrinth, Ariadne gives your AI agent the thread to navigate its own memories.

## Why Ariadne?

Current AI memory options tend to force a choice:

- **Fast but simple** — keyword search, no semantic understanding
- **Semantic but heavy** — a vector store you still have to run and operate
- **Feature-rich but hosted** — requires a running server, database, or API keys

Ariadne bundles all of it — vector + keyword + graph retrieval, deduplication,
and a retention model — into a single Python library backed by one SQLite file,
with no daemon and no external dependencies.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Ariadne                        │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  FAISS   │  │  FTS5    │  │  Knowledge   │  │
│  │  Vector  │  │  Keyword │  │  Graph       │  │
│  │  (cosine)│  │  (BM25)  │  │  (recursive) │  │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│       │              │               │           │
│       └──────┬───────┴───────────────┘           │
│              │                                   │
│     ┌────────▼─────────┐                         │
│     │ Reciprocal Rank  │                         │
│     │    Fusion        │                         │
│     └────────┬─────────┘                         │
│              │                                   │
│  ┌───────────▼────────────────────────────────┐  │
│  │           SQLite (WAL mode)                │  │
│  │  memories │ entities │ edges │ FTS5 index  │  │
│  └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

Vectors are stored as BLOBs in SQLite and the FAISS index is rebuilt from them on
open, so the whole system is one `.db` file. For latency, measure on your own
hardware with the [benchmarks harness](../benchmarks).

## What Makes It Different

1. **SQLite + FAISS, in-process** — SQLite for metadata, FTS5, and the graph;
   FAISS for vectors. No server, no sqlite-vec row scan, no PostgreSQL.

2. **Ebbinghaus forgetting curve** — memories strengthen with each access
   (`retention_strength` grows, capped) and fade without it.

3. **Priority-based retention** — when you run eviction, the lowest-priority
   memories (importance, recency, access count, retention) are soft-deleted
   first.

4. **Auto-deduplication** — MinHash LSH catches near-duplicates before they enter
   the store, and the index is rebuilt from the database on open so it survives
   restarts.

5. **Knowledge graph** — entities and typed relationships with multi-hop,
   bidirectional traversal. Find connections vector search alone would miss.
