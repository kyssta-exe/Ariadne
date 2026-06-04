---
title: "Quick Start — Ariadne"
description: "Get started with Ariadne in 5 minutes. Store memories, search with hybrid retrieval, traverse the knowledge graph."
---


Get Ariadne running in under 60 seconds.

## Install

```bash
pip install ariadne sentence-transformers
```

## Store, Search, Connect

```python
from sentence_transformers import SentenceTransformer
from arriadne import AriadneMemory

# Load an embedding model (384-dim)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Ariadne — creates the SQLite DB and FAISS index
mem = AriadneMemory(db_path="agent_memory.db", embedding_dim=384)

# ── Store memories ──────────────────────────────────────
# Each remember() call runs dedup + contradiction detection automatically

mem.remember(
    content="User prefers dark mode in all applications",
    memory_type="semantic",
    importance=0.8,
    entities=["user", "preferences"],
)

mem.remember(
    content="Deploy to production by running: make deploy-prod",
    memory_type="procedural",
    importance=0.9,
    entities=["deployment", "production"],
)

mem.remember(
    content="Met with DevOps team to discuss Kubernetes migration",
    memory_type="episodic",
    importance=0.6,
    entities=["DevOps", "Kubernetes"],
)

# ── Search (hybrid: FTS5 + FAISS + RRF fusion) ─────────

query = "how to deploy"
query_embedding = model.encode(query).tolist()

results = mem.recall(query, embedding=query_embedding, k=5)
for r in results:
    print(f"  [{r['search_type']}] score={r['score']:.4f} | {r['content'][:60]}")

# Output:
#   [hybrid] score=0.0324 | Deploy to production by running: make deploy-prod
#   [fts]    score=0.0181 | Met with DevOps team to discuss Kubernetes migration

# ── Build a knowledge graph ─────────────────────────────

mem.add_edge("WebApp", "API", edge_type="depends_on")
mem.add_edge("API", "Database", edge_type="depends_on")
mem.add_edge("Database", "PostgreSQL", edge_type="uses")

# Traverse the dependency chain
graph = mem.graph("WebApp", hops=3)
print(graph["nodes"])
# ['WebApp', 'API', 'Database', 'PostgreSQL']

# ── Check what's in the database ────────────────────────

stats = mem.stats()
print(f"Active: {stats['active_memories']}, Graph: {stats['total_entities']} entities, {stats['total_edges']} edges")
# Active: 3, Graph: 7 entities, 3 edges

mem.close()
```

## REST API

Start the built-in HTTP server:

```python
from arriadne import AriadneMemory
from arriadne.server import start_server

mem = AriadneMemory(db_path="memory.db")
start_server(mem, port=8420)
```

Or via CLI:

```bash
ariadne serve --db-path memory.db --port 8420
```

Then use standard HTTP:

```bash
# Store a memory
curl -X POST http://localhost:8420/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "memory_type": "semantic", "importance": 0.8}'

# Search
curl -X POST http://localhost:8420/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "dark mode", "k": 5}'

# Health check
curl http://localhost:8420/health
```

See the full [REST API reference](/guide/rest-api) for all endpoints.

## CLI Usage

Prefer the terminal?

```bash
# Initialize
ariadne init --db-path memory.db

# Add memories
ariadne add "User prefers dark mode" --type semantic --importance 0.8
ariadne add "Deploy with make deploy-prod" --type procedural --importance 0.9

# Search
ariadne search "deploy" -k 5

# Start REST server
ariadne serve --port 8420

# Stats
ariadne stats
```

## What Just Happened

1. **`AriadneMemory()`** opened (or created) a SQLite database in WAL mode and loaded/created a FAISS index
2. **`remember()`** ran SHA-256 exact dedup, MinHash LSH near-duplicate check, negation-based contradiction detection, stored the memory with an L2-normalized embedding in FAISS, and associated entities in the knowledge graph
3. **`recall()`** ran hybrid search: FTS5 keyword → FAISS vector → Reciprocal Rank Fusion (merges both ranked lists into one)
4. **`add_edge()`** upserted entities and created a typed, weighted relationship between them
5. **`graph()`** ran BFS traversal via SQLite recursive CTE — following edges bidirectionally up to 3 hops
6. **`close()`** saved the FAISS index to `.faiss` file and closed the SQLite connection

## Next Steps

- [REST API reference](/guide/rest-api) -- all HTTP endpoints documented
- [Choose an embedding model](/guide/embeddings) -- `all-MiniLM-L6-v2` is a good default
- [Understand search](/guide/search) -- when to use vector vs keyword vs hybrid
- [Build a knowledge graph](/guide/graph) -- entities, relationships, multi-hop traversal
- [Configure retention](/guide/lifecycle) -- Ebbinghaus curves, eviction, consolidation
- [Set up deduplication](/guide/deduplication) -- MinHash thresholds, contradiction detection
- [Observability](/guide/observability) -- metrics, health checks, monitoring

---

## Frequently Asked Questions

### How do I install Ariadne?
```bash
pip install arriadne
```
No system packages, no Docker, no external services.

### What Python version do I need?
Python 3.10 or later.

### Can I use Ariadne without an embedding model?
Yes. Without embeddings, you get keyword-only search (FTS5), the knowledge graph, deduplication, and retention modeling.

### How much disk space does Ariadne use?
About **5MB per 1,000 memories**. Both SQLite and FAISS are compact and auto-managed.

### Can I use Ariadne with multiple agents?
Yes. Ariadne supports a shared surface database for cross-agent memory sharing.

