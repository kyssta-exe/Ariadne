# Quick Start

## 5-Minute Setup

```python
from arriadne import AriadneMemory
import numpy as np

# Initialize (creates database + FAISS index)
mem = AriadneMemory("~/.ariadne/memory.db", embedding_dim=384)

# Store a memory
result = mem.remember(
    content="User prefers dark mode in all applications",
    content_type="semantic",
    importance=0.8,
)
print(f"Stored memory {result['memory_id']}")

# Search for memories
results = mem.recall("dark mode preferences", k=5)
for r in results:
    print(f"  [{r['rrf_score']:.4f}] {r['content'][:60]}...")

# Build a knowledge graph
mem.db.add_edge("User", "person", "Hermes", "project", "uses")
mem.db.add_edge("Hermes", "project", "Ariadne", "component", "replaces")

# Traverse the graph
nodes = mem.graph("User", "person", max_hops=3)
for node in nodes:
    print(f"  depth={node['depth']}: {node['name']} ({node['type']})")

# Check stats
print(mem.stats())
```

## CLI Usage

```bash
# Initialize database
ariadne init --db-path ~/.ariadne/memory.db

# Add memories
ariadne add "User prefers dark mode" --type semantic --importance 0.8
ariadne add "VPS has 4 cores 8GB RAM" --type semantic

# Search
ariadne search "dark mode"
ariadne search "server configuration" --k 10

# Stats
ariadne stats
```

## What Just Happened

1. **AriadneMemory** opened (or created) a SQLite database and loaded a FAISS index
2. **remember()** checked for duplicates (content hash + MinHash LSH), then stored the memory with an embedding
3. **recall()** ran hybrid search: FTS5 keyword search + FAISS vector search, fused with Reciprocal Rank Fusion
4. **add_edge()** created entities and a typed relationship in the knowledge graph
5. **graph()** ran BFS traversal via recursive CTE query
