---
layout: home

hero:
  name: "Ariadne"
  text: "Memory for AI agents"
  tagline: "Sub-millisecond search. Zero infrastructure."
  actions:
    - theme: brand
      text: Get Started
      link: /guide/quick-start
    - theme: alt
      text: GitHub
      link: https://github.com/kyssta-exe/Ariadne

features:
  - title: "302us vector search"
    details: "FAISS-powered. 6.5x faster than ChromaDB, 3.3x faster than sqlite-vec. Scales to millions of memories."
  - title: "Hybrid retrieval"
    details: "Vector similarity + keyword search + graph traversal. Reciprocal Rank Fusion. 90%+ recall (with semantic embeddings)."
  - title: "Knowledge graph"
    details: "Typed entities, relationships, multi-hop traversal. One query walks the full chain. 72us per traversal."
  - title: "Cognitive retention"
    details: "Ebbinghaus forgetting curve. Memories strengthen with use, fade without it."
  - title: "Auto-dedup"
    details: "MinHash LSH catches near-duplicates before they enter the system."
  - title: "Zero infrastructure"
    details: "SQLite + FAISS. One .db file. No Docker, no Redis, no API keys, no daemon."
---

<div class="hero-benchmarks">

### Quick Start

```python
from arriadne import AriadneMemory

memory = AriadneMemory("./my-memory.db")

# Store a memory
memory.remember(
    content="VPS has 4 cores, 8GB RAM, Ubuntu 24.04",
    memory_type="semantic",
    importance=0.8,
)

# Search — vector + keyword + graph, fused
results = memory.recall("server specs", k=5)
# → [<Memory content="VPS has 4 cores..." score=0.94>]
```

</div>

<div class="hero-compare">

| | Ariadne | ChromaDB | sqlite-vec |
|---|:---:|:---:|:---:|
| Vector search (p50) | **0.30ms** | 1.96ms | 1.0ms |
| FTS search (p50) | **0.55ms** | -- | -- |
| Hybrid search (p50) | **1.21ms** | -- | -- |
| Graph traversal (p50) | **0.07ms** | -- | -- |
| Requires daemon | No | No | No |
| Knowledge graph | Yes | No | No |

</div>

<div class="hero-install">

```bash
pip install arriadne
```

</div>
