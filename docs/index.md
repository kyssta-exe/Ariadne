---
layout: home

hero:
  name: "Ariadne"
  text: "Memory for AI agents"
  tagline: "Local-first hybrid search + knowledge graph. Zero infrastructure."
  actions:
    - theme: brand
      text: Get Started
      link: /guide/quick-start
    - theme: alt
      text: GitHub
      link: https://github.com/kyssta-exe/Ariadne

features:
  - title: "Vector search (FAISS)"
    details: "In-process FAISS index. Auto-upgrades from exact (Flat) to approximate (IVF) as your data grows."
  - title: "Hybrid retrieval"
    details: "Vector similarity + FTS5 keywords, fused with Reciprocal Rank Fusion."
  - title: "Knowledge graph"
    details: "Typed entities and relationships, multi-hop traversal. Edges are walked in both directions."
  - title: "Cognitive retention"
    details: "Ebbinghaus forgetting curve. Memories strengthen with use, fade without it."
  - title: "Auto-dedup"
    details: "MinHash LSH catches near-duplicates before they enter the store. Survives restarts."
  - title: "Zero infrastructure"
    details: "SQLite + FAISS, one .db file. No Docker, no Redis, no API keys, no daemon. Thread-safe."
---

<div class="hero-benchmarks">

### Quick Start

```python
from arriadne import AriadneMemory
from arriadne.embeddings import SentenceTransformerEmbedder

# An embedder turns text into vectors so semantic recall works automatically.
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")  # 384-dim

mem = AriadneMemory(db_path="memory.db", embedding_dim=embedder.dim, embedder=embedder)

# Store a memory
mem.remember("VPS has 4 cores, 8GB RAM, Ubuntu 24.04", importance=0.8)

# Search — vector + keyword, fused
results = mem.recall("server specs", k=5)
# → [{"content": "VPS has 4 cores...", "score": 0.94, ...}]
```

</div>

<div class="hero-compare">

| Capability | Ariadne | Chroma | sqlite-vec | Mem0 |
|---|:---:|:---:|:---:|:---:|
| Vector search | ✅ | ✅ | ✅ | ✅ |
| Keyword + hybrid (RRF) | ✅ | ⚠️ | ❌ | ⚠️ |
| Knowledge graph | ✅ | ❌ | ❌ | ⚠️ |
| Near-dup dedup | ✅ | ❌ | ❌ | ⚠️ |
| Local, no daemon | ✅ | ✅ | ✅ | ❌ |

<sub>Capability comparison, not a benchmark. ✅ built-in · ⚠️ partial/varies · ❌ not available.</sub>

</div>

<div class="hero-install">

```bash
pip install "arriadne[embeddings]"
```

</div>
