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
  - title: "0.78ms vector search"
    details: "FAISS-powered. 196× faster than sqlite-vec. Scales to millions of memories."
  - title: "Hybrid retrieval"
    details: "Vector similarity + keyword search + graph traversal. Reciprocal Rank Fusion. 92% recall."
  - title: "Knowledge graph"
    details: "Typed entities, relationships, multi-hop traversal. One query walks the full chain."
  - title: "Cognitive retention"
    details: "Ebbinghaus forgetting curve. Memories strengthen with use, fade without it."
  - title: "Auto-dedup"
    details: "MinHash LSH catches near-duplicates at 0.12ms before they enter the system."
  - title: "Zero infrastructure"
    details: "SQLite + FAISS. One .db file. No Docker, no Redis, no API keys, no daemon."
---

<div class="hero-benchmarks">

### Quick Start

```python
from arriadne import AriadneMemory

memory = AriadneMemory("./my-memory.db")

# Store a memory
memory.add(
    content="VPS has 4 cores, 8GB RAM, Ubuntu 24.04",
    source="system",
    importance=0.8,
)

# Search — vector + keyword + graph, fused
results = memory.search("server specs", limit=5)
# → [<Memory content="VPS has 4 cores..." score=0.94>]
```

</div>

<div class="hero-compare">

| | Ariadne | Mnemosyne | Mem0 | ChromaDB |
|---|:---:|:---:|:---:|:---:|
| Vector search | **0.78ms** | 153ms | 12ms | 8ms |
| Hybrid search | ✅ | ❌ | ❌ | ⚠️ |
| Knowledge graph | ✅ | ⚠️ | ❌ | ❌ |
| Runs locally | ✅ | ✅ | ❌ | ✅ |
| No daemon | ✅ | ✅ | ❌ | ❌ |

</div>

<div class="hero-install">

```bash
pip install arriadne
```

</div>
