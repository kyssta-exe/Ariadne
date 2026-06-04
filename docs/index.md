---
layout: home

hero:
  name: Ariadne
  text: Memory that works like memory
  tagline: Sub-millisecond hybrid search. Cognitive retention. Knowledge graph traversal. Zero infrastructure. One pip install.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/
    - theme: alt
      text: View on GitHub
      link: https://github.com/kyssta-exe/Ariadne
    - theme: alt
      text: API Reference
      link: /api/

features:
  - icon: ⚡
    title: 196× Faster Than sqlite-vec
    details: FAISS-powered vector search at 0.78ms across 10K memories. No cloud API, no GPU — just a local index that auto-upgrades from exact to approximate search as your data grows.
  - icon: 🔀
    title: Hybrid Search with RRF
    details: Vector similarity + BM25 keyword search + graph traversal, fused with Reciprocal Rank Fusion. 92% recall@10. Finds what pure vector or pure keyword search misses.
  - icon: 🔗
    title: Traversable Knowledge Graph
    details: Typed entities and relationships with multi-hop BFS traversal via SQLite recursive CTEs. Store "WebApp depends_on API depends_on Database" and traverse the chain in one call.
  - icon: 🧠
    title: Cognitive Retention
    details: Ebbinghaus forgetting curve with stability growth on each access. Priority-weighted scoring from importance, recency, access count, and retention. Memories that strengthen with use and fade without it.
  - icon: 🎯
    title: Auto-Deduplication
    details: MinHash LSH catches near-duplicates at 0.12ms before they enter the system. No more redundant memories cluttering search results.
  - icon: 📦
    title: Zero Infrastructure
    details: SQLite + FAISS. No Docker, no PostgreSQL, no Redis, no API keys, no daemon. Your entire memory system is a single .db file you can back up, version control, or rsync.

---
