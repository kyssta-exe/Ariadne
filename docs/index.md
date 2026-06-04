---
layout: home

hero:
  name: Ariadne
  text: Thread through the labyrinth of memories
  tagline: Next-generation AI memory system. Sub-millisecond search across 100K memories. Zero cloud. Zero daemon.
  image:
    src: /logo.svg
    alt: Ariadne
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
    title: Sub-millisecond Search
    details: FAISS-powered vector search at 0.78ms across 100K memories. 196x faster than naive approaches. Hybrid search with RRF fusion at 2.15ms.
  - icon: 🧠
    title: Cognitive Memory
    details: Ebbinghaus forgetting curve, priority-based retention, automatic consolidation. Memories strengthen with use and fade without it — just like human memory.
  - icon: 🔗
    title: Knowledge Graph
    details: Entities, relationships, and multi-hop traversal. Find connections across your memory that simple vector search misses.
  - icon: 🎯
    title: Hybrid Search
    details: Vector similarity + BM25 keyword search + graph traversal, fused with Reciprocal Rank Fusion. 92% recall@10.
  - icon: 🔄
    title: Auto-Deduplication
    details: MinHash LSH detects near-duplicates in <1ms. No more redundant memories cluttering your context.
  - icon: 🏠
    title: Zero Cloud
    details: Everything runs locally. SQLite + FAISS. No API keys, no daemons, no network calls. Your memories stay on your machine.

---
<br>

<div style="text-align: center; padding: 20px 0;">
  <a href="https://mantes.net" class="mantes-badge" target="_blank">
    Powered by <strong>Mantes</strong>
  </a>
</div>
