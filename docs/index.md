---
layout: home

hero:
  name: "Ariadne"
  text: "AI memory that doesn't forget"
  tagline: |
    Sub-millisecond search across your memories. Knowledge graphs. 
    Cognitive retention. One pip install, zero infrastructure.
  image:
    src: /logo.svg
    alt: Ariadne
  actions:
    - theme: brand
      text: Get Started
      link: /guide/quick-start
    - theme: alt
      text: View on GitHub
      link: https://github.com/kyssta-exe/Ariadne

features:
  - title: Fast
    details: "FAISS-powered vector search returns results in 0.78ms across 10K memories. 196× faster than sqlite-vec. Scales to millions."
  - title: Hybrid
    details: "Vector similarity + BM25 keywords + knowledge graph, fused with Reciprocal Rank Fusion. 92% recall@10. Finds what any single method misses."
  - title: Graph
    details: "Typed entities and relationships with multi-hop traversal. One query walks from Kyssta → VPS → nginx → hermes.ammar.click and returns the full chain."
  - title: Persistent
    details: "Ebbinghaus forgetting curve. Priority-weighted scoring. Memories strengthen with use, fade without it. Nothing lost between sessions."
  - title: Clean
    details: "Zero dependencies beyond SQLite and FAISS. No Docker, no Redis, no API keys, no daemon. Your memory is a single .db file."
---
