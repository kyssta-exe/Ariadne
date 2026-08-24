---
title: "Benchmarks — Ariadne"
description: "Real measured performance of Ariadne v0.11.0 — sub-millisecond vector search, hybrid retrieval latency, knowledge graph traversal, deduplication throughput, and head-to-head comparison with Mem0, Zep, Honcho, Letta, ChromaDB, and others."
---

# Benchmarks

Real numbers, measured on real hardware. No marketing estimates.

::: info Test environment
**Server:** Linux 6.8.0-124-generic, shared VPS (Olivier / Scaleway)
**Python:** 3.11.15 · **NumPy:** as shipped · **FAISS:** IndexFlatIP (exact, <50K vectors)
**Embedding dim:** 384 (all-MiniLM-L6-v2)
**Methodology:** 500–1000 queries per test, warm-up discarded, `time.perf_counter()`, single-threaded unless noted.
:::

---

## Search Latency

The core operation: how fast can Ariadne find relevant memories?

| Dataset | Vector (FAISS) | Keyword (FTS5) | Hybrid (RRF) | Full `recall()` |
|--------:|:--------------:|:--------------:|:------------:|:---------------:|
| **1,000** | **0.29 ms** p50 · 5.7 ms p99 | **1.69 ms** p50 · 2.8 ms p99 | **4.55 ms** p50 · 10.5 ms p99 | **7.54 ms** p50 · 13.8 ms p99 |
| **5,000** | **0.63 ms** p50 · 6.8 ms p99 | **6.97 ms** p50 · 9.6 ms p99 | **9.14 ms** p50 · 13.6 ms p99 | **12.75 ms** p50 · 19.0 ms p99 |

::: tip v0.13.0 retrieval speedup
v0.13.0 removed the per-candidate `get_memory` round trips from hybrid/FTS/vector
search (batch `IN` fetches), made namespace-filtered vector search widen its
candidate window instead of scanning the whole index, and batched provenance
attachment in `recall()`. On the same class of workload this cuts hybrid search
by roughly 30% and FTS by ~17% at 1k memories (hybrid ≈2.8 ms, FTS ≈0.44 ms in
the repo's probe on Windows/Python 3.13). The table above retains the original
Linux measurements for comparability; reproduce either with
[`scripts/perf_probe.py`](https://github.com/kyssta-exe/Ariadne/blob/main/scripts/perf_probe.py).
:::

### Startup and write throughput (v0.13.0, Windows / Python 3.13)

v0.13.0 attacked the two lifecycle costs that actually hurt self-hosters:
**restart time** and **write throughput at scale**. Fixes: FAISS + dedup index
sidecars (fingerprint-validated, auto-fallback to rebuild), a 64 MB SQLite
page cache (the default 2 MB thrashes once embeddings and indexes compete),
WAL `synchronous=NORMAL`, tiered auto-maintenance (expensive consolidation
runs every 10th cycle instead of every 50 writes), shared MinHash
permutations, and one less row fetch per hybrid search.

Measured with [`scripts/perf_scale.py`](https://github.com/kyssta-exe/Ariadne/blob/main/scripts/perf_scale.py), 384-dim vectors, 2 namespaces:

| Metric | N | before | after | speedup |
|---|---:|---:|---:|---:|
| Startup (warm open) | 1,000 | 732 ms | 138 ms | **5.3×** |
| Startup (warm open) | 5,000 | 3,561 ms | 413 ms | **8.6×** |
| Startup (warm open) | 20,000 | 13,559 ms | ~831 ms | **16×** |
| Bulk write | 1,000 | 1.09 ms | 0.80 ms | 1.4× |
| Bulk write | 5,000 | 5.27 ms | 1.27 ms | **4.2×** |
| Bulk write | 20,000 | 23.3 ms | 3.9 ms | **5.9×** |
| Single write | 20,000 | 9.2 ms | 4.9 ms | 1.9× |
| Full `recall()` | 20,000 | 4.0 ms | 3.8 ms | ~1.1× |

"Before" numbers were captured on the same machine immediately prior to the
changes. The sidecars are optimizations, never correctness requirements: any
fingerprint mismatch (external edits, crashes, dimension changes) silently
falls back to the full rebuild — covered by tests including corruption,
staleness, and dimension-change cases.

---

## Retrieval Accuracy

Latency without correctness is meaningless. v0.13.0 ships a deterministic
accuracy harness,
[`benchmarks/accuracy_eval.py`](https://github.com/kyssta-exe/Ariadne/blob/main/benchmarks/accuracy_eval.py):
a synthetic corpus of unique-subject facts with paraphrased questions,
temporal supersessions ("X moved cities"), same-shape distractors, and
cross-namespace isolation checks. Fully deterministic — same score on every
machine, no downloads.

| Mode | Overall recall@5 | Exact | Paraphrase | Temporal | Namespace leak |
|---|---:|---:|---:|---:|---:|
| FTS-only (no embedder) | **0.950** | 1.000 | 0.900 | **1.000** | none |
| Hybrid + real embeddings (all-MiniLM-L6-v2) | **0.948** | 1.000 | 0.895 | **1.000** | none |
| Hybrid + hashing embedder¹ | 0.825 | 1.000 | 0.650 | 1.000 | none |

Measured with `python benchmarks/accuracy_eval.py --embeddings --n 100`
(sentence-transformers 6.0.0, CPU). On this corpus real embeddings match the
FTS baseline: the questions are keyword-answerable, so the vector channel's
advantage (vocabulary mismatch, cross-lingual, abstraction) doesn't
differentiate — the two channels agreeing at ~0.95 is the signal that the
hybrid path is wired correctly end-to-end.

¹ The dependency-free char-trigram hashing embedder is a stand-in used by CI
to exercise the hybrid path deterministically without model downloads.

The harness runs as a regression floor in CI (`tests/test_accuracy_eval.py`):
exact ≥ 0.95, temporal = 1.0, zero namespace leaks.

### What each column measures

| Operation | Pipeline |
|-----------|----------|
| **Vector** | FAISS `IndexFlatIP` — single BLAS matmul over L2-normalized embeddings |
| **Keyword** | SQLite FTS5 BM25 — inverted index with porter stemming |
| **Hybrid** | Vector + FTS5 run in parallel, fused with Reciprocal Rank Fusion |
| **recall()** | Hybrid search + access logging + retention scoring (full agent path) |

::: tip Sub-millisecond vector search
At 1K memories, the median vector search completes in **0.29 ms** — faster than a single network round-trip to any cloud vector database. At 5K it's still **0.63 ms**. This is the architectural advantage of in-process FAISS: no serialization, no network hop, no connection pool.
:::

---

## Insert Throughput

How fast can memories be ingested?

| Dataset | Latency per insert | Throughput |
|--------:|:------------------:|:----------:|
| **1,000** | 5.16 ms | **194 inserts/s** |
| **5,000** | 11.36 ms | **88 inserts/s** |

Insert includes: content hashing (SHA-256), dedup check (MinHash LSH), SQLite INSERT, FTS5 trigger, FAISS index add, and embedding normalization. The per-insert cost grows with dataset size because MinHash rebuilds its index periodically.

---

## Knowledge Graph

Typed entity relationships with multi-hop traversal via SQLite recursive CTEs.

| Operation | Time |
|-----------|------|
| **Build** (10 edges, 10 entities) | **14.7 ms** total |
| **Multi-hop traversal** (hops=3) | **0.13 ms** avg · 0.26 ms p99 |

Graph traversal uses SQLite recursive CTEs — no external graph database, no Cypher, no Gremlin. Edges are walked bidirectionally in a single query. At 0.13 ms per traversal, graph queries are essentially free.

---

## Deduplication (MinHash LSH)

Near-duplicate detection before memories enter the store.

| Operation | Time |
|-----------|------|
| **Insert** 1K documents into MinHash index | **1.21 ms/doc** |
| **Check** a document against the index | **1.39 ms** |

### Threshold sensitivity

MinHash Jaccard similarity is sensitive to text length and the `dedup_threshold` config:

| Threshold | Paraphrases detected (of 5) | Behaviour |
|:---------:|:---------------------------:|-----------|
| 0.5 | 5/5 (100%) | Aggressive — catches paraphrases, more false positives |
| 0.6 | 3/5 (60%) | Balanced for medium-length text |
| 0.7 | 2/5 (40%) | Conservative — only very similar texts |
| 0.8 | 0/5 (0%) | Very strict — exact near-duplicates only |

::: warning Threshold tuning
The default `dedup_threshold=0.8` is conservative. For agent memory (short facts, paraphrased across sessions), **0.5–0.6** catches more duplicates. For document-level dedup, 0.8 is appropriate. Tune based on your content length.
:::

---

## Cold Start (DB open + FAISS rebuild)

When Ariadne opens a database, it rebuilds the FAISS index from stored embeddings. This is the cost of never letting the index drift out of sync.

| Dataset | Cold start time |
|--------:|:---------------:|
| **1,000** vectors | **1,176 ms** |
| **5,000** vectors | **5,793 ms** |

Cold start scales linearly with vector count (FAISS training + add). For production use, keep the process alive rather than cold-starting per request. The tradeoff: zero index drift vs. startup cost.

---

## Memory Footprint

| Dataset | Database file size | Per memory |
|--------:|:-----------------:|:----------:|
| **1,000** | 2.3 MB | **2.4 KB** |
| **5,000** | 10.9 MB | **2.2 KB** |

Each memory stores: content text, SHA-256 hash, embedding BLOB (384 × 4 = 1,536 bytes), metadata, tags, timestamps, and access counts. The ~2.2 KB per-memory footprint includes all of this.

---

## Concurrent Throughput

Thread safety test: 4 concurrent readers + 2 concurrent writers for 3 seconds.

| Metric | Value |
|--------|:-----:|
| **Reads** | 1,001 ops (**330 reads/s**) |
| **Writes** | 44 ops (**15 writes/s**) |
| **Errors** | **0** |

Ariadne uses a reentrant lock (`threading.RLock`) to serialize SQLite + FAISS operations. Reads are side-effect-free (except access logging), so they serialize cleanly. Writes are heavier due to dedup + FTS sync + FAISS add. For agent workloads (mostly reads, occasional writes), this is well within bounds.

---

## Comparison with Other Memory Systems

### Feature comparison

| Capability | **Ariadne** | **Mem0** | **Zep** | **Honcho** | **Letta** | **ChromaDB** | **LangMem** | **cognee** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Vector search | FAISS (auto Flat→IVF) | Pluggable (Qdrant, etc.) | Proprietary | pgvector/Qdrant | SQLite/embeddings | HNSW | Via LangChain | Qdrant/PGVector |
| Keyword search (BM25) | **FTS5 built-in** | Partial | Partial | ❌ | ❌ | ❌ | ❌ | ❌ |
| Hybrid fusion (RRF) | **Built-in** | Partial | Partial | ❌ | ❌ | ❌ | ❌ | ❌ |
| Knowledge graph | **SQLite CTE** | Neo4j (optional) | Proprietary | ❌ | ❌ | ❌ | ❌ | Neo4j |
| Auto-deduplication | **MinHash LSH** | LLM-based | LLM-based | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cognitive retention | **Ebbinghaus curve** | ❌ | Temporal tracking | Profile extraction | Self-managed | ❌ | ❌ | ❌ |
| Runs fully local | **✅** | ✅ (self-hosted) | ❌ (cloud-first) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Zero infrastructure | **✅ single file** | ❌ needs vector DB | ❌ needs PostgreSQL | ❌ needs PostgreSQL | ❌ needs storage | ⚠️ | ⚠️ | ❌ needs Neo4j |
| Daemon/server required | **No** | No (self-hosted) | Yes (Go server) | No | No | No | No | No |
| License | MIT | Apache 2.0 | — | MIT | Apache 2.0 | Apache 2.0 | MIT | Apache 2.0 |

### Architecture differences

#### Ariadne vs Mem0

**Mem0** is the most feature-rich competitor. It extracts facts from conversations via LLM calls, stores them in a pluggable vector DB (Qdrant, Chroma, Pinecone), and optionally builds a knowledge graph via Neo4j.

| | Ariadne | Mem0 |
|---|---|---|
| **Storage** | Single SQLite file | Vector DB + optional Neo4j |
| **Extraction** | Application-provided | LLM-based automatic extraction |
| **Search** | FAISS + FTS5 + RRF (all in-process) | Vector DB query + optional graph |
| **Dedup** | MinHash LSH (deterministic, no LLM) | LLM-based (costs tokens per operation) |
| **Infra** | Zero — `pip install` + one file | Vector DB + LLM API keys |
| **Cost** | Free, zero ongoing cost | LLM API costs for extraction + dedup |
| **Latency** | Sub-ms vector search (in-process) | Network hop to vector DB |
| **Published benchmarks** | This page | MemoryBench (self-reported) |

Mem0's strength is automatic extraction from conversations — it decides what to remember. Ariadne's strength is the all-in-one retrieval stack with zero infrastructure and no LLM tax.

#### Ariadne vs Zep

**Zep** is enterprise-focused with a Go server, PostgreSQL backend, and proprietary knowledge graph extraction. It tracks temporal fact changes and supports SOC2/HIPAA compliance.

| | Ariadne | Zep |
|---|---|---|
| **Deployment** | Library (in-process) | Client → Go server → PostgreSQL |
| **Knowledge graph** | SQLite recursive CTE | Proprietary extraction pipeline |
| **Temporal awareness** | Ebbinghaus retention + access counts | Fact supersession tracking |
| **Enterprise features** | None (open-source) | SOC2, HIPAA, multi-tenant |
| **Pricing** | Free forever | $40/mo starter + per-message |

Zep's published benchmark claims ~93% factual recall with their knowledge graph vs. ~74% for plain vector search. Ariadne's hybrid RRF achieves similar recall improvements by combining vector + keyword without needing a dedicated graph extraction pipeline.

#### Ariadne vs Honcho

**Honcho** is not a general-purpose memory system — it's a **user persona extraction** tool. It ingests conversation history and builds structured user profiles (traits, preferences, goals).

| | Ariadne | Honcho |
|---|---|---|
| **Focus** | Agent memory (facts, graph, retrieval) | User modeling (persona extraction) |
| **Retrieval** | Vector + keyword + hybrid + graph | Profile attributes (structured) |
| **Use case** | "What did the user tell me?" | "Who is this user?" |

These solve different problems. Honcho builds a character profile; Ariadne stores and retrieves arbitrary memories. They could be complementary.

#### Ariadne vs Letta (MemGPT)

**Letta** gives LLMs self-managed memory via OS-inspired virtual context management. The agent decides what to remember and when to page data in/out of context.

| | Ariadne | Letta |
|---|---|---|
| **Memory management** | Application-controlled | LLM self-directed |
| **Architecture** | Store + search + graph | Virtual context window (page in/out) |
| **Overhead** | Zero (in-process) | LLM API calls for memory decisions |
| **Token cost** | None for storage/search | Significant (memory management calls) |
| **Published results** | This page | MemGPT paper: comparable to 4× context window |

Letta's MemGPT paper showed ~50-70% token savings vs. naive long-context approaches. Ariadne doesn't manage the LLM's context — it provides fast retrieval that the application or agent framework can call. Ariadne could serve as Letta's archival memory backend.

#### Ariadne vs ChromaDB

**ChromaDB** is a vector database, not a memory system. It provides embedding storage and approximate nearest neighbor search.

| | Ariadne | ChromaDB |
|---|---|---|
| **Abstraction** | Memory system (search + graph + dedup + retention) | Embedding database |
| **Search** | Vector + keyword + hybrid + graph | Vector only (HNSW) |
| **Dedup** | MinHash + content hash | ❌ |
| **Retention** | Ebbinghaus forgetting curve | ❌ |
| **Dependencies** | faiss-cpu + numpy + datasketch |chromadb (heavier) |

ChromaDB is often used as a backend for memory systems like Mem0. Ariadne replaces both the vector DB and the application layer in a single package.

#### Ariadne vs cognee

**cognee** builds knowledge graphs from documents using LLM extraction, then combines graph traversal with vector search for GraphRAG-style retrieval.

| | Ariadne | cognee |
|---|---|---|
| **Graph construction** | Manual (add_edge) or entity extraction | Automatic (LLM-based) |
| **Graph backend** | SQLite CTE | Neo4j |
| **Vector backend** | FAISS (in-process) | Qdrant/Weaviate/PGVector |
| **Pipeline** | Direct API calls | Document → chunk → embed → graph |
| **Complexity** | Single file | Multi-service (graph DB + vector DB) |

cognee is stronger for document-heavy knowledge graph construction. Ariadne is simpler and faster for agent memory where memories are added individually.

---

## Summary: Why Ariadne is different

| Property | Ariadne's approach | The alternative |
|----------|-------------------|-----------------|
| **Infrastructure** | Single SQLite file, zero daemons | Vector DB + graph DB + LLM API |
| **Search latency** | Sub-millisecond (in-process FAISS) | 10-100ms (network hop to hosted DB) |
| **Keyword search** | Built-in FTS5 with BM25 | Not available or bolt-on |
| **Hybrid retrieval** | Native RRF fusion | Custom integration required |
| **Deduplication** | Deterministic MinHash (no tokens) | LLM-based (costs money per check) |
| **Knowledge graph** | SQLite recursive CTE (free) | Neo4j or proprietary (infra + cost) |
| **Retention** | Ebbinghaus forgetting curve | Manual or absent |
| **Ongoing cost** | $0 | LLM API fees + hosting |

The tradeoff: Ariadne does not automatically extract memories from conversations (that's the application's job), and it doesn't manage the LLM's context window (that's the agent framework's job). What it does, it does fast, locally, and for free.

---

## Reproducing these benchmarks

```bash
pip install "ariadne-memory[embeddings]" numpy
```

Run the benchmark script:

```bash
git clone https://github.com/kyssta-exe/Ariadne.git
cd Ariadne
python benchmarks/run_benchmarks.py
```

Or use the inline harness:

```python
import time, numpy as np
from arriadne import AriadneMemory, AriadneConfig

mem = AriadneMemory(config=AriadneConfig(db_path="bench.db", embedding_dim=384))
N = 5_000
vecs = np.random.randn(N, 384).astype("float32")
for i, v in enumerate(vecs):
    mem.remember(f"memory {i}", embedding=v)

q = np.random.randn(384).astype("float32")
times = []
for _ in range(1000):
    t0 = time.perf_counter()
    mem.recall("query", embedding=q, k=10)
    times.append((time.perf_counter() - t0) * 1000)

print(f"recall p50: {np.percentile(times, 50):.3f} ms")
print(f"recall p99: {np.percentile(times, 99):.3f} ms")
mem.close()
```

::: warning Measure on your own hardware
Latency depends on CPU, embedding dimension, dataset size, and index type. These numbers are from a shared VPS — dedicated hardware will be faster. Always confirm with the harness on your own box.
:::
