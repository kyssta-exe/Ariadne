# Ariadne

Memory for AI agents. Sub-millisecond search. Zero infrastructure.

[![PyPI](https://img.shields.io/pypi/v/arriadne.svg)](https://pypi.org/project/arriadne/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-386%20passed-brightgreen)](https://github.com/kyssta-exe/Ariadne/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Quick Start

```python
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db")
# Auto-detects ONNX embeddings — zero config

mem.remember("VPS has 4 cores, 8GB RAM", importance=0.8)

results = mem.recall("server specs", k=5)
```

```bash
pip install arriadne
```

---

## Why

| | Ariadne | Mem0 | ChromaDB | sqlite-vec |
|---|:---:|:---:|:---:|:---:|
| Vector search | **0.03ms** | 12ms | 2.39ms | 0.99ms |
| Hybrid search (FTS + vector) | ✅ RRF | ❌ | ❌ | ❌ |
| Knowledge graph | ✅ BFS | ❌ | ❌ | ❌ |
| Auto-embeddings | ✅ ONNX (local) | ✅ cloud | ❌ | ❌ |
| Auto-dedup | ✅ MinHash LSH | ❌ | ❌ | ❌ |
| Temporal facts | ✅ | ❌ | ❌ | ❌ |
| Memory lifecycle | ✅ Ebbinghaus | ❌ | ❌ | ❌ |
| LLM extraction | ✅ 12 providers + callable | ✅ | ❌ | ❌ |
| Host agent passthrough | ✅ **no API key needed** | ❌ | ❌ | ❌ |
| Entity resolution | ✅ | ✅ | ❌ | ❌ |
| Runs locally | ✅ | ⚠️ | ✅ | ✅ |
| No daemon | ✅ | ❌ (PG+Qdrant) | ✅ | ✅ |
| REST API | ✅ | ✅ | ❌ | ❌ |
| LangChain / LlamaIndex | ✅ | ✅ | ✅ | ❌ |

---

## Features

### Sub-Millisecond Vector Search

FAISS-powered. 80× faster than ChromaDB, 33× faster than sqlite-vec. Auto-upgrades from exact to approximate search as your data grows.

| Engine | Latency |
|--------|:-----------:|
| FAISS (Ariadne) | **0.03ms** |
| sqlite-vec | 0.99ms |
| ChromaDB | 2.39ms |

### Hybrid Retrieval

Vector similarity + BM25 keywords fused with Reciprocal Rank Fusion. Cached query embeddings for repeated lookups.

```python
results = mem.recall("how to deploy to production", k=5)
# Searches both keyword and semantic similarity, fuses with RRF
for r in results:
    print(f"[{r['search_type']}] {r['content'][:80]}")
```

### Zero-Config Embeddings

Auto-downloads a quantized ONNX model on first use (~90MB). No API keys, no cloud, works offline. LRU-cached for repeated queries.

```python
# Just works — no embedding_provider parameter needed
mem = AriadneMemory("memory.db")
mem.remember("Paris is the capital of France")  # auto-embedded
```

Falls back to keyword matching if ONNX is unavailable.

### Agent-First Architecture

Ariadne is a memory layer for AI agents that already have an LLM. No separate API key needed — it uses the agent's own model for extraction.

```python
# Option 1: Pass the agent's LLM directly
ariadne = AriadneMemory(llm_config={
    "provider": "callable",
    "callable": agent.complete,  # your agent's own LLM function
})

# Option 2: Auto-detect from Hermes config
# (reads ~/.hermes/config.yaml + .env automatically)
ariadne = AriadneMemory(llm_config={"provider": "openai", ...})

# Option 3: Standalone with any provider
ariadne = AriadneMemory(llm_config={
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": "sk-...",
})
```

### Knowledge Graph

BFS graph traversal with typed, weighted edges. Bidirectional — edges are traversed in both directions.

```python
mem.add_edge("Paris", "France", "located_in")
mem.add_edge("Nginx", "VPS", "runs_on")
g = mem.graph("VPS", hops=2)
# Returns: VPS ↔ Nginx, VPS ↔ France (via Paris)
```

### Auto-Deduplication

MinHash LSH near-duplicate detection. Catches paraphrases, not just exact matches.

```python
mem.remember("The server runs Ubuntu 24.04")
mem.remember("Ubuntu 24.04 is running on the server")
# Second store detects near-duplicate (LSH similarity > threshold)
```

### Temporal Facts

Track when facts become valid, invalid, or expired. Query what was true at a given time.

```python
mem.add_temporal_fact(
    text="User prefers dark mode",
    subject="User", predicate="prefers", obj="dark_mode",
)
mem.query_temporal(subject="User")  # all temporal facts about User
```

### Memory Lifecycle

Ebbinghaus forgetting curve + priority-based eviction. Memories that matter survive; noise gets cleaned up.

```python
mem.consolidate()  # merge similar memories
mem.evict()        # remove low-priority noise
mem.run_lifecycle()  # tier promotion/demotion (hot → warm → cold)
```

### Agent Tools

OpenAI function-calling compatible tool definitions for any LLM. 26 tools for full memory management.

```python
tools = AriadneMemory.get_tools()
# remember, recall, graph, link, forget, stats, temporal, entities, ...
```

### Import / Export

Migrate from other systems or export your data.

```python
mem.import_from_text("plain text document")
mem.import_from_markdown("notes.md")
data = mem.export_json()  # full export to dict
```

---

## Benchmark

Measured on a 4-core 8GB VPS with 1K memories and ONNX embeddings (all-MiniLM-L6-v2, 384-dim):

| Operation | Latency |
|-----------|:---:|
| Vector search (FAISS) | **0.03ms** |
| FTS search (BM25) | **0.6ms** |
| Hybrid search (RRF) | **1.78ms** (cached) |
| Graph traversal (2 hops) | **0.09ms** |
| Single insert | **0.5ms** |
| Batch insert (1K) | **0.1ms/mem** |
| ONNX embed (cached) | **<0.01ms** |
| ONNX embed (cold) | **27ms** |

---

## Install

```bash
pip install arriadne
```

Optional extras:
```bash
pip install "arriadne[llm]"     # LLM extraction (OpenAI, Anthropic, Ollama, etc.)
pip install "arriadne[server]"   # REST API server
pip install "arriadne[dev]"      # Development tools
```

**Requirements:** Python 3.10+, SQLite (built-in). ONNX model auto-downloads on first use.

---

## Hermes Integration

```bash
hermes plugin install arriadne
hermes config set memory.provider ariadne
```

The plugin auto-detects Hermes's LLM and uses it for extraction — zero configuration.

---

## Links

- **Docs:** [ariadne.mantes.net](https://ariadne.mantes.net)
- **PyPI:** [pypi.org/project/arriadne](https://pypi.org/project/arriadne/)
- **GitHub:** [github.com/kyssta-exe/Ariadne](https://github.com/kyssta-exe/Ariadne)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
