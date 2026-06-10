# Ariadne

Memory for AI agents. Local-first hybrid search + knowledge graph. Zero infrastructure.

[![PyPI](https://img.shields.io/pypi/v/ariadne-memory.svg)](https://pypi.org/project/ariadne-memory/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-166%20passed-brightgreen)](https://github.com/kyssta-exe/Ariadne/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Quick Start

```bash
pip install "ariadne-memory[embeddings]"
```

```python
from arriadne import AriadneMemory
from arriadne.embeddings import SentenceTransformerEmbedder

# An embedder turns text into vectors so semantic recall works automatically.
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")  # 384-dim

mem = AriadneMemory(db_path="memory.db", embedding_dim=embedder.dim, embedder=embedder)

mem.remember("VPS has 4 cores, 8GB RAM", importance=0.8)

# Semantic match — "server specs" finds the memory despite sharing no keywords.
results = mem.recall("server specs", k=5)
```

Without the `[embeddings]` extra (or without an `embedder`), Ariadne still works
as a fast **keyword** store — pass your own vectors to `remember`/`recall` for
semantic search, or omit them for FTS-only matching:

```python
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db")          # no embedder
mem.remember("deploy script lives in infra/deploy.sh")
mem.recall("deploy script", k=5)                  # keyword match
```

---

## Why

Most "agent memory" options make you choose: a bare vector store (Chroma,
sqlite-vec), or a hosted service (Mem0). Ariadne bundles vector + keyword +
graph retrieval, deduplication, and a retention model into one local SQLite
file — no daemon, no server, no API keys.

| Capability | Ariadne | Chroma | sqlite-vec | Mem0 |
|---|:---:|:---:|:---:|:---:|
| Vector search | ✅ FAISS (auto Flat→IVF) | ✅ | ✅ | ✅ |
| Keyword search (BM25/FTS5) | ✅ | ❌ | ❌ | ⚠️ |
| Hybrid fusion (RRF) | ✅ | ⚠️ basic | ❌ | ⚠️ |
| Knowledge graph (multi-hop) | ✅ | ❌ | ❌ | ⚠️ |
| Near-duplicate dedup (MinHash) | ✅ | ❌ | ❌ | ⚠️ |
| Retention / forgetting curve | ✅ | ❌ | ❌ | ⚠️ |
| Runs fully local, no daemon | ✅ | ✅ | ✅ | ❌ |
| Single file, zero infra | ✅ | ⚠️ | ✅ | ❌ |

Capability comparison, not a benchmark — for latency, measure on your own
hardware (see [Performance](#performance)). ✅ built-in · ⚠️ partial/varies · ❌ not available.

---

## Features

### Vector search (FAISS)

In-process FAISS index. Starts as exact `IndexFlatIP` and auto-upgrades to
`IndexIVFFlat` once the dataset grows past `ivf_threshold`. Vectors are keyed by
the memory's own id (`IndexIDMap2`) and rebuilt from the database on open, so the
index can never drift out of sync after deletes or restarts.

### Hybrid retrieval

Vector similarity + BM25 keywords (SQLite FTS5), fused with Reciprocal Rank
Fusion. Keyword matching tries AND first (precise) and falls back to OR (recall).

```python
results = mem.recall("how to deploy to production", k=5)
# Runs keyword + vector search and fuses the rankings
```

### Knowledge graph

Typed entities and relationships with multi-hop traversal via SQLite recursive
CTEs. Edges are walked in both directions:

```python
mem.add_edge("WebApp", "API", edge_type="depends_on")
mem.add_edge("API", "Database", edge_type="depends_on")
mem.graph("WebApp", hops=2)   # → API, Database
```

### Cognitive retention

Ebbinghaus forgetting curve `R = e^(-t/S)`. Stability `S` grows each time a
memory is recalled (`retention_growth_factor`, capped) — memories strengthen
with use and fade without it. Priority-weighted scoring from importance,
recency, access count, and retention drives eviction.

### Auto-deduplication

MinHash LSH catches near-duplicates before they enter the store; the index is
rebuilt from the database on open so it survives restarts. Exact duplicates are
caught by a SHA-256 content hash.

### Built for agents

Thread-safe (a single `AriadneMemory` can be shared across threads), reads are
side-effect-free, and housekeeping (`evict` / `consolidate` / `prune_access_log`
/ `purge_deleted`, or `maintenance()` for all four) keeps the store bounded.

---

## Performance

Latency depends on your hardware, embedding dimension, and dataset size, so
Ariadne ships no canned numbers — measure on your own box:

```bash
pip install "ariadne-memory[embeddings]"
```

```python
import time, numpy as np
from arriadne import AriadneMemory, AriadneConfig

mem = AriadneMemory(config=AriadneConfig(db_path="bench.db", embedding_dim=384))
vecs = np.random.randn(10_000, 384).astype("float32")
for i, v in enumerate(vecs):
    mem.remember(f"memory {i}", embedding=v)

q = np.random.randn(384).astype("float32")
t = time.perf_counter()
for _ in range(1000):
    mem.recall("query", embedding=q, k=10)
print(f"recall avg: {(time.perf_counter() - t):.3f} ms/query")
mem.close()
```

Architecturally: FAISS does similarity as a single BLAS matrix multiply (and
switches to an inverted-file index at scale), keyword search rides SQLite's FTS5
BM25 index, and graph traversal is a recursive CTE — all in-process, no network
hops. See the [benchmarks guide](https://ariadne.mantes.net/benchmarks) for a
fuller harness.

---

## Hermes Agent Integration

Ariadne works as a drop-in memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com/),
giving your agent durable hybrid search memory with zero infrastructure.

### Plugin Setup

```bash
git clone https://github.com/kyssta-exe/Ariadne.git /tmp/ariadne-repo
cp -r /tmp/ariadne-repo/plugin ~/.hermes/plugins/ariadne
```

Then configure Hermes to use Ariadne:

```bash
hermes config set memory.provider ariadne
hermes restart
```

Alternatively, set the provider in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: ariadne
```

The plugin automatically creates its database at `~/.hermes/ariadne/memory.db`
(plus a shared surface at `~/.hermes/ariadne/shared/memory.db` for cross-agent
memory).

### Available Tools

The plugin exposes these `ariadne_*` tools to Hermes:

| Tool | Description |
|------|-------------|
| `ariadne_remember` | Store a durable memory (fact, preference, insight, etc.) |
| `ariadne_recall` | Hybrid search — FTS5 text + FAISS vector ranking |
| `ariadne_stats` | Return memory system statistics |
| `ariadne_forget` | Permanently delete a memory by ID |
| `ariadne_update` | Update content or importance of an existing memory |
| `ariadne_invalidate` | Soft-delete (mark as superseded) a memory |
| `ariadne_export` | Export all memories to a JSON file |
| `ariadne_import` | Import memories from a JSON file |
| `ariadne_graph_query` | Traverse the knowledge graph from a seed entity |
| `ariadne_graph_link` | Declare a relationship between two entities |
| `ariadne_sleep` | Run memory consolidation (compress old working memories) |
| `ariadne_diagnose` | Run diagnostics on the Ariadne installation |
| `ariadne_scratchpad_write` | Write a temporary note to the scratchpad |
| `ariadne_scratchpad_read` | Read scratchpad entries |
| `ariadne_scratchpad_clear` | Clear all scratchpad entries |
| `ariadne_shared_remember` | Store a memory in the shared surface DB (cross-agent) |
| `ariadne_shared_recall` | Search the shared surface DB |
| `ariadne_shared_forget` | Delete a shared surface memory |
| `ariadne_shared_stats` | Return shared surface DB stats |

Full guide: [ariadne.mantes.net/guide/hermes](https://ariadne.mantes.net/guide/hermes)

---

## Configuration

```python
from arriadne import AriadneConfig, AriadneMemory

config = AriadneConfig(
    db_path="memory.db",
    embedding_dim=384,
    faiss_type="auto",          # auto | flat_ip | ivf_flat
    dedup_threshold=0.8,
    retention_half_life=86400,  # 1 day
)

mem = AriadneMemory(config=config)
```

---

## Documentation

**[ariadne.mantes.net](https://ariadne.mantes.net)**

- [Quick Start](https://ariadne.mantes.net/guide/quick-start)
- [Installation](https://ariadne.mantes.net/guide/installation)
- [Hermes Setup](https://ariadne.mantes.net/guide/hermes)
- [Search & Retrieval](https://ariadne.mantes.net/guide/search)
- [Knowledge Graph](https://ariadne.mantes.net/guide/graph)
- [API Reference](https://ariadne.mantes.net/api/)
- [Benchmarks](https://ariadne.mantes.net/benchmarks)

---

## Backup & Restore

Ariadne supports full database backup and restore through the CLI, the web
dashboard, and the Python API. Backups are consistent SQLite snapshots (WAL
checkpoint + file copy) — no daemon restart required.

### CLI Commands

```bash
# Create a timestamped backup (default: arriadne-backup-YYYYMMDDTHHMMSS.db)
ariadne backup

# Backup to a specific file
ariadne backup -o /backups/my-memory.db

# Restore from a backup (creates a safety backup of the current DB first)
ariadne restore /backups/my-memory.db

# Restore without safety backup
ariadne restore /backups/my-memory.db --no-safety-backup

# Export all memories as JSON (to stdout or a file)
ariadne export
ariadne export -o memories.json

# Import memories from a JSON file
ariadne import memories.json
```

### Dashboard UI

Launch the dashboard and use the backup/restore controls:

```bash
ariadne dashboard
```

The dashboard exposes two endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/backup` | `GET` | Download the current database as a `.db` file |
| `/api/restore` | `POST` | Upload a `.db` file to restore (creates a safety backup automatically) |

### Python API

```python
from arriadne import AriadneMemory, AriadneConfig

mem = AriadneMemory(config=AriadneConfig(db_path="memory.db"))

# Export all memories to a dict
data = mem.export_json()
# data contains {"memories": [...], "stats": {...}}

# Import from a previously exported dict
imported_count = mem.import_json(data)
print(f"Imported {imported_count} memories")

mem.close()
```

---

## Addons

Ariadne supports domain-specific addons that extend the core memory system
with specialized extractors, entity types, CLI commands, and API endpoints.
Addons are separate pip packages discovered automatically via Python entry points.

### Available Addons

| Addon | Description | Install |
|-------|-------------|---------|
| [ariadne-finance](addons/finance/) | Finance research — PDF/Excel extraction, ticker recognition, financial knowledge graph | `pip install ariadne-finance` |

### Installing an Addon

```bash
# Install the finance addon (Excel + CSV only)
pip install ariadne-finance

# With PDF support
pip install "ariadne-finance[pdf]"

# Full (PDF + yfinance for market data)
pip install "ariadne-finance[full]"
```

Once installed, the addon is auto-discovered — no configuration needed:

```python
from arriadne.addons import AddonRegistry

registry = AddonRegistry()
registry.discover()  # finds all installed addons
print(registry.addon_names)  # ['ariadne-finance']

# Use addon extractors
extractor = registry.get_extractor_for_file("report.pdf")
result = extractor.extract("report.pdf")

registry.shutdown()
```

### Creating Your Own Addon

See [docs/addons/index.md](docs/addons/index.md) for the full addon authoring guide.
Quick start:

```python
from arriadne.addons import BaseAddon, ExtractorBase, EntityType

class MyAddon(BaseAddon):
    name = "my-addon"
    version = "0.1.0"
    description = "My custom addon"

    def get_extractors(self):
        return [MyExtractor()]

    def get_entity_types(self):
        return [EntityType(name="custom", display_name="Custom Entity")]
```

Register in your `pyproject.toml`:

```toml
[project.entry-points."ariadne.addons"]
my-addon = "my_addon:MyAddon"
```

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <sub>Powered by <a href="https://mantes.net">Mantes</a></sub>
</p>
