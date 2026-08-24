# Ariadne

Memory for AI agents. Local-first hybrid search + knowledge graph. Zero infrastructure.

[![PyPI](https://img.shields.io/pypi/v/ariadne-memory.svg)](https://pypi.org/project/ariadne-memory/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-277%20passed-brightgreen)](https://github.com/kyssta-exe/Ariadne/actions)
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
| Session search & digests | ✅ | ❌ | ❌ | ⚠️ |
| Trust scoring (dynamic confidence) | ✅ | ❌ | ❌ | ⚠️ |
| Core memory blocks (always-in-context) | ✅ | ❌ | ❌ | ⚠️ |
| Cross-encoder reranking | ✅ | ❌ | ❌ | ✅ |
| Semantic (paraphrase) dedup | ✅ | ❌ | ❌ | ⚠️ |
| Entity resolution (merge/alias) | ✅ | ❌ | ❌ | ⚠️ |
| Async API | ⚠️ | ❌ | ❌ | ✅ |
| LLM-free operation possible | ✅ | ✅ | ✅ | ❌ |
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
Stored confidence from memory provenance/feedback is applied after retrieval, so
approved facts outrank rejected ones without hiding their history. Results
include `score_parts` explaining the RRF/FTS and confidence contribution.

```python
results = mem.recall("how to deploy to production", k=5)
# Runs keyword + vector search and fuses the rankings

context = mem.context_pack("how to deploy to production", token_budget=800)
# Compact, deterministic memory block ready for an agent prompt
```

### Knowledge graph

Typed entities and relationships with multi-hop traversal via SQLite recursive
CTEs. Edges are walked in both directions:

```python
mem.add_edge("WebApp", "API", edge_type="depends_on")
mem.add_edge("API", "Database", edge_type="depends_on")
mem.graph("WebApp", hops=2)   # → API, Database
```

Recall results can also be *expanded* through the entity graph — memories that
share entities with the direct hits join the result set at a decayed score, so
associations never outrank actual matches:

```python
results = mem.recall("user data", k=3)
expanded = mem.expand(results, hops=1)
```

### Session intelligence

Raw conversation turns are recorded as immutable episodes, and those episodes
are first-class search targets — the "what did I try last week?" surface that
distilled memories can't answer:

```python
mem.process_turn("The auth bug was token expiry", "Fixed by refreshing tokens.")

mem.search_episodes("auth bug token", k=5)     # search raw history
mem.digest_session("session-42")               # compact digest memory, with
                                               # provenance back to the turns
mem.session_context()                          # "recent session context" block
mem.context_pack("deploy", include_sessions=True)  # digests + recall, budgeted
```

Digests are deterministic and LLM-free (role weighting + recurring-term
salience), idempotent per session, and supersede cleanly on `force=True`.

### Trust scoring

Confidence is dynamic, not a write-time guess. A new write that contradicts a
stored memory decays that memory's confidence; explicit reinforcement and
curation winners gain it back; retrieval re-weights rankings accordingly:

```python
mem.remember("The API port is 8080")
mem.remember("The API port is not 8080, it is 9090")  # old fact loses trust

mem.reinforce(memory_id)   # confirmed useful → confidence rises
```

### Autonomous memory (optional LLM)

`LLMMemoryManager` turns raw turns into structured memories via any
prompt→text callable, with a deterministic, LLM-free update policy in the
mem0 style: each candidate fact is resolved against the closest stored memory
as ADD (novel), UPDATE (contradiction → supersede, history preserved), or NOOP
(near-duplicate). KV facts (`subject.attribute = value`) upsert precisely via
`set_fact()`.

### Core memory blocks (Letta-style)

A small set of named, always-in-context blocks — persona, user profile,
project state — that the agent self-edits during a run. They live outside the
memory lifecycle: no dedup, no decay, no eviction. The Hermes plugin injects
them into every turn; MCP exposes `ariadne_core_view` / `ariadne_core_append`
/ `ariadne_core_replace`; the dashboard has a REST API:

```python
mem.core_append("user_profile", "Prefers concise answers. ")
mem.core_set("project_state", "Migration phase 2 of 3.")
mem.core_pack()                      # → "Core memory:" context section
mem.context_pack("deploy", include_core=True, include_sessions=True)
```

### Reranking (second retrieval stage)

RRF fusion is retrieval-grade; a cross-encoder reads `(query, document)`
jointly and orders the top-k far better. Opt in per query — the model loads
lazily and degrades to fused order when `sentence-transformers` isn't
installed:

```python
results = mem.recall("how to deploy to production", k=5, rerank=True)
# results[0]["score_parts"] carries both "fused" and "rerank" scores
```

### Semantic deduplication

MinHash is lexical — paraphrases slip through. When an embedder is
configured, `remember()` checks the nearest stored vector and treats
near-identical *meaning* (cosine ≥ `semantic_dedup_threshold`, default 0.92)
as a duplicate, reinforcing the existing memory instead of storing noise.

### Entity resolution

`merge_entities("pg", "postgres")` re-points memory links and graph edges
onto the canonical entity and removes the alias — repairing the graph
fragmentation that abbreviations cause.

### Async-first

Every modern agent framework is asyncio-native. `AsyncAriadneMemory` mirrors
the hot path as coroutines (delegating to `asyncio.to_thread`, so the event
loop never stalls on SQLite/FAISS work):

```python
from arriadne import AsyncAriadneMemory

async with AsyncAriadneMemory(db_path="memory.db") as mem:
    await mem.remember("Async agents need memory too")
    hits = await mem.recall("async memory", k=5, rerank=True)
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

v0.13.0 lifecycle numbers (Windows/Python 3.13, 384-dim, measured with
`scripts/perf_scale.py`): **warm restart ≈0.83 s at 20k memories** (was 13.6 s
— FAISS + dedup sidecar indexes), **bulk ingest ≈3.9 ms/write at 20k** (was
23 ms — 64 MB page cache, WAL `synchronous=NORMAL`, tiered maintenance).
Retrieval stays flat with size: hybrid ≈3 ms and full `recall()` ≈3.8 ms at
20k memories. An accuracy harness (`benchmarks/accuracy_eval.py`) keeps the
other side honest: 0.950 answer recall@5 FTS-only, 1.000 on temporal
supersession, zero namespace leaks — run `ariadne doctor` to check a live
store's health.

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
| `ariadne_context_pack` | Pack relevant memories under a token budget |
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
