---
name: ariadne
description: "Ariadne memory system for Hermes — FAISS + FTS5 + knowledge graph + cognitive retention. Local-first, zero daemon."
version: 1.0.0
author: hermes
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [memory, ai-agent, FAISS, knowledge-graph, search, hermes-plugin]
    homepage: https://ariadne-memory.readthedocs.io
---

# Ariadne — Memory System for Hermes

Ariadne is a local-first memory system for AI agents. It replaces Mnemosyne as the Hermes memory provider with a hybrid search engine: **FAISS** vector search + **SQLite FTS5** keyword search + **Reciprocal Rank Fusion**, plus a knowledge graph with typed edges and multi-hop traversal, MinHash LSH near-duplicate deduplication, and an Ebbinghaus forgetting curve for cognitive retention. No cloud, no daemon, no API keys.

## Features at a Glance

| Capability | Details |
|---|---|
| Vector search | FAISS (auto Flat→IVF at scale) |
| Keyword search | SQLite FTS5 BM25 |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) |
| Knowledge graph | Typed edges, multi-hop traversal via recursive CTEs |
| Deduplication | MinHash LSH + SHA-256 content hash |
| Retention | Ebbinghaus forgetting curve (stability grows with use) |
| Shared memory | Cross-agent surface via separate DB |
| Storage | Single SQLite file (~/.hermes/ariadne/memory.db) |

## Setup

### 1. Install the Python package

```bash
pip install ariadne-memory
```

With embeddings support (recommended):

```bash
pip install "ariadne-memory[embeddings]"
```

### 2. Install the Hermes plugin

```bash
git clone https://github.com/kyssta-exe/Ariadne.git /tmp/ariadne-repo
cp -r /tmp/ariadne-repo/plugin ~/.hermes/plugins/ariadne
rm -rf /tmp/ariadne-repo
```

Or manually create `~/.hermes/plugins/ariadne/` with `__init__.py` and `plugin.yaml` from the repo.

### 3. Switch the provider

```bash
hermes config set memory.provider ariadne
```

Or edit `~/.hermes/config.yaml`:

```yaml
memory:
  provider: ariadne
```

### 4. Migrate from Mnemosyne (optional)

If you have existing Mnemosyne memories:

```bash
ariadne migrate ~/.hermes/mnemosyne/data/mnemosyne.db \
  --db-path ~/.hermes/ariadne/memory.db
```

Or export to JSON first and import:

```bash
ariadne export -o /tmp/mnemosyne-export.json \
  --db-path ~/.hermes/mnemosyne/data/mnemosyne.db
ariadne import /tmp/mnemosyne-export.json \
  --db-path ~/.hermes/ariadne/memory.db
```

### 5. Restart and verify

```bash
hermes restart
```

Then ask Hermes: **"Check memory status"** — you should see `"engine": "Ariadne"` in the output.

### File Layout

```
~/.hermes/
├── plugins/
│   └── ariadne/
│       ├── __init__.py     # MemoryProvider implementation
│       └── plugin.yaml     # Plugin metadata + tool schemas
└── ariadne/
    ├── memory.db           # SQLite: memories, graph, embeddings, metadata
    └── shared/
        └── memory.db       # Cross-agent shared memory surface
```

## Plugin Tools

Ariadne exposes these tools through the Hermes MemoryProvider interface with the `ariadne_` prefix.

### Core Memory Tools

| Tool | Purpose | Key Params |
|---|---|---|
| `ariadne_remember` | Store a new memory | `content`, `importance` (0-1), `memory_type`, `entities`, `metadata` |
| `ariadne_recall` | Search memories (hybrid vector+keyword) | `query`, `limit`, `type_filter`, `importance_min` |
| `ariadne_forget` | Delete a memory by ID | `memory_id` |
| `ariadne_update` | Update memory content or metadata | `memory_id`, `content`, `importance`, `metadata` |
| `ariadne_invalidate` | Soft-delete / mark memory inactive | `memory_id` |
| `ariadne_stats` | Database statistics | (none) |

### Knowledge Graph Tools

| Tool | Purpose | Key Params |
|---|---|---|
| `ariadne_graph_link` | Create a typed edge between entities | `source`, `target`, `relationship`, `weight` |
| `ariadne_graph_query` | Traverse the graph from an entity | `entity`, `hops` (default 2) |

### Export / Import

| Tool | Purpose | Key Params |
|---|---|---|
| `ariadne_export` | Export all memories as JSON | `output_path` |
| `ariadne_import` | Import memories from JSON | `input_path` |

### Scratchpad (ephemeral notes)

| Tool | Purpose | Key Params |
|---|---|---|
| `ariadne_scratchpad_write` | Write a scratchpad note | `content` |
| `ariadne_scratchpad_read` | Read all scratchpad notes | (none) |
| `ariadne_scratchpad_clear` | Clear scratchpad | (none) |

### Shared Memory (cross-agent)

| Tool | Purpose | Key Params |
|---|---|---|
| `ariadne_shared_remember` | Store in shared surface | `content`, `importance`, `memory_type` |
| `ariadne_shared_recall` | Search shared surface | `query`, `limit` |
| `ariadne_shared_forget` | Delete from shared surface | `memory_id` |
| `ariadne_shared_stats` | Shared surface statistics | (none) |

### Diagnostics & Maintenance

| Tool | Purpose | Key Params |
|---|---|---|
| `ariadne_diagnose` | Health check + version info | (none) |
| `ariadne_sleep` | Run full maintenance cycle (consolidate + lifecycle) | `dry_run` |

## CLI Commands

The `ariadne` CLI provides direct database access outside of Hermes:

```bash
# Initialize a new database
ariadne init --db-path ~/.hermes/ariadne/memory.db --dim 384

# Add a memory
ariadne add "Deploy script lives in infra/deploy.sh" \
  --type semantic --importance 0.8 --db-path ~/.hermes/ariadne/memory.db

# Search memories
ariadne search "deploy script" -k 5 --db-path ~/.hermes/ariadne/memory.db

# Show stats
ariadne stats --db-path ~/.hermes/ariadne/memory.db

# Export/import JSON
ariadne export -o backup.json --db-path ~/.hermes/ariadne/memory.db
ariadne import backup.json --db-path ~/.hermes/ariadne/memory.db

# Run maintenance (consolidate + evict + prune)
ariadne maintain --db-path ~/.hermes/ariadne/memory.db
```

### Backup & Restore

```bash
# Create a consistent backup (WAL checkpoint + copy)
ariadne backup --db-path ~/.hermes/ariadne/memory.db \
  -o ~/.hermes/ariadne/backup-$(date +%Y%m%dT%H%M%S).db

# Restore from backup (creates safety backup first)
ariadne restore ~/.hermes/ariadne/backup-20260101T120000.db \
  --db-path ~/.hermes/ariadne/memory.db

# Restore without safety backup
ariadne restore ~/.hermes/ariadne/backup-20260101T120000.db \
  --db-path ~/.hermes/ariadne/memory.db --no-safety-backup
```

Quick backup shortcut (copies all files):

```bash
cp -r ~/.hermes/ariadne ~/backup/ariadne-$(date +%Y%m%d)
```

### Dashboard

Ariadne includes a web dashboard for visualizing memory stats, the knowledge graph, and search results.

```bash
# Install dashboard dependency
pip install "ariadne[dashboard]"

# Launch dashboard
ariadne dashboard --host 0.0.0.0 --port 8765

# Without auto-opening browser
ariadne dashboard --host 0.0.0.0 --port 8765 --no-browser
```

Open `http://localhost:8765` in a browser. The dashboard provides:
- Memory count and type breakdown
- Knowledge graph visualization (interactive)
- Search interface
- Lifecycle tier distribution (hot/warm/cold)
- FAISS index stats

## Best Practices

### Memory Quality
- **Set importance thoughtfully** (0.0–1.0): facts the agent needs long-term should be 0.7+; ephemeral context can be 0.3–0.5.
- **Use `memory_type`** to categorize: `semantic` (facts/preferences), `procedural` (how-to), `episodic` (events/conversations).
- **Add entities** when storing memories — this feeds the knowledge graph and improves graph traversal queries.

### Maintenance
- **Run `ariadne_sleep`** periodically (e.g., on cron or at session end) to consolidate duplicates and age out stale memories.
- **Use `dry_run: true`** on consolidate/prune first to see what would be affected before committing.
- **Back up before maintenance**: `ariadne backup` before large consolidation or pruning runs.

### Graph Usage
- **Link related entities** with `ariadne_graph_link` as the agent discovers relationships.
- **Query with hops=2 or 3** for most use cases; higher hops are slower and usually unnecessary.

### Shared Memory
- Use `ariadne_shared_*` tools for knowledge that should persist across agent sessions or be shared between multiple Hermes agents.
- The shared surface lives at `~/.hermes/ariadne/shared/memory.db` — back it up separately if needed.

### Performance
- FAISS auto-upgrades from `FlatIP` to `IndexIVFFlat` as the dataset grows — no manual tuning needed.
- The vector index is rebuilt from the database on every startup, so there's no separate index file to manage.
- For very large datasets (100K+ memories), the IVF index provides sub-millisecond recall.

## Reverting to Mnemosyne

```bash
hermes config set memory.provider mnemosyne
hermes restart
```

Or disable the plugin without removing it:

```bash
mv ~/.hermes/plugins/ariadne ~/.hermes/plugins/ariadne.disabled
hermes restart
```

## Addons

Ariadne supports domain-specific addons via Python entry_points. Addons are separate pip packages that register via the `ariadne.addons` entry_points group and are auto-discovered at runtime.

### Available Addons

| Addon | Description | Install |
|-------|-------------|---------|
| `ariadne-finance` | PDF/Excel extraction, ticker recognition, financial knowledge graph | `pip install ariadne-finance` |

### Finance Addon

```bash
# Core (Excel + CSV only)
pip install ariadne-finance

# With PDF support (marker-pdf + pdfplumber)
pip install "ariadne-finance[pdf]"

# Full (PDF + yfinance market data)
pip install "ariadne-finance[full]"
```

CLI usage:
```bash
ariadne finance ingest report.pdf --importance 0.8
ariadne finance search "NVDA revenue Q3"
ariadne finance tickers report.txt
```

Dashboard API: `GET /api/finance/tickers?text=...`, `GET /api/finance/classify?text=...`, `POST /api/finance/ingest`

### Creating Custom Addons

```python
from arriadne.addons import BaseAddon, ExtractorBase, EntityType

class MyAddon(BaseAddon):
    name = "my-addon"
    version = "0.1.0"
    description = "My custom domain addon"

    def get_extractors(self):
        return [MyExtractor()]

    def get_entity_types(self):
        return [EntityType(name="custom", display_name="Custom Entity")]

    def get_cli_commands(self):
        from arriadne.addons import CLICommand
        return [CLICommand(name="my-cmd", help_text="My command", handler=my_handler)]
```

Register in `pyproject.toml`:
```toml
[project.entry-points."ariadne.addons"]
my-addon = "my_addon:Addon"
```

Addons provide: extractors, entity types, CLI commands, API routes, search filters, graph relationships. See `docs/addons/index.md` for the full authoring guide.

### Pitfalls

- **Non-editable install required.** Use `pip install .` not `pip install -e .`. Editable installs create `.pth` files that can shadow other packages in the same namespace (e.g., `ariadne_finance` shadowed by stale `ariadne` editable finder).
- **Optional deps guard.** PDF extraction requires `marker-pdf` and `pdfplumber`. Always wrap in try/except ImportError with helpful install message.
- **Entity canonicalization.** Add canonical entity names on ingress (lowercase, strip, deduplicate) to prevent fragmentation: `"Mailcow"` vs `"mailcow"` vs `" mailcow "`.

## References

- **Docs**: https://ariadne-memory.readthedocs.io
- **GitHub**: https://github.com/kyssta-exe/Ariadne
- **PyPI**: https://pypi.org/project/ariadne-memory/
- **Hermes Guide**: https://ariadne-memory.readthedocs.io/guide/hermes
- **Addon docs**: `docs/addons/index.md`, `docs/addons/finance.md`
- **Finance addon**: `addons/finance/`
