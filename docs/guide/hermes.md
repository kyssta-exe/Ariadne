---
title: "Setup with Hermes Agent — Ariadne"
description: "Use Ariadne as the memory provider in Hermes Agent. One command setup, fast hybrid search, knowledge graph, deduplication."
---


Ariadne is a memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com/): fast vector + keyword search, hybrid retrieval, knowledge graph, and deduplication. It exposes its own `ariadne_*` memory tools.

---

## Option 1 — Autonomous Setup (Recommended)

Paste this entire block into Hermes as a single message. It contains everything Hermes needs — what Ariadne is, where to get it, how to install, and how to migrate:

```
Replace my memory provider with Ariadne. Here's everything you need:

ARIADNE
- PyPI: https://pypi.org/project/ariadne-memory/
- GitHub: https://github.com/kyssta-exe/Ariadne
- Docs: https://ariadne.mantes.net
- Plugin: https://github.com/kyssta-exe/Ariadne/tree/main/plugin

WHAT IT IS
Ariadne is a local memory system for AI agents. It uses FAISS for vector search, FTS5 for keywords, Reciprocal Rank Fusion to combine both, a knowledge graph with typed edges and multi-hop traversal, MinHash LSH deduplication, and an Ebbinghaus forgetting curve for retention. No cloud, no daemon, no API keys.

INSTALL
1. Install the Python package: pip install arriadne
2. Download the plugin files from https://github.com/kyssta-exe/Ariadne/tree/main/plugin and save them to ~/.hermes/plugins/ariadne/__init__.py and ~/.hermes/plugins/ariadne/plugin.yaml
   (Or clone the repo and copy: git clone https://github.com/kyssta-exe/Ariadne.git /tmp/ariadne && cp -r /tmp/ariadne/plugin ~/.hermes/plugins/ariadne)
3. Set the provider: hermes config set memory.provider ariadne
4. Migrate existing memories: if ~/.hermes/mnemosyne/data/mnemosyne.db exists, export all memories from it and import them into ~/.hermes/ariadne/memory.db using the ariadne CLI or the ariadne_import tool
5. Restart: hermes restart

DO THIS NOW. Install it, migrate my memories, and switch to Ariadne as my memory provider. After restarting, run ariadne_stats to confirm Ariadne is active.
```

Hermes will:
1. Install `ariadne` from PyPI
2. Download and place the plugin files
3. Switch the config to `memory.provider: ariadne`
4. Export your existing Mnemosyne memories and import them into Ariadne
5. Restart itself
6. Verify with `ariadne_stats`

After it finishes, send **"Check memory status"** — you should see `"engine": "Ariadne"` in the output.

---

## Option 2 — Manual Setup

Step-by-step if you prefer to do it yourself.

### Step 1 — Install the Python package

```bash
pip install arriadne
```

### Step 2 — Download the plugin

```bash
git clone https://github.com/kyssta-exe/Ariadne.git /tmp/ariadne-repo
mkdir -p ~/.hermes/plugins/ariadne
cp /tmp/ariadne-repo/plugin/__init__.py ~/.hermes/plugins/ariadne/
cp /tmp/ariadne-repo/plugin/plugin.yaml ~/.hermes/plugins/ariadne/
rm -rf /tmp/ariadne-repo
```

### Step 3 — Switch the provider

```bash
hermes config set memory.provider ariadne
```

Or edit `~/.hermes/config.yaml` directly:

```yaml
memory:
  provider: ariadne
```

### Step 4 — Migrate existing memories (optional)

If you were using Mnemosyne before and have memories to keep, export them to
JSON and import with the CLI:

```bash
ariadne migrate ~/mnemosyne_export.json --db-path ~/.hermes/ariadne/memory.db
```

### Step 5 — Restart

```bash
hermes restart
```

### Step 6 — Verify

Send to Hermes: **"Check memory status"**

You should see:

```json
{
  "engine": "Ariadne v0.1.2",
  "active_memories": 234,
  "graph_nodes": 58,
  "graph_edges": 60
}
```

If it says `engine: Ariadne` — you're done.

---

## What Changes

| | Before (Mnemosyne) | After (Ariadne) |
|---|---|---|
| Vector search | sqlite-vec (row scan) | **FAISS** (Flat → IVF) |
| Hybrid search | ❌ | **FTS5 + FAISS + RRF** |
| Dedup | ❌ | **MinHash LSH near-duplicates** |
| Knowledge graph | Basic edges | **Typed edges + multi-hop traversal** |
| Retention | ❌ | **Ebbinghaus forgetting curve** |

Ariadne exposes its own memory tools (`ariadne_remember`, `ariadne_recall`,
`ariadne_graph_query`, `ariadne_sleep`, etc.) through the standard Hermes
MemoryProvider interface, so the natural-language experience is unchanged — you
just talk to Hermes and it manages memory for you.

---

## Reverting

Switch back to Mnemosyne:

```bash
hermes config set memory.provider mnemosyne
hermes restart
```

Or manually:

```bash
mv ~/.hermes/plugins/ariadne ~/.hermes/plugins/ariadne.disabled
# Edit ~/.hermes/config.yaml: set provider back to mnemosyne
hermes restart
```

---

## File Layout

```
~/.hermes/
├── plugins/
│   └── ariadne/
│       ├── __init__.py     # MemoryProvider implementation
│       └── plugin.yaml     # Plugin metadata + tool schemas
└── ariadne/
    ├── memory.db           # SQLite: memories, graph, metadata, embeddings
    └── shared/
        └── memory.db       # Cross-agent shared surface
```

The FAISS vector index is rebuilt from the embeddings stored in `memory.db` on
startup, so there is no separate index file to back up or keep in sync.

Back up with: `cp -r ~/.hermes/ariadne ~/backup/ariadne-$(date +%Y%m%d)`
