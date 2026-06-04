---
title: "Setup with Hermes Agent — Ariadne"
description: "Replace Mnemosyne with Ariadne in Hermes Agent. One command setup, 196x faster search, same tool names."
---


Ariadne replaces the default Mnemosyne memory provider in [Hermes Agent](https://hermes-agent.nousresearch.com/). 196× faster search, hybrid retrieval, knowledge graph, deduplication. Same tool names, same interface.

---

## Option 1 — Autonomous Setup (Recommended)

Paste this entire block into Hermes as a single message. It contains everything Hermes needs — what Ariadne is, where to get it, how to install, and how to migrate:

```
Replace my memory provider with Ariadne. Here's everything you need:

ARIADENE
- PyPI: https://pypi.org/project/arriadne/
- GitHub: https://github.com/kyssta-exe/Ariadne
- Docs: https://ariadne.mantes.net
- Plugin: https://github.com/kyssta-exe/Ariadne/tree/main/plugin

WHAT IT IS
Ariadne is a local memory system for AI agents. It uses FAISS for vector search (0.78ms vs Mnemosyne's 153ms), FTS5 for keywords, Reciprocal Rank Fusion to combine both, a knowledge graph with typed edges and multi-hop traversal, MinHash LSH deduplication, and an Ebbinghaus forgetting curve for retention. No cloud, no daemon, no API keys.

INSTALL
1. Install the Python package: pip install arriadne
2. Download the plugin files from https://github.com/kyssta-exe/Ariadne/tree/main/plugin and save them to ~/.hermes/plugins/ariadne/__init__.py and ~/.hermes/plugins/ariadne/plugin.yaml
   (Or clone the repo and copy: git clone https://github.com/kyssta-exe/Ariadne.git /tmp/ariadne && cp -r /tmp/ariadne/plugin ~/.hermes/plugins/ariadne)
3. Set the provider: hermes config set memory.provider ariadne
4. Migrate existing memories: if ~/.hermes/mnemosyne/data/mnemosyne.db exists, export all memories from it and import them into ~/.hermes/ariadne/memory.db using the ariadne CLI or the mnemosyne_import tool
5. Restart: hermes restart

DO THIS NOW. Install it, migrate my memories, and switch to Ariadne as my memory provider. After restarting, run mnemosyne_stats to confirm Ariadne is active.
```

Hermes will:
1. Install `ariadne` from PyPI
2. Download and place the plugin files
3. Switch the config to `memory.provider: ariadne`
4. Export your existing Mnemosyne memories and import them into Ariadne
5. Restart itself
6. Verify with `mnemosyne_stats`

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

If you were using Mnemosyne before and have memories to keep:

```bash
ariadne migrate --from mnemosyne --to ~/.hermes/ariadne/memory.db
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
| Vector search | 153ms (sqlite-vec) | **0.78ms** (FAISS) |
| Hybrid search | ❌ | **FTS5 + FAISS + RRF** |
| Dedup | ❌ | **MinHash LSH (0.12ms)** |
| Knowledge graph | Basic edges | **Typed edges + BFS traversal** |
| Retention | ❌ | **Ebbinghaus forgetting curve** |

**Nothing else changes.** Same tool names (`mnemosyne_remember`, `mnemosyne_recall`, etc.), same natural language interface, same conversations. Hermes calls the same tools — Ariadne just makes them 196× faster.

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
    ├── memory.db           # SQLite: memories, graph, metadata
    ├── memory.faiss        # FAISS: vector index (auto-created)
    └── shared/
        └── memory.db       # Cross-agent shared surface
```

Back up with: `cp -r ~/.hermes/ariadne ~/backup/ariadne-$(date +%Y%m%d)`

---

## Frequently Asked Questions

### Will Ariadne break my existing Hermes setup?
No. Ariadne uses the same tool names as Mnemosyne. All conversations, cron jobs, and memory references work unchanged.

### Can I switch back to Mnemosyne?
Yes. Run `hermes config set memory.provider mnemosyne` and restart.

### Does Ariadne work with Hermes cron jobs?
Yes. Cron jobs using `mnemosyne_remember`/`mnemosyne_recall` work identically.

### How do I know if Ariadne is active?
Send "Check memory status" to Hermes. If the output shows `"engine": "Ariadne"`, it's working.

