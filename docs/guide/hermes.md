# Hermes Agent Integration

Ariadne ships as a drop-in memory provider for [Hermes Agent](https://github.com/hermes-agent/hermes-agent). One command to set up, zero configuration required.

## One-Command Setup

Tell Hermes to switch to Ariadne:

```
Switch my memory provider to ariadne
```

Hermes will:
1. Install the `ariadne` plugin in `~/.hermes/plugins/ariadne/`
2. Set `memory.provider: ariadne` in `~/.hermes/config.yaml`
3. Create the database at `~/.hermes/ariadne/memory.db`
4. Restart the memory system with Ariadne active

That's it. No manual config editing, no plugin files to copy.

## Verify It's Working

```
Check memory status
```

Hermes will run `mnemosyne_stats` (the tool name stays the same for compatibility) and return:

```json
{
  "engine": "Ariadne v0.1.2",
  "active_memories": 234,
  "working_memories": 184,
  "episodic_memories": 12,
  "graph_nodes": 58,
  "graph_edges": 60
}
```

If you see `engine: Ariadne` — you're good.

## What Changes (and What Doesn't)

### Nothing breaks

Ariadne uses the same tool names as Mnemosyne (`mnemosyne_remember`, `mnemosyne_recall`, etc.). Your existing conversations, cron jobs, and memory references all work identically.

### What gets better

| Before (Mnemosyne) | After (Ariadne) |
|---------------------|-----------------|
| 153ms vector search | **0.78ms** |
| No hybrid search | **FTS5 + FAISS + RRF fusion** |
| No dedup | **MinHash LSH at 0.12ms** |
| Basic graph | **Typed edges + multi-hop traversal** |
| No retention model | **Ebbinghaus forgetting curve** |

### What the agent can do

Once Ariadne is active, Hermes gains these capabilities through the same tools it already uses:

**Store any fact:**
```
Remember that my VPS is at 51.75.73.169 with 4 cores and 8GB RAM
```
→ Calls `mnemosyne_remember` → Ariadne stores it with FAISS vector embedding + FTS5 keyword index.

**Search with hybrid retrieval:**
```
What do I know about my server setup?
```
→ Calls `mnemosyne_recall("server setup")` → Returns results ranked by vector similarity + keyword match + graph connections.

**Build a knowledge graph:**
```
Link VPS depends_on nginx, nginx depends_on SSL certificates
```
→ Calls `mnemosyne_graph_link` twice → Creates typed edges. Later, `mnemosyne_graph_query("VPS")` traverses the full chain.

**Run diagnostics:**
```
Run memory diagnostics
```
→ Calls `mnemosyne_diagnose` → Returns health check: DB size, FAISS index type, graph connectivity, dedup stats.

## Migrating from Mnemosyne

If you were using Mnemosyne before, your existing data can be migrated:

```
Import my old mnemosyne memories into ariadne
```

Hermes will:
1. Export from `~/.hermes/mnemosyne/data/mnemosyne.db`
2. Import into `~/.hermes/ariadne/memory.db`
3. Preserve graph edges, importance scores, and memory types

Or manually:
```bash
ariadne migrate --from mnemosyne --to ~/.hermes/ariadne/memory.db
```

## Reverting to Mnemosyne

If something goes wrong:
```
Switch my memory provider back to mnemosyne
```

Or manually:
```bash
# Re-enable old plugin
mv ~/.hermes/plugins/mnemosyne.disabled ~/.hermes/plugins/mnemosyne

# Update config
hermes config set memory.provider mnemosyne
```

## Architecture

```
~/.hermes/ariadne/
├── memory.db          # SQLite: memories, graph, metadata
├── memory.faiss       # FAISS: vector index (auto-created)
├── memory.faiss.idmap # FAISS: ID mapping (auto-created)
├── shared/
│   └── memory.db      # Cross-agent shared surface
└── *.wal              # Write-ahead log (auto-managed)
```

Total disk usage: ~5MB for 1,000 memories. ~50MB for 10,000.

The database is a single directory you can back up with:
```bash
cp -r ~/.hermes/ariadne ~/backup/ariadne-$(date +%Y%m%d)
```
