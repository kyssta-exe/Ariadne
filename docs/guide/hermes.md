# Hermes Agent Integration

Ariadne ships as a drop-in memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com/). Replace the default Mnemosyne memory with Ariadne for 196× faster search, hybrid retrieval, and a knowledge graph.

## Prerequisites

- Hermes Agent installed ([docs](https://hermes-agent.nousresearch.com/docs))
- Ariadne installed: `pip install arriadne`

## Step 1 — Copy the Plugin

Ariadne includes a Hermes plugin that implements the `MemoryProvider` interface. Copy it into Hermes's plugin directory:

```bash
# From the Ariadne repo (or download the plugin/ directory from GitHub)
git clone https://github.com/kyssta-exe/Ariadne.git /tmp/ariadne-repo
cp -r /tmp/ariadne-repo/plugin ~/.hermes/plugins/ariadne
```

Or if you have the PyPI package installed, create the plugin manually:

```bash
mkdir -p ~/.hermes/plugins/ariadne
cat > ~/.hermes/plugins/ariadne/__init__.py << 'PLUGIN'
"""Ariadne Memory Provider for Hermes — drop-in replacement for Mnemosyne."""
# See: https://github.com/kyssta-exe/Ariadne/tree/main/plugin
PLUGIN

# Download the actual plugin from GitHub
curl -sL https://raw.githubusercontent.com/kyssta-exe/Ariadne/main/plugin/__init__.py \
  -o ~/.hermes/plugins/ariadne/__init__.py
curl -sL https://raw.githubusercontent.com/kyssta-exe/Ariadne/main/plugin/plugin.yaml \
  -o ~/.hermes/plugins/ariadne/plugin.yaml
```

## Step 2 — Switch the Provider

Edit `~/.hermes/config.yaml` and change:

```yaml
memory:
  provider: mnemosyne    # ← change this
```

To:

```yaml
memory:
  provider: ariadne
```

## Step 3 — Restart Hermes

```bash
hermes restart
```

Or if Hermes is running as a service:

```bash
systemctl restart hermes
```

## Step 4 — Verify

Open a conversation with Hermes and send:

```
Check memory status
```

Hermes should call `mnemosyne_stats` (tool name unchanged for compatibility) and return:

```json
{
  "engine": "Ariadne v0.1.2",
  "active_memories": 234,
  "working_memories": 184,
  "graph_nodes": 58,
  "graph_edges": 60
}
```

If you see `engine: Ariadne` — you're done.

## What the Agent Can Do

Once active, these natural language commands work through Ariadne:

**Store a fact:**
```
Remember that my VPS is at 51.75.73.169 with 4 cores and 8GB RAM
```

**Search memories:**
```
What do I know about my server setup?
```

**Build a knowledge graph:**
```
Link VPS depends_on nginx, nginx depends_on SSL certificates
```

**Run diagnostics:**
```
Run memory diagnostics
```

## Migrating from Mnemosyne

If you had memories in Mnemosyne before:

```bash
# Export from old DB
ariadne migrate --from mnemosyne --to ~/.hermes/ariadne/memory.db
```

Or let Hermes handle it — it reads the old Mnemosyne DB automatically during first run if `~/.hermes/mnemosyne/data/mnemosyne.db` exists.

## Reverting

Switch back to Mnemosyne:

```bash
mv ~/.hermes/plugins/ariadne ~/.hermes/plugins/ariadne.disabled
hermes config set memory.provider mnemosyne
hermes restart
```

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
    ├── memory.faiss.idmap  # FAISS: ID mapping (auto-created)
    └── shared/
        └── memory.db       # Cross-agent shared surface
```

Total disk: ~5MB per 1,000 memories. Back up with:

```bash
cp -r ~/.hermes/ariadne ~/backup/ariadne-$(date +%Y%m%d)
```
