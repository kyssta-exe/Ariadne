---
title: "Architecture — Ariadne"
description: "Ariadne internals: SQLite + FAISS storage engine, search pipeline, knowledge graph, and memory lifecycle."
---


Ariadne is a single-process, zero-daemon memory system built on **SQLite** (with WAL mode), **FAISS** (vector index), and **MinHash LSH** (deduplication). No external servers, no cloud dependencies.

## Storage Layers

```
┌─────────────────────────────────────────────────────────────┐
│                        AriadneDB                            │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │    FAISS     │  │    SQLite    │  │   MinHash LSH     │  │
│  │ IndexIDMap2  │  │  Metadata +  │  │   Dedup Index     │  │
│  │ (Flat / IVF) │  │  FTS5 + Graph│  │   (in-memory)     │  │
│  │  cosine sim  │  │  + embeddings│  │                   │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
│                                                             │
│  rebuilt on open   .db file           rebuilt on open       │
└─────────────────────────────────────────────────────────────┘
```

The SQLite `.db` file is the single source of truth. Embeddings are stored as
BLOBs in the `memories` table; the FAISS index and the MinHash dedup index are
both **rebuilt from the database on open**, so there is no separate index file
to keep in sync.

| Layer | Technology | Purpose | Persistence |
|-------|-----------|---------|-------------|
| Vector Search | FAISS `IndexIDMap2` over `IndexFlatIP` / `IndexIVFFlat` | Semantic similarity | Embeddings in `.db`; index rebuilt on open |
| Metadata & FTS | SQLite 3 (WAL mode) | Structured data, keyword search, graph | `.db` file |
| Dedup Index | MinHash LSH (datasketch) | Near-duplicate detection | In-memory (rebuilds from DB) |

## SQLite Schema

Ariadne uses the following tables:

### `memories`

The primary table storing all memory records.

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,           -- SHA-256 for exact dedup
    memory_type TEXT NOT NULL DEFAULT 'semantic',
    importance REAL NOT NULL DEFAULT 0.5,
    embedding BLOB,                       -- L2-normalized float32 vector
    created_at REAL NOT NULL,             -- Unix timestamp
    updated_at REAL NOT NULL,
    accessed_at REAL NOT NULL,            -- Last access time
    access_count INTEGER NOT NULL DEFAULT 0,
    retention_strength REAL NOT NULL DEFAULT 1.0,
    is_deleted INTEGER NOT NULL DEFAULT 0, -- Soft-delete flag
    deleted_at REAL,                       -- When soft-deleted
    metadata TEXT                          -- JSON metadata
);
```

Indexes:
- `idx_memories_content_hash` — exact dedup lookups
- `idx_memories_type` — type filtering
- `idx_memories_importance` — priority sorting
- `idx_memories_deleted` — active memory queries
- `idx_memories_created` — time range filtering

### `entities`

Named entities in the knowledge graph.

```sql
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT DEFAULT 'general',
    created_at REAL NOT NULL
);

CREATE INDEX idx_entities_name ON entities(name);
```

### `edges`

Directed relationships between entities.

```sql
CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    edge_type TEXT NOT NULL DEFAULT 'related',
    weight REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    FOREIGN KEY (source_id) REFERENCES entities(id),
    FOREIGN KEY (target_id) REFERENCES entities(id)
);

CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
```

### `memory_entities`

Many-to-many link between memories and entities.

```sql
CREATE TABLE memory_entities (
    memory_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY (memory_id, entity_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);
```

### `memory_links`

Direct links between memories (for related memory discovery).

```sql
CREATE TABLE memory_links (
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    link_type TEXT NOT NULL DEFAULT 'related',
    strength REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);
```

### `consolidations`

Tracks memory consolidation groups.

```sql
CREATE TABLE consolidations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_ids TEXT NOT NULL,              -- JSON array of memory IDs
    consolidated_content TEXT NOT NULL,
    consolidated_importance REAL NOT NULL,
    created_at REAL NOT NULL
);
```

### `access_log`

Records every memory access for retention computation.

```sql
CREATE TABLE access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    accessed_at REAL NOT NULL,
    query TEXT,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE INDEX idx_access_log_memory ON access_log(memory_id);
```

### `memories_fts` (FTS5 Virtual Table)

Full-text search index synced via triggers.

```sql
CREATE VIRTUAL TABLE memories_fts
USING fts5(
    content,
    content_rowid='id',
    tokenize='porter unicode61'
);
```

### FTS Sync Triggers

Three triggers keep the FTS index in sync with the `memories` table:

```sql
-- On INSERT: add to FTS
CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content)
    VALUES (new.id, new.content);
END;

-- On DELETE: remove from FTS
CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE rowid = old.id;
END;

-- On UPDATE: replace in FTS
CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
    DELETE FROM memories_fts WHERE rowid = old.id;
    INSERT INTO memories_fts(rowid, content)
    VALUES (new.id, new.content);
END;
```

## FAISS Index Strategy

Every index is wrapped in `IndexIDMap2` and keyed on the memory's own primary
key, so search returns ids directly and the mapping can't drift after deletes.
The underlying index is chosen by vector count:

| Mode | Underlying index | Switches to IVF when |
|------|------------------|----------------------|
| `flat_ip` | `IndexFlatIP` (exact) | never |
| `auto` (default) | `IndexFlatIP`, then `IndexIVFFlat` | `count ≥ ivf_threshold` (default 50,000) |
| `ivf_flat` | `IndexFlatIP`, then `IndexIVFFlat` | `count ≥ ivf_min_points` (default 1,000) |

### Staged upgrade (no untrained IVF)

An IVF index can't be added to until it's trained, and training needs enough
samples. So **all** modes start on `IndexFlatIP` and switch to IVF only once
there's enough data — `ivf_flat` no longer crashes on the first insert. The
switch is a full rebuild from the database (see below).

```python
from arriadne import AriadneConfig

config = AriadneConfig(
    faiss_type="auto",       # default
    ivf_threshold=50_000,    # auto: upgrade to IVF at this many vectors
    ivf_nlist=128,           # cells; effective nlist = min(ivf_nlist, sqrt(n))
)
```

### Rebuild from the database

On open — and whenever the index upgrades to IVF — Ariadne rebuilds the index
from the embeddings stored in the `memories` table:

1. Read `id, embedding` for every active (non-deleted) memory
2. Create the appropriate base index with `nlist = min(ivf_nlist, √n)`
3. Train it (IVF only) on those vectors
4. `add_with_ids(vectors, ids)` and wrap in `IndexIDMap2`

Because the database is the source of truth, soft-deleted vectors are pruned on
the next open and the index can never disagree with stored metadata.

## Concurrency & WAL Mode

Ariadne uses SQLite's **Write-Ahead Logging (WAL)** mode for concurrent read/write access:

```sql
PRAGMA journal_mode=WAL;
PRAGMA wal_autocheckpoint=1000;  -- Checkpoint every 1000 pages
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;        -- 5 second busy timeout
```

### WAL Benefits

- **Readers don't block writers** — concurrent reads during writes
- **Writers don't block readers** — concurrent writes during reads
- **Crash recovery** — WAL provides atomic transaction guarantees
- **Better performance** — sequential writes instead of random I/O

### Thread Safety

A single `AriadneDB` / `AriadneMemory` is safe to share across threads. The
SQLite connection is opened with `check_same_thread=False`, and every public
entry point is guarded by a reentrant lock (the SQLite + FAISS state in
`AriadneDB`, and the in-memory MinHash index in `AriadneMemory`). Operations are
serialized for correctness rather than run in parallel.

```python
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

# Safe to call concurrently from multiple threads
def worker():
    mem.remember("...")
    mem.recall("...")
```

## File Layout

```
arriadne.db          # SQLite database (metadata, FTS5, graph, embeddings)
arriadne.db-wal      # WAL log (SQLite)
arriadne.db-shm      # Shared memory (SQLite)
```

There is no separate FAISS index file: vectors live in the `.db` and the index
is rebuilt from them on open.

## Zero External Dependencies

Ariadne runs entirely locally with no external services:

| Component | Technology | Alternative |
|-----------|-----------|-------------|
| Vector search | FAISS (local) | No cloud API |
| Metadata | SQLite (local) | No PostgreSQL |
| FTS | SQLite FTS5 | No Elasticsearch |
| Graph | SQLite recursive CTEs | No Neo4j |
| Dedup | MinHash LSH (in-memory) | No external service |

