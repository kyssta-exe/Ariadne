# Architecture

Ariadne is a single-process, zero-daemon memory system built on **SQLite** (with WAL mode), **FAISS** (vector index), and **MinHash LSH** (deduplication). No external servers, no cloud dependencies.

## Storage Layers

```
┌─────────────────────────────────────────────────────────────┐
│                        AriadneDB                            │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │    FAISS     │  │    SQLite    │  │   MinHash LSH     │  │
│  │  Vector Index│  │  Metadata +  │  │   Dedup Index     │  │
│  │              │  │  FTS5 + Graph│  │   (in-memory)     │  │
│  │  cosine sim  │  │  WAL mode    │  │   ~1ms lookup     │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
│                                                             │
│  .faiss file      .db file            RAM only             │
└─────────────────────────────────────────────────────────────┘
```

| Layer | Technology | Purpose | Persistence |
|-------|-----------|---------|-------------|
| Vector Search | FAISS (IndexFlatIP / IndexIVFFlat) | Semantic similarity | `.faiss` file |
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

Ariadne automatically selects the optimal FAISS index based on vector count:

| Condition | Index Type | Algorithm | Use Case |
|-----------|-----------|-----------|----------|
| `< 1,000` vectors | `IndexFlatIP` | Brute-force inner product | Small datasets, exact search |
| `≥ 1,000` vectors | `IndexIVFFlat` | Inverted file index | Large datasets, approximate search |

### Auto-Upgrade

When `faiss_type="auto"` (default), Ariadne starts with `IndexFlatIP` and automatically upgrades to `IndexIVFFlat` when the vector count exceeds `ivf_threshold` (default: 1,000).

```python
from arriadne import AriadneConfig

config = AriadneConfig(
    faiss_type="auto",       # Default: auto-select
    ivf_threshold=1000,      # Upgrade to IVFFlat at 1K vectors
    ivf_nlist=256,           # Number of Voronoi cells
)
```

### Manual Index Selection

```python
config = AriadneConfig(
    faiss_type="flat_ip",    # Always use exact search
    # OR
    faiss_type="ivf_flat",   # Always use approximate search
)
```

### IVFFlat Training

When upgrading to IVFFlat, Ariadne:
1. Extracts all vectors from the FlatIP index
2. Creates a new IVFFlat index with `nlist = min(ivf_nlist, initial_size // 10)`
3. Trains the index on existing vectors
4. Re-adds all vectors to the new index

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

Ariadne's SQLite connection is not thread-safe by default. For multi-threaded applications:

```python
import sqlite3

# Each thread should create its own connection
conn = sqlite3.connect("arriadne.db")
conn.execute("PRAGMA journal_mode=WAL")
```

## File Layout

```
arriadne.db          # SQLite database (metadata, FTS5, graph)
arriadne.db.faiss    # FAISS vector index
arriadne.db-wal      # WAL log (SQLite)
arriadne.db-shm      # Shared memory (SQLite)
```

## Zero External Dependencies

Ariadne runs entirely locally with no external services:

| Component | Technology | Alternative |
|-----------|-----------|-------------|
| Vector search | FAISS (local) | No cloud API |
| Metadata | SQLite (local) | No PostgreSQL |
| FTS | SQLite FTS5 | No Elasticsearch |
| Graph | SQLite recursive CTEs | No Neo4j |
| Dedup | MinHash LSH (in-memory) | No external service |

