---
title: "CLI Reference — Ariadne"
description: "Ariadne CLI: init, add, search, stats, migrate. Command-line interface for managing your memory database."
---


Ariadne provides a command-line interface for common operations without writing Python code.

## Installation

The CLI is installed automatically with the `ariadne-memory` package:

```bash
pip install ariadne-memory
```

Or run directly:

```bash
python -m arriadne.cli [command] [options]
```

## Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose (DEBUG) logging |
| `--db-path PATH` | Path to database file (default: `arriadne.db`) |

---

## `init`

Initialize a new Ariadne database.

```bash
ariadne init [--db-path PATH] [--dim DIMENSION]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dim` | `384` | Embedding dimension |

### Example

```bash
# Create a new database with default settings
ariadne init

# Create with custom path and dimension
ariadne init --db-path ~/.ariadne/memory.db --dim 768
```

### Output

```
Initialized Ariadne database at ariadne.db
  Embedding dimension: 384
  Total memories: 0
```

---

## `add`

Add a memory to the database.

```bash
ariadne add CONTENT [--type TYPE] [--importance SCORE] [--entities LIST] [--metadata JSON]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--type` | `semantic` | Memory type: `semantic`, `episodic`, `procedural` |
| `--importance` | `0.5` | Importance score (0.0–1.0) |
| `--entities` | — | Comma-separated entity names |
| `--metadata` | — | JSON metadata string |

### Examples

```bash
# Add a simple fact
ariadne add "User prefers dark mode" --type semantic --importance 0.8

# Add with entities
ariadne add "Deployed v2.3.1 to production" --type episodic --entities "production,v2.3.1"

# Add with metadata
ariadne add "Python 3.12 released" --type semantic --importance 0.6 --metadata '{"version": "3.12"}'

# Custom database path
ariadne --db-path ~/.ariadne/memory.db add "New memory"
```

### Output

```
Memory added: id=42
```

Or if a duplicate is detected:

```
Duplicate memory detected (similar to 1)
```

---

## `search`

Search memories by query.

```bash
ariadne search QUERY [-k NUM] [--type TYPE] [--min-importance SCORE]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-k` | `10` | Number of results to return |
| `--type` | — | Filter by memory type |
| `--min-importance` | — | Minimum importance threshold |

### Examples

```bash
# Basic search
ariadne search "dark mode"

# Limit results
ariadne search "server configuration" -k 5

# Filter by type
ariadne search "deployment" --type procedural

# Filter by importance
ariadne search "critical" --min-importance 0.8

# Combine filters
ariadne search "database" --type semantic --min-importance 0.7 -k 3
```

### Output

```
Found 3 results:

[1] (id=12, score=0.0324, type=semantic)
    User prefers dark mode in all applications
    importance=0.80 created=1748947200

[2] (id=45, score=0.0281, type=semantic)
    Dark mode reduces eye strain during night usage
    importance=0.60 created=1748860800

[3] (id=67, score=0.0195, type=episodic)
    Configured dark mode theme for the dashboard
    importance=0.50 created=1748774400
```

---

## `stats`

Show comprehensive database statistics.

```bash
ariadne stats [--db-path PATH]
```

### Example

```bash
ariadne stats
```

### Output

```
Ariadne Database Statistics
========================================
  Active memories:     1523
  Deleted memories:    47
  Total memories:      1570
  Total entities:      234
  Total edges:         512
  Total memory links:  89
  Consolidations:      12
  FAISS vectors:       1523
  FAISS type:          IndexFlatIP
  FAISS dimension:     384
  Avg importance:      0.6234
  Dedup index size:    1523
  DB size:             4.82 MB

  By type:
    episodic: 312
    procedural: 187
    semantic: 1024
```

---

## `migrate`

Import memories from a JSON export file.

```bash
ariadne migrate SOURCE [--db-path PATH]
```

### Options

| Option | Description |
|--------|-------------|
| `source` | Path to JSON export file |

### Input Formats

Ariadne supports two JSON formats:

**List format:**
```json
[
  {"question": "What is Python?", "answer": "A programming language", "importance": 7},
  {"content": "User prefers dark mode", "importance": 0.8}
]
```

**Object format:**
```json
{
  "cards": [
    {"question": "What is Python?", "answer": "A programming language"}
  ]
}
```

### Field Mapping

| Source Field | Ariadne Field | Notes |
|-------------|---------------|-------|
| `content` or `question` | `content` | Primary text |
| `answer` | `content` | Fallback if question empty |
| `importance` | `importance` | Divided by 10 if > 1.0 |
| `tags` | `entities` | String or list |

### Examples

```bash
# Migrate from Mnemosyne export
ariadne migrate ~/mnemosyne_export.json

# Migrate to specific database
ariadne migrate memories.json --db-path ~/.ariadne/memory.db
```

### Output

```
Migration complete: 847 imported, 12 skipped
```

---

## Examples

### Full Workflow

```bash
# 1. Initialize
ariadne init --db-path project_memory.db --dim 384

# 2. Add memories
ariadne add "Project uses PostgreSQL for main database" --type semantic --importance 0.9 --entities "PostgreSQL,project"
ariadne add "Deployed v2.3.1 to production on 2026-01-15" --type episodic --entities "production,v2.3.1"
ariadne add "Run make deploy-prod to deploy to production" --type procedural --importance 0.8

# 3. Search
ariadne search "how to deploy" --db-path project_memory.db
ariadne search "PostgreSQL" --type semantic --db-path project_memory.db

# 4. Check stats
ariadne stats --db-path project_memory.db

# 5. Migrate from existing data
ariadne migrate existing_notes.json --db-path project_memory.db
```

### Integration with Other Tools

```bash
# Pipe content from other commands
echo "Server has 4 cores and 8GB RAM" | xargs -I {} ariadne add "{}" --type semantic

# Search and format
ariadne search "server config" -k 3 2>/dev/null | grep "score=" | cut -d'|' -f2
```

