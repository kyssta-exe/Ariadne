# Migration

Ariadne supports importing memories from **Mnemosyne** (flashcard tool) and **custom JSON files**. You can migrate via the CLI or programmatically.

## From Mnemosyne (CLI)

Mnemosyne exports cards as JSON with `question`/`answer` fields.

### Export from Mnemosyne

In Mnemosyne, go to **File → Export** and select JSON format. Save the export file.

### Run Migration

```bash
ariadne migrate path/to/mnemosyne_export.json --db-path my_memory.db
```

### What Happens

1. Reads the JSON file (supports both list format and `{cards: [...]}` format)
2. For each card:
   - Uses `question` as content (falls back to `answer` if question is empty)
   - Normalizes `importance` from Mnemosyne's 0–10 scale to Ariadne's 0.0–1.0 scale
   - Maps `tags` to entity names
3. Skips duplicates (content hash + MinHash LSH)
4. Reports final counts

### Example

```bash
$ ariadne migrate ~/mnemosyne_export.json
Migration complete: 847 imported, 12 skipped
```

## From JSON (CLI)

Any JSON file with a list of objects containing text content works:

```json
[
  {"content": "PostgreSQL supports JSONB columns", "importance": 0.8},
  {"content": "User prefers dark mode", "importance": 0.9},
  {"content": "Deploy with make deploy-prod", "importance": 0.7}
]
```

```bash
ariadne migrate memories.json --db-path memory.db
```

The CLI looks for these fields in each object:
- `content` or `question` — the text to store
- `importance` — importance score (0.0–1.0)
- `tags` — entity names (string or list)

## From JSON (Programmatic)

### Basic Import

```python
import json
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

with open("memories.json") as f:
    data = json.load(f)

imported = 0
for item in data:
    content = item.get("content", "")
    if not content:
        continue
    
    result = mem.remember(
        content=content,
        memory_type=item.get("type", "semantic"),
        importance=item.get("importance", 0.5),
        entities=item.get("tags", None),
        metadata=item.get("metadata", None),
    )
    if result["status"] == "created":
        imported += 1

mem.close()
print(f"Imported {imported} memories")
```

### Import with Embeddings

```python
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from arriadne import AriadneMemory

model = SentenceTransformer("all-MiniLM-L6-v2")
mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

with open("memories.json") as f:
    data = json.load(f)

# Batch encode for efficiency
texts = [item["content"] for item in data if item.get("content")]
embeddings = model.encode(texts)

for item, embedding in zip(data, embeddings):
    content = item.get("content", "")
    if not content:
        continue
    
    mem.remember(
        content=content,
        memory_type=item.get("type", "semantic"),
        importance=item.get("importance", 0.5),
        embedding=embedding.tolist(),
        entities=item.get("tags"),
    )

mem.close()
```

### Import from CSV

```python
import csv
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

with open("memories.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        mem.remember(
            content=row["content"],
            memory_type=row.get("type", "semantic"),
            importance=float(row.get("importance", 0.5)),
            entities=row.get("tags", "").split(",") if row.get("tags") else None,
        )

mem.close()
```

## Migration Script

Complete migration script with progress reporting:

```python
#!/usr/bin/env python3
"""Migrate memories from various sources to Ariadne."""

import json
import sys
from pathlib import Path
from arriadne import AriadneMemory

def migrate_json(source_path: str, db_path: str):
    """Migrate from a JSON file."""
    source = Path(source_path)
    if not source.exists():
        print(f"File not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    with open(source) as f:
        data = json.load(f)

    # Handle both list and {cards: [...]} formats
    cards = data if isinstance(data, list) else data.get("cards", [])
    if not cards:
        print("No cards found in export file.", file=sys.stderr)
        sys.exit(1)

    mem = AriadneMemory(db_path=db_path, embedding_dim=384)
    
    imported = 0
    skipped = 0
    for i, card in enumerate(cards, 1):
        content = card.get("content") or card.get("question") or card.get("answer", "")
        if not content:
            skipped += 1
            continue

        importance = card.get("importance", 0.5)
        if isinstance(importance, (int, float)) and importance > 1.0:
            importance = min(1.0, importance / 10.0)  # Normalize Mnemosyne scale

        tags = card.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        result = mem.remember(
            content=content,
            memory_type=card.get("type", "semantic"),
            importance=importance,
            entities=tags if tags else None,
        )
        
        if result["status"] == "created":
            imported += 1
            if imported % 100 == 0:
                print(f"  Progress: {imported}/{len(cards)} imported...")
        else:
            skipped += 1

    mem.close()
    print(f"Migration complete: {imported} imported, {skipped} skipped")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python migrate.py <source.json> <target.db>")
        sys.exit(1)
    
    migrate_json(sys.argv[1], sys.argv[2])
```

### Usage

```bash
python migrate.py mnemosyne_export.json ariadne_memory.db
```

## Data Format Reference

### Mnemosyne Format

```json
[
  {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "importance": 7,
    "tags": ["geography", "france"]
  }
]
```

### Ariadne Native Format

```json
[
  {
    "content": "Paris is the capital of France",
    "type": "semantic",
    "importance": 0.9,
    "tags": ["geography", "france"],
    "metadata": {"category": "facts"}
  }
]
```

### Field Mapping

| Mnemosyne | Ariadne | Notes |
|-----------|---------|-------|
| `question` | `content` | Primary text |
| `answer` | `content` | Fallback if question empty |
| `importance` | `importance` | Divided by 10 (0–10 → 0.0–1.0) |
| `tags` | `entities` | String or list |

