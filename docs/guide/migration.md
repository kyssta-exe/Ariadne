---
title: Migration & Import/Export
description: Import memories from other systems, export your data, and migrate seamlessly
---

# Migration & Import/Export

Ariadne provides comprehensive import/export tools to migrate from other memory systems or simply backup your data.

## Export

### Export to JSON

```python
from arriadne import AriadneMemory
from arriadne.migration import export_json

mem = AriadneMemory("my_memory.db")
export_json(mem, "backup.json")
```

### Export to Markdown

```python
from arriadne.migration import export_markdown

export_markdown(mem, "memories.md")
```

Output format:
```markdown
# Ariadne Memory Export
Generated: 2026-06-04 12:00:00
Total memories: 290

## Memory #1
- **ID**: 1
- **Category**: semantic
- **Created**: 2026-05-30
- **Importance**: 0.85
- **Entities**: Paris, France

Paris is the capital of France, a beautiful city...

---
```

## Import

### From JSON (Ariadne format)

```python
from arriadne.migration import import_json

count = import_json(mem, "backup.json")
print(f"Imported {count} memories")
```

### From Plain Text

Each paragraph becomes a separate memory:

```python
from arriadne.migration import import_from_text

count = import_from_text(mem, "notes.txt")
```

### From Markdown

Headers become entity associations:

```python
from arriadne.migration import import_from_markdown

count = import_from_markdown(mem, "knowledge_base.md")
```

### From ChromaDB

```python
from arriadne.migration import import_from_chromadb

count = import_from_chromadb(
    mem,
    "/path/to/chromadb/collection"
)
```

### From Mem0

```python
from arriadne.migration import import_from_mem0

count = import_from_mem0(mem, "mem0_export.json")
```

## CLI Commands

```bash
# Export all memories
ariadne export --format json --output backup.json
ariadne export --format markdown --output memories.md

# Import from various sources
ariadne import --format json --input backup.json
ariadne import --format text --input notes.txt
ariadne import --format markdown --input knowledge_base.md

# Migrate from other systems
ariadne migrate --source mnemosyne --input mnemosyne_export.json
```

## Dedup on Import

All import functions automatically check for duplicate memories. If a memory with similar content already exists, it's skipped. This prevents duplicate buildup when re-importing the same data.

## Backup Strategy

For production use, combine export with cron:

```bash
# Daily backup
ariadne export --format json --output /backups/ariadne_$(date +%Y%m%d).json
```
