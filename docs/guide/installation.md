---
title: "Installation — Ariadne"
description: "Install Ariadne with pip install ariadne. No system packages, no Docker, no external services. Python 3.10+ required."
---


## From PyPI

```bash
pip install ariadne
```

That's the only step. `faiss-cpu`, `numpy`, and `datasketch` install automatically as dependencies. No system packages, no Docker, no external services.

### Verify

```bash
ariadne --help
```

If the CLI works, you're ready.

## From Source

```bash
git clone https://github.com/kyssta-exe/Ariadne.git
cd Ariadne
pip install -e .
```

### Development Install

```bash
pip install -e ".[dev]"  # Includes pytest, mypy, ruff
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `faiss-cpu` | ≥ 1.7.4 | Vector similarity search (FlatIP, IVFFlat) |
| `numpy` | ≥ 1.24.0 | Embedding array operations |
| `datasketch` | ≥ 1.5.0 | MinHash LSH for deduplication |

All are pure Python wheels with precompiled C extensions — no build tools required.

## Optional: Embeddings

Ariadne is model-agnostic — any embedding model works. For convenience, the most common pairing:

```bash
pip install sentence-transformers
```

Then:

```python
from sentence_transformers import SentenceTransformer
from arriadne import AriadneMemory

model = SentenceTransformer("all-MiniLM-L6-v2")
mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

text = "User prefers dark mode"
embedding = model.encode(text).tolist()
mem.remember(content=text, embedding=embedding, importance=0.8)
```

See the [Embeddings Guide](/guide/embeddings) for model selection, Matryoshka embeddings, and quantization.

## Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.10+ |
| RAM | ~45 MB (for 10K memories) |
| Disk | ~11 MB (for 10K memories with FAISS index) |
| OS | Linux, macOS, Windows (tested on Linux) |

No GPU required. FAISS CPU is used by default and is fast enough for 100K+ vectors.

## Supported Platforms

| Platform | Status |
|----------|--------|
| Linux (x86_64) | ✅ Tested |
| macOS (Apple Silicon) | ✅ Should work (faiss-cpu has wheels) |
| macOS (Intel) | ✅ Should work |
| Windows | ⚠️ faiss-cpu may need Visual C++ Redistributable |

## Where Files Live

```
~/.ariadne/
  └── memory.db          # SQLite database (memories, entities, edges, FTS5)
  └── memory.db.faiss    # FAISS vector index
  └── memory.db-wal      # SQLite WAL log (auto-cleaned)
  └── memory.db-shm      # SQLite shared memory (auto-cleaned)
```

You can back up, rsync, or commit `memory.db` and `memory.db.faiss` anywhere. The WAL and SHM files are temporary and regenerated on open.
