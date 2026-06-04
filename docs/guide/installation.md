# Installation

## From PyPI

```bash
pip install arriadne
```

## From Source

```bash
git clone https://github.com/kyssta-exe/Ariadne.git
cd Ariadne
pip install -e .
```

## Dependencies

- **faiss-cpu** — Vector similarity search (no GPU required)
- **numpy** — Numerical operations
- **datasketch** — MinHash LSH for deduplication

All dependencies install automatically. No system packages needed.

## Verify Installation

```bash
ariadne --help
ariadne init --db-path ~/.ariadne/memory.db
```

## Requirements

- Python 3.10+
- ~80MB RAM for 100K memories
- ~50MB disk for 100K memories with embeddings
