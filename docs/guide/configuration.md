# Configuration

## AriadneConfig

```python
from arriadne.config import AriadneConfig

config = AriadneConfig(
    db_path="~/.ariadne/memory.db",  # SQLite database path
    embedding_dim=384,                # Vector dimensions
    faiss_type="auto",                # auto|flat|ivf
    dedup_threshold=0.5,              # MinHash Jaccard threshold
    dedup_num_perm=128,               # MinHash permutations
    eviction_budget=100000,           # Max active memories
    soft_delete_days=7,               # Days before hard delete
    consolidation_threshold=0.85,     # Jaccard for consolidation
    consolidation_min_age_hours=24,   # Min age before consolidation
    priority_weights={                # Retention scoring weights
        "importance": 0.3,
        "recency": 0.25,
        "access_count": 0.25,
        "reference_count": 0.2,
    },
)
```

## Defaults

Most settings have sensible defaults. The only required parameter is `db_path`.

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_dim` | 384 | Matches all-MiniLM-L6-v2 |
| `faiss_type` | "auto" | FlatIP <1K vectors, IVFFlat ≥1K |
| `dedup_threshold` | 0.5 | Jaccard similarity for duplicate detection |
| `eviction_budget` | 100000 | Max memories before eviction kicks in |
| `soft_delete_days` | 7 | Grace period before hard delete |

## Environment Variables

```bash
ARIADNE_DB_PATH=/path/to/memory.db    # Override db_path
ARIADNE_LOG_LEVEL=DEBUG                # Set log level
ARIADNE_EMBEDDING_DIM=768              # Override embedding dimension
```
