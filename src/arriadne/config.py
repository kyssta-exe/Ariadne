"""Configuration for Ariadne memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AriadneConfig:
    """Configuration options for Ariadne memory system.

    Attributes:
        db_path: Path to SQLite database file.
        embedding_dim: Dimension of embedding vectors.
        faiss_type: FAISS index type - "auto", "flat_ip", or "ivf_flat".
        ivf_threshold: Vector count at which to auto-upgrade to IVFFlat.
        ivf_nlist: Number of Voronoi cells for IVFFlat index.
        dedup_threshold: MinHash similarity threshold for deduplication (0.0-1.0).
        dedup_num_perm: Number of MinHash permutations.
        consolidation_threshold: Jaccard similarity for memory consolidation.
        consolidation_min_group: Minimum group size for consolidation.
        eviction_budget: Maximum fraction of memories to evict per run.
        retention_half_life: Ebbinghaus half-life in seconds.
        priority_weights: Weights for priority scoring components.
        max_graph_depth: Maximum depth for graph traversal.
        batch_size: Batch size for bulk operations.
        wal_autocheckpoint: SQLite WAL auto-checkpoint interval in pages.
        fts_tokenizer: FTS5 tokenizer configuration.
    """

    db_path: str | Path = "arriadne.db"
    embedding_dim: int = 384
    faiss_type: str = "auto"
    ivf_threshold: int = 50000
    ivf_nlist: int = 128
    dedup_threshold: float = 0.8
    dedup_num_perm: int = 128
    consolidation_threshold: float = 0.7
    consolidation_min_group: int = 2
    eviction_budget: float = 0.1
    retention_half_life: float = 86400.0  # 1 day in seconds
    retention_growth_factor: float = 1.5  # stability multiplier applied per access
    retention_strength_cap: float = 100.0  # ceiling for accrued retention strength
    priority_weights: dict[str, float] = field(default_factory=lambda: {
        "importance": 0.4,
        "recency": 0.3,
        "access_count": 0.2,
        "retention": 0.1,
    })
    max_graph_depth: int = 10
    batch_size: int = 1000
    wal_autocheckpoint: int = 1000
    fts_tokenizer: str = "porter unicode61"
    ivf_min_points: int = 1000  # min vectors before an explicit ivf_flat index trains
    max_access_log_per_memory: int = 50  # access_log rows kept per memory after pruning
    purge_retention_seconds: float = 604800.0  # soft-deleted rows kept recoverable for 7 days

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.embedding_dim < 1:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if not 0.0 <= self.dedup_threshold <= 1.0:
            raise ValueError(f"dedup_threshold must be in [0, 1], got {self.dedup_threshold}")
        if not 0.0 <= self.consolidation_threshold <= 1.0:
            raise ValueError(
                f"consolidation_threshold must be in [0, 1], got {self.consolidation_threshold}"
            )
        if not 0.0 < self.eviction_budget <= 1.0:
            raise ValueError(f"eviction_budget must be in (0, 1], got {self.eviction_budget}")
        if self.retention_half_life <= 0:
            raise ValueError(f"retention_half_life must be positive, got {self.retention_half_life}")
        if self.retention_growth_factor < 1.0:
            raise ValueError(
                f"retention_growth_factor must be >= 1.0, got {self.retention_growth_factor}"
            )
        if self.retention_strength_cap < 1.0:
            raise ValueError(
                f"retention_strength_cap must be >= 1.0, got {self.retention_strength_cap}"
            )
        if self.ivf_min_points < 1:
            raise ValueError(f"ivf_min_points must be >= 1, got {self.ivf_min_points}")
        if self.purge_retention_seconds < 0:
            raise ValueError(
                f"purge_retention_seconds must be >= 0, got {self.purge_retention_seconds}"
            )
        if self.max_access_log_per_memory < 1:
            raise ValueError(
                f"max_access_log_per_memory must be >= 1, got {self.max_access_log_per_memory}"
            )
        match self.faiss_type:
            case "auto" | "flat_ip" | "ivf_flat":
                pass
            case _:
                raise ValueError(f"Unknown faiss_type: {self.faiss_type!r}")
        self.db_path = Path(self.db_path)
