"""Configuration for Ariadne memory system."""

from __future__ import annotations

import os
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
        trust_contradiction_penalty: Confidence subtracted from existing
            memories when a new write contradicts them (trust scoring).
        trust_reinforce_delta: Confidence added by ``reinforce()`` when an
            agent confirms a memory (trust scoring).
        session_digest_max_turns: Max episode turns included in a session digest.
        maintenance_interval: Writes between automatic maintenance cycles.
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
    trust_contradiction_penalty: float = 0.1  # confidence lost when contradicted by a new write
    trust_reinforce_delta: float = 0.1  # confidence gained per explicit reinforcement
    session_digest_max_turns: int = 12  # episode turns kept in a session digest
    maintenance_interval: int = 50  # writes between auto-maintenance cycles
    semantic_dedup: bool = True  # embedding-based paraphrase dedup (needs an embedder)
    semantic_dedup_threshold: float = 0.92  # cosine above which two memories are duplicates
    core_block_char_limit: int = 10000  # max characters per core memory block
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # lazy-loaded default
    # SQLite synchronous mode. NORMAL is the recommended WAL pairing: commits
    # skip the per-transaction fsync (checkpoints handle it), which removes an
    # fsync from the hot write path. The database stays crash-consistent; only
    # un-checkpointed tail transactions may be lost on power failure. Use
    # "FULL" to trade write speed for maximum durability.
    synchronous: str = "NORMAL"
    # SQLite page cache in MB (negative cache_size pragma). The default 2 MB
    # thrashes on memory-sized workloads; 64 MB measured 6x faster bulk writes.
    cache_mb: int = 64
    # Heavy auto-maintenance (consolidate + evict) runs every N writes, where
    # N = heavy_maintenance_factor * maintenance_interval. Light maintenance
    # (access-log pruning, purge) runs every maintenance_interval writes.
    # Consolidation cost grows with store size; without this factor, write
    # throughput decays as the store grows.
    heavy_maintenance_factor: int = 10

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
            raise ValueError(
                f"retention_half_life must be positive, got {self.retention_half_life}"
            )
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
        if not 0.0 <= self.trust_contradiction_penalty <= 1.0:
            raise ValueError(
                "trust_contradiction_penalty must be in [0, 1], "
                f"got {self.trust_contradiction_penalty}"
            )
        if not 0.0 <= self.trust_reinforce_delta <= 1.0:
            raise ValueError(
                f"trust_reinforce_delta must be in [0, 1], got {self.trust_reinforce_delta}"
            )
        if self.session_digest_max_turns < 1:
            raise ValueError(
                f"session_digest_max_turns must be >= 1, got {self.session_digest_max_turns}"
            )
        if self.maintenance_interval < 1:
            raise ValueError(
                f"maintenance_interval must be >= 1, got {self.maintenance_interval}"
            )
        if not 0.0 <= self.semantic_dedup_threshold <= 1.0:
            raise ValueError(
                "semantic_dedup_threshold must be in [0, 1], "
                f"got {self.semantic_dedup_threshold}"
            )
        if self.core_block_char_limit < 1:
            raise ValueError(
                f"core_block_char_limit must be >= 1, got {self.core_block_char_limit}"
            )
        if self.synchronous.upper() not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError(
                f"synchronous must be one of OFF/NORMAL/FULL/EXTRA, got {self.synchronous!r}"
            )
        if self.heavy_maintenance_factor < 1:
            raise ValueError(
                "heavy_maintenance_factor must be >= 1, "
                f"got {self.heavy_maintenance_factor}"
            )
        if self.cache_mb < 1:
            raise ValueError(f"cache_mb must be >= 1, got {self.cache_mb}")
        match self.faiss_type:
            case "auto" | "flat_ip" | "ivf_flat":
                pass
            case _:
                raise ValueError(f"Unknown faiss_type: {self.faiss_type!r}")
        self.db_path = Path(self.db_path)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> AriadneConfig:
        """Build a config from ``ARIADNE_*`` environment variables.

        Recognized variables (all optional; invalid values raise the same
        ``ValueError`` as the constructor, never a silent fallback):

        - ``ARIADNE_DB_PATH`` — database path
        - ``ARIADNE_EMBEDDING_DIM`` — embedding dimension
        - ``ARIADNE_FAISS_TYPE`` — auto | flat_ip | ivf_flat
        - ``ARIADNE_DEDUP_THRESHOLD`` — MinHash dedup threshold
        - ``ARIADNE_RETENTION_HALF_LIFE`` — Ebbinghaus half-life in seconds
        - ``ARIADNE_MAINTENANCE_INTERVAL`` — writes per auto-maintenance cycle
        - ``ARIADNE_TRUST_CONTRADICTION_PENALTY`` — trust decay per contradiction
        - ``ARIADNE_TRUST_REINFORCE_DELTA`` — trust gain per reinforcement
        """
        source = dict(os.environ) if env is None else env
        overrides: dict[str, object] = {}

        def _read_str(key: str, field_name: str) -> None:
            value = source.get(key)
            if value is not None and value.strip():
                overrides[field_name] = value.strip()

        def _read_number(key: str, field_name: str, cast: type) -> None:
            raw = source.get(key)
            if raw is None or not raw.strip():
                return
            try:
                overrides[field_name] = cast(raw.strip())
            except ValueError as exc:
                raise ValueError(f"{key} must be a valid {cast.__name__}: {raw!r}") from exc

        _read_str("ARIADNE_DB_PATH", "db_path")
        _read_number("ARIADNE_EMBEDDING_DIM", "embedding_dim", int)
        _read_str("ARIADNE_FAISS_TYPE", "faiss_type")
        _read_number("ARIADNE_DEDUP_THRESHOLD", "dedup_threshold", float)
        _read_number("ARIADNE_RETENTION_HALF_LIFE", "retention_half_life", float)
        _read_number("ARIADNE_MAINTENANCE_INTERVAL", "maintenance_interval", int)
        _read_number("ARIADNE_TRUST_CONTRADICTION_PENALTY", "trust_contradiction_penalty", float)
        _read_number("ARIADNE_TRUST_REINFORCE_DELTA", "trust_reinforce_delta", float)
        return cls(**overrides)  # type: ignore[arg-type]
