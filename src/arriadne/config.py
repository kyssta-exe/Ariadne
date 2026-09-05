"""Configuration for Ariadne memory system.

Configs can be built in Python, loaded from the environment (``ARIADNE_*``
variables via :meth:`AriadneConfig.from_env`), or loaded from a TOML file via
:meth:`AriadneConfig.from_toml`. The environment layer means a deployment only
needs ``ARIADNE_DB_PATH=/data/agent.db`` instead of code changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any


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
        max_memories: Soft capacity bound. When the store grows past this many
            active memories, ``evict()`` removes the lowest-priority overflow.
            ``None`` (default) disables implicit eviction entirely — data is
            never destroyed without asking; use ``curator.decay()`` or a
            capacity to bound the store explicitly.
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
    max_memories: int | None = None  # soft capacity; None = never evict implicitly

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
        if self.max_memories is not None and self.max_memories < 1:
            raise ValueError(f"max_memories must be >= 1 or None, got {self.max_memories}")
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

    # ------------------------------------------------------------------
    # Serialization / environment loading
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain JSON-safe dict."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = str(value) if isinstance(value, Path) else value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base: AriadneConfig | None = None) -> AriadneConfig:
        """Build a config from a dict, ignoring unknown keys.

        Unknown keys are ignored (with no error) so config files can carry
        extra sections for other tools. Values are type-coerced from the
        declared field annotations, so ``"384"`` works for an int field.
        """
        hints = cls._type_hints()
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in hints:
                continue
            kwargs[key] = cls._coerce(key, value, hints[key])
        if base is not None:
            merged = base.to_dict()
            merged.update(kwargs)
            return cls(**merged)
        return cls(**kwargs)

    @classmethod
    @lru_cache(maxsize=1)
    def _type_hints(cls) -> dict[str, Any]:
        """Resolved field annotations (cached; dataclass stores strings)."""
        import dataclasses
        import typing

        return {
            f.name: hint
            for f in dataclasses.fields(cls)
            for hint in [typing.get_type_hints(cls).get(f.name, Any)]
        }

    @staticmethod
    def _coerce(key: str, value: Any, hint: Any) -> Any:
        """Coerce an external value (env/TOML/JSON) to the field's type."""
        import types as _types
        import typing as _t

        if value is None:
            return None
        text = str(value).strip()
        none_like = text.lower() in {"none", "null", ""}
        # Unwrap Optional[X] / X | None to the concrete type. PEP 604 unions
        # (`int | None`) have origin types.UnionType, typing.Optional has
        # origin typing.Union — both must be handled.
        origin = _t.get_origin(hint)
        if origin is _t.Union or origin is _types.UnionType:
            args = [a for a in _t.get_args(hint) if a is not type(None)]
            if none_like:
                return None
            if args:
                hint = args[0]
                origin = _t.get_origin(hint)
        if hint is bool:
            if isinstance(value, bool):
                return value
            return text.lower() in {"1", "true", "yes", "on"}
        if origin is dict or hint is dict:
            if isinstance(value, dict):
                return value
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                raise ValueError(f"{key} must be a JSON object, got {value!r}") from None
        if hint is int:
            if none_like:
                return None
            return int(float(text))
        if hint is float:
            if none_like:
                return None
            return float(text)
        if hint is Path:
            return Path(text)
        return text

    @classmethod
    def from_env(
        cls,
        prefix: str = "ARIADNE",
        *,
        base: AriadneConfig | None = None,
        environ: dict[str, str] | None = None,
    ) -> AriadneConfig:
        """Build a config from ``<PREFIX>_<FIELD>`` environment variables.

        Every dataclass field can be set, e.g. ``ARIADNE_DB_PATH``,
        ``ARIADNE_EMBEDDING_DIM``, ``ARIADNE_MAX_MEMORIES``,
        ``ARIADNE_RETENTION_HALF_LIFE``. Values are coerced to the field
        type; unset variables fall back to ``base`` (or the defaults).
        ``ARIADNE_PRIORITY_WEIGHTS`` accepts a JSON object.
        """
        env = os.environ if environ is None else environ
        base = base or cls()
        hints = cls._type_hints()
        overrides: dict[str, Any] = {}
        for name, hint in hints.items():
            key = f"{prefix}_{name}".upper()
            if key not in env:
                continue
            raw = env[key]
            if name == "priority_weights":
                try:
                    overrides[name] = json.loads(raw)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"{prefix}_PRIORITY_WEIGHTS must be a JSON object, got {raw!r}"
                    ) from None
            else:
                overrides[name] = cls._coerce(name, raw, hint)
        merged = base.to_dict()
        merged.update(overrides)
        return cls(**merged)

    @classmethod
    def from_toml(cls, path: str | Path, *, base: AriadneConfig | None = None) -> AriadneConfig:
        """Build a config from a TOML file.

        Top-level keys map to fields; keys under an optional ``[ariadne]``
        table are also accepted. Unknown keys are ignored, so the same file
        can host other tools' settings.
        """
        try:
            import tomllib  # Python 3.11+
        except ImportError:  # pragma: no cover - Python 3.10
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError as exc:
                raise ImportError(
                    "TOML config on Python 3.10 requires 'tomli'. "
                    "Install it with: pip install tomli"
                ) from exc
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("rb") as fh:
            raw = tomllib.load(fh)
        # Top-level scalars map to fields; an optional [ariadne] table also
        # maps to fields and wins on collision. from_dict drops unknown keys,
        # so other tools' tables in the same file are harmless.
        data = dict(raw)
        data.pop("ariadne", None)
        data.update(raw.get("ariadne", {}))
        return cls.from_dict(data, base=base)
