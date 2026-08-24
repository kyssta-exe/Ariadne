"""Core storage layer for Ariadne memory system.

Provides AriadneDB with SQLite (WAL, FTS5, graph tables) and FAISS vector index.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import faiss
import numpy as np

from arriadne.config import AriadneConfig

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])


def _synchronized(method: _F) -> _F:
    """Serialize a method on the instance's reentrant ``_lock``.

    Guards every public SQLite + FAISS entry point so a single AriadneDB can be
    shared across threads. The lock is reentrant, so synchronized methods may
    freely call one another (e.g. search -> get_memory).
    """

    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return cast("_F", wrapper)


class _SupportsNumpy(Protocol):
    def __array__(self) -> np.ndarray: ...


Schema = sqlite3.Connection


def _now() -> float:
    """Return current Unix timestamp."""
    return time.time()


def _hash_content(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _clamp01(value: float) -> float:
    """Clamp a value into the documented [0.0, 1.0] range."""
    return max(0.0, min(1.0, float(value)))


def _memory_confidence(memory: dict[str, Any] | None) -> float:
    """Return a safe confidence value for legacy or hand-edited rows."""
    value = memory.get("confidence", 1.0) if memory else 1.0
    try:
        return _clamp01(1.0 if value is None else float(value))
    except (TypeError, ValueError):
        return 0.0


# Pre-compiled regex for FTS5 query parsing (Optimization: avoids re-compilation on every call)
_FTS_WORD_RE = re.compile(r"\w+")


def _fts_escape(query: str, op: str = "OR") -> str:
    """Escape FTS5 special characters and join terms with ``op`` (OR/AND).

    Each word is quoted individually to handle special characters. ``fts_search``
    uses AND first (precise) and falls back to OR (recall) when AND finds
    nothing, so broad and narrow queries both behave sensibly.

    Optimization: Uses pre-compiled _FTS_WORD_RE pattern instead of
    re.findall(r"\\w+", query) to avoid re-compilation on every search call.
    """
    words = _FTS_WORD_RE.findall(query)
    if not words:
        return '""'
    escaped = [f'"{word.replace(chr(34), chr(34) * 2)}"' for word in words]
    joiner = " AND " if op.upper() == "AND" else " OR "
    return joiner.join(escaped)


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union) if union else 0.0


class AriadneDB:
    """Core memory database with SQLite storage and FAISS vector indexing.

    Supports vector search, full-text search, hybrid search, graph operations,
    memory lifecycle management, and consolidation.

    Args:
        config: AriadneConfig instance with all settings.

    Example:
        >>> db = AriadneDB(config=AriadneConfig(db_path="test.db"))
        >>> with db:
        ...     db.add_memory("hello", np.zeros(384), memory_type="semantic")
        ...     results = db.vector_search(np.zeros(384), k=5)
    """

    def __init__(self, config: AriadneConfig | None = None) -> None:
        self._config = config or AriadneConfig()
        self._conn: sqlite3.Connection | None = None
        # FAISS index is an IndexIDMap2 keyed on each memory's own primary key,
        # so vector positions can never drift out of sync with the database.
        self._faiss_index: faiss.Index | None = None
        self._initialized = False
        # Reentrant lock guarding all SQLite + FAISS state. The connection is
        # opened with check_same_thread=False so a single AriadneDB can be shared
        # across threads (the common case for an agent serving concurrent turns).
        self._lock = threading.RLock()
        # WAL checkpoint tracking (Optimization: prevents WAL file from growing unbounded)
        self._write_count: int = 0
        self._wal_checkpoint_interval: int = 1000  # Checkpoint every N writes

    @property
    def config(self) -> AriadneConfig:
        """Return the configuration."""
        return self._config

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the database connection, opening if needed."""
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    def _faiss_sidecar_path(self) -> Path | None:
        """Path of the serialized FAISS index, or None for in-memory DBs."""
        db_path = str(self._config.db_path)
        if db_path in (":memory:", ""):
            return None
        return Path(db_path + ".faiss")

    def _vector_fingerprint(self) -> str:
        """Cheap state fingerprint of the vector set stored in SQLite.

        ``"{count}:{max id}"`` over vector-bearing active rows. Any realistic
        add/remove combination changes at least one of the two, so a sidecar
        index whose embedded fingerprint disagrees with the database is stale
        and gets rebuilt instead of trusted.
        """
        assert self._conn is not None
        row = self._conn.execute(
            """SELECT COUNT(*), COALESCE(MAX(id), 0) FROM memories
               WHERE embedding IS NOT NULL AND is_deleted = 0"""
        ).fetchone()
        return f"{int(row[0])}:{int(row[1])}"

    def _load_faiss_index(self) -> None:
        """Load the FAISS index: sidecar file first, rebuild as fallback.

        The ``memories`` table remains the source of truth — the sidecar is
        only trusted when the fingerprint stored next to it (``<db>.faiss.fp``)
        matches the database's current vector set, so external edits, crashes
        between write and persist, or dimension changes all fall back to the
        (correct, slower) rebuild.
        """
        sidecar = self._faiss_sidecar_path()
        fingerprint = self._vector_fingerprint()
        if sidecar is not None and sidecar.exists():
            try:
                fp_path = sidecar.parent / (sidecar.name + ".fp")
                stored_fp = (
                    fp_path.read_text(encoding="utf-8").strip() if fp_path.exists() else ""
                )
                candidate = faiss.read_index(str(sidecar))
                if stored_fp == fingerprint and candidate.d == self._config.embedding_dim:
                    self._faiss_index = candidate
                    logger.info(
                        "Loaded FAISS index from sidecar (%d vectors)", candidate.ntotal
                    )
                    return
                logger.info("FAISS sidecar stale (fingerprint mismatch); rebuilding")
            except Exception as e:
                logger.warning("FAISS sidecar unreadable (%s); rebuilding", e)
        self._faiss_index = self._build_faiss_from_db()

    def _persist_faiss(self) -> None:
        """Serialize the FAISS index (and its fingerprint) to the sidecar.

        Called on close (and after full rebuilds) so the next open skips the
        O(n) rebuild. Failures are logged and ignored — persistence is an
        optimization, never a correctness requirement.
        """
        sidecar = self._faiss_sidecar_path()
        if sidecar is None or self._faiss_index is None:
            return
        try:
            faiss.write_index(self._faiss_index, str(sidecar))
            fp_path = sidecar.parent / (sidecar.name + ".fp")
            fp_path.write_text(self._vector_fingerprint(), encoding="utf-8")
        except Exception as e:  # pragma: no cover - depends on filesystem state
            logger.warning("Could not persist FAISS sidecar: %s", e)

    def open(self) -> None:
        """Open database connection and initialize schema + FAISS index."""
        if self._initialized:
            return
        db_path = str(self._config.db_path)
        logger.info("Opening database at %s", db_path)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL is the recommended WAL pairing: commits no longer fsync the
        # WAL on every transaction (the checkpoint does), which removes an
        # fsync from the hot write path. The database stays consistent under
        # crashes; only the very last un-checkpointed transactions may be lost
        # on a power failure.
        self._conn.execute(f"PRAGMA synchronous={self._config.synchronous}")
        # Page cache: the default ~2 MB thrashes once embeddings (~1.5 KB per
        # row), 19 secondary indexes, and FTS5 compete — measured 4.3 -> 0.7
        # ms/write on a 5k-memory ingest by raising it to 64 MB.
        self._conn.execute(f"PRAGMA cache_size=-{int(self._config.cache_mb) * 1024}")
        self._conn.execute(f"PRAGMA wal_autocheckpoint={self._config.wal_autocheckpoint}")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._create_schema()
        self._load_faiss_index()
        self._initialized = True

    def close(self) -> None:
        """Close the database connection.

        Every embedding still lives in the ``memories`` table as a normalized
        BLOB — the database alone is always sufficient to reconstruct
        everything. As a startup optimization the FAISS index is additionally
        serialized to a ``<db>.faiss`` sidecar (validated by fingerprint on
        the next open) so reopening a large store does not pay the O(n)
        rebuild.
        """
        if self._conn is not None:
            # Final WAL checkpoint before close (Optimization: flush pending WAL writes)
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
        self._persist_faiss()
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._initialized = False
        logger.info("Database closed")

    def _checkpoint_if_needed(self) -> None:
        """Periodically checkpoint WAL file to prevent unbounded growth.

        Optimization: After N writes (default 1000), calls PRAGMA wal_checkpoint(PASSIVE)
        which checkpoints without blocking readers/writers. This prevents the WAL
        file from growing unbounded while avoiding the performance hit of full
        checkpoints on every write.

        SQLite's wal_autocheckpoint handles automatic checkpointing based on WAL
        page count, but manual checkpoints provide finer-grained control for
        write-heavy workloads (like bulk memory ingestion).
        """
        self._write_count += 1
        if self._write_count >= self._wal_checkpoint_interval:
            assert self._conn is not None
            try:
                self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                logger.debug("WAL checkpoint after %d writes", self._write_count)
            except Exception as e:
                logger.debug("WAL checkpoint skipped: %s", e)
            self._write_count = 0

    def _commit(self) -> None:
        """Commit the current transaction and trigger WAL checkpoint if needed.

        Optimization: Wraps conn.commit() with periodic WAL checkpointing to
        prevent the WAL file from growing unbounded during write-heavy workloads.
        """
        assert self._conn is not None
        self._conn.commit()
        self._checkpoint_if_needed()

    def __enter__(self) -> AriadneDB:
        self.open()
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def _create_schema(self) -> None:
        """Create all SQLite tables, indexes, and FTS5 virtual tables."""
        assert self._conn is not None
        cursor = self._conn.cursor()

        # Migrate the base table before creating indexes that reference newer
        # columns. SQLite validates an index immediately, so doing this after
        # the CREATE INDEX block makes old databases unopenable.
        cursor.execute("PRAGMA table_info(memories)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if existing_cols:
            migrations = {
                "tags": "ALTER TABLE memories ADD COLUMN tags TEXT DEFAULT '[]'",
                "namespace": (
                    "ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
                ),
                "scope": "ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'session'",
                "user_id": "ALTER TABLE memories ADD COLUMN user_id TEXT",
                "agent_id": "ALTER TABLE memories ADD COLUMN agent_id TEXT",
                "session_id": "ALTER TABLE memories ADD COLUMN session_id TEXT",
                "project_id": "ALTER TABLE memories ADD COLUMN project_id TEXT",
                "event_at": "ALTER TABLE memories ADD COLUMN event_at REAL",
                "valid_from": "ALTER TABLE memories ADD COLUMN valid_from REAL",
                "valid_to": "ALTER TABLE memories ADD COLUMN valid_to REAL",
                "supersedes_id": "ALTER TABLE memories ADD COLUMN supersedes_id INTEGER",
                "confidence": "ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 1.0",
            }
            for col, ddl in migrations.items():
                if col not in existing_cols:
                    cursor.execute(ddl)
                    logger.info("Added %r column to memories table (migration)", col)

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'semantic',
                importance REAL NOT NULL DEFAULT 0.5,
                embedding BLOB,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                retention_strength REAL NOT NULL DEFAULT 1.0,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                deleted_at REAL,
                metadata TEXT,
                tags TEXT DEFAULT '[]',
                namespace TEXT NOT NULL DEFAULT 'default',
                scope TEXT NOT NULL DEFAULT 'session',
                user_id TEXT,
                agent_id TEXT,
                session_id TEXT,
                project_id TEXT,
                event_at REAL,
                valid_from REAL,
                valid_to REAL,
                supersedes_id INTEGER,
                confidence REAL DEFAULT 1.0
            );

            CREATE INDEX IF NOT EXISTS idx_memories_content_hash
                ON memories(content_hash);
            CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_memories_deleted
                ON memories(is_deleted);
            CREATE INDEX IF NOT EXISTS idx_memories_created
                ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_namespace
                ON memories(namespace);
            CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories(scope);
            CREATE INDEX IF NOT EXISTS idx_memories_user
                ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_agent
                ON memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_memories_session
                ON memories(session_id);
            CREATE INDEX IF NOT EXISTS idx_memories_project
                ON memories(project_id);
            CREATE INDEX IF NOT EXISTS idx_memories_valid_from
                ON memories(valid_from);
            CREATE INDEX IF NOT EXISTS idx_memories_valid_to
                ON memories(valid_to);
            CREATE INDEX IF NOT EXISTS idx_memories_supersedes
                ON memories(supersedes_id);
            CREATE INDEX IF NOT EXISTS idx_memories_event_at
                ON memories(event_at);
            CREATE INDEX IF NOT EXISTS idx_memories_confidence
                ON memories(confidence);

            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                entity_type TEXT DEFAULT 'general',
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_entities_name
                ON entities(name);

            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'related',
                weight REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source
                ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target
                ON edges(target_id);

            -- Deduplicate identical edges so repeated add_edge() calls for the
            -- same (source, target, type) don't accumulate rows. A unique index
            -- is used instead of a table constraint because SQLite can't ALTER
            -- TABLE ADD CONSTRAINT; this achieves the same effect on new and
            -- existing databases. INSERT OR IGNORE in add_edge() honours it.
            CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_uniq
                ON edges(source_id, target_id, edge_type);

            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL,
                PRIMARY KEY (memory_id, entity_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id),
                FOREIGN KEY (entity_id) REFERENCES entities(id)
            );

            CREATE TABLE IF NOT EXISTS memory_links (
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                link_type TEXT NOT NULL DEFAULT 'related',
                strength REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES memories(id),
                FOREIGN KEY (target_id) REFERENCES memories(id)
            );

            CREATE TABLE IF NOT EXISTS consolidations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_ids TEXT NOT NULL,
                consolidated_content TEXT NOT NULL,
                consolidated_importance REAL NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                accessed_at REAL NOT NULL,
                query TEXT,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );

            CREATE INDEX IF NOT EXISTS idx_access_log_memory
                ON access_log(memory_id);
        """)

        # FTS5 virtual table
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(
                content,
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """)

        # FTS sync triggers - use DELETE for proper FTS5 cleanup
        cursor.executescript("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content)
                VALUES (new.id, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                DELETE FROM memories_fts WHERE rowid = old.id;
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                DELETE FROM memories_fts WHERE rowid = old.id;
                INSERT INTO memories_fts(rowid, content)
                VALUES (new.id, new.content);
            END;

            -- Optimization: Auto-log memory access via trigger instead of
            -- separate INSERT in get_memory(). Reduces _touch_memory from
            -- 2 writes (UPDATE + INSERT) to 1 write (UPDATE) by having
            -- the trigger handle the access_log INSERT automatically.
            -- Fires only when access_count is incremented (i.e., during
            -- get_memory/touch, not during regular update_memory calls).
            CREATE TRIGGER IF NOT EXISTS memories_access_log AFTER UPDATE ON memories
            WHEN NEW.access_count > OLD.access_count
            BEGIN
                INSERT INTO access_log (memory_id, accessed_at)
                VALUES (NEW.id, NEW.accessed_at);
            END;
        """)

        # Episode table: immutable raw evidence
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                role TEXT NOT NULL,
                source TEXT,
                event_at REAL NOT NULL,
                metadata TEXT,
                namespace TEXT NOT NULL DEFAULT 'default',
                scope TEXT NOT NULL DEFAULT 'session',
                user_id TEXT,
                agent_id TEXT,
                session_id TEXT,
                project_id TEXT,
                created_at REAL NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_event_at ON episodes(event_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_namespace ON episodes(namespace)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id)")

        # Episode full-text index: lets agents search raw history the way
        # ctx-style tools do ("what did I try last week?") without loading the
        # whole episode table. External-content FTS5 kept in sync by triggers.
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
            USING fts5(
                content,
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """)
        cursor.executescript("""
            CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                INSERT INTO episodes_fts(rowid, content)
                VALUES (new.id, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
                DELETE FROM episodes_fts WHERE rowid = old.id;
            END;
        """)
        # Backfill legacy episodes into a freshly-created FTS table.
        cursor.execute(
            """INSERT INTO episodes_fts(rowid, content)
               SELECT e.id, e.content
               FROM episodes AS e
               WHERE NOT EXISTS (
                   SELECT 1 FROM episodes_fts AS f WHERE f.rowid = e.id
               )"""
        )

        # Source table: provenance records attached to memories
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                episode_id INTEGER,
                source TEXT NOT NULL,
                source_id TEXT,
                span TEXT,
                confidence REAL DEFAULT 1.0,
                created_at REAL NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id),
                FOREIGN KEY (episode_id) REFERENCES episodes(id)
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sources_memory ON sources(memory_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sources_episode ON sources(episode_id)")

        # Feedback table: user/agent approval, rejection, correction
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                confidence_delta REAL DEFAULT 0.0,
                note TEXT,
                actor TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_memory ON feedback(memory_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_action ON feedback(action)")

        # Core memory blocks (Letta-style working memory): small named blocks
        # that are ALWAYS loaded into context — persona, human profile,
        # project state. Kept separate from `memories` on purpose: they are
        # mutable working state, not searchable facts, so dedup/eviction/
        # retention must never touch them.
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS core_memory_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                namespace TEXT NOT NULL DEFAULT 'default',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE (name, namespace)
            );
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_core_blocks_ns ON core_memory_blocks(namespace)"
        )

        # A newly-created standalone FTS5 table has no rows for legacy
        # memories. Populate missing rowids so opening an old database does
        # not silently destroy keyword recall.
        cursor.execute(
            """INSERT INTO memories_fts(rowid, content)
               SELECT m.id, m.content
               FROM memories AS m
               WHERE NOT EXISTS (
                   SELECT 1 FROM memories_fts AS f WHERE f.rowid = m.id
               )"""
        )
        self._commit()
        logger.debug("Schema created/verified")

    def _ivf_plan(self, n: int) -> int:
        """Return the IVF ``nlist`` to use for ``n`` vectors, or 0 to stay flat.

        An IVF index can only be added to *after* it is trained, and training
        needs enough samples. So for every mode we keep a plain FlatIP index
        until there is enough data, then switch to IVF (via a full rebuild that
        trains on all current vectors). This is why ``faiss_type='ivf_flat'`` no
        longer crashes on the first insert — it simply stages through FlatIP.
        """
        if self._config.faiss_type == "flat_ip":
            return 0
        if self._config.faiss_type == "auto":
            threshold = self._config.ivf_threshold
        else:  # ivf_flat — switch as soon as we can train a usable index
            threshold = self._config.ivf_min_points
        if n < threshold:
            return 0
        # nlist scales with sqrt(n) and is always <= n, so training is valid.
        return min(self._config.ivf_nlist, max(1, int(n**0.5)))

    def _create_base_index(self, initial_size: int) -> faiss.Index:
        """Create the underlying (un-wrapped) FAISS index based on config and size."""
        dim = self._config.embedding_dim
        if self._config.faiss_type not in ("flat_ip", "ivf_flat", "auto"):
            raise ValueError(f"Unknown faiss_type: {self._config.faiss_type!r}")
        nlist = self._ivf_plan(initial_size)
        if nlist <= 0:
            return faiss.IndexFlatIP(dim)
        quantizer = faiss.IndexFlatIP(dim)
        return faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    def _build_faiss_from_db(self) -> faiss.Index:
        """Construct a fresh IndexIDMap2 from all active embeddings in the DB.

        Vectors are keyed by the memory's own primary key. Soft-deleted memories
        are excluded, so dead vectors do not accumulate across restarts.
        """
        assert self._conn is not None
        dim = self._config.embedding_dim
        cursor = self._conn.execute(
            "SELECT id, embedding FROM memories "
            "WHERE embedding IS NOT NULL AND is_deleted = 0 ORDER BY id"
        )
        ids: list[int] = []
        vectors: list[np.ndarray] = []
        for row in cursor.fetchall():
            blob = row[1]
            if blob is None:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] != dim:
                logger.warning(
                    "Skipping memory %d: stored embedding dim %d != configured %d",
                    row[0],
                    vec.shape[0],
                    dim,
                )
                continue
            ids.append(int(row[0]))
            vectors.append(vec)

        base = self._create_base_index(len(ids))
        if ids:
            matrix = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
            if not base.is_trained:
                base.train(matrix)
        index = faiss.IndexIDMap2(base)
        if ids:
            index.add_with_ids(matrix, np.asarray(ids, dtype=np.int64))
        logger.info("Built FAISS index from DB (%d vectors)", len(ids))
        return index

    def _maybe_upgrade_faiss_index(self) -> None:
        """Upgrade from FlatIP to IVFFlat once enough vectors exist to train it."""
        assert self._faiss_index is not None
        if self._config.faiss_type == "flat_ip":
            return
        if self._ivf_plan(self._faiss_index.ntotal) <= 0:
            return
        base = self._faiss_index
        if isinstance(base, faiss.IndexIDMap2):
            base = faiss.downcast_index(base.index)
        if isinstance(base, faiss.IndexFlat):
            logger.info(
                "Upgrading FAISS index from FlatIP to IVFFlat (%d vectors)",
                self._faiss_index.ntotal,
            )
            self._faiss_index = self._build_faiss_from_db()

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """L2-normalize embedding vector for cosine similarity."""
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            return embedding
        return embedding / norm

    @_synchronized
    def add_memory(
        self,
        content: str,
        embedding: np.ndarray | None = None,
        memory_type: str = "semantic",
        importance: float = 0.5,
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        namespace: str = "default",
        scope: str = "session",
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
        event_at: float | None = None,
        valid_from: float | None = None,
        valid_to: float | None = None,
        supersedes_id: int | None = None,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """Add a new memory to the database.

        Args:
            content: Text content of the memory.
            embedding: Optional embedding vector (will be L2-normalized).
            memory_type: Category of memory (semantic, episodic, procedural).
            importance: Importance score (0.0-1.0).
            entities: List of entity names to associate.
            metadata: Optional JSON-serializable metadata dict.
            tags: Optional list of tags.
            namespace: Namespace for isolation.
            scope: Scope for isolation.
            user_id: User identifier.
            agent_id: Agent identifier.
            session_id: Session identifier.
            project_id: Project identifier.
            event_at: Unix timestamp when the fact/event occurred.
            valid_from: Unix timestamp when the memory becomes valid.
            valid_to: Unix timestamp when the memory expires (None = forever).
            supersedes_id: ID of memory this supersedes (for temporal updates).
            confidence: Confidence score (0.0-1.0).

        Returns:
            Dict with memory_id and status ('created' or 'duplicate').
        """
        assert self._conn is not None
        try:
            importance = _clamp01(importance)
            confidence = _clamp01(confidence)
            content_hash = _hash_content(content)

            # Dedup check
            cursor = self._conn.execute(
                """SELECT id FROM memories
                   WHERE content_hash = ? AND namespace = ? AND is_deleted = 0""",
                (content_hash, namespace),
            )
            existing = cursor.fetchone()
            if existing is not None:
                logger.info("Duplicate memory detected (hash=%s), id=%d", content_hash, existing[0])
                return {"memory_id": existing[0], "status": "duplicate"}

            now = _now()
            emb: np.ndarray | None = None
            embedding_blob = None
            if embedding is not None:
                emb = self._normalize_embedding(np.asarray(embedding, dtype=np.float32))
                if emb.ndim != 1 or emb.shape[0] != self._config.embedding_dim:
                    raise ValueError(
                        f"embedding dimension {emb.shape} != configured "
                        f"({self._config.embedding_dim},)"
                    )
                embedding_blob = emb.tobytes()

            metadata_json = json.dumps(metadata) if metadata is not None else None
            tags_json = json.dumps(tags) if tags else "[]"

            cursor = self._conn.execute(
                """INSERT INTO memories
                   (content, content_hash, memory_type, importance, embedding,
                    created_at, updated_at, accessed_at, access_count,
                    retention_strength, is_deleted, metadata, tags, namespace,
                    scope, user_id, agent_id, session_id, project_id,
                    event_at, valid_from, valid_to, supersedes_id, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",  # noqa: E501 - 24-column placeholder list
                (
                    content,
                    content_hash,
                    memory_type,
                    importance,
                    embedding_blob,
                    now,
                    now,
                    now,
                    0,
                    1.0,
                    0,
                    metadata_json,
                    tags_json,
                    namespace,
                    scope,
                    user_id,
                    agent_id,
                    session_id,
                    project_id,
                    event_at,
                    valid_from,
                    valid_to,
                    supersedes_id,
                    confidence,
                ),
            )
            memory_id = cursor.lastrowid
            assert memory_id is not None
            memory_id = int(memory_id)

            # Associate entities
            if entities:
                for entity_name in entities:
                    entity_id = self._get_or_create_entity(entity_name)
                    self._conn.execute(
                        "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",  # noqa: E501 - 24-column placeholder list
                        (memory_id, entity_id),
                    )

            self._commit()

            # Add to FAISS only after the row is durably committed, so the index
            # can never hold a vector for a row that was rolled back.
            if emb is not None:
                vec = np.ascontiguousarray(emb.reshape(1, -1), dtype=np.float32)
                self._faiss_index.add_with_ids(vec, np.asarray([memory_id], dtype=np.int64))
                self._maybe_upgrade_faiss_index()

            logger.info(
                "Added memory id=%d type=%s importance=%.2f content_preview=%.50s",
                memory_id,
                memory_type,
                importance,
                content,
            )
            return {"memory_id": memory_id, "status": "created"}

        except sqlite3.Error as e:
            logger.error("Database error adding memory: %s", e)
            self._conn.rollback()
            raise
        except Exception as e:
            logger.error("Unexpected error adding memory: %s", e)
            self._conn.rollback()
            raise

    @_synchronized
    def add_memories_bulk(
        self,
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add multiple memories in a single transaction.

        Each memory dict supports: content, embedding, memory_type,
        importance, entities, metadata, tags, namespace, scope, user_id,
        agent_id, session_id, project_id, event_at, valid_from, valid_to,
        supersedes_id, confidence.

        Returns:
            List of result dicts with memory_id and status per memory.
        """
        assert self._conn is not None
        results = []
        pending_vectors: list[tuple[int, np.ndarray]] = []
        now = _now()
        try:
            for mem in memories:
                content = mem.get("content", "")
                importance = _clamp01(mem.get("importance", 0.5))
                confidence = _memory_confidence(mem)
                content_hash = _hash_content(content)
                memory_type = mem.get("memory_type", "semantic")
                namespace = mem.get("namespace", "default")
                scope = mem.get("scope", "session")
                user_id = mem.get("user_id")
                agent_id = mem.get("agent_id")
                session_id = mem.get("session_id")
                project_id = mem.get("project_id")
                event_at = mem.get("event_at")
                valid_from = mem.get("valid_from")
                valid_to = mem.get("valid_to")
                supersedes_id = mem.get("supersedes_id")

                # Dedup check
                cursor = self._conn.execute(
                    """SELECT id FROM memories
                       WHERE content_hash = ? AND namespace = ? AND is_deleted = 0""",
                    (content_hash, namespace),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    results.append({"memory_id": existing[0], "status": "duplicate"})
                    continue

                emb = None
                embedding_blob = None
                embedding = mem.get("embedding")
                if embedding is not None:
                    emb = self._normalize_embedding(np.asarray(embedding, dtype=np.float32))
                    if emb.ndim != 1 or emb.shape[0] != self._config.embedding_dim:
                        raise ValueError(
                            f"embedding dimension {emb.shape} != configured "
                            f"({self._config.embedding_dim},)"
                        )
                    embedding_blob = emb.tobytes()

                metadata_json = json.dumps(mem.get("metadata")) if mem.get("metadata") else None
                tags_json = json.dumps(mem.get("tags")) if mem.get("tags") else "[]"

                cursor = self._conn.execute(
                    """INSERT INTO memories
                       (content, content_hash, memory_type, importance, embedding,
                        created_at, updated_at, accessed_at, access_count,
                        retention_strength, is_deleted, metadata, tags, namespace,
                        scope, user_id, agent_id, session_id, project_id,
                        event_at, valid_from, valid_to, supersedes_id, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",  # noqa: E501 - 24-column placeholder list
                    (
                        content,
                        content_hash,
                        memory_type,
                        importance,
                        embedding_blob,
                        now,
                        now,
                        now,
                        0,
                        1.0,
                        0,
                        metadata_json,
                        tags_json,
                        namespace,
                        scope,
                        user_id,
                        agent_id,
                        session_id,
                        project_id,
                        event_at,
                        valid_from,
                        valid_to,
                        supersedes_id,
                        confidence,
                    ),
                )
                memory_id = cursor.lastrowid
                assert memory_id is not None
                memory_id = int(memory_id)

                # Defer FAISS insertion until after the transaction commits
                if emb is not None:
                    pending_vectors.append((memory_id, emb))

                # Associate entities
                entities = mem.get("entities")
                if entities:
                    for entity_name in entities:
                        entity_id = self._get_or_create_entity(entity_name)
                        self._conn.execute(
                            "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",  # noqa: E501 - 24-column placeholder list
                            (memory_id, entity_id),
                        )

                results.append({"memory_id": memory_id, "status": "created"})

            self._commit()

            if pending_vectors:
                ids = np.asarray([mid for mid, _ in pending_vectors], dtype=np.int64)
                matrix = np.ascontiguousarray(
                    np.vstack([v.reshape(1, -1) for _, v in pending_vectors]), dtype=np.float32
                )
                self._faiss_index.add_with_ids(matrix, ids)
                self._maybe_upgrade_faiss_index()

            logger.info(
                "Bulk add: %d memories, %d created, %d duplicates",
                len(memories),
                sum(1 for r in results if r["status"] == "created"),
                sum(1 for r in results if r["status"] == "duplicate"),
            )
            return results
        except sqlite3.Error as e:
            logger.error("Database error in bulk add: %s", e)
            self._conn.rollback()
            raise

    @_synchronized
    def export_all(self) -> dict[str, Any]:
        """Export all active memories, entities, and links as a JSON-safe dict."""
        assert self._conn is not None
        memories = []
        for row in self._conn.execute(
            """SELECT id, content, content_hash, memory_type, importance,
                      created_at, updated_at, accessed_at, access_count,
                      retention_strength, metadata, tags, embedding, namespace,
                      scope, user_id, agent_id, session_id, project_id,
                      event_at, valid_from, valid_to, supersedes_id, confidence
               FROM memories WHERE is_deleted = 0 ORDER BY id"""
        ).fetchall():
            mem = {
                "id": row[0],
                "content": row[1],
                "content_hash": row[2],
                "memory_type": row[3],
                "importance": row[4],
                "created_at": row[5],
                "updated_at": row[6],
                "accessed_at": row[7],
                "access_count": row[8],
                "retention_strength": row[9],
                "metadata": json.loads(row[10]) if row[10] else None,
                "tags": json.loads(row[11]) if row[11] else [],
                "embedding": (
                    np.frombuffer(row[12], dtype=np.float32).tolist()
                    if row[12] is not None
                    else None
                ),
                "namespace": row[13],
                "scope": row[14],
                "user_id": row[15],
                "agent_id": row[16],
                "session_id": row[17],
                "project_id": row[18],
                "event_at": row[19],
                "valid_from": row[20],
                "valid_to": row[21],
                "supersedes_id": row[22],
                "confidence": row[23],
            }
            # Get entities for this memory
            entity_rows = self._conn.execute(
                """SELECT e.name FROM memory_entities me
                   JOIN entities e ON e.id = me.entity_id
                   WHERE me.memory_id = ?""",
                (mem["id"],),
            ).fetchall()
            mem["entities"] = [r[0] for r in entity_rows]
            # Get sources for this memory
            source_rows = self._conn.execute(
                """SELECT s.id, s.episode_id, s.source, s.source_id, s.span,
                          s.confidence, s.created_at
                   FROM sources s WHERE s.memory_id = ?""",
                (mem["id"],),
            ).fetchall()
            mem["sources"] = [
                {
                    "id": r[0],
                    "episode_id": r[1],
                    "source": r[2],
                    "source_id": r[3],
                    "span": r[4],
                    "confidence": r[5],
                    "created_at": r[6],
                }
                for r in source_rows
            ]
            memories.append(mem)

        entities = [
            {"id": r[0], "name": r[1], "entity_type": r[2], "created_at": r[3]}
            for r in self._conn.execute(
                "SELECT id, name, entity_type, created_at FROM entities"
            ).fetchall()
        ]

        memory_links = [
            {
                "source_id": r[0],
                "target_id": r[1],
                "link_type": r[2],
                "strength": r[3],
                "created_at": r[4],
            }
            for r in self._conn.execute(
                "SELECT source_id, target_id, link_type, strength, created_at FROM memory_links"
            ).fetchall()
        ]

        edges = [
            {
                "source_id": r[0],
                "target_id": r[1],
                "edge_type": r[2],
                "weight": r[3],
                "created_at": r[4],
            }
            for r in self._conn.execute(
                "SELECT source_id, target_id, edge_type, weight, created_at FROM edges"
            ).fetchall()
        ]

        # Episodes
        episodes = [
            {
                "id": r[0],
                "content": r[1],
                "role": r[2],
                "source": r[3],
                "event_at": r[4],
                "metadata": json.loads(r[5]) if r[5] else None,
                "namespace": r[6],
                "scope": r[7],
                "user_id": r[8],
                "agent_id": r[9],
                "session_id": r[10],
                "project_id": r[11],
                "created_at": r[12],
            }
            for r in self._conn.execute(
                """SELECT id, content, role, source, event_at, metadata,
                                  namespace, scope, user_id, agent_id, session_id,
                                  project_id, created_at
                           FROM episodes"""
            ).fetchall()
        ]

        return {
            "version": 1,
            "memories": memories,
            "entities": entities,
            "edges": edges,
            "memory_links": memory_links,
            "episodes": episodes,
        }

    @_synchronized
    def import_all(self, data: dict[str, Any]) -> int:
        """Import memories from an export dict. Returns count imported.

        Imported rows receive fresh autoincrement ids, so exported ids are
        remapped (old -> new) and ``memory_links`` are rewritten through that
        map. Links whose endpoints were not part of the import are skipped
        rather than attached to whatever rows happen to share the old ids.
        """
        assert self._conn is not None
        count = 0
        now = _now()
        dim = self._config.embedding_dim
        id_map: dict[int, int] = {}
        try:
            for mem in data.get("memories", []):
                content = mem.get("content", "")
                content_hash = _hash_content(content)
                old_id = mem.get("id")
                namespace = mem.get("namespace", "default")
                scope = mem.get("scope", "session")
                cursor = self._conn.execute(
                    """SELECT id FROM memories
                       WHERE content_hash = ? AND namespace = ? AND is_deleted = 0""",
                    (content_hash, namespace),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    # Duplicate: map the exported id onto the existing row so
                    # links referencing it still resolve.
                    if old_id is not None:
                        id_map[int(old_id)] = int(existing[0])
                    continue

                metadata_json = json.dumps(mem.get("metadata")) if mem.get("metadata") else None
                tags_json = json.dumps(mem.get("tags", []))
                importance = _clamp01(mem.get("importance", 0.5))
                confidence = _memory_confidence(mem)

                embedding_blob = None
                embedding = mem.get("embedding")
                if embedding is not None:
                    emb = self._normalize_embedding(np.asarray(embedding, dtype=np.float32))
                    if emb.shape[0] == dim:
                        embedding_blob = emb.tobytes()
                    else:
                        logger.warning(
                            "Import: dropping embedding with dim %d != configured %d",
                            emb.shape[0],
                            dim,
                        )

                cursor = self._conn.execute(
                    """INSERT INTO memories
                       (content, content_hash, memory_type, importance, embedding,
                        created_at, updated_at, accessed_at, access_count,
                        retention_strength, metadata, tags, namespace, scope,
                        user_id, agent_id, session_id, project_id,
                        event_at, valid_from, valid_to, supersedes_id, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",  # noqa: E501 - 24-column placeholder list
                    (
                        content,
                        content_hash,
                        mem.get("memory_type", "semantic"),
                        importance,
                        embedding_blob,
                        mem.get("created_at", now),
                        now,
                        mem.get("accessed_at", now),
                        mem.get("access_count", 0),
                        mem.get("retention_strength", 1.0),
                        metadata_json,
                        tags_json,
                        namespace,
                        scope,
                        mem.get("user_id"),
                        mem.get("agent_id"),
                        mem.get("session_id"),
                        mem.get("project_id"),
                        mem.get("event_at"),
                        mem.get("valid_from"),
                        mem.get("valid_to"),
                        mem.get("supersedes_id"),
                        confidence,
                    ),
                )
                memory_id = cursor.lastrowid
                assert memory_id is not None
                memory_id = int(memory_id)
                if old_id is not None:
                    id_map[int(old_id)] = memory_id

                # Re-associate entities (create if needed)
                for entity_name in mem.get("entities", []):
                    entity_id = self._get_or_create_entity(entity_name)
                    self._conn.execute(
                        "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",  # noqa: E501 - 24-column placeholder list
                        (memory_id, entity_id),
                    )
                count += 1

            # Re-import entity edges after remapping exported entity ids.
            entity_id_map: dict[int, int] = {}
            for entity in data.get("entities", []):
                old_entity_id = entity.get("id")
                if old_entity_id is None:
                    continue
                current = self._get_or_create_entity(entity.get("name", ""))
                entity_id_map[int(old_entity_id)] = current
            for edge in data.get("edges", []):
                source_id = entity_id_map.get(edge.get("source_id"))
                target_id = entity_id_map.get(edge.get("target_id"))
                if source_id is None or target_id is None:
                    continue
                self._conn.execute(
                    """INSERT INTO edges (source_id, target_id, edge_type, weight, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        source_id,
                        target_id,
                        edge.get("edge_type", "related"),
                        edge.get("weight", 1.0),
                        edge.get("created_at", now),
                    ),
                )

            # Re-import memory_links, remapping endpoints to their new ids.
            skipped_links = 0
            for link in data.get("memory_links", []):
                source_id = id_map.get(link.get("source_id"))
                target_id = id_map.get(link.get("target_id"))
                if source_id is None or target_id is None:
                    skipped_links += 1
                    continue
                self._conn.execute(
                    """INSERT OR IGNORE INTO memory_links
                    (source_id, target_id, link_type, strength, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        source_id,
                        target_id,
                        link.get("link_type", "related"),
                        link.get("strength", 1.0),
                        link.get("created_at", now),
                    ),
                )
            if skipped_links:
                logger.warning(
                    "Import: skipped %d links with unresolvable endpoints", skipped_links
                )

            # Import episodes
            for episode in data.get("episodes", []):
                cursor = self._conn.execute(
                    """INSERT INTO episodes
                       (content, role, source, event_at, metadata, namespace, scope,
                        user_id, agent_id, session_id, project_id, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        episode.get("content", ""),
                        episode.get("role", "user"),
                        episode.get("source"),
                        episode.get("event_at"),
                        json.dumps(episode.get("metadata")) if episode.get("metadata") else None,
                        episode.get("namespace", "default"),
                        episode.get("scope", "session"),
                        episode.get("user_id"),
                        episode.get("agent_id"),
                        episode.get("session_id"),
                        episode.get("project_id"),
                        episode.get("created_at", now),
                    ),
                )

            # Import sources
            for source in data.get("sources", []):
                cursor = self._conn.execute(
                    """INSERT INTO sources
                       (memory_id, episode_id, source, source_id, span, confidence, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        id_map.get(source.get("memory_id")),
                        source.get("episode_id"),
                        source.get("source", "imported"),
                        source.get("source_id"),
                        source.get("span"),
                        source.get("confidence", 1.0),
                        source.get("created_at", now),
                    ),
                )

            self._commit()
            # Imported rows bypassed the live FAISS index; rebuild so their
            # embeddings become searchable immediately, not only after reopen.
            if count:
                self._faiss_index = self._build_faiss_from_db()
            logger.info("Import: %d memories imported", count)
            return count
        except sqlite3.Error as e:
            logger.error("Database error in import: %s", e)
            self._conn.rollback()
            raise

    def _get_or_create_entity(self, name: str) -> int:
        """Get entity ID by name, creating if it doesn't exist."""
        assert self._conn is not None
        cursor = self._conn.execute("SELECT id FROM entities WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row is not None:
            return int(row[0])

        cursor = self._conn.execute(
            "INSERT INTO entities (name, created_at) VALUES (?, ?)",
            (name, _now()),
        )
        entity_id = cursor.lastrowid
        assert entity_id is not None
        return int(entity_id)

    @_synchronized
    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        """Retrieve a memory by ID.

        This is a **pure read**: it never mutates access statistics. Recording an
        access is done explicitly via ``touch_memory``/``touch_memories`` (and
        ``AriadneMemory.recall`` does it for the memories it actually surfaces).
        Keeping reads side-effect free is what makes search fast — a single
        search no longer issues one UPDATE + commit per candidate hit.

        Args:
            memory_id: The memory's unique ID.

        Returns:
            Memory dict with all fields, or None if not found.
        """
        assert self._conn is not None
        try:
            cursor = self._conn.execute(
                """SELECT id, content, content_hash, memory_type, importance,
                          created_at, updated_at, accessed_at, access_count,
                          retention_strength, is_deleted, metadata, tags, namespace,
                          scope, user_id, agent_id, session_id, project_id,
                          event_at, valid_from, valid_to, supersedes_id, confidence
                   FROM memories WHERE id = ?""",
                (memory_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return self._row_to_memory(row)
        except sqlite3.Error as e:
            logger.error("Database error getting memory %d: %s", memory_id, e)
            return None

    def _row_to_memory(self, row: Any) -> dict[str, Any]:
        """Map a memories-table row (column order of ``get_memory``) to a dict."""
        metadata_raw, tags_raw = row[11], row[12]
        return {
            "id": row[0],
            "content": row[1],
            "content_hash": row[2],
            "memory_type": row[3],
            "importance": row[4],
            "created_at": row[5],
            "updated_at": row[6],
            "accessed_at": row[7],
            "access_count": row[8],
            "retention_strength": row[9],
            "is_deleted": bool(row[10]),
            # Fast path: NULL / stored defaults skip a json.loads per row, which
            # dominates bulk row mapping at search time.
            "metadata": json.loads(metadata_raw) if metadata_raw else None,
            "tags": json.loads(tags_raw) if tags_raw and tags_raw != "[]" else [],
            "namespace": row[13],
            "scope": row[14],
            "user_id": row[15],
            "agent_id": row[16],
            "session_id": row[17],
            "project_id": row[18],
            "event_at": row[19],
            "valid_from": row[20],
            "valid_to": row[21],
            "supersedes_id": row[22],
            "confidence": row[23],
        }

    @_synchronized
    def get_memories_bulk(self, memory_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch several memories by id in one query.

        Returns a dict keyed by memory id; missing and soft-deleted ids are
        absent from the result. Search pipelines use this instead of one
        ``get_memory`` call per candidate (the old N+1 pattern).
        """
        assert self._conn is not None
        unique_ids = list(dict.fromkeys(int(i) for i in memory_ids if i >= 0))
        if not unique_ids:
            return {}
        placeholders = ",".join("?" * len(unique_ids))
        rows = self._conn.execute(
            f"""SELECT id, content, content_hash, memory_type, importance,
                       created_at, updated_at, accessed_at, access_count,
                       retention_strength, is_deleted, metadata, tags, namespace,
                       scope, user_id, agent_id, session_id, project_id,
                       event_at, valid_from, valid_to, supersedes_id, confidence
                FROM memories WHERE id IN ({placeholders})""",
            unique_ids,
        ).fetchall()
        return {int(row[0]): self._row_to_memory(row) for row in rows}

    @_synchronized
    def recent_memories(
        self,
        limit: int = 20,
        namespace: str | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """List the most recently created memories (newest first).

        Backs listing surfaces (CLI ``list``, adapter fallbacks) that need
        recent items rather than query-ranked ones.
        """
        assert self._conn is not None
        sql = """SELECT id, content, content_hash, memory_type, importance,
                        created_at, updated_at, accessed_at, access_count,
                        retention_strength, is_deleted, metadata, tags, namespace,
                        scope, user_id, agent_id, session_id, project_id,
                        event_at, valid_from, valid_to, supersedes_id, confidence
                 FROM memories"""
        params: list[Any] = []
        conditions = []
        if not include_deleted:
            conditions.append("is_deleted = 0")
        if namespace is not None:
            conditions.append("namespace = ?")
            params.append(namespace)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    @_synchronized
    def touch_memory(self, memory_id: int) -> None:
        """Record a single memory access. See ``touch_memories``."""
        self.touch_memories([memory_id])

    @_synchronized
    def touch_memories(self, memory_ids: list[int]) -> None:
        """Record an access for several memories in a single transaction.

        Increments ``access_count``, refreshes ``accessed_at``, and grows the
        stored ``retention_strength`` by ``retention_growth_factor`` (capped at
        ``retention_strength_cap``). This is the spacing-effect model — memories
        strengthen each time they are recalled — and feeds back into
        ``compute_retention_strength``. The ``memories_access_log`` trigger logs
        each access automatically.
        """
        if not memory_ids:
            return
        assert self._conn is not None
        now = _now()
        growth = self._config.retention_growth_factor
        cap = self._config.retention_strength_cap
        placeholders = ",".join("?" * len(memory_ids))
        try:
            self._conn.execute(
                f"""UPDATE memories
                    SET accessed_at = ?,
                        access_count = access_count + 1,
                        retention_strength = MIN(?, retention_strength * ?)
                    WHERE id IN ({placeholders}) AND is_deleted = 0""",
                [now, cap, growth, *memory_ids],
            )
            self._commit()
        except sqlite3.Error as e:
            logger.error("Error touching memories: %s", e)
            self._conn.rollback()

    @_synchronized
    def update_memory(
        self,
        memory_id: int,
        content: str | None = None,
        importance: float | None = None,
        embedding: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        """Update an existing memory.

        Args:
            memory_id: The memory's unique ID.
            content: New content (replaces old).
            importance: New importance score.
            embedding: New embedding vector.
            metadata: New metadata dict.

        Returns:
            True if updated, False if memory not found.
        """
        assert self._conn is not None
        try:
            existing = self.get_memory(memory_id)
            if existing is None:
                return False
            had_embedding = bool(
                self._conn.execute(
                    "SELECT embedding IS NOT NULL FROM memories WHERE id = ?",
                    (memory_id,),
                ).fetchone()[0]
            )

            updates = ["updated_at = ?"]
            params: list[Any] = [_now()]

            if content is not None:
                updates.append("content = ?")
                params.append(content)
                updates.append("content_hash = ?")
                params.append(_hash_content(content))
                if embedding is None and had_embedding:
                    # A text edit without a replacement vector must not leave
                    # the old semantic meaning attached to the new content.
                    updates.append("embedding = NULL")

            if importance is not None:
                updates.append("importance = ?")
                params.append(_clamp01(importance))

            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))

            if tags is not None:
                updates.append("tags = ?")
                params.append(json.dumps(tags))

            emb: np.ndarray | None = None
            if embedding is not None:
                emb = self._normalize_embedding(np.asarray(embedding, dtype=np.float32))
                if emb.ndim != 1 or emb.shape[0] != self._config.embedding_dim:
                    raise ValueError(
                        f"embedding dimension {emb.shape} != configured "
                        f"({self._config.embedding_dim},)"
                    )
                updates.append("embedding = ?")
                params.append(emb.tobytes())

            params.append(memory_id)
            self._conn.execute(
                f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
                params,
            )

            self._commit()

            # Keep FAISS in sync after the commit: IndexIDMap2 lets us replace
            # a vector by its id.
            if emb is not None:
                id_array = np.asarray([memory_id], dtype=np.int64)
                try:
                    self._faiss_index.remove_ids(id_array)
                except Exception as e:  # pragma: no cover - depends on index state
                    logger.debug("FAISS remove during update skipped: %s", e)
                vec = np.ascontiguousarray(emb.reshape(1, -1), dtype=np.float32)
                self._faiss_index.add_with_ids(vec, id_array)
                self._maybe_upgrade_faiss_index()
            elif content is not None and had_embedding:
                try:
                    self._faiss_index.remove_ids(np.asarray([memory_id], dtype=np.int64))
                except Exception as e:  # pragma: no cover - depends on index state
                    logger.debug("FAISS remove during content update skipped: %s", e)

            logger.info("Updated memory %d", memory_id)
            return True

        except sqlite3.Error as e:
            logger.error("Database error updating memory %d: %s", memory_id, e)
            self._conn.rollback()
            return False

    @_synchronized
    def delete_memory(self, memory_id: int, hard: bool = False) -> bool:
        """Delete a memory (soft or hard).

        Args:
            memory_id: The memory's unique ID.
            hard: If True, permanently delete. Otherwise soft-delete.

        Returns:
            True if deleted, False if memory not found.
        """
        assert self._conn is not None
        try:
            cursor = self._conn.execute("SELECT id FROM memories WHERE id = ?", (memory_id,))
            if cursor.fetchone() is None:
                return False

            if hard:
                # Children first: with foreign_keys=ON, deleting the parent row
                # while memory_entities/memory_links/access_log still reference
                # it fails with "FOREIGN KEY constraint failed".
                self._conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (memory_id,))
                self._conn.execute(
                    "DELETE FROM memory_links WHERE source_id = ? OR target_id = ?",
                    (memory_id, memory_id),
                )
                self._conn.execute("DELETE FROM access_log WHERE memory_id = ?", (memory_id,))
                self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                # Remove the vector from FAISS by its id (IndexIDMap2)
                if self._faiss_index is not None:
                    try:
                        self._faiss_index.remove_ids(np.asarray([memory_id], dtype=np.int64))
                    except Exception as e:  # pragma: no cover - depends on index state
                        logger.debug("FAISS remove_ids during hard delete skipped: %s", e)
            else:
                now = _now()
                self._conn.execute(
                    "UPDATE memories SET is_deleted = 1, deleted_at = ? WHERE id = ?",
                    (now, memory_id),
                )
                if self._faiss_index is not None:
                    try:
                        self._faiss_index.remove_ids(np.asarray([memory_id], dtype=np.int64))
                    except Exception as e:  # pragma: no cover - depends on index state
                        logger.debug("FAISS remove during soft delete skipped: %s", e)

            self._commit()
            logger.info("Deleted memory %d (hard=%s)", memory_id, hard)
            return True

        except sqlite3.Error as e:
            logger.error("Database error deleting memory %d: %s", memory_id, e)
            self._conn.rollback()
            return False

    def _namespaces_for_ids(self, memory_ids: list[int]) -> dict[int, str]:
        """Light id -> namespace projection used to pre-filter vector candidates."""
        assert self._conn is not None
        if not memory_ids:
            return {}
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._conn.execute(
            f"SELECT id, namespace FROM memories WHERE id IN ({placeholders})",
            memory_ids,
        ).fetchall()
        return {int(r[0]): str(r[1]) for r in rows}

    @_synchronized
    def vector_search(
        self, embedding: np.ndarray, k: int = 10, namespace: str | None = None
    ) -> list[dict[str, Any]]:
        """Search memories by vector similarity (cosine).

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            namespace: Optional namespace filter. The FAISS index is not
                namespace-partitioned, so a filtered search widens its candidate
                window geometrically (instead of scanning the whole index) until
                k in-namespace hits are found; only the returned rows are
                hydrated in full.

        Returns:
            List of memory dicts ordered by similarity (descending).
        """
        assert self._faiss_index is not None
        if self._faiss_index.ntotal == 0:
            return []

        try:
            emb = self._normalize_embedding(np.asarray(embedding, dtype=np.float32))
            vec = emb.reshape(1, -1)
            selected: list[tuple[float, int]] = []
            probe_k = k if namespace is None else max(k * 4, 64)
            while True:
                requested_k = min(probe_k, self._faiss_index.ntotal)
                distances, indices = self._faiss_index.search(vec, requested_k)

                # With IndexIDMap2 the returned label is the memory's own id.
                candidates = [
                    (float(dist), int(idx))
                    for dist, idx in zip(distances[0], indices[0], strict=True)
                    if idx >= 0
                ]
                if namespace is None:
                    selected = candidates[:k]
                    break
                ns_map = self._namespaces_for_ids([idx for _, idx in candidates])
                selected = [
                    (dist, idx)
                    for dist, idx in candidates
                    if ns_map.get(idx) == namespace
                ][:k]
                if len(selected) >= k or requested_k >= self._faiss_index.ntotal:
                    break
                probe_k *= 2

            by_id = self.get_memories_bulk([idx for _, idx in selected])
            results: list[dict[str, Any]] = []
            for dist, idx in selected:
                memory = by_id.get(idx)
                if memory is None or memory["is_deleted"]:
                    continue
                memory["score"] = dist
                memory["search_type"] = "vector"
                results.append(memory)
            return results[:k]

        except Exception as e:
            logger.error("Vector search error: %s", e)
            return []

    @_synchronized
    def search_vector_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        namespace: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Search memories by vector similarity for multiple queries at once.

        Optimization: Uses FAISS's built-in batched search (index.search with
        multiple query vectors) which is significantly faster than looping
        vector_search() for each query individually. FAISS parallelizes the
        search internally and avoids repeated overhead.

        Args:
            query_embeddings: 2D array of shape (num_queries, dim) with
                              query embedding vectors.
            k: Number of results per query.

        Returns:
            List of lists, where each inner list contains memory dicts
            for the corresponding query, ordered by similarity (descending).
        """
        assert self._faiss_index is not None
        if self._faiss_index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]

        try:
            queries = np.asarray(query_embeddings, dtype=np.float32)
            if queries.ndim == 1:
                queries = queries.reshape(1, -1)

            # Normalize all queries at once (vectorized)
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            queries = queries / norms

            search_k = (
                self._faiss_index.ntotal
                if namespace is not None
                else min(k, self._faiss_index.ntotal)
            )
            distances, indices = self._faiss_index.search(queries, search_k)

            all_results: list[list[dict[str, Any]]] = []
            for query_idx in range(len(queries)):
                results = []
                for dist, idx in zip(distances[query_idx], indices[query_idx], strict=True):
                    if idx < 0:
                        continue
                    # With IndexIDMap2 the returned label is the memory's own id.
                    memory = self.get_memory(int(idx))
                    if memory is not None and not memory["is_deleted"]:
                        if namespace is not None and memory.get("namespace") != namespace:
                            continue
                        memory["score"] = float(dist)
                        memory["search_type"] = "vector_batch"
                        results.append(memory)
                        if len(results) >= k:
                            break
                all_results.append(results)
            return all_results

        except Exception as e:
            logger.error("Batch vector search error: %s", e)
            return [[] for _ in range(len(query_embeddings))]

    @_synchronized
    def fts_search(
        self,
        query: str,
        k: int = 10,
        namespace: str | None = None,
        confidence_weighted: bool = True,
    ) -> list[dict[str, Any]]:
        """Search memories by full-text keyword matching.

        Args:
            query: Search query string.
            k: Number of results.
            namespace: Optional namespace filter.
            confidence_weighted: Re-rank matches by stored confidence.
                Disable for hybrid search, which applies confidence once after
                RRF fusion.

        Returns:
            List of memory dicts ordered by relevance.
        """
        assert self._conn is not None
        try:
            # Precision first (AND), then fall back to recall (OR) when AND finds
            # nothing. Confidence is applied in SQL before LIMIT so a lower BM25
            # match can legitimately outrank a rejected top match.
            rows: list[Any] = []
            weight_expr = (
                "(0.5 + 0.5 * MAX(0.0, MIN(1.0, COALESCE(CAST(m.confidence AS REAL), 1.0))))"
                if confidence_weighted
                else "1.0"
            )
            for op in ("AND", "OR"):
                fts_query = _fts_escape(query, op=op)
                sql = f"""SELECT f.rowid, f.rank,
                              ABS(f.rank) * {weight_expr} AS score
                         FROM memories_fts AS f
                         JOIN memories AS m ON m.id = f.rowid
                        WHERE f.memories_fts MATCH ?
                          AND m.is_deleted = 0"""
                params: list[Any] = [fts_query]
                if namespace is not None:
                    sql += " AND m.namespace = ?"
                    params.append(namespace)
                sql += " ORDER BY score DESC, f.rowid ASC LIMIT ?"
                params.append(k)
                cursor = self._conn.execute(sql, params)
                rows = cursor.fetchall()
                if rows:
                    break

            results = []
            row_map = self.get_memories_bulk([int(rowid) for rowid, _, _ in rows])
            for rowid, rank, score in rows:
                memory = row_map.get(int(rowid))
                if memory is not None and not memory["is_deleted"]:
                    memory["score"] = float(score)
                    memory["search_type"] = "fts"
                    if confidence_weighted:
                        confidence = _memory_confidence(memory)
                        memory["score_parts"] = {
                            "fts": abs(float(rank)),
                            "confidence": confidence,
                            "confidence_weight": 0.5 + 0.5 * confidence,
                        }
                    results.append(memory)
            return results

        except sqlite3.Error as e:
            logger.error("FTS search error: %s", e)
            return []

    @_synchronized
    def hybrid_search(
        self,
        query: str,
        embedding: np.ndarray | None = None,
        k: int = 10,
        rrf_k: int = 60,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector and FTS results with Reciprocal Rank Fusion.

        Args:
            query: Text query for FTS.
            embedding: Query embedding for vector search (optional).
            k: Number of results to return.
            rrf_k: RRF parameter (higher = less weight on top ranks).

        Returns:
            List of memory dicts ordered by fused score.

        Optimization: Early termination — if either FTS or vector returns 0 results,
        skip the RRF fusion step and return results directly from the non-empty source.
        This avoids unnecessary computation when one search modality finds nothing.
        """
        fts_results = self.fts_search(
            query, k=k * 2, namespace=namespace, confidence_weighted=False
        )
        vector_results: list[dict[str, Any]] = []
        if embedding is not None:
            vector_results = self.vector_search(embedding, k=k * 2, namespace=namespace)

        # Early termination optimization: skip RRF fusion if one side is empty
        if not fts_results and not vector_results:
            return []
        if not fts_results:
            # Only vector results — apply confidence before returning.
            for rank, mem in enumerate(vector_results, 1):
                base_score = 1.0 / (rrf_k + rank)
                confidence = _memory_confidence(mem)
                weight = 0.5 + 0.5 * confidence
                mem["score"] = base_score * weight
                mem["search_type"] = "hybrid"
                mem["score_parts"] = {
                    "rrf": base_score,
                    "confidence": confidence,
                    "confidence_weight": weight,
                }
            vector_results.sort(key=lambda item: (-item["score"], item.get("id", 0)))
            return vector_results[:k]
        if not vector_results:
            # Only FTS results — apply confidence before returning.
            for rank, mem in enumerate(fts_results, 1):
                base_score = 1.0 / (rrf_k + rank)
                confidence = _memory_confidence(mem)
                weight = 0.5 + 0.5 * confidence
                mem["score"] = base_score * weight
                mem["search_type"] = "hybrid"
                mem["score_parts"] = {
                    "rrf": base_score,
                    "confidence": confidence,
                    "confidence_weight": weight,
                }
            fts_results.sort(key=lambda item: (-item["score"], item.get("id", 0)))
            return fts_results[:k]

        # Build rank maps
        fts_ranks: dict[int, int] = {}
        for rank, mem in enumerate(fts_results):
            fts_ranks[mem["id"]] = rank + 1

        vector_ranks: dict[int, int] = {}
        for rank, mem in enumerate(vector_results):
            vector_ranks[mem["id"]] = rank + 1

        # Reciprocal Rank Fusion with confidence reweighting.
        # Memories with higher confidence float; rejected/stale ones sink.
        # Multiplier maps confidence [0,1] -> [0.5, 1.0], so default (1.0) is
        # unchanged and a rejected memory (0.1) drops to ~0.55x its fused score.
        # Both input lists already carry full memory dicts (fetched inside
        # fts_search / vector_search), so no third row fetch is needed.
        row_map: dict[int, dict[str, Any]] = {}
        for mem in fts_results:
            row_map.setdefault(mem["id"], mem)
        for mem in vector_results:
            row_map.setdefault(mem["id"], mem)
        fused_scores: dict[int, float] = {}
        for mid in row_map:
            score = 0.0
            if mid in fts_ranks:
                score += 1.0 / (rrf_k + fts_ranks[mid])
            if mid in vector_ranks:
                score += 1.0 / (rrf_k + vector_ranks[mid])
            conf = _memory_confidence(row_map.get(mid))
            fused_scores[mid] = score * (0.5 + 0.5 * conf)

        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: (-fused_scores[x], x))[:k]

        results = []
        for mid in sorted_ids:
            memory = row_map.get(mid)
            if memory is not None and not memory["is_deleted"]:
                memory["score"] = fused_scores[mid]
                memory["search_type"] = "hybrid"
                confidence = _memory_confidence(memory)
                memory["score_parts"] = {
                    "rrf": fused_scores[mid] / (0.5 + 0.5 * confidence),
                    "confidence": confidence,
                    "confidence_weight": 0.5 + 0.5 * confidence,
                }
                results.append(memory)
        return results

    @_synchronized
    def add_edge(
        self,
        source_entity: str,
        target_entity: str,
        edge_type: str = "related",
        weight: float = 1.0,
    ) -> None:
        """Add a directed edge between two entities.

        Args:
            source_entity: Name of the source entity.
            target_entity: Name of the target entity.
            edge_type: Type of relationship.
            weight: Edge weight (0.0-1.0).
        """
        assert self._conn is not None
        try:
            source_id = self._get_or_create_entity(source_entity)
            target_id = self._get_or_create_entity(target_entity)
            # UPSERT on the (source, target, type) unique index: a brand-new edge
            # is inserted, while re-declaring the same edge refreshes its weight
            # (the latest declaration wins) without creating duplicate rows.
            self._conn.execute(
                """INSERT INTO edges (source_id, target_id, edge_type, weight, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(source_id, target_id, edge_type)
                   DO UPDATE SET weight = excluded.weight""",
                (source_id, target_id, edge_type, weight, _now()),
            )
            self._commit()
            logger.debug("Added edge %s ->%s (type=%s)", source_entity, target_entity, edge_type)
        except sqlite3.Error as e:
            logger.error("Error adding edge: %s", e)
            self._conn.rollback()

    @_synchronized
    def traverse_graph(
        self,
        entity_name: str,
        hops: int = 1,
        edge_type: str | None = None,
    ) -> dict[str, Any]:
        """Traverse the knowledge graph from an entity using BFS.

        Uses recursive CTE for efficient traversal.

        Args:
            entity_name: Starting entity name.
            hops: Maximum number of hops.
            edge_type: Optional filter on edge type.

        Returns:
            Dict with 'nodes' (entity names) and 'edges' (connections).
        """
        assert self._conn is not None
        try:
            # Find source entity
            cursor = self._conn.execute("SELECT id FROM entities WHERE name = ?", (entity_name,))
            row = cursor.fetchone()
            if row is None:
                return {"nodes": [entity_name], "edges": []}

            source_id = int(row[0])
            hops = min(hops, self._config.max_graph_depth)

            edge_filter = ""
            params: list[Any] = [source_id, hops]
            if edge_type:
                edge_filter = "AND e.edge_type = ?"
                params.append(edge_type)

            # Recursive CTE for BFS. Edges are treated as undirected: from the
            # current node we step to *the other* endpoint of any incident edge
            # (the previous version chose the next node by depth, so it only ever
            # followed outgoing edges and missed incoming ones). UNION dedupes
            # (node_id, depth) pairs, which also terminates cycles.
            query = f"""
                WITH RECURSIVE graph_traverse(node_id, depth) AS (
                    SELECT ?, 0
                    UNION
                    SELECT
                        CASE WHEN e.source_id = gt.node_id
                             THEN e.target_id ELSE e.source_id END,
                        gt.depth + 1
                    FROM graph_traverse gt
                    JOIN edges e ON (
                        e.source_id = gt.node_id
                        OR e.target_id = gt.node_id
                    )
                    WHERE gt.depth < ?
                    {edge_filter}
                )
                SELECT DISTINCT gt.node_id, en.name
                FROM graph_traverse gt
                JOIN entities en ON en.id = gt.node_id
            """

            cursor = self._conn.execute(query, params)
            nodes = []
            seen_ids: set[int] = set()
            for node_id, name in cursor.fetchall():
                if node_id not in seen_ids:
                    nodes.append(name)
                    seen_ids.add(node_id)

            # Get edges between discovered nodes
            if len(nodes) > 1:
                placeholders = ",".join("?" * len(seen_ids))
                edge_query = f"""
                    SELECT e.source_id, e.target_id, e.edge_type, e.weight,
                           s.name, t.name
                    FROM edges e
                    JOIN entities s ON s.id = e.source_id
                    JOIN entities t ON t.id = e.target_id
                    WHERE e.source_id IN ({placeholders})
                      AND e.target_id IN ({placeholders})
                """
                node_list = list(seen_ids)
                cursor = self._conn.execute(edge_query, node_list + node_list)
                edges = []
                for row in cursor.fetchall():
                    edges.append(
                        {
                            "source": row[4],
                            "target": row[5],
                            "type": row[2],
                            "weight": row[3],
                        }
                    )
            else:
                edges = []

            return {"nodes": nodes, "edges": edges}

        except sqlite3.Error as e:
            logger.error("Graph traversal error: %s", e)
            return {"nodes": [entity_name], "edges": []}

    def compute_retention_strength(self, memory: dict[str, Any]) -> float:
        """Compute Ebbinghaus retention score for a memory.

        Uses R = e^(-t/S) where t is time since last access and S is the
        stability. Stability grows with both ``importance`` and the accrued
        ``retention_strength`` — which ``touch_memories`` multiplies up on every
        access — so frequently recalled memories decay more slowly (the spacing
        effect). This is the behaviour the docs promised but the column never
        actually fed into before.

        Args:
            memory: Memory dict with timing fields.

        Returns:
            Retention strength (0.0-1.0).

        Optimization: Delegates to a cached helper that takes hashable primitives
        with time-bucketed now value (1-second granularity).
        """
        return _cached_retention_strength(
            memory["accessed_at"],
            memory["importance"],
            float(memory.get("retention_strength", 1.0) or 0.0),
            self._config.retention_half_life,
            int(_now()),  # Bucket to 1-second granularity for cache hits
        )

    def compute_priority_score(self, memory: dict[str, Any]) -> float:
        """Compute priority score using weighted formula.

        Priority = w_imp * importance + w_rec * recency + w_acc * access_norm + w_ret * retention

        Args:
            memory: Memory dict with all scoring fields.

        Returns:
            Priority score (0.0-1.0).

        Optimization: Delegates to a cached helper that takes hashable primitives
        with time-bucketed now value. Combined with the cached retention_strength,
        this dramatically reduces recomputation during bulk eviction scoring.
        """
        now_bucket = int(_now())  # 1-second granularity
        # Compute retention through the cached path
        retention = self.compute_retention_strength(memory)
        weights = self._config.priority_weights
        return _cached_priority_score(
            memory["importance"],
            memory["created_at"],
            memory["access_count"],
            retention,
            weights["importance"],
            weights["recency"],
            weights["access_count"],
            weights["retention"],
            now_bucket,
        )

    @_synchronized
    def evict(self) -> int:
        """Evict low-priority memories via soft delete.

        Removes memories with the lowest priority scores up to
        the configured eviction budget, then prunes the access log so it does
        not grow without bound.

        Returns:
            Number of memories evicted.
        """
        assert self._conn is not None
        try:
            cursor = self._conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
            total = cursor.fetchone()[0]
            if total == 0:
                return 0

            budget = max(1, int(total * self._config.eviction_budget))

            cursor = self._conn.execute(
                """SELECT id, importance, created_at, accessed_at,
                          access_count, retention_strength
                   FROM memories WHERE is_deleted = 0"""
            )
            rows = cursor.fetchall()
            memories = []
            for row in rows:
                mem = {
                    "id": row[0],
                    "importance": row[1],
                    "created_at": row[2],
                    "accessed_at": row[3],
                    "access_count": row[4],
                    "retention_strength": row[5],
                }
                mem["priority"] = self.compute_priority_score(mem)
                memories.append(mem)

            memories.sort(key=lambda m: m["priority"])
            to_evict = memories[:budget]

            now = _now()
            evicted = 0
            for mem in to_evict:
                self._conn.execute(
                    "UPDATE memories SET is_deleted = 1, deleted_at = ? WHERE id = ?",
                    (now, mem["id"]),
                )
                evicted += 1

            self._commit()
            self.prune_access_log()
            logger.info("Evicted %d low-priority memories", evicted)
            return evicted

        except sqlite3.Error as e:
            logger.error("Eviction error: %s", e)
            self._conn.rollback()
            return 0

    @_synchronized
    def consolidate(self) -> int:
        """Consolidate similar memories.

        Groups memories whose token sets exceed ``consolidation_threshold``
        (Jaccard) and, for each group, creates a single merged memory (with the
        mean of the members' embeddings so it stays vector-searchable), then
        soft-deletes the originals and links them to the consolidated memory.
        Previously this only wrote a dangling summary row and left every
        original in place, so nothing was ever actually consolidated.

        Returns:
            Number of consolidation groups created.
        """
        assert self._conn is not None
        dim = self._config.embedding_dim
        try:
            cursor = self._conn.execute(
                """SELECT id, content, importance, embedding, created_at,
                          namespace, scope, user_id, agent_id, session_id, project_id
                   FROM memories
                   WHERE is_deleted = 0
                   ORDER BY created_at DESC
                   LIMIT 5000"""
            )
            rows = cursor.fetchall()
            if len(rows) < self._config.consolidation_min_group:
                return 0

            token_sets: dict[int, set[str]] = {}
            memories: list[dict[str, Any]] = []
            for row in rows:
                mid = row[0]
                token_sets[mid] = set(row[1].lower().split())
                memories.append(
                    {
                        "id": mid,
                        "content": row[1],
                        "importance": row[2],
                        "embedding": row[3],
                        "namespace": row[5],
                        "scope": row[6],
                        "user_id": row[7],
                        "agent_id": row[8],
                        "session_id": row[9],
                        "project_id": row[10],
                    }
                )

            # Consolidate only within an identical isolation key. Never merge
            # data across namespaces, users, projects, agents, or sessions.
            grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            for memory in memories:
                key = (
                    memory["namespace"],
                    memory["scope"],
                    memory["user_id"],
                    memory["agent_id"],
                    memory["session_id"],
                    memory["project_id"],
                )
                grouped.setdefault(key, []).append(memory)

            # Greedy grouping by pairwise Jaccard similarity within each key.
            used: set[int] = set()
            groups: list[list[dict[str, Any]]] = []
            for scoped_memories in grouped.values():
                for i, mem_a in enumerate(scoped_memories):
                    if mem_a["id"] in used:
                        continue
                    group = [mem_a]
                    used.add(mem_a["id"])
                    for j in range(i + 1, len(scoped_memories)):
                        mem_b = scoped_memories[j]
                        if mem_b["id"] in used:
                            continue
                        sim = _jaccard_similarity(token_sets[mem_a["id"]], token_sets[mem_b["id"]])
                        if sim >= self._config.consolidation_threshold:
                            group.append(mem_b)
                            used.add(mem_b["id"])
                    if len(group) >= self._config.consolidation_min_group:
                        groups.append(group)

            consolidated = 0
            for group in groups:
                merged_content = " | ".join(m["content"] for m in group)
                importance = max(m["importance"] for m in group)
                group_ids = [m["id"] for m in group]

                # Mean embedding across members that carry one of the right dim.
                vecs = []
                for m in group:
                    blob = m["embedding"]
                    if blob is None:
                        continue
                    v = np.frombuffer(blob, dtype=np.float32)
                    if v.shape[0] == dim:
                        vecs.append(v)
                mean_emb = np.mean(np.vstack(vecs), axis=0) if vecs else None

                new = self.add_memory(
                    merged_content,
                    embedding=mean_emb,
                    memory_type="semantic",
                    importance=importance,
                    metadata={"consolidated_from": group_ids},
                    namespace=group[0]["namespace"],
                    scope=group[0]["scope"],
                    user_id=group[0]["user_id"],
                    agent_id=group[0]["agent_id"],
                    session_id=group[0]["session_id"],
                    project_id=group[0]["project_id"],
                )
                new_id = new["memory_id"]

                self._conn.execute(
                    """INSERT INTO consolidations
                       (memory_ids, consolidated_content, consolidated_importance, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (json.dumps(group_ids), merged_content, importance, _now()),
                )

                now = _now()
                for mid in group_ids:
                    self._conn.execute(
                        "UPDATE memories SET is_deleted = 1, deleted_at = ? WHERE id = ?",
                        (now, mid),
                    )
                    self._conn.execute(
                        """INSERT OR IGNORE INTO memory_links
                           (source_id, target_id, link_type, strength, created_at)
                           VALUES (?, ?, 'consolidated', 1.0, ?)""",
                        (new_id, mid, now),
                    )
                    # The retired original's vector is dropped from the live index.
                    if self._faiss_index is not None:
                        try:
                            self._faiss_index.remove_ids(np.asarray([mid], dtype=np.int64))
                        except Exception:  # pragma: no cover - depends on index state
                            pass
                consolidated += 1

            self._commit()
            logger.info("Consolidated %d groups", consolidated)
            return consolidated

        except sqlite3.Error as e:
            logger.error("Consolidation error: %s", e)
            self._conn.rollback()
            return 0

    @_synchronized
    def prune_access_log(self, keep_per_memory: int | None = None) -> int:
        """Keep only the most recent ``keep_per_memory`` access_log rows per memory.

        The access log gains a row on every recall; without pruning it grows
        without bound. Returns the number of rows deleted.
        """
        assert self._conn is not None
        keep = keep_per_memory or self._config.max_access_log_per_memory
        try:
            cursor = self._conn.execute(
                """DELETE FROM access_log
                   WHERE id IN (
                       SELECT id FROM (
                           SELECT id, ROW_NUMBER() OVER (
                               PARTITION BY memory_id
                               ORDER BY accessed_at DESC, id DESC
                           ) AS rn
                           FROM access_log
                       ) WHERE rn > ?
                   )""",
                (keep,),
            )
            deleted = cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
            self._commit()
            if deleted:
                logger.info("Pruned %d old access_log rows", deleted)
            return deleted
        except sqlite3.Error as e:
            logger.error("Access log prune error: %s", e)
            self._conn.rollback()
            return 0

    @_synchronized
    def purge_deleted(self, older_than_seconds: float = 0.0) -> int:
        """Permanently remove soft-deleted memories (and their vectors/links).

        ``older_than_seconds`` keeps recently soft-deleted rows recoverable; pass
        0 to purge everything currently marked deleted. Returns the count purged.
        """
        assert self._conn is not None
        cutoff = _now() - older_than_seconds
        try:
            cursor = self._conn.execute(
                """SELECT id FROM memories
                   WHERE is_deleted = 1 AND (deleted_at IS NULL OR deleted_at <= ?)""",
                (cutoff,),
            )
            ids = [int(r[0]) for r in cursor.fetchall()]
            for mid in ids:
                # Children first (foreign_keys=ON), then the memory row itself.
                self._conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (mid,))
                self._conn.execute(
                    "DELETE FROM memory_links WHERE source_id = ? OR target_id = ?", (mid, mid)
                )
                self._conn.execute("DELETE FROM access_log WHERE memory_id = ?", (mid,))
                self._conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
                if self._faiss_index is not None:
                    try:
                        self._faiss_index.remove_ids(np.asarray([mid], dtype=np.int64))
                    except Exception:  # pragma: no cover - depends on index state
                        pass
            self._commit()
            logger.info("Purged %d soft-deleted memories", len(ids))
            return len(ids)
        except sqlite3.Error as e:
            logger.error("Purge error: %s", e)
            self._conn.rollback()
            return 0

    @_synchronized
    def stats(self) -> dict[str, Any]:
        """Get comprehensive database statistics.

        Returns:
            Dict with counts, sizes, index info, and performance metrics.
        """
        assert self._conn is not None
        try:
            result: dict[str, Any] = {}

            # Memory counts
            cursor = self._conn.execute("SELECT COUNT(*) FROM memories")
            result["total_memories"] = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
            result["active_memories"] = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 1")
            result["deleted_memories"] = cursor.fetchone()[0]

            # By type
            cursor = self._conn.execute(
                """SELECT memory_type, COUNT(*) FROM memories
                   WHERE is_deleted = 0 GROUP BY memory_type"""
            )
            result["by_type"] = {row[0]: row[1] for row in cursor.fetchall()}

            # By namespace (active rows)
            cursor = self._conn.execute(
                """SELECT namespace, COUNT(*) FROM memories
                   WHERE is_deleted = 0 GROUP BY namespace"""
            )
            result["by_namespace"] = {row[0]: row[1] for row in cursor.fetchall()}

            # Entity counts
            cursor = self._conn.execute("SELECT COUNT(*) FROM entities")
            result["total_entities"] = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(*) FROM edges")
            result["total_edges"] = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(*) FROM memory_links")
            result["total_memory_links"] = cursor.fetchone()[0]

            # Consolidation count
            cursor = self._conn.execute("SELECT COUNT(*) FROM consolidations")
            result["total_consolidations"] = cursor.fetchone()[0]

            # FAISS index
            if self._faiss_index is not None:
                result["faiss_vectors"] = self._faiss_index.ntotal
                base_index = self._faiss_index
                if isinstance(base_index, faiss.IndexIDMap2):
                    base_index = faiss.downcast_index(base_index.index)
                result["faiss_type"] = type(base_index).__name__
                result["faiss_dimension"] = self._config.embedding_dim
            else:
                result["faiss_vectors"] = 0
                result["faiss_type"] = "none"
                result["faiss_dimension"] = self._config.embedding_dim

            # Average importance
            cursor = self._conn.execute("SELECT AVG(importance) FROM memories WHERE is_deleted = 0")
            avg_row = cursor.fetchone()
            result["avg_importance"] = round(float(avg_row[0] or 0.0), 4)

            # Database file size
            db_path = Path(str(self._config.db_path))
            if db_path.exists():
                result["db_size_bytes"] = db_path.stat().st_size
            else:
                result["db_size_bytes"] = 0

            return result

        except sqlite3.Error as e:
            logger.error("Stats error: %s", e)
            return {"error": str(e)}

    # ── Episode / Provenance / Feedback ─────────────────────────────────

    @_synchronized
    def add_episode(
        self,
        content: str,
        role: str,
        source: str | None = None,
        event_at: float | None = None,
        metadata: dict[str, Any] | None = None,
        namespace: str = "default",
        scope: str = "session",
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Add an immutable episode (raw evidence)."""
        assert self._conn is not None
        now = _now()
        # event_at is NOT NULL in the schema; default to "now" when the caller
        # omits it instead of letting the INSERT violate the constraint.
        event_time = now if event_at is None else event_at
        try:
            cursor = self._conn.execute(
                """INSERT INTO episodes
                   (content, role, source, event_at, metadata, namespace, scope,
                    user_id, agent_id, session_id, project_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    content,
                    role,
                    source,
                    event_time,
                    json.dumps(metadata) if metadata else None,
                    namespace,
                    scope,
                    user_id,
                    agent_id,
                    session_id,
                    project_id,
                    now,
                ),
            )
            episode_id = cursor.lastrowid
            self._commit()
            return {"episode_id": episode_id, "status": "created"}
        except sqlite3.Error as e:
            logger.error("Error adding episode: %s", e)
            self._conn.rollback()
            raise

    @_synchronized
    def expand_by_entities(
        self,
        seed_ids: list[int],
        limit: int = 10,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find active memories that share entities with the seed memories.

        This is the associative step of recall: given direct hits, surface the
        knowledge-graph neighbourhood without a full graph traversal. Candidates
        are ranked by the number of entities shared with the seeds. Returned
        dicts carry ``shared_entities`` and ``search_type='graph_expansion'``
        (no ``score`` — the caller scales relative to the seeds).
        """
        assert self._conn is not None
        seeds = [int(i) for i in dict.fromkeys(seed_ids)]
        if not seeds:
            return []
        seed_placeholders = ",".join("?" * len(seeds))
        sql = f"""SELECT me2.memory_id, COUNT(DISTINCT me2.entity_id) AS shared
                  FROM memory_entities AS me1
                  JOIN memory_entities AS me2
                    ON me1.entity_id = me2.entity_id
                   AND me2.memory_id != me1.memory_id
                  JOIN memories AS m ON m.id = me2.memory_id
                  WHERE me1.memory_id IN ({seed_placeholders})
                    AND m.is_deleted = 0"""
        params: list[Any] = list(seeds)
        if namespace is not None:
            sql += " AND m.namespace = ?"
            params.append(namespace)
        sql += " GROUP BY me2.memory_id ORDER BY shared DESC, me2.memory_id ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        candidates = {int(r[0]): int(r[1]) for r in rows}
        if not candidates:
            return []
        row_map = self.get_memories_bulk(list(candidates.keys()))
        results: list[dict[str, Any]] = []
        for mid, shared in candidates.items():
            memory = row_map.get(mid)
            if memory is None or mid in seeds:
                continue
            memory["shared_entities"] = shared
            memory["search_type"] = "graph_expansion"
            results.append(memory)
        results.sort(key=lambda mem: (-mem["shared_entities"], mem["id"]))
        return results

    @_synchronized
    def search_episodes(
        self,
        query: str,
        k: int = 10,
        namespace: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search raw episode history by keyword (BM25 over ``episodes_fts``).

        This is the "search your past sessions" surface: it finds the actual
        turns, not the distilled memories, so agents can recover detail that
        extraction dropped. Precision first (AND), then OR fallback, mirroring
        ``fts_search``.
        """
        assert self._conn is not None
        try:
            for op in ("AND", "OR"):
                fts_query = _fts_escape(query, op=op)
                sql = """SELECT e.id, e.content, e.role, e.source, e.event_at,
                                e.metadata, e.namespace, e.session_id,
                                e.created_at, ABS(f.rank) AS score
                         FROM episodes_fts AS f
                         JOIN episodes AS e ON e.id = f.rowid
                         WHERE f.episodes_fts MATCH ?"""
                params: list[Any] = [fts_query]
                if namespace is not None:
                    sql += " AND e.namespace = ?"
                    params.append(namespace)
                if session_id is not None:
                    sql += " AND e.session_id = ?"
                    params.append(session_id)
                sql += " ORDER BY score DESC, e.id ASC LIMIT ?"
                params.append(k)
                rows = self._conn.execute(sql, params).fetchall()
                if rows:
                    break
            return [
                {
                    "id": r[0],
                    "content": r[1],
                    "role": r[2],
                    "source": r[3],
                    "event_at": r[4],
                    "metadata": json.loads(r[5]) if r[5] else None,
                    "namespace": r[6],
                    "session_id": r[7],
                    "created_at": r[8],
                    "score": float(r[9]),
                }
                for r in rows
            ]
        except sqlite3.Error as e:
            logger.error("Episode search error: %s", e)
            return []

    @_synchronized
    def get_episodes(
        self,
        session_id: str | None = None,
        namespace: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """List episodes (oldest first), optionally scoped to a session."""
        assert self._conn is not None
        sql = """SELECT id, content, role, source, event_at, metadata,
                        namespace, session_id, created_at
                 FROM episodes"""
        params: list[Any] = []
        conditions = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if namespace is not None:
            conditions.append("namespace = ?")
            params.append(namespace)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY event_at ASC, id ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "id": r[0],
                "content": r[1],
                "role": r[2],
                "source": r[3],
                "event_at": r[4],
                "metadata": json.loads(r[5]) if r[5] else None,
                "namespace": r[6],
                "session_id": r[7],
                "created_at": r[8],
            }
            for r in rows
        ]

    @_synchronized
    def list_sessions(
        self, namespace: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Summarize recorded sessions: turn counts and time ranges, newest first."""
        assert self._conn is not None
        sql = """SELECT session_id, COUNT(*) AS turns,
                        MIN(event_at) AS first_at, MAX(event_at) AS last_at
                 FROM episodes
                 WHERE session_id IS NOT NULL"""
        params: list[Any] = []
        if namespace is not None:
            sql += " AND namespace = ?"
            params.append(namespace)
        sql += " GROUP BY session_id ORDER BY last_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "session_id": r[0],
                "turns": int(r[1]),
                "first_event_at": r[2],
                "last_event_at": r[3],
            }
            for r in rows
        ]

    @_synchronized
    def list_session_digests(
        self, namespace: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return the most recent session-digest memories (newest first).

        Digests are ordinary memories whose metadata carries
        ``{"kind": "session_digest", "session_id": ...}``; JSON1 extracts the
        kind so the filter never depends on serialized-string formatting.
        """
        assert self._conn is not None
        sql = """SELECT id, content, created_at, event_at, namespace,
                        metadata, importance, memory_type
                 FROM memories
                 WHERE is_deleted = 0
                   AND json_extract(metadata, '$.kind') = 'session_digest'"""
        params: list[Any] = []
        if namespace is not None:
            sql += " AND namespace = ?"
            params.append(namespace)
        sql += " ORDER BY COALESCE(event_at, created_at) DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "id": r[0],
                "content": r[1],
                "created_at": r[2],
                "event_at": r[3],
                "namespace": r[4],
                "metadata": json.loads(r[5]) if r[5] else None,
                "importance": r[6],
                "memory_type": r[7],
            }
            for r in rows
        ]

    @_synchronized
    def add_source(
        self,
        memory_id: int,
        episode_id: int | None,
        source: str,
        source_id: str | None = None,
        span: str | None = None,
        confidence: float = 1.0,
    ) -> int:
        """Attach a provenance source record to a memory."""
        assert self._conn is not None
        now = _now()
        try:
            cursor = self._conn.execute(
                """INSERT INTO sources
                   (memory_id, episode_id, source, source_id, span, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (memory_id, episode_id, source, source_id, span, _clamp01(confidence), now),
            )
            source_id_val = cursor.lastrowid
            self._commit()
            return source_id_val
        except sqlite3.Error as e:
            logger.error("Error adding source: %s", e)
            self._conn.rollback()
            raise

    @_synchronized
    def add_feedback(
        self,
        memory_id: int,
        action: str,
        confidence_delta: float = 0.0,
        note: str | None = None,
        actor: str | None = None,
    ) -> int:
        """Record user/agent feedback on a memory."""
        assert self._conn is not None
        now = _now()
        try:
            cursor = self._conn.execute(
                """INSERT INTO feedback
                   (memory_id, action, confidence_delta, note, actor, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (memory_id, action, confidence_delta, note, actor, now),
            )
            feedback_id = cursor.lastrowid
            self._commit()
            # Optionally adjust memory confidence based on feedback
            if confidence_delta != 0.0:
                self._conn.execute(
                    """UPDATE memories SET confidence = MIN(1.0, MAX(0.0, confidence + ?))
                       WHERE id = ? AND is_deleted = 0""",
                    (confidence_delta, memory_id),
                )
                self._commit()
            return feedback_id
        except sqlite3.Error as e:
            logger.error("Error adding feedback: %s", e)
            self._conn.rollback()
            raise

    @_synchronized
    def adjust_confidence(self, memory_id: int, delta: float) -> float | None:
        """Shift a memory's confidence by ``delta`` (clamped to [0, 1]).

        This is the trust-scoring primitive: recall reinforcement, contradiction
        penalties, and explicit agent feedback all flow through here, so trust
        evolves as memories are confirmed or contested instead of being a static
        write-time guess. Returns the new confidence, or None if not found.
        """
        assert self._conn is not None
        try:
            cursor = self._conn.execute(
                """UPDATE memories
                   SET confidence = MIN(1.0, MAX(0.0, COALESCE(confidence, 1.0) + ?))
                   WHERE id = ? AND is_deleted = 0""",
                (float(delta), memory_id),
            )
            if cursor.rowcount == 0:
                return None
            self._commit()
            row = self._conn.execute(
                "SELECT confidence FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            return float(row[0]) if row is not None else None
        except sqlite3.Error as e:
            logger.error("Error adjusting confidence for %d: %s", memory_id, e)
            self._conn.rollback()
            return None

    @_synchronized
    def get_feedback(self, memory_id: int) -> list[dict[str, Any]]:
        """Get all feedback for a memory."""
        assert self._conn is not None
        rows = self._conn.execute(
            """SELECT id, action, confidence_delta, note, actor, created_at
               FROM feedback WHERE memory_id = ? ORDER BY created_at DESC""",
            (memory_id,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "action": r[1],
                "confidence_delta": r[2],
                "note": r[3],
                "actor": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    @_synchronized
    def get_feedback_bulk(self, memory_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        """Fetch feedback for several memories in one query (keyed by memory id)."""
        assert self._conn is not None
        unique_ids = list(dict.fromkeys(int(i) for i in memory_ids))
        if not unique_ids:
            return {}
        placeholders = ",".join("?" * len(unique_ids))
        rows = self._conn.execute(
            f"""SELECT memory_id, id, action, confidence_delta, note, actor, created_at
                FROM feedback WHERE memory_id IN ({placeholders})
                ORDER BY created_at DESC""",
            unique_ids,
        ).fetchall()
        result: dict[int, list[dict[str, Any]]] = {mid: [] for mid in unique_ids}
        for r in rows:
            result[int(r[0])].append(
                {
                    "id": r[1],
                    "action": r[2],
                    "confidence_delta": r[3],
                    "note": r[4],
                    "actor": r[5],
                    "created_at": r[6],
                }
            )
        return result

    @_synchronized
    def recall_with_temporal(
        self,
        query: str,
        embedding: np.ndarray | None = None,
        k: int = 10,
        namespace: str | None = None,
        as_of: float | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memories with temporal filtering (as_of parameter)."""
        assert self._conn is not None
        try:
            # Base search
            if embedding is not None:
                results = self.hybrid_search(
                    query, embedding=embedding, k=k * 3, namespace=namespace
                )
            else:
                results = self.fts_search(query, k=k * 3, namespace=namespace)

            # Temporal filtering
            if as_of is not None:
                filtered = []
                for mem in results:
                    if mem.get("is_deleted"):
                        continue
                    valid_from = mem.get("valid_from")
                    valid_to = mem.get("valid_to")
                    if valid_from is not None and valid_from > as_of:
                        continue
                    if valid_to is not None and valid_to <= as_of:
                        continue
                    filtered.append(mem)
                results = filtered

            return results[:k]
        except Exception as e:
            logger.error("Temporal recall error: %s", e)
            return []

    @_synchronized
    def get_supersession_chain(self, memory_id: int) -> list[dict[str, Any]]:
        """Follow supersedes_id chain to get history of a memory."""
        assert self._conn is not None
        chain = []
        current_id = memory_id
        while current_id is not None:
            mem = self.get_memory(current_id)
            if mem is None:
                break
            chain.append(mem)
            current_id = mem.get("supersedes_id")
        return chain

    @_synchronized
    def get_sources_for_memory(self, memory_id: int) -> list[dict[str, Any]]:
        """Get all provenance sources for a memory."""
        assert self._conn is not None
        rows = self._conn.execute(
            """SELECT s.id, s.episode_id, s.source, s.source_id, s.span,
                      s.confidence, s.created_at, e.content as episode_content
               FROM sources s
               LEFT JOIN episodes e ON e.id = s.episode_id
               WHERE s.memory_id = ?
               ORDER BY s.created_at""",
            (memory_id,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "episode_id": r[1],
                "source": r[2],
                "source_id": r[3],
                "span": r[4],
                "confidence": r[5],
                "created_at": r[6],
                "episode_content": r[7],
            }
            for r in rows
        ]

    @_synchronized
    def get_sources_bulk(self, memory_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        """Fetch provenance sources for several memories in one query."""
        assert self._conn is not None
        unique_ids = list(dict.fromkeys(int(i) for i in memory_ids))
        if not unique_ids:
            return {}
        placeholders = ",".join("?" * len(unique_ids))
        rows = self._conn.execute(
            f"""SELECT s.memory_id, s.id, s.episode_id, s.source, s.source_id,
                       s.span, s.confidence, s.created_at, e.content
                FROM sources AS s
                LEFT JOIN episodes AS e ON e.id = s.episode_id
                WHERE s.memory_id IN ({placeholders})
                ORDER BY s.created_at""",
            unique_ids,
        ).fetchall()
        result: dict[int, list[dict[str, Any]]] = {mid: [] for mid in unique_ids}
        for r in rows:
            result[int(r[0])].append(
                {
                    "id": r[1],
                    "episode_id": r[2],
                    "source": r[3],
                    "source_id": r[4],
                    "span": r[5],
                    "confidence": r[6],
                    "created_at": r[7],
                    "episode_content": r[8],
                }
            )
        return result

    # ── Core memory blocks (Letta-style always-in-context state) ──────────

    @staticmethod
    def _core_block_row(row: Any) -> dict[str, Any]:
        return {
            "id": int(row[0]),
            "name": str(row[1]),
            "content": str(row[2]),
            "namespace": str(row[3]),
            "created_at": float(row[4]),
            "updated_at": float(row[5]),
        }

    @_synchronized
    def core_block_set(
        self, name: str, content: str, namespace: str = "default"
    ) -> dict[str, Any]:
        """Upsert one core memory block and return it."""
        assert self._conn is not None
        now = _now()
        self._conn.execute(
            """INSERT INTO core_memory_blocks (name, content, namespace, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (name, namespace)
               DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at""",
            (name, content, namespace, now, now),
        )
        self._commit()
        block = self.core_block_get(name, namespace)
        assert block is not None
        return block

    @_synchronized
    def core_block_get(
        self, name: str, namespace: str = "default"
    ) -> dict[str, Any] | None:
        """Return one core memory block, or None."""
        assert self._conn is not None
        row = self._conn.execute(
            """SELECT id, name, content, namespace, created_at, updated_at
               FROM core_memory_blocks WHERE name = ? AND namespace = ?""",
            (name, namespace),
        ).fetchone()
        return self._core_block_row(row) if row is not None else None

    @_synchronized
    def core_block_append(
        self,
        name: str,
        text: str,
        namespace: str = "default",
        char_limit: int = 10000,
    ) -> dict[str, Any]:
        """Append text to a core block (created if missing), bounded by ``char_limit``.

        Letta-style agents append observations continuously; the bound keeps a
        runaway loop from growing the block past what fits in a prompt. When
        the append would exceed the limit, the *oldest* content is dropped
        from the front so the most recent observations survive.
        """
        assert self._conn is not None
        existing = self.core_block_get(name, namespace)
        content = (existing["content"] if existing else "") + text
        if len(content) > char_limit:
            content = content[len(content) - char_limit :]
        return self.core_block_set(name, content, namespace)

    @_synchronized
    def core_block_delete(self, name: str, namespace: str = "default") -> bool:
        """Delete one core memory block. Returns True if it existed."""
        assert self._conn is not None
        cursor = self._conn.execute(
            "DELETE FROM core_memory_blocks WHERE name = ? AND namespace = ?",
            (name, namespace),
        )
        deleted = cursor.rowcount > 0
        self._commit()
        return deleted

    @_synchronized
    def core_blocks_list(self, namespace: str = "default") -> list[dict[str, Any]]:
        """List a namespace's core memory blocks, ordered by name."""
        assert self._conn is not None
        rows = self._conn.execute(
            """SELECT id, name, content, namespace, created_at, updated_at
               FROM core_memory_blocks WHERE namespace = ? ORDER BY name""",
            (namespace,),
        ).fetchall()
        return [self._core_block_row(row) for row in rows]

    # ── Entity resolution ─────────────────────────────────────────────────

    @_synchronized
    def merge_entities(self, source: str, target: str) -> int:
        """Merge entity ``source`` into ``target`` (alias resolution).

        Re-points every ``memory_entities`` link and both directions of every
        graph edge onto the target entity (existing links win on conflict),
        then removes the source entity. Returns how many links were moved.
        """
        assert self._conn is not None
        source_id = self._get_or_create_entity(source)
        target_id = self._get_or_create_entity(target)
        if source_id == target_id:
            return 0
        moved = 0
        try:
            cur = self._conn.execute(
                """INSERT OR IGNORE INTO memory_entities (memory_id, entity_id)
                   SELECT memory_id, ? FROM memory_entities WHERE entity_id = ?""",
                (target_id, source_id),
            )
            moved += cur.rowcount
            self._conn.execute("DELETE FROM memory_entities WHERE entity_id = ?", (source_id,))

            # Both edge directions, honouring the (source, target, type)
            # unique index: insert-or-ignore first, then delete the originals.
            cur = self._conn.execute(
                """INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, weight, created_at)
                   SELECT ?, target_id, edge_type, weight, created_at
                   FROM edges WHERE source_id = ?""",
                (target_id, source_id),
            )
            moved += cur.rowcount
            self._conn.execute("DELETE FROM edges WHERE source_id = ?", (source_id,))

            cur = self._conn.execute(
                """INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, weight, created_at)
                   SELECT source_id, ?, edge_type, weight, created_at
                   FROM edges WHERE target_id = ?""",
                (target_id, source_id),
            )
            moved += cur.rowcount
            self._conn.execute("DELETE FROM edges WHERE target_id = ?", (source_id,))

            self._conn.execute("DELETE FROM entities WHERE id = ?", (source_id,))
            self._commit()
        except sqlite3.Error as e:
            logger.error("Error merging entities %r -> %r: %s", source, target, e)
            self._conn.rollback()
            raise
        return moved


@lru_cache(maxsize=4096)
def _cached_retention_strength(
    accessed_at: float,
    importance: float,
    retention_strength: float,
    retention_half_life: float,
    now_bucket: int,
) -> float:
    """Cached Ebbinghaus retention score computation (module-level for lru_cache).

    Stability S = half_life * importance * retention_strength. retention_strength
    starts at 1.0 and is multiplied up on each access (capped), so repeatedly
    recalled memories decay more slowly.

    Args:
        accessed_at: Last access timestamp.
        importance: Memory importance score.
        retention_strength: Accrued stability multiplier (>= 0).
        retention_half_life: Configured half-life in seconds.
        now_bucket: Current time floored to integer seconds.
    """
    time_since_access = float(now_bucket) - accessed_at
    half_life = retention_half_life * max(importance, 0.0) * max(retention_strength, 0.0)
    if half_life <= 0:
        return 0.0
    return math.exp(-time_since_access / half_life)


@lru_cache(maxsize=4096)
def _cached_priority_score(
    importance: float,
    created_at: float,
    access_count: int,
    retention: float,
    w_importance: float,
    w_recency: float,
    w_access: float,
    w_retention: float,
    now_bucket: int,
) -> float:
    """Cached priority score computation (module-level for lru_cache).

    Args:
        importance: Memory importance score.
        created_at: Creation timestamp.
        access_count: Number of times accessed.
        retention: Pre-computed retention strength.
        w_importance, w_recency, w_access, w_retention: Priority weights.
        now_bucket: Current time floored to integer seconds.
    """
    age = float(now_bucket) - created_at
    recency = 1.0 / (1.0 + age / 86400.0)  # Normalized to days
    access_norm = min(1.0, access_count / 100.0)

    score = (
        w_importance * importance
        + w_recency * recency
        + w_access * access_norm
        + w_retention * retention
    )
    return round(score, 6)
