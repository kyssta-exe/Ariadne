"""Hermes-compatible memory interface for Ariadne.

Wraps AriadneDB and Deduplicator into a clean API matching
the Hermes agent memory protocol.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from arriadne.config import AriadneConfig
from arriadne.dedup import ContradictionDetector, Deduplicator
from arriadne.embeddings import Embedder, resolve_embedder
from arriadne.rerank import Reranker, resolve_reranker
from arriadne.storage import AriadneDB, _now

logger = logging.getLogger(__name__)

# Session-digest scoring helpers: 3+ char word tokens, and per-role salience
# weights (assistant turns tend to carry decisions/conclusions).
_WORD_RE = re.compile(r"\w{3,}")
_ROLE_WEIGHTS = {"assistant": 1.2, "turn": 1.1, "user": 1.0, "tool": 0.8, "system": 0.6}


class AriadneMemory:
    """Hermes-compatible memory interface.

    Provides remember, recall, forget, update, graph, and stats methods
    that match the Hermes agent memory protocol.

    Args:
        config: AriadneConfig instance, or None for defaults.
        db_path: Alternative way to set database path.
        embedding_dim: Alternative way to set embedding dimension.

    Example:
        >>> mem = AriadneMemory(db_path="my_memory.db", embedding_dim=384)
        >>> result = mem.remember("Paris is the capital of France", importance=0.9)
        >>> results = mem.recall("capital of France", k=5)
    """

    def __init__(
        self,
        config: AriadneConfig | None = None,
        db_path: str | Path | None = None,
        embedding_dim: int | None = None,
        embedder: Embedder | str | None = None,
        reranker: Reranker | str | None = None,
    ) -> None:
        if config is None:
            kwargs: dict[str, Any] = {}
            if db_path is not None:
                kwargs["db_path"] = db_path
            if embedding_dim is not None:
                kwargs["embedding_dim"] = embedding_dim
            config = AriadneConfig(**kwargs)

        self._config = config
        # Reentrant lock guarding the (not thread-safe) MinHash dedup index.
        # AriadneDB has its own lock for SQLite + FAISS.
        self._lock = threading.RLock()
        # Per-instance auto-maintenance write counter.
        self._write_count = 0
        # Writes between auto-maintenance cycles (configurable, was hardcoded 50).
        self._maintenance_interval = max(1, int(config.maintenance_interval))
        self._maintenance_cycles = 0

        # Optional embedder: when set, remember()/recall() auto-embed content and
        # queries so semantic search works without the caller wiring vectors.
        self._embedder = resolve_embedder(embedder)
        if self._embedder is not None:
            emb_dim = getattr(self._embedder, "dim", None)
            if emb_dim is not None and emb_dim != config.embedding_dim:
                raise ValueError(
                    f"embedder output dim {emb_dim} != config.embedding_dim "
                    f"{config.embedding_dim}; set embedding_dim={emb_dim}"
                )

        # Optional cross-encoder reranker for recall(..., rerank=True). May
        # stay None and be lazily loaded on first use from rerank_model.
        self._reranker: Reranker | None = resolve_reranker(reranker)
        self._reranker_unavailable = False

        self._db = AriadneDB(config)
        self._dedup_by_namespace: dict[str, Deduplicator] = {}
        self._contradiction_detector = ContradictionDetector()
        self._db.open()
        self._load_dedup_from_db()
        logger.info("AriadneMemory initialized (db=%s)", config.db_path)

    def _get_reranker(self) -> Reranker | None:
        """Return the configured reranker, lazily loading the default model.

        Returns None (and remembers the failure) when sentence-transformers
        is not installed, so ``rerank=True`` degrades to fused order instead
        of crashing — the dependency stays optional.
        """
        if self._reranker is not None:
            return self._reranker
        if self._reranker_unavailable:
            return None
        from arriadne.rerank import CrossEncoderReranker

        try:
            self._reranker = CrossEncoderReranker(self._config.rerank_model)
        except ImportError:
            logger.warning(
                "recall(rerank=True) requested but sentence-transformers is not "
                "installed; returning fused order. Install with: "
                'pip install "arriadne[embeddings]"'
            )
            self._reranker_unavailable = True
            return None
        return self._reranker

    def _dedup_for(self, namespace: str) -> Deduplicator:
        """Return the near-duplicate index for one namespace."""
        if namespace not in self._dedup_by_namespace:
            self._dedup_by_namespace[namespace] = Deduplicator(
                threshold=self._config.dedup_threshold,
                num_perm=self._config.dedup_num_perm,
            )
        return self._dedup_by_namespace[namespace]

    def _load_dedup_from_db(self) -> None:
        """Rebuild the in-memory MinHash dedup index from stored memories.

        The Deduplicator lives only in memory. Without this rebuild on open,
        near-duplicate detection would silently reset to empty on every restart,
        so only exact (hash-identical) duplicates would be caught across runs.

        As a startup fast path, a previously persisted state file
        (``<db>.dedup.pkl``) is loaded when its fingerprint matches the
        database — reconstructing MinHashes from stored hash values skips the
        per-document hashing that dominates restart time on large stores.
        """
        import pickle

        try:
            sidecar = self._dedup_sidecar_path()
            if sidecar is not None and sidecar.exists():
                with sidecar.open("rb") as fh:
                    state = pickle.load(fh)
                fingerprint = self._active_fingerprint()
                if (
                    state.get("fingerprint") == fingerprint
                    and state.get("threshold") == self._config.dedup_threshold
                    and state.get("num_perm") == self._config.dedup_num_perm
                ):
                    rows = self._db.conn.execute(
                        "SELECT id, content, namespace FROM memories WHERE is_deleted = 0"
                    ).fetchall()
                    contents_by_ns: dict[str, dict[str, str]] = {}
                    for row in rows:
                        contents_by_ns.setdefault(row[2], {})[str(row[0])] = row[1]
                    for ns, ns_state in state.get("namespaces", {}).items():
                        dedup = self._dedup_for(ns)
                        try:
                            dedup.restore_state(ns_state, contents_by_ns.get(ns, {}))
                        except Exception as exc:
                            logger.warning(
                                "Dedup sidecar invalid (%s); rebuilding", exc
                            )
                            self._dedup_by_namespace = {}
                            break
                    total = sum(d.size for d in self._dedup_by_namespace.values())
                    if total:
                        logger.info(
                            "Loaded dedup index from sidecar (%d memories)", total
                        )
                        return
        except Exception as e:
            logger.warning("Failed to load dedup sidecar: %s", e)

        try:
            cursor = self._db.conn.execute(
                "SELECT id, content, namespace FROM memories WHERE is_deleted = 0"
            )
            count = 0
            for row in cursor.fetchall():
                self._dedup_for(row[2]).add(row[1], doc_id=str(row[0]))
                count += 1
            if count:
                logger.info("Loaded %d memories into dedup index", count)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to load dedup index from DB: %s", e)

    def _dedup_sidecar_path(self) -> Path | None:
        """Path of the persisted dedup state, or None for in-memory DBs."""
        db_path = str(self._config.db_path)
        if db_path in (":memory:", ""):
            return None
        return Path(db_path + ".dedup.pkl")

    def _active_fingerprint(self) -> str:
        """Fingerprint over ALL active rows (the dedup index's true domain).

        Distinct from the FAISS vector fingerprint: dedup also indexes
        memories stored without embeddings.
        """
        row = self._db.conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM memories WHERE is_deleted = 0"
        ).fetchone()
        return f"{int(row[0])}:{int(row[1])}"

    def _persist_dedup(self) -> None:
        """Persist the dedup index state so the next open skips re-hashing."""
        import pickle

        sidecar = self._dedup_sidecar_path()
        if sidecar is None or not self._dedup_by_namespace:
            return
        try:
            state = {
                "fingerprint": self._active_fingerprint(),
                "threshold": self._config.dedup_threshold,
                "num_perm": self._config.dedup_num_perm,
                "namespaces": {
                    ns: dedup.dump_state()
                    for ns, dedup in self._dedup_by_namespace.items()
                },
            }
            with sidecar.open("wb") as fh:
                pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:  # pragma: no cover - depends on filesystem state
            logger.warning("Could not persist dedup sidecar: %s", e)

    def close(self) -> None:
        """Close the memory system, saving all state."""
        self._persist_dedup()
        self._db.close()

    def __enter__(self) -> AriadneMemory:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def remember(
        self,
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.5,
        embedding: list[float] | np.ndarray | None = None,
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
        """Remember something by storing it in memory.

        Performs deduplication check and optional contradiction detection.

        Args:
            content: Text content to remember.
            memory_type: Category (semantic, episodic, procedural, etc.).
            importance: Importance score (0.0-1.0).
            embedding: Optional embedding vector.
            entities: Optional list of entity names.
            metadata: Optional metadata dict.
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
            Dict with 'memory_id', 'status' ('created' or 'duplicate'),
            and optionally 'contradictions' list.
        """
        result: dict[str, Any] = {
            "memory_id": None,
            "status": "error",
        }

        try:
            with self._lock:
                # Check for contradictions with existing memories
                contradictions = self._check_contradictions(content, namespace=namespace)
                if contradictions:
                    result["contradictions"] = contradictions
                    # Trust scoring: a fresh statement that contradicts stored
                    # knowledge makes the stored version less trustworthy. The
                    # penalty is small and repeatable, so a memory contradicted
                    # by several later writes sinks in ranking instead of being
                    # deleted outright. The incoming write still lands and gets
                    # a fair hearing from future reinforcement.
                    penalty = self._config.trust_contradiction_penalty
                    if penalty > 0:
                        seen_ids: set[int] = set()
                        for c in contradictions:
                            existing_id = c.get("existing_memory_id")
                            if existing_id is None or existing_id in seen_ids:
                                continue
                            seen_ids.add(existing_id)
                            self._db.adjust_confidence(int(existing_id), -penalty)

                # Check dedup (MinHash near-duplicate detection)
                dedup = self._dedup_for(namespace)
                if dedup.is_duplicate(content):
                    duplicates = dedup.find_duplicates(content)
                    if duplicates:
                        result["memory_id"] = None
                        result["status"] = "duplicate"
                        result["duplicate_of"] = duplicates[0]["id"]
                        return result

                # Auto-embed when an embedder is configured and no vector given.
                if embedding is None and self._embedder is not None:
                    embedding = self._embedder(content)

                emb_array = None
                if embedding is not None:
                    emb_array = np.asarray(embedding, dtype=np.float32)

                # Semantic dedup: MinHash is lexical, so paraphrases ("I live
                # in Paris" / "Paris is my home") slip through. When an
                # embedder is available, check the nearest stored vector and
                # treat near-identical meaning as a duplicate — and as mild
                # confirmation of the existing memory (trust scoring).
                if (
                    emb_array is not None
                    and self._config.semantic_dedup
                    and self._db._faiss_index is not None
                    and self._db._faiss_index.ntotal > 0
                ):
                    probe = emb_array.astype(np.float32)
                    norm = float(np.linalg.norm(probe))
                    if norm > 1e-10:
                        probe = probe / norm
                    hits = self._db.vector_search(
                        probe.reshape(1, -1), k=1, namespace=namespace
                    )
                    if hits and hits[0]["score"] >= self._config.semantic_dedup_threshold:
                        existing_id = int(hits[0]["id"])
                        result["memory_id"] = None
                        result["status"] = "duplicate"
                        result["duplicate_of"] = existing_id
                        result["semantic_duplicate"] = True
                        try:
                            self._db.adjust_confidence(
                                existing_id, self._config.trust_reinforce_delta
                            )
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.debug("Reinforce on semantic dup failed: %s", exc)
                        return result

                # Canonicalize entities to prevent fragmentation (e.g., "Mailcow" vs "mailcow")
                canonical_entities = None
                if entities:
                    canonical_entities = list(
                        {e.strip().lower() for e in entities if isinstance(e, str) and e.strip()}
                    )

                # Add to storage
                storage_result = self._db.add_memory(
                    content=content,
                    embedding=emb_array,
                    memory_type=memory_type,
                    importance=importance,
                    entities=canonical_entities,
                    metadata=metadata,
                    tags=tags,
                    namespace=namespace,
                    scope=scope,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    project_id=project_id,
                    event_at=event_at,
                    valid_from=valid_from,
                    valid_to=valid_to,
                    supersedes_id=supersedes_id,
                    confidence=confidence,
                )

                memory_id = storage_result["memory_id"]
                result["memory_id"] = memory_id
                result["status"] = storage_result["status"]

                # Add to dedup index
                if storage_result["status"] == "created":
                    dedup.add(content, doc_id=str(memory_id))
                    # Parity with the bulk path: single writes also count
                    # toward periodic auto-maintenance.
                    self._maybe_maintain()

            logger.info(
                "Remember: id=%s status=%s type=%s",
                memory_id,
                result["status"],
                memory_type,
            )
            return result

        except Exception as e:
            logger.error("Error in remember: %s", e)
            result["status"] = "error"
            result["error"] = str(e)
            return result

    def _check_contradictions(
        self, content: str, namespace: str = "default"
    ) -> list[dict[str, Any]]:
        """Check if new content contradicts existing memories.

        Args:
            content: New content to check.

        Returns:
            List of contradiction details.
        """
        try:
            # Get recent memories to check against
            results = self._db.fts_search(content, k=5, namespace=namespace)
            contradictions = []
            for mem in results:
                if mem.get("is_deleted"):
                    continue
                detected = self._contradiction_detector.detect_contradictions(
                    content, mem["content"]
                )
                for c in detected:
                    c["existing_memory_id"] = mem["id"]
                    contradictions.append(c)
            return contradictions
        except Exception as e:
            logger.debug("Contradiction check error: %s", e)
            return []

    def forget(self, memory_id: int, hard: bool = False) -> bool:
        """Forget (delete) a memory.

        Args:
            memory_id: Memory to forget.
            hard: If True, permanently delete. Otherwise soft-delete.

        Returns:
            True if forgotten, False if not found.
        """
        try:
            with self._lock:
                result = self._db.delete_memory(memory_id, hard=hard)
                # Remove from dedup index only once the delete actually happened
                if result:
                    for dedup in self._dedup_by_namespace.values():
                        dedup.remove(str(memory_id))
            logger.info("Forget: id=%d hard=%s result=%s", memory_id, hard, result)
            return result
        except Exception as e:
            logger.error("Error in forget: %s", e)
            return False

    def update(
        self,
        memory_id: int,
        content: str | None = None,
        importance: float | None = None,
        embedding: list[float] | np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        """Update an existing memory.

        Args:
            memory_id: Memory to update.
            content: New content.
            importance: New importance score.
            embedding: New embedding vector.
            metadata: New metadata.

        Returns:
            True if updated, False if not found.
        """
        try:
            with self._lock:
                # Auto-embed the new content if an embedder is set and the caller
                # changed content but didn't supply a fresh vector.
                if embedding is None and content is not None and self._embedder is not None:
                    embedding = self._embedder(content)

                emb_array = None
                if embedding is not None:
                    emb_array = np.asarray(embedding, dtype=np.float32)

                result = self._db.update_memory(
                    memory_id,
                    content=content,
                    importance=importance,
                    embedding=emb_array,
                    metadata=metadata,
                    tags=tags,
                )

                # Update dedup index if content changed
                if result and content is not None:
                    existing = self._db.get_memory(memory_id)
                    namespace = existing.get("namespace", "default") if existing else "default"
                    dedup = self._dedup_for(namespace)
                    dedup.remove(str(memory_id))
                    dedup.add(content, doc_id=str(memory_id))

            logger.info("Update: id=%d result=%s", memory_id, result)
            return result
        except Exception as e:
            logger.error("Error in update: %s", e)
            return False

    def graph(
        self,
        entity: str,
        edge_type: str | None = None,
        hops: int = 1,
    ) -> dict[str, Any]:
        """Traverse the knowledge graph from an entity.

        Args:
            entity: Starting entity name.
            edge_type: Optional edge type filter.
            hops: Maximum traversal depth.

        Returns:
            Dict with 'nodes' and 'edges'.
        """
        try:
            result = self._db.traverse_graph(entity, hops=hops, edge_type=edge_type)
            logger.info(
                "Graph: entity=%s hops=%d nodes=%d edges=%d",
                entity,
                hops,
                len(result["nodes"]),
                len(result["edges"]),
            )
            return result
        except Exception as e:
            logger.error("Error in graph traversal: %s", e)
            return {"nodes": [entity], "edges": [], "error": str(e)}

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str = "related",
        weight: float = 1.0,
    ) -> None:
        """Add an edge between two entities in the knowledge graph.

        Args:
            source: Source entity name.
            target: Target entity name.
            edge_type: Relationship type.
            weight: Edge weight.
        """
        self._db.add_edge(source, target, edge_type, weight)

    def stats(self) -> dict[str, Any]:
        """Get comprehensive memory system statistics.

        Returns:
            Dict with memory counts, graph info, dedup index size, etc.
        """
        try:
            db_stats = self._db.stats()
            db_stats["dedup_index_size"] = sum(
                dedup.size for dedup in self._dedup_by_namespace.values()
            )
            return db_stats
        except Exception as e:
            logger.error("Error getting stats: %s", e)
            return {"error": str(e)}

    def consolidate(self) -> int:
        """Run memory consolidation.

        Returns:
            Number of consolidation groups created.
        """
        with self._lock:
            count = self._db.consolidate()
            # Consolidation retires originals and adds merged memories; resync the
            # in-memory dedup index so it reflects what is actually active.
            if count:
                self._dedup_by_namespace = {}
                self._load_dedup_from_db()
            return count

    def evict(self) -> int:
        """Run priority-based memory eviction.

        Returns:
            Number of memories evicted.
        """
        return self._db.evict()

    def purge_deleted(self, older_than_seconds: float = 0.0) -> int:
        """Permanently remove soft-deleted memories. See AriadneDB.purge_deleted."""
        return self._db.purge_deleted(older_than_seconds)

    def prune_access_log(self, keep_per_memory: int | None = None) -> int:
        """Trim the access log to bound its growth. See AriadneDB.prune_access_log."""
        return self._db.prune_access_log(keep_per_memory)

    def maintenance(self, *, light: bool = False) -> dict[str, int]:
        """Run a housekeeping cycle: consolidate, evict, prune, purge.

        Args:
            light: Run only the cheap steps (access-log pruning and purge).
                Consolidation and eviction cost grows with store size, so the
                automatic cycles use light mode most of the time and full
                mode every ``heavy_maintenance_factor`` cycles — otherwise
                write throughput decays as the store grows. Manual calls
                default to the full cycle.

        Returns a summary dict of how much each step changed.
        """
        with self._lock:
            consolidated = 0
            evicted = 0
            if not light:
                consolidated = self.consolidate()
                evicted = self._db.evict()
            pruned = self._db.prune_access_log()
            # Keep recently soft-deleted rows recoverable; purging with 0 here
            # would permanently destroy everything evict() just soft-deleted.
            purged = self._db.purge_deleted(older_than_seconds=self._config.purge_retention_seconds)
        return {
            "consolidated": consolidated,
            "evicted": evicted,
            "access_log_pruned": pruned,
            "purged": purged,
        }

    def _maybe_maintain(self) -> None:
        """Trigger maintenance every ``_maintenance_interval`` writes.

        Light cycles run every interval; the expensive consolidate+evict pass
        only every ``heavy_maintenance_factor`` intervals, because its cost
        grows with store size and a fixed 50-write cadence makes write
        throughput decay from ~1 ms to ~20 ms as the store reaches 20k
        memories.
        """
        # Per-instance counter: a class-level counter would couple unrelated
        # AriadneMemory instances (e.g. the primary and shared DBs).
        self._write_count += 1
        if self._write_count >= self._maintenance_interval:
            self._write_count = 0
            self._maintenance_cycles += 1
            light = self._maintenance_cycles % self._config.heavy_maintenance_factor != 0
            try:
                self.maintenance(light=light)
                logger.info("Auto-maintenance completed (light=%s)", light)
            except Exception as e:
                logger.warning("Auto-maintenance failed: %s", e)

    def remember_many(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add multiple memories efficiently in a single transaction.

        Each item supports: content, memory_type, importance, entities, metadata, tags,
        namespace, scope, user_id, agent_id, session_id, project_id.

        Returns:
            List of result dicts per item.
        """
        with self._lock:
            # Batch-embed if embedder is configured. Embedders are per-text
            # callables (str -> vector); use embed_batch when available,
            # otherwise embed each text individually.
            if self._embedder is not None:
                to_embed = [
                    (i, item.get("content", ""))
                    for i, item in enumerate(items)
                    if item.get("embedding") is None
                ]
                if to_embed:
                    texts = [text for _, text in to_embed]
                    embed_batch = getattr(self._embedder, "embed_batch", None)
                    if callable(embed_batch):
                        embeddings = embed_batch(texts)
                    else:
                        embeddings = [self._embedder(text) for text in texts]
                    for (i, _), emb in zip(to_embed, embeddings, strict=True):
                        items[i]["embedding"] = emb

            results = self._db.add_memories_bulk(items)

            # Update dedup index for created memories. add_memories_bulk
            # returns one result per item, in order, so pair them positionally.
            for item, res in zip(items, results, strict=True):
                if res["status"] == "created" and item.get("content"):
                    self._dedup_for(item.get("namespace", "default")).add(
                        item["content"], doc_id=str(res["memory_id"])
                    )

            self._maybe_maintain()
            return results

    def export_json(self) -> dict[str, Any]:
        """Export all memories, entities, and links as JSON-safe dict."""
        return self._db.export_all()

    # ── Episode / Provenance / Feedback ────────────────────────────────

    def record_episode(
        self,
        content: str,
        role: str,
        source: str | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        event_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an immutable episode (raw evidence turn)."""
        with self._lock:
            return self._db.add_episode(
                content=content,
                role=role,
                source=source,
                event_at=event_at,
                metadata=metadata,
                namespace=namespace,
                session_id=session_id,
            )

    def remember_with_provenance(
        self,
        content: str,
        episode_id: int | None,
        source: str,
        source_id: str | None = None,
        span: str | None = None,
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Remember a memory with explicit provenance linking to an episode."""
        with self._lock:
            result = self.remember(content, **kwargs)
            if result.get("status") == "created":
                memory_id = result["memory_id"]
                self._db.add_source(
                    memory_id=memory_id,
                    episode_id=episode_id,
                    source=source,
                    source_id=source_id,
                    span=span,
                    confidence=confidence,
                )
            return result

    def supersede(
        self,
        old_memory_id: int,
        new_content: str,
        episode_id: int | None = None,
        source: str = "user",
        source_id: str | None = None,
        span: str | None = None,
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new memory that supersedes an old one, preserving history."""
        with self._lock:
            old = self._db.get_memory(old_memory_id)
            if old is None:
                return {"status": "error", "error": "memory not found"}

            now = _now()
            # Create new memory pointing to old
            result = self.remember(
                content=new_content,
                event_at=now,
                valid_from=now,
                supersedes_id=old_memory_id,
                confidence=confidence,
                **kwargs,
            )

            if result.get("status") == "created":
                new_id = result["memory_id"]
                # Link sources
                if episode_id is not None or source_id is not None or span is not None:
                    self._db.add_source(
                        memory_id=new_id,
                        episode_id=episode_id,
                        source=source,
                        source_id=source_id,
                        span=span,
                        confidence=confidence,
                    )
                # Copy old sources to new memory for continuity
                old_sources = self._db.get_sources_for_memory(old_memory_id)
                for src in old_sources:
                    self._db.add_source(
                        memory_id=new_id,
                        episode_id=src.get("episode_id"),
                        source=src.get("source", "derived"),
                        source_id=src.get("source_id"),
                        span=src.get("span"),
                        confidence=src.get("confidence", 1.0) * 0.9,  # Slight decay
                    )

            return result

    def feedback(
        self,
        memory_id: int,
        action: str,
        confidence_delta: float = 0.0,
        note: str | None = None,
        actor: str | None = None,
    ) -> dict[str, Any]:
        """Record feedback (approve/reject/correct) on a memory."""
        with self._lock:
            feedback_id = self._db.add_feedback(
                memory_id=memory_id,
                action=action,
                confidence_delta=confidence_delta,
                note=note,
                actor=actor,
            )
            return {"feedback_id": feedback_id, "status": "recorded"}

    def get_feedback(self, memory_id: int) -> list[dict[str, Any]]:
        """Get all feedback for a memory."""
        return self._db.get_feedback(memory_id)

    def recall(
        self,
        query: str,
        embedding: list[float] | np.ndarray | None = None,
        k: int = 10,
        type_filter: str | None = None,
        time_range: tuple[float, float] | None = None,
        importance_min: float | None = None,
        namespace: str | None = None,
        as_of: float | None = None,
        rerank: bool = False,
    ) -> list[dict[str, Any]]:
        """Recall memories matching a query.

        Uses hybrid search (vector + FTS) when embedding is provided,
        falls back to FTS-only otherwise.

        Args:
            query: Text query.
            embedding: Optional query embedding.
            k: Number of results.
            type_filter: Optional memory type filter.
            time_range: Optional (start, end) timestamps.
            importance_min: Optional minimum importance threshold.
            namespace: Namespace to search (None = all namespaces).
            as_of: Unix timestamp for temporal point-in-time recall.
            rerank: Rerank the final set with a cross-encoder (second-stage
                retrieval quality, Mem0/Zep-style). Loads
                ``AriadneConfig.rerank_model`` lazily on first use; degrades
                to fused order when sentence-transformers is not installed.

        Returns:
            List of matching memory dicts with sources attached.
        """
        try:
            # Auto-embed the query when an embedder is configured.
            if embedding is None and self._embedder is not None and query.strip():
                embedding = self._embedder(query)

            emb_array = None
            if embedding is not None:
                emb_array = np.asarray(embedding, dtype=np.float32)

            if emb_array is not None:
                search = self._db.hybrid_search
            else:
                search = self._db.fts_search

            # Storage can cheaply return a larger candidate window, but filters
            # applied here must keep expanding it until k eligible memories are
            # found. A fixed 3x window creates false empty results when excluded
            # rows rank above the requested type/time/importance.
            candidate_k = max(k, 1)
            results: list[dict[str, Any]] = []
            while True:
                if emb_array is not None:
                    if as_of is not None:
                        # Use temporal search when as_of is specified
                        results = self._db.recall_with_temporal(
                            query,
                            embedding=emb_array,
                            k=candidate_k,
                            namespace=namespace,
                            as_of=as_of,
                        )
                    else:
                        results = search(
                            query, embedding=emb_array, k=candidate_k, namespace=namespace
                        )
                else:
                    if as_of is not None:
                        # Use temporal FTS search when as_of is specified
                        results = self._db.recall_with_temporal(
                            query, embedding=None, k=candidate_k, namespace=namespace, as_of=as_of
                        )
                    else:
                        results = search(query, k=candidate_k, namespace=namespace)

                filtered = []
                for mem in results:
                    if mem.get("is_deleted"):
                        continue
                    if type_filter and mem.get("memory_type") != type_filter:
                        continue
                    if time_range:
                        start, end = time_range
                        if not (start <= mem["created_at"] <= end):
                            continue
                    if importance_min is not None and mem.get("importance", 0) < importance_min:
                        continue
                    filtered.append(mem)

                if len(filtered) >= k or len(results) < candidate_k:
                    break
                candidate_k *= 2

            final = filtered[:k]

            # For current recall (no as_of), filter out superseded memories
            # Only keep memories that are not superseded by another active memory
            if as_of is None:
                superseded_ids = set()
                for mem in final:
                    supersedes_id = mem.get("supersedes_id")
                    if supersedes_id is not None:
                        superseded_ids.add(supersedes_id)
                final = [m for m in final if m["id"] not in superseded_ids]

            # Attach sources and feedback summary to each result. Provenance is
            # fetched in two bulk queries instead of 3 x k round trips; the
            # supersession chain is only worth a lookup for memories that
            # actually supersede something (for fresh memories the chain is
            # trivially the memory itself).
            if final:
                ids = [m["id"] for m in final]
                sources_map = self._db.get_sources_bulk(ids)
                feedback_map = self._db.get_feedback_bulk(ids)
                for mem in final:
                    mem["sources"] = sources_map.get(mem["id"], [])
                    mem["feedback"] = feedback_map.get(mem["id"], [])
                    mem["supersession_chain"] = (
                        self._db.get_supersession_chain(mem["id"])
                        if mem.get("supersedes_id") is not None
                        else [mem]
                    )

            # Optional second stage: cross-encoder reranking of the final set.
            # RRF fusion is retrieval-grade; the reranker reads (query,
            # document) jointly and orders the top-k. The fused score is kept
            # in score_parts so explainability survives.
            if rerank and len(final) > 1:
                reranker = self._get_reranker()
                if reranker is not None:
                    try:
                        rerank_scores = reranker(
                            query, [mem.get("content") or "" for mem in final]
                        )
                        for mem, rs in zip(final, rerank_scores, strict=False):
                            parts = mem.get("score_parts") or {}
                            parts["fused"] = mem.get("score")
                            parts["rerank"] = float(rs)
                            mem["score_parts"] = parts
                            mem["score"] = float(rs)
                            mem["search_type"] = f"{mem.get('search_type', 'hybrid')}+rerank"
                        final.sort(key=lambda item: (-item.get("score", 0.0), item.get("id", 0)))
                    except Exception as exc:
                        logger.warning("Reranking failed, keeping fused order: %s", exc)

            # Record access for the memories actually surfaced
            if final:
                self._db.touch_memories([m["id"] for m in final])

            logger.info(
                "Recall query=%.50s results=%d filtered=%d",
                query,
                len(results),
                len(filtered),
            )
            return final

        except Exception as e:
            logger.error("Error in recall: %s", e)
            return []

    def get_history(self, memory_id: int) -> dict[str, Any]:
        """Get full temporal history and provenance for a memory."""
        with self._lock:
            chain = self._db.get_supersession_chain(memory_id)
            sources = self._db.get_sources_for_memory(memory_id)
            feedback = self._db.get_feedback(memory_id)
            return {
                "memory_id": memory_id,
                "supersession_chain": chain,
                "sources": sources,
                "feedback": feedback,
            }

    def invalidate(self, memory_id: int, hard: bool = False) -> bool:
        """Invalidate a memory (soft delete by default)."""
        return self.forget(memory_id, hard=hard)

    # ── Session intelligence (ctx-style episodic recall) ─────────────────

    def search_episodes(
        self,
        query: str,
        k: int = 10,
        namespace: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search raw session history (episodes) by keyword.

        Unlike ``recall``, which searches distilled memories, this searches the
        immutable recorded turns — the "what did I actually say/do last week"
        surface. Results include role, session id, and timestamps.

        Args:
            query: Search query string.
            k: Number of episode results.
            namespace: Optional namespace filter.
            session_id: Optional session filter.

        Returns:
            List of episode dicts ordered by BM25 relevance.
        """
        with self._lock:
            return self._db.search_episodes(query, k=k, namespace=namespace, session_id=session_id)

    def list_sessions(
        self, namespace: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """List recorded sessions (newest activity first) with turn counts."""
        with self._lock:
            return self._db.list_sessions(namespace=namespace, limit=limit)

    def digest_session(
        self,
        session_id: str,
        namespace: str = "default",
        max_turns: int | None = None,
        importance: float = 0.6,
        force: bool = False,
    ) -> dict[str, Any]:
        """Distill a session's raw episodes into a compact digest memory.

        Deterministic and LLM-free: episodes are scored by role weight and by
        how much they mention the session's recurring terms (files, errors,
        decisions), the top turns are kept, and the digest is stored as an
        ordinary memory with ``kind=session_digest`` metadata plus provenance
        links back to the selected episodes. Recalling it next session gives
        the agent "what happened last time" without replaying every turn.

        Args:
            session_id: Session to digest.
            namespace: Namespace the episodes live in.
            max_turns: Max turns kept in the digest (default from config).
            importance: Importance for the stored digest memory.
            force: Re-digest even if one exists (supersedes the old digest).

        Returns:
            Dict with 'status' ('created' | 'exists' | 'empty'), 'memory_id',
            'digest' text, and 'episodes' (turn count).
        """
        if max_turns is None:
            max_turns = self._config.session_digest_max_turns
        with self._lock:
            episodes = self._db.get_episodes(session_id=session_id, namespace=namespace)
            if not episodes:
                return {"status": "empty", "session_id": session_id, "digest": None}

            existing = None
            for digest_mem in self._db.list_session_digests(namespace=namespace, limit=100):
                if (digest_mem.get("metadata") or {}).get("session_id") == session_id:
                    existing = digest_mem
                    break
            if existing is not None and not force:
                return {
                    "status": "exists",
                    "memory_id": existing["id"],
                    "digest": existing["content"],
                    "episodes": len(episodes),
                }

            # A forced re-digest replaces the old one. Retire the old row
            # first: remember() would otherwise flag the (near-)identical
            # digest content as a duplicate and drop the replacement.
            if existing is not None:
                self.forget(existing["id"], hard=False)

            # Session-level term document frequency: terms that recur across
            # turns mark the session's real topics.
            term_df: dict[str, int] = {}
            for ep in episodes:
                for term in set(_WORD_RE.findall(ep.get("content", "").lower())):
                    term_df[term] = term_df.get(term, 0) + 1

            scored: list[tuple[float, int, dict[str, Any]]] = []
            for idx, ep in enumerate(episodes):
                terms = set(_WORD_RE.findall(ep.get("content", "").lower()))
                salience = sum(min(term_df.get(t, 0), 4) for t in terms)
                role_weight = _ROLE_WEIGHTS.get(ep.get("role", "user"), 1.0)
                score = role_weight * salience / math.sqrt(len(terms) + 1.0)
                scored.append((score, idx, ep))

            chosen = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_turns]
            chosen.sort(key=lambda item: item[1])  # present chronologically

            lines = [f"Session {session_id} digest ({len(episodes)} turns):"]
            for _, _, ep in chosen:
                text = " ".join(ep.get("content", "").split())
                if len(text) > 300:
                    text = text[:297] + "..."
                lines.append(f"- [{ep.get('role', 'user')}] {text}")
            digest_text = "\n".join(lines)

            last_event = max(
                (ep.get("event_at") or ep.get("created_at") or 0.0) for ep in episodes
            )
            remember_result = self.remember(
                content=digest_text,
                memory_type="episodic",
                importance=importance,
                namespace=namespace,
                session_id=session_id,
                event_at=last_event,
                metadata={
                    "kind": "session_digest",
                    "session_id": session_id,
                    "turn_count": len(episodes),
                    "episode_ids": [ep["id"] for _, _, ep in chosen],
                },
                supersedes_id=existing["id"] if existing is not None else None,
            )

            memory_id = remember_result.get("memory_id")
            if remember_result.get("status") == "created" and memory_id is not None:
                for _, _, ep in chosen:
                    self._db.add_source(
                        memory_id=memory_id,
                        episode_id=ep["id"],
                        source="session_digest",
                        confidence=1.0,
                    )

            return {
                "status": remember_result.get("status", "error"),
                "memory_id": memory_id,
                "digest": digest_text,
                "episodes": len(episodes),
            }

    def session_context(
        self,
        namespace: str = "default",
        max_sessions: int = 3,
        char_budget: int = 1500,
    ) -> str:
        """Assemble a compact 'what happened in recent sessions' block.

        Built from stored session digests, newest first, under a character
        budget. Prepend it (or use ``context_pack(include_sessions=True)``) at
        the start of an agent session for continuity without replaying history.

        Returns:
            Context string, or '' when no digests exist.
        """
        digests = self._db.list_session_digests(namespace=namespace, limit=max_sessions)
        if not digests:
            return ""
        per_digest = max(200, char_budget // len(digests))
        lines = ["Recent session context:"]
        for digest_mem in digests:
            content = " ".join((digest_mem.get("content") or "").split())
            if len(content) > per_digest:
                content = content[: per_digest - 3] + "..."
            lines.append(f"- {content}")
        return "\n".join(lines)

    # ── Associative expansion ─────────────────────────────────────────────

    def expand(
        self,
        results: list[dict[str, Any]],
        *,
        hops: int = 1,
        limit: int = 10,
        decay: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Expand recall results through the entity graph and re-merge.

        For each hop, memories sharing entities with the current frontier are
        surfaced (ranked by shared-entity count) and scored as a decaying
        fraction of the best seed score, so direct hits always outrank
        associations. The merged list is re-sorted by score.

        Args:
            results: Seed results (typically a ``recall()`` return).
            hops: Expansion rounds (1 = direct entity neighbours).
            limit: Max added memories per hop.
            decay: Score multiplier per hop.

        Returns:
            Merged, re-scored, re-sorted result list.
        """
        if not results or hops < 1:
            return results
        merged = list(results)
        present_ids = {mem["id"] for mem in merged}
        best_score = max((mem.get("score", 0.0) for mem in merged), default=1.0 / 61.0)
        if best_score <= 0:
            best_score = 1.0 / 61.0
        frontier = [mem["id"] for mem in merged]

        for hop in range(1, hops + 1):
            neighbors = self._db.expand_by_entities(frontier, limit=limit * 2)
            added: list[dict[str, Any]] = []
            for mem in neighbors:
                if mem["id"] in present_ids:
                    continue
                shared = min(mem.get("shared_entities", 1), 3)
                mem["score"] = best_score * (decay**hop) * (0.5 + 0.5 * shared / 3.0)
                mem["score_parts"] = {
                    "expansion_hop": hop,
                    "shared_entities": mem.get("shared_entities", 0),
                }
                added.append(mem)
                if len(added) >= limit:
                    break
            if not added:
                break
            merged.extend(added)
            present_ids.update(mem["id"] for mem in added)
            frontier = [mem["id"] for mem in added]

        merged.sort(key=lambda item: (-item.get("score", 0.0), item.get("id", 0)))
        return merged

    # ── Trust scoring ──────────────────────────────────────────────────────

    def reinforce(self, memory_id: int, delta: float | None = None) -> float | None:
        """Confirm a memory, raising its trust (confidence).

        The affirmative counterpart of the automatic contradiction penalty:
        agents or users call this when a recalled memory proved useful or was
        explicitly verified, so trust accrues from use instead of being a
        static write-time guess.

        Args:
            memory_id: Memory to reinforce.
            delta: Confidence increment (default from config).

        Returns:
            New confidence, or None if the memory does not exist.
        """
        increment = self._config.trust_reinforce_delta if delta is None else float(delta)
        return self._db.adjust_confidence(memory_id, increment)

    # ── Core memory blocks (Letta-style always-in-context state) ──────────

    def core_set(self, name: str, content: str, namespace: str = "default") -> dict[str, Any]:
        """Create or replace a core memory block.

        Core blocks are the agent's working memory — persona, human profile,
        project state — always loaded into context, never deduplicated,
        decayed, or evicted. This is the layer an agent self-edits during a
        run (Letta/MemGPT's ``core_memory_replace``).
        """
        with self._lock:
            return self._db.core_block_set(name, content, namespace)

    def core_get(self, name: str, namespace: str = "default") -> dict[str, Any] | None:
        """Return one core memory block, or None."""
        with self._lock:
            return self._db.core_block_get(name, namespace)

    def core_append(
        self,
        name: str,
        text: str,
        namespace: str = "default",
        char_limit: int | None = None,
    ) -> dict[str, Any]:
        """Append to a core memory block (created if missing).

        The block is bounded by ``char_limit`` (default from config): when the
        append would overflow, the oldest content is trimmed so the most
        recent observations survive. Letta/MemGPT's ``core_memory_append``.
        """
        limit = self._config.core_block_char_limit if char_limit is None else char_limit
        with self._lock:
            return self._db.core_block_append(name, text, namespace, char_limit=limit)

    def core_delete(self, name: str, namespace: str = "default") -> bool:
        """Delete a core memory block. Returns True if it existed."""
        with self._lock:
            return self._db.core_block_delete(name, namespace)

    def core_blocks(self, namespace: str = "default") -> list[dict[str, Any]]:
        """List a namespace's core memory blocks, ordered by name."""
        with self._lock:
            return self._db.core_blocks_list(namespace)

    def core_pack(self, namespace: str = "default", char_budget: int = 4000) -> str:
        """Render core memory blocks as a compact context section.

        Empty blocks are skipped; the budget is split across blocks and each
        is trimmed from the front (keeping the most recent tail) if needed.
        """
        blocks = self.core_blocks(namespace)
        if not blocks:
            return ""
        non_empty = [b for b in blocks if b["content"].strip()]
        if not non_empty:
            return ""
        per_block = max(200, char_budget // len(non_empty))
        lines = ["Core memory:"]
        for block in non_empty:
            content = block["content"]
            if len(content) > per_block:
                content = "…" + content[len(content) - per_block :]
            lines.append(f"### {block['name']}\n{content}")
        return "\n".join(lines)

    # ── Entity resolution ─────────────────────────────────────────────────

    def merge_entities(self, source: str, target: str) -> int:
        """Merge entity ``source`` into ``target`` (alias resolution).

        Re-points memory links and graph edges onto the target, then removes
        the source — repairing the fragmentation caused by "PG" vs "postgres"
        style variants. Returns the number of links moved.
        """
        with self._lock:
            return self._db.merge_entities(source.strip().lower(), target.strip().lower())

    def context_pack(
        self,
        query: str,
        token_budget: int = 2000,
        per_memory_overhead: int = 8,
        include_scores: bool = False,
        namespaces: list[str] | None = None,
        include_sessions: bool = False,
        include_core: bool = False,
        **recall_kwargs: Any,
    ) -> str:
        """Assemble a token-budget-aware context string from recalled memories.

        Uses a deterministic token estimate (chars / 4). Returns a compact
        block of top memories ordered by relevance, under the estimated budget.
        Feed the result straight into an LLM prompt.

        Args:
            query: Text query, forwarded to recall().
            token_budget: Maximum tokens in the returned string.
            per_memory_overhead: Estimated tokens of formatting wrapped
                around each memory (label, spacing).
            include_scores: Prepend a relevance score to each entry.
            namespaces: Optional explicit namespace allow-list. Results from
                those namespaces are merged and re-ranked before packing.
            include_sessions: Prepend a 'recent session context' block built
                from stored session digests (namespace = first allowed
                namespace, or 'default'). Renewed continuity between sessions.
            include_core: Prepend the core memory blocks (always-in-context
                working state) rendered by ``core_pack()``.
            **recall_kwargs: Extra recall() filters (type_filter...).
        """
        limit = recall_kwargs.pop("k", 20)
        session_block = ""
        if include_sessions:
            ns = namespaces[0] if namespaces else (
                recall_kwargs.get("namespace") or "default"
            )
            session_block = self.session_context(namespace=str(ns))
        core_block = ""
        if include_core:
            core_ns = namespaces[0] if namespaces else (
                recall_kwargs.get("namespace") or "default"
            )
            core_block = self.core_pack(namespace=str(core_ns))
        results = self._gather_results(query, limit, namespaces, recall_kwargs)

        results = sorted(
            results,
            key=lambda item: (-item.get("score", 0.0), item.get("id", 0)),
        )
        lines: list[str] = []
        used = 0
        budget = max(0, int(token_budget))
        overhead = max(0, int(per_memory_overhead))
        for block in (core_block, session_block):
            if not block:
                continue
            block_est = max(1, (len(block) + 3) // 4)
            if block_est <= budget:
                lines.append(block)
                used += block_est
        for mem in results:
            content = (mem.get("content") or "").strip()
            if not content:
                continue
            score = mem.get("score", 0.0)
            prefix = f"[{score:.3f}] " if include_scores else "- "
            line = prefix + content.replace("\n", " ")
            est = max(1, (len(line) + 3) // 4) + overhead
            if used + est > budget:
                continue
            lines.append(line)
            used += est
        return "\n".join(lines)

    def _gather_results(
        self,
        query: str,
        limit: int,
        namespaces: list[str] | None,
        recall_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Recall from one or many namespaces and merge by score."""
        if namespaces is None:
            return self.recall(query, k=limit, **recall_kwargs)
        recall_kwargs.pop("namespace", None)
        by_id: dict[object, dict[str, Any]] = {}
        for namespace in dict.fromkeys(str(item) for item in namespaces):
            for result in self.recall(query, k=limit, namespace=namespace, **recall_kwargs):
                memory_id = result.get("id")
                previous = by_id.get(memory_id)
                if previous is None or result.get("score", 0.0) > previous.get("score", 0.0):
                    by_id[memory_id] = result
        return sorted(
            by_id.values(),
            key=lambda item: (-item.get("score", 0.0), item.get("id", 0)),
        )[:limit]

    def import_json(self, data: dict[str, Any]) -> int:
        """Import from a previously exported JSON dict. Returns count imported."""
        count = self._db.import_all(data)
        # Rebuild dedup index
        self._dedup_by_namespace = {}
        self._load_dedup_from_db()
        return count
