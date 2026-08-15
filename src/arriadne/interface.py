"""Hermes-compatible memory interface for Ariadne.

Wraps AriadneDB and Deduplicator into a clean API matching
the Hermes agent memory protocol.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from arriadne.config import AriadneConfig
from arriadne.dedup import ContradictionDetector, Deduplicator
from arriadne.embeddings import Embedder, resolve_embedder
from arriadne.storage import AriadneDB, _now

logger = logging.getLogger(__name__)


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

        self._db = AriadneDB(config)
        self._dedup_by_namespace: dict[str, Deduplicator] = {}
        self._contradiction_detector = ContradictionDetector()
        self._db.open()
        self._load_dedup_from_db()
        logger.info("AriadneMemory initialized (db=%s)", config.db_path)

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
        """
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

    def close(self) -> None:
        """Close the memory system, saving all state."""
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

    def maintenance(self) -> dict[str, int]:
        """Run a full housekeeping cycle: consolidate, evict, prune, purge.

        Returns a summary dict of how much each step changed.
        """
        with self._lock:
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

    # Auto-maintenance: call every N writes to keep things tidy
    _maintenance_interval: int = 50

    def _maybe_maintain(self) -> None:
        """Trigger maintenance every _maintenance_interval writes."""
        # Per-instance counter: a class-level counter would couple unrelated
        # AriadneMemory instances (e.g. the primary and shared DBs).
        self._write_count += 1
        if self._write_count >= self._maintenance_interval:
            self._write_count = 0
            try:
                self.maintenance()
                logger.info("Auto-maintenance completed")
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
                    for (i, _), emb in zip(to_embed, embeddings):
                        items[i]["embedding"] = emb

            results = self._db.add_memories_bulk(items)

            # Update dedup index for created memories. add_memories_bulk
            # returns one result per item, in order, so pair them positionally.
            for item, res in zip(items, results):
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
        **kwargs,
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
        **kwargs,
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

            # Attach sources and feedback summary to each result
            for mem in final:
                mem["sources"] = self._db.get_sources_for_memory(mem["id"])
                mem["feedback"] = self._db.get_feedback(mem["id"])
                mem["supersession_chain"] = self._db.get_supersession_chain(mem["id"])

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

    def context_pack(
        self,
        query: str,
        token_budget: int = 2000,
        per_memory_overhead: int = 8,
        include_scores: bool = False,
        namespaces: list[str] | None = None,
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
            **recall_kwargs: Extra recall() filters (type_filter...).
        """
        limit = recall_kwargs.pop("k", 20)
        if namespaces is None:
            results = self.recall(query, k=limit, **recall_kwargs)
        else:
            recall_kwargs.pop("namespace", None)
            by_id: dict[object, dict[str, Any]] = {}
            for namespace in dict.fromkeys(str(item) for item in namespaces):
                for result in self.recall(query, k=limit, namespace=namespace, **recall_kwargs):
                    memory_id = result.get("id")
                    previous = by_id.get(memory_id)
                    if previous is None or result.get("score", 0.0) > previous.get("score", 0.0):
                        by_id[memory_id] = result
            results = sorted(
                by_id.values(),
                key=lambda item: (-item.get("score", 0.0), item.get("id", 0)),
            )[:limit]

        results = sorted(
            results,
            key=lambda item: (-item.get("score", 0.0), item.get("id", 0)),
        )
        lines: list[str] = []
        used = 0
        budget = max(0, int(token_budget))
        overhead = max(0, int(per_memory_overhead))
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

    def import_json(self, data: dict[str, Any]) -> int:
        """Import from a previously exported JSON dict. Returns count imported."""
        count = self._db.import_all(data)
        # Rebuild dedup index
        self._dedup_by_namespace = {}
        self._load_dedup_from_db()
        return count
