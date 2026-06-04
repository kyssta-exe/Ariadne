"""Hermes-compatible memory interface for Ariadne.

Wraps AriadneDB and Deduplicator into a clean API matching
the Hermes agent memory protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from arriadne.config import AriadneConfig
from arriadne.dedup import ContradictionDetector, Deduplicator
from arriadne.storage import AriadneDB

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
    ) -> None:
        if config is None:
            kwargs: dict[str, Any] = {}
            if db_path is not None:
                kwargs["db_path"] = db_path
            if embedding_dim is not None:
                kwargs["embedding_dim"] = embedding_dim
            config = AriadneConfig(**kwargs)

        self._config = config
        self._db = AriadneDB(config)
        self._dedup = Deduplicator(
            threshold=config.dedup_threshold,
            num_perm=config.dedup_num_perm,
        )
        self._contradiction_detector = ContradictionDetector()
        self._db.open()
        self._rebuild_dedup_index()
        logger.info("AriadneMemory initialized (db=%s)", config.db_path)

    def _rebuild_dedup_index(self) -> None:
        """Rebuild the in-memory dedup index from existing memories.

        The Deduplicator is in-memory only (MinHash LSH doesn't persist).
        On startup, we load all active memories into it so that
        near-duplicate detection works across sessions.
        """
        try:
            cursor = self._db.conn.execute(
                "SELECT id, content FROM memories WHERE is_deleted = 0"
            )
            count = 0
            for row in cursor.fetchall():
                self._dedup.add(row[1], doc_id=str(row[0]))
                count += 1
            logger.info("Rebuilt dedup index with %d memories", count)
        except Exception as e:
            logger.warning("Failed to rebuild dedup index: %s", e)

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

        Returns:
            Dict with 'memory_id', 'status' ('created' or 'duplicate'),
            and optionally 'contradictions' list.
        """
        result: dict[str, Any] = {
            "memory_id": None,
            "status": "error",
        }

        try:
            # Check for contradictions with existing memories
            contradictions = self._check_contradictions(content)
            if contradictions:
                result["contradictions"] = contradictions

            # Check dedup
            if self._dedup.is_duplicate(content):
                # Find the existing duplicate
                duplicates = self._dedup.find_duplicates(content)
                if duplicates:
                    result["memory_id"] = None
                    result["status"] = "duplicate"
                    result["duplicate_of"] = duplicates[0]["id"]
                    return result

            # Convert embedding
            emb_array = None
            if embedding is not None:
                emb_array = np.asarray(embedding, dtype=np.float32)

            # Add to storage
            storage_result = self._db.add_memory(
                content=content,
                embedding=emb_array,
                memory_type=memory_type,
                importance=importance,
                entities=entities,
                metadata=metadata,
            )

            memory_id = storage_result["memory_id"]
            result["memory_id"] = memory_id
            result["status"] = storage_result["status"]

            # Add to dedup index
            if storage_result["status"] == "created":
                self._dedup.add(content, doc_id=str(memory_id))

            logger.info(
                "Remember: id=%s status=%s type=%s",
                memory_id, result["status"], memory_type,
            )
            return result

        except Exception as e:
            logger.error("Error in remember: %s", e)
            result["status"] = "error"
            result["error"] = str(e)
            return result

    def _check_contradictions(self, content: str) -> list[dict[str, Any]]:
        """Check if new content contradicts existing memories.

        Args:
            content: New content to check.

        Returns:
            List of contradiction details.
        """
        try:
            # Get recent memories to check against
            results = self._db.fts_search(content, k=5)
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

    def recall(
        self,
        query: str,
        embedding: list[float] | np.ndarray | None = None,
        k: int = 10,
        type_filter: str | None = None,
        time_range: tuple[float, float] | None = None,
        importance_min: float | None = None,
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

        Returns:
            List of matching memory dicts.
        """
        try:
            emb_array = None
            if embedding is not None:
                emb_array = np.asarray(embedding, dtype=np.float32)

            if emb_array is not None:
                results = self._db.hybrid_search(query, embedding=emb_array, k=k * 3)
            else:
                results = self._db.fts_search(query, k=k * 3)

            # Apply filters
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

            logger.info(
                "Recall query=%.50s results=%d filtered=%d",
                query, len(results), len(filtered),
            )
            return filtered[:k]

        except Exception as e:
            logger.error("Error in recall: %s", e)
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
            # Remove from dedup index
            self._dedup.remove(str(memory_id))

            result = self._db.delete_memory(memory_id, hard=hard)
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
            emb_array = None
            if embedding is not None:
                emb_array = np.asarray(embedding, dtype=np.float32)

            result = self._db.update_memory(
                memory_id,
                content=content,
                importance=importance,
                embedding=emb_array,
                metadata=metadata,
            )

            # Update dedup index if content changed
            if result and content is not None:
                self._dedup.remove(str(memory_id))
                self._dedup.add(content, doc_id=str(memory_id))

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
                entity, hops, len(result["nodes"]), len(result["edges"]),
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
            db_stats["dedup_index_size"] = self._dedup.size
            return db_stats
        except Exception as e:
            logger.error("Error getting stats: %s", e)
            return {"error": str(e)}

    def consolidate(self) -> int:
        """Run memory consolidation.

        Returns:
            Number of consolidation groups created.
        """
        return self._db.consolidate()

    def evict(self) -> int:
        """Run priority-based memory eviction.

        Returns:
            Number of memories evicted.
        """
        return self._db.evict()
