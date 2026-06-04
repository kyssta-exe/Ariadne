"""Unified memory interface for Ariadne.

Provides AriadneMemory — the single entry point that bundles:
- Storage (SQLite + FAISS)
- Embeddings (auto-detected: ONNX / SentenceTransformers / keyword)
- Deduplication (MinHash LSH with persistence)
- Knowledge graph
- Conversation memory
- Agent tools (OpenAI function calling)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from arriadne.config import AriadneConfig
from arriadne.conversation import AgentTools, ConversationTracker
from arriadne.dedup import ContradictionDetector, Deduplicator
from arriadne.embeddings import EmbeddingProvider, auto_detect_provider
from arriadne.storage import AriadneDB

logger = logging.getLogger(__name__)


class AriadneMemory:
    """The single entry point for Ariadne memory.

    Zero-config: just pass a path and everything works.
    Auto-detects the best embedding provider, auto-embeds text,
    auto-extracts entities, auto-deduplicates.

    Example:
        >>> mem = AriadneMemory("my_memory.db")
        >>> mem.remember("Paris is the capital of France")
        >>> results = mem.recall("What's the capital of France?")

    With conversation tracking:
        >>> mem = AriadneMemory("agent.db")
        >>> mem.sync_turn("user", "Deploy the app to production")
        >>> mem.sync_turn("assistant", "I'll deploy to prod now.")
        >>> mem.get_context("deployment")  # relevant past turns
    """

    def __init__(
        self,
        config: AriadneConfig | None = None,
        db_path: str | Path | None = None,
        embedding_dim: int | None = None,
        embedding_provider: EmbeddingProvider | str | None = None,
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
        self._contradiction_detector = ContradictionDetector()
        self._db.open()

        # Initialize embedding provider
        if isinstance(embedding_provider, EmbeddingProvider):
            self._embedder = embedding_provider
        else:
            self._embedder = auto_detect_provider(
                dimension=config.embedding_dim,
                preferred=embedding_provider or config.embedding_provider,
            )

        # Sync config dimension with embedding provider's actual dimension.
        # Some providers (ONNX, SentenceTransformer) override the requested
        # dimension based on the model's actual output. The FAISS index and
        # all downstream code must use the provider's actual dimension.
        if self._embedder.dimension != config.embedding_dim:
            logger.info(
                "Syncing embedding_dim: %d -> %d (provider: %s)",
                config.embedding_dim, self._embedder.dimension, self._embedder.name,
            )
            config.embedding_dim = self._embedder.dimension
            # Recreate FAISS index with correct dimension
            self._db._config.embedding_dim = self._embedder.dimension
            self._db._faiss_index = self._db._create_faiss_index(0)

        # Initialize deduplicator with persistence
        self._dedup = Deduplicator(
            threshold=config.dedup_threshold,
            num_perm=config.dedup_num_perm,
        )
        self._rebuild_dedup_index()

        # Initialize conversation tracker
        self._conversation = ConversationTracker(self._db)

        logger.info(
            "AriadneMemory initialized (db=%s, embedding=%s)",
            config.db_path, self._embedder.name,
        )

    def _rebuild_dedup_index(self) -> None:
        """Rebuild the in-memory dedup index from existing memories."""
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

    def _auto_embed(self, text: str) -> np.ndarray | None:
        """Generate embedding using the configured provider.

        Returns None if the provider is keyword-based (sparse embeddings
        shouldn't be used for cosine similarity search).
        """
        if self._embedder.name == "keyword":
            return None
        try:
            return self._embedder.embed(text)
        except Exception as e:
            logger.warning("Embedding generation failed: %s", e)
            return None

    def _auto_embed_batch(self, texts: list[str]) -> list[np.ndarray | None]:
        """Generate embeddings for a batch of texts."""
        if self._embedder.name == "keyword":
            return [None] * len(texts)
        try:
            vecs = self._embedder.embed_batch(texts)
            return [vecs[i] for i in range(len(texts))]
        except Exception as e:
            logger.warning("Batch embedding generation failed: %s", e)
            return [None] * len(texts)

    # ─── Core API ───────────────────────────────────────────────

    def remember(
        self,
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.5,
        embedding: list[float] | np.ndarray | None = None,
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        auto_embed: bool = True,
    ) -> dict[str, Any]:
        """Store a memory. Auto-embeds if no embedding provided.

        Args:
            content: Text content to remember.
            memory_type: Category (semantic, episodic, procedural, preference).
            importance: Importance score (0.0-1.0).
            embedding: Optional pre-computed embedding vector.
            entities: Optional entity names to associate.
            metadata: Optional metadata dict.
            auto_embed: Whether to auto-generate embedding (default True).

        Returns:
            Dict with memory_id, status, and optional contradictions.
        """
        result: dict[str, Any] = {"memory_id": None, "status": "error"}

        try:
            # Check for contradictions
            contradictions = self._check_contradictions(content)
            if contradictions:
                result["contradictions"] = contradictions

            # Check dedup
            if self._dedup.is_duplicate(content):
                duplicates = self._dedup.find_duplicates(content)
                if duplicates:
                    result["status"] = "duplicate"
                    result["duplicate_of"] = duplicates[0]["id"]
                    return result

            # Auto-embed if needed
            emb_array = None
            if embedding is not None:
                emb_array = np.asarray(embedding, dtype=np.float32)
            elif auto_embed:
                emb = self._auto_embed(content)
                if emb is not None:
                    emb_array = emb

            # Store
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

            return result

        except Exception as e:
            logger.error("Error in remember: %s", e)
            result["error"] = str(e)
            return result

    def recall(
        self,
        query: str,
        embedding: list[float] | np.ndarray | None = None,
        k: int = 10,
        type_filter: str | None = None,
        time_range: tuple[float, float] | None = None,
        importance_min: float | None = None,
        auto_embed: bool = True,
    ) -> list[dict[str, Any]]:
        """Search memories. Auto-embeds query if no embedding provided.

        When an embedding provider is configured, automatically performs
        hybrid search (vector + FTS5 + RRF fusion). Without embeddings,
        falls back to FTS5 keyword search.
        """
        try:
            emb_array = None
            if embedding is not None:
                emb_array = np.asarray(embedding, dtype=np.float32)
            elif auto_embed:
                emb = self._auto_embed(query)
                if emb is not None:
                    emb_array = emb

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

            return filtered[:k]

        except Exception as e:
            logger.error("Error in recall: %s", e)
            return []

    def forget(self, memory_id: int, hard: bool = False) -> bool:
        """Forget (delete) a memory."""
        try:
            self._dedup.remove(str(memory_id))
            return self._db.delete_memory(memory_id, hard=hard)
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
        """Update an existing memory."""
        try:
            emb_array = None
            if embedding is not None:
                emb_array = np.asarray(embedding, dtype=np.float32)

            result = self._db.update_memory(
                memory_id, content=content, importance=importance,
                embedding=emb_array, metadata=metadata,
            )

            if result and content is not None:
                self._dedup.remove(str(memory_id))
                self._dedup.add(content, doc_id=str(memory_id))

            return result
        except Exception as e:
            logger.error("Error in update: %s", e)
            return False

    # ─── Knowledge Graph ────────────────────────────────────────

    def graph(
        self,
        entity: str,
        edge_type: str | None = None,
        hops: int = 1,
    ) -> dict[str, Any]:
        """Traverse the knowledge graph from an entity."""
        try:
            return self._db.traverse_graph(entity, hops=hops, edge_type=edge_type)
        except Exception as e:
            return {"nodes": [entity], "edges": [], "error": str(e)}

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str = "related",
        weight: float = 1.0,
    ) -> None:
        """Add an edge between two entities."""
        self._db.add_edge(source, target, edge_type, weight)

    # ─── Conversation Memory ────────────────────────────────────

    def sync_turn(
        self,
        role: str,
        content: str,
        extract_facts: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a conversation turn and extract memories.

        Processes a message from a conversation, stores it as episodic
        memory, and optionally extracts facts and entities.

        Args:
            role: Speaker role ("user", "assistant", "system").
            content: Message content.
            extract_facts: Extract structured facts (default True).
            metadata: Optional extra metadata.

        Returns:
            Dict with facts, entities, and memory IDs.
        """
        return self._conversation.sync_turn(
            role, content, extract_facts=extract_facts, metadata=metadata,
        )

    def get_context(
        self,
        query: str,
        max_turns: int = 10,
        max_tokens_estimate: int = 2000,
    ) -> list[dict[str, Any]]:
        """Get relevant conversation context for an LLM prompt."""
        return self._conversation.get_context(
            query, max_turns=max_turns, max_tokens_estimate=max_tokens_estimate,
        )

    # ─── Embedding API ──────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text using the configured provider."""
        return self._embedder.embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts."""
        return self._embedder.embed_batch(texts)

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Return the active embedding provider."""
        return self._embedder

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._embedder.dimension

    # ─── Agent Tools ────────────────────────────────────────────

    @staticmethod
    def get_tools() -> list[dict[str, Any]]:
        """Return OpenAI function calling compatible tool definitions."""
        return AgentTools.get_tools()

    @property
    def tool_executor(self) -> _ToolExecutor:
        """Return a tool executor that maps tool calls to methods."""
        return _ToolExecutor(self)

    # ─── Lifecycle ──────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Get comprehensive memory system statistics."""
        try:
            db_stats = self._db.stats()
            db_stats["dedup_index_size"] = self._dedup.size
            db_stats["embedding_provider"] = self._embedder.name
            db_stats["embedding_dimension"] = self._embedder.dimension
            return db_stats
        except Exception as e:
            return {"error": str(e)}

    def consolidate(self) -> int:
        """Run memory consolidation."""
        return self._db.consolidate()

    def evict(self) -> int:
        """Run priority-based memory eviction."""
        return self._db.evict()

    def close(self) -> None:
        """Close the memory system, saving all state."""
        self._db.close()

    def __enter__(self) -> AriadneMemory:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def _check_contradictions(self, content: str) -> list[dict[str, Any]]:
        """Check if new content contradicts existing memories."""
        try:
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
        except Exception:
            return []


class _ToolExecutor:
    """Maps agent tool calls to AriadneMemory methods."""

    def __init__(self, memory: AriadneMemory) -> None:
        self._memory = memory

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool call by name with arguments."""
        handlers = {
            "remember": self._handle_remember,
            "recall": self._handle_recall,
            "recall_graph": self._handle_graph,
            "link_entities": self._handle_link,
            "forget": self._handle_forget,
            "memory_stats": self._handle_stats,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}

        return handler(arguments)

    def _handle_remember(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self._memory.remember(
            content=args["content"],
            memory_type=args.get("memory_type", "semantic"),
            importance=args.get("importance", 0.5),
            entities=args.get("entities"),
        )
        return result

    def _handle_recall(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        results = self._memory.recall(
            query=args["query"],
            k=args.get("k", 5),
            type_filter=args.get("memory_type"),
        )
        # Simplify for LLM consumption
        return [
            {
                "id": r["id"],
                "content": r["content"],
                "score": r.get("score", 0),
                "type": r.get("memory_type", ""),
            }
            for r in results
        ]

    def _handle_graph(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._memory.graph(
            entity=args["entity"],
            hops=args.get("hops", 2),
        )

    def _handle_link(self, args: dict[str, Any]) -> dict[str, Any]:
        self._memory.add_edge(
            source=args["source"],
            target=args["target"],
            edge_type=args.get("relationship", "related"),
        )
        return {"status": "created"}

    def _handle_forget(self, args: dict[str, Any]) -> dict[str, Any]:
        success = self._memory.forget(args["memory_id"])
        return {"status": "forgotten" if success else "not_found"}

    def _handle_stats(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._memory.stats()
