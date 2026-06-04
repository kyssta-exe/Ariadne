"""Unified memory interface for Ariadne.

Provides AriadneMemory — the single entry point that bundles:
- Storage (SQLite + FAISS + FTS5)
- Embeddings (auto-detected: ONNX / SentenceTransformers / keyword)
- Deduplication (MinHash LSH with persistence)
- Knowledge graph
- Conversation memory
- Agent tools (OpenAI function calling)
- LLM-powered extraction (OpenAI / Anthropic / Ollama)
- Entity resolution (spaCy NER + vector matching)
- Temporal knowledge graph (valid_at / invalid_at)
- Memory consolidation (similarity / topic / temporal)
- Three-tier lifecycle (hot / warm / cold)
"""

from __future__ import annotations

import logging
import time
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
        llm_config: dict[str, Any] | None = None,
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
        if self._embedder.dimension != config.embedding_dim:
            logger.info(
                "Syncing embedding_dim: %d -> %d (provider: %s)",
                config.embedding_dim, self._embedder.dimension, self._embedder.name,
            )
            config.embedding_dim = self._embedder.dimension
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

        # === NEW: LLM-powered intelligence (lazy-init) ===
        self._llm_provider = None
        self._llm_config = llm_config or {}
        if self._llm_config:
            try:
                from arriadne.llm import LLMProvider
                self._llm_provider = LLMProvider.from_config(self._llm_config)
                logger.info("LLM provider initialized: %s", self._llm_provider.name)
            except Exception as e:
                logger.warning("Failed to initialize LLM provider: %s", e)

        # === NEW: Entity resolution (lazy-init) ===
        self._entity_resolver = None

        # === NEW: Temporal graph (lazy-init) ===
        self._temporal = None

        # === NEW: Consolidator (lazy-init) ===
        self._consolidator = None

        # === NEW: Lifecycle manager (lazy-init) ===
        self._lifecycle = None

        # === NEW: Extraction engine (lazy-init) ===
        self._extractor = None

        logger.info(
            "AriadneMemory initialized (db=%s, embedding=%s, llm=%s)",
            config.db_path, self._embedder.name,
            self._llm_provider.name if self._llm_provider else "none",
        )

    def _get_llm(self):
        """Get or initialize the LLM provider."""
        if self._llm_provider is None and self._llm_config:
            from arriadne.llm import LLMProvider
            self._llm_provider = LLMProvider.from_config(self._llm_config)
        if self._llm_provider is None:
            try:
                from arriadne.llm import LLMProvider
                self._llm_provider = LLMProvider.auto_detect()
            except Exception:
                return None
        return self._llm_provider

    def _get_extractor(self):
        """Get or initialize the extraction engine."""
        if self._extractor is None:
            llm = self._get_llm()
            if llm is None:
                return None
            from arriadne.extraction import MemoryExtractor
            self._extractor = MemoryExtractor(llm)
        return self._extractor

    def _get_entity_resolver(self):
        """Get or initialize the entity resolver."""
        if self._entity_resolver is None:
            from arriadne.entity_resolution import EntityResolver
            self._entity_resolver = EntityResolver(
                embedding_provider=self._embedder if self._embedder.name != "keyword" else None,
                llm_provider=self._get_llm(),
            )
        return self._entity_resolver

    def _get_temporal(self):
        """Get or initialize the temporal graph."""
        if self._temporal is None:
            from arriadne.temporal import TemporalGraph
            self._temporal = TemporalGraph(self._db.conn)
        return self._temporal

    def _get_consolidator(self):
        """Get or initialize the consolidator."""
        if self._consolidator is None:
            from arriadne.consolidation import MemoryConsolidator
            self._consolidator = MemoryConsolidator(
                db_conn=self._db.conn,
                embedding_provider=self._embedder if self._embedder.name != "keyword" else None,
                llm_provider=self._get_llm(),
            )
        return self._consolidator

    def _get_lifecycle(self):
        """Get or initialize the lifecycle manager."""
        if self._lifecycle is None:
            from arriadne.lifecycle import MemoryLifecycle
            self._lifecycle = MemoryLifecycle(self._db.conn)
        return self._lifecycle

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
        tenant_id: str = "default",
        category: str = "semantic",
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
            tenant_id: Multi-tenant isolation key (default: "default").
            category: Memory lifecycle category (episodic/semantic/procedural/working).

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
                tenant_id=tenant_id,
                category=category,
            )

            memory_id = storage_result["memory_id"]
            result["memory_id"] = memory_id
            result["status"] = storage_result["status"]

            # Add to dedup index
            if storage_result["status"] == "created":
                self._dedup.add(content, doc_id=str(memory_id))

            # Auto-resolve entities
            try:
                resolver = self._get_entity_resolver()
                resolver.resolve(content, memory_id=str(memory_id))
            except Exception:
                pass

            return result

        except Exception as e:
            logger.error("Error in remember: %s", e)
            result["error"] = str(e)
            return result

    def remember_batch(
        self,
        contents: list[str],
        memory_type: str = "semantic",
        importance: float = 0.5,
        auto_embed: bool = True,
        batch_size: int = 500,
        category: str = "semantic",
    ) -> list[dict[str, Any]]:
        """Store multiple memories in a batch for maximum throughput.

        Uses the optimized ``add_memory_batch`` path with a single SQLite
        transaction and a single FAISS batch add. 10-50x faster than calling
        ``remember()`` in a loop.

        Args:
            contents: List of text contents to remember.
            memory_type: Category for all memories (semantic, episodic, etc.).
            importance: Importance score for all memories (0.0-1.0).
            auto_embed: Whether to auto-generate embeddings (default True).
            batch_size: Sub-batch size for embedding generation (default 500).

        Returns:
            List of result dicts with "memory_id" and "status".
        """
        if not contents:
            return []

        all_results: list[dict[str, Any]] = []

        # Process in sub-batches to limit memory usage for embedding generation
        for start in range(0, len(contents), batch_size):
            chunk = contents[start:start + batch_size]

            # Batch-embed all texts at once
            embeddings: list[np.ndarray | None] = []
            if auto_embed:
                embeddings = self._auto_embed_batch(chunk)

            # Build items list for storage layer
            items: list[dict[str, Any]] = []
            for i, content in enumerate(chunk):
                item: dict[str, Any] = {
                    "content": content,
                    "memory_type": memory_type,
                    "importance": importance,
                    "category": category,
                }
                if i < len(embeddings) and embeddings[i] is not None:
                    item["embedding"] = embeddings[i]
                items.append(item)

            # Single batch call to storage layer
            results = self._db.add_memory_batch(items)

            # Add to dedup index
            for content_text, res in zip(chunk, results):
                if res["status"] == "created":
                    self._dedup.add(content_text, doc_id=str(res["memory_id"]))

            all_results.extend(results)

        logger.info(
            "Batch remembered %d items (%d created, %d duplicates)",
            len(contents),
            sum(1 for r in all_results if r["status"] == "created"),
            sum(1 for r in all_results if r["status"] == "duplicate"),
        )
        return all_results

    def recall(
        self,
        query: str,
        embedding: list[float] | np.ndarray | None = None,
        k: int = 10,
        type_filter: str | None = None,
        time_range: tuple[float, float] | None = None,
        importance_min: float | None = None,
        auto_embed: bool = True,
        category_filter: str | None = None,
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
                if category_filter and mem.get("category", "semantic") != category_filter:
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

    # ─── LLM-Powered Intelligence ─────────────────────────────

    def extract_from_conversation(
        self,
        messages: list[dict[str, str]],
        auto_store: bool = False,
        observation_date: str | None = None,
    ) -> list[Any]:
        """Extract memories from a conversation using LLM.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            auto_store: Automatically store extracted memories
            observation_date: Current date for temporal grounding

        Returns:
            List of ExtractedMemory objects
        """
        extractor = self._get_extractor()
        if extractor is None:
            logger.warning("No LLM provider for extraction")
            return []

        # Get recent memory texts for dedup
        recent_texts = []
        try:
            cursor = self._db.conn.execute(
                "SELECT content FROM memories WHERE is_deleted = 0 ORDER BY created_at DESC LIMIT 10"
            )
            recent_texts = [row[0] for row in cursor.fetchall()]
        except Exception:
            pass

        extracted = extractor.extract_from_conversation(
            messages,
            observation_date=observation_date,
            existing_memory_texts=recent_texts,
        )

        if auto_store:
            for mem in extracted:
                self.remember(
                    content=mem.text,
                    importance=mem.importance / 10.0,
                    entities=mem.entities,
                    metadata={"attributed_to": mem.attributed_to, "topic": mem.topic},
                )

        return extracted

    def detect_contradictions_llm(
        self,
        new_memory: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """LLM-powered contradiction detection against existing memories."""
        extractor = self._get_extractor()
        if extractor is None:
            return self._check_contradictions(new_memory)

        # Get candidate memories
        candidates = self.recall(new_memory, k=limit)
        existing = [{"id": str(m["id"]), "text": m["content"]} for m in candidates]

        if not existing:
            return []

        return extractor.detect_contradictions(new_memory, existing)

    def consolidate_with_llm(
        self,
        method: str = "similarity",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Consolidate related memories using LLM-powered merging."""
        consolidator = self._get_consolidator()
        result = consolidator.consolidate_all(method=method, dry_run=dry_run)
        return {
            "groups_processed": result.groups_processed,
            "memories_before": result.memories_before,
            "memories_after": result.memories_after,
            "compression_ratio": result.compression_ratio,
            "latency_ms": result.latency_ms,
        }

    def run_lifecycle(self) -> dict[str, Any]:
        """Run the memory lifecycle process (hot/warm/cold tiers)."""
        lifecycle = self._get_lifecycle()
        return lifecycle.run_lifecycle()

    def get_entities(
        self,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get resolved entities from the knowledge graph."""
        resolver = self._get_entity_resolver()
        if entity_type:
            entities = resolver.get_entities_by_type(entity_type)
        else:
            entities = resolver.get_all_entities()

        return [
            {
                "id": e.id,
                "name": e.name,
                "type": e.entity_type,
                "aliases": e.aliases,
                "mention_count": e.mention_count,
                "linked_memories": len(e.linked_memory_ids),
            }
            for e in entities[:limit]
        ]

    def graph_search(
        self,
        entity: str,
        hops: int = 2,
    ) -> dict[str, Any]:
        """Search the knowledge graph from an entity."""
        return self.graph(entity, hops=hops)

    def graph_add_edge(
        self,
        source: str,
        target: str,
        relation: str = "related",
        weight: float = 1.0,
    ) -> None:
        """Add an edge to the knowledge graph."""
        self.add_edge(source, target, relation, weight)

    def add_temporal_fact(
        self,
        text: str,
        subject: str,
        predicate: str,
        obj: str,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a fact to the temporal knowledge graph."""
        temporal = self._get_temporal()
        fact = temporal.add_fact(
            text=text,
            subject=subject,
            predicate=predicate,
            obj=obj,
            source_memory_id=memory_id,
        )
        return {
            "fact_id": fact.fact_id,
            "text": fact.text,
            "subject": fact.subject,
            "valid_at": fact.valid_at,
            "is_current": fact.is_current,
        }

    def query_temporal(
        self,
        subject: str | None = None,
        at_time: float | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query temporal facts."""
        temporal = self._get_temporal()
        facts = temporal.find_facts(subject=subject, at_time=at_time, limit=limit)
        return [
            {
                "fact_id": f.fact_id,
                "text": f.text,
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "valid_at": f.valid_at,
                "invalid_at": f.invalid_at,
                "is_current": f.is_current,
            }
            for f in facts
        ]

    # ─── Compatibility aliases (server API) ─────────────────────

    def store(
        self,
        content: str,
        topic: str = "general",
        importance: int = 5,
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Store a memory (alias for remember, server-compatible)."""
        result = self.remember(
            content=content,
            memory_type=topic,
            importance=importance / 10.0,
            entities=entities,
            metadata=metadata,
            tenant_id=tenant_id,
        )
        # Enrich with fields the server expects
        mem_id = result.get("memory_id")
        return {
            "id": str(mem_id) if mem_id else None,
            "content": content,
            "topic": topic,
            "importance": importance,
            "status": result.get("status", "unknown"),
            "created_at": time.time(),
            "metadata": metadata or {},
        }

    def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.5,
        use_hybrid: bool = True,
        include_graph: bool = False,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """Search memories (alias for recall, server-compatible)."""
        results = self.recall(query, k=limit)
        enriched = []
        for r in results:
            enriched.append({
                "id": str(r["id"]),
                "content": r["content"],
                "score": r.get("score", 0),
                "topic": r.get("memory_type", ""),
                "importance": r.get("importance", 5),
                "created_at": r.get("created_at", 0),
                "metadata": r.get("metadata", {}),
            })

        if include_graph and enriched:
            # Add graph connections for top results
            try:
                resolver = self._get_entity_resolver()
                for r in enriched[:3]:
                    entities = resolver._extractor.extract_names(r["content"])
                    if entities:
                        r["entities"] = entities
            except Exception:
                pass

        return enriched

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a memory by ID."""
        try:
            cursor = self._db.conn.execute(
                "SELECT id, content, memory_type, importance, created_at, metadata FROM memories WHERE id = ? AND is_deleted = 0",
                (int(memory_id),),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": str(row[0]),
                "content": row[1],
                "topic": row[2] or "",
                "importance": int(row[3] or 5),
                "created_at": row[4],
                "metadata": row[5] or {},
            }
        except (ValueError, TypeError):
            return None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID string."""
        try:
            return self.forget(int(memory_id))
        except (ValueError, TypeError):
            return False

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

    # ─── Advanced Features ─────────────────────────────────────

    def get_category_stats(self) -> dict[str, Any]:
        """Get statistics for each memory category."""
        from arriadne.categories import MemoryCategoryManager
        manager = MemoryCategoryManager()
        return manager.get_category_stats(self._db.conn)

    def get_importance_stats(self) -> dict[str, Any]:
        """Get importance score distribution statistics."""
        cursor = self._db.conn.execute(
            """SELECT importance, COUNT(*) as cnt
               FROM memories WHERE is_deleted = 0
               GROUP BY ROUND(importance * 10) / 10
               ORDER BY importance"""
        )
        distribution = {row[0]: row[1] for row in cursor.fetchall()}

        cursor = self._db.conn.execute(
            """SELECT AVG(importance), MIN(importance), MAX(importance),
                      AVG(access_count)
               FROM memories WHERE is_deleted = 0"""
        )
        row = cursor.fetchone()

        return {
            "distribution": distribution,
            "avg_importance": round(float(row[0] or 0), 4),
            "min_importance": round(float(row[1] or 0), 4),
            "max_importance": round(float(row[2] or 0), 4),
            "avg_access_count": round(float(row[3] or 0), 2),
        }

    def recompute_importance(self) -> int:
        """Recompute importance for all memories based on access patterns and category."""
        from arriadne.categories import MemoryCategoryManager

        manager = MemoryCategoryManager()
        cursor = self._db.conn.execute(
            """SELECT id, importance, category, access_count, accessed_at, created_at
               FROM memories WHERE is_deleted = 0"""
        )
        updated = 0
        now = time.time()

        for row in cursor.fetchall():
            mem_id = row[0]
            importance = row[1] or 0.5
            category = row[2] or "semantic"
            access_count = row[3] or 0
            accessed_at = row[4] or row[5]

            config = manager.get_config(category)
            days_since_access = (now - accessed_at) / 86400.0

            # Apply decay
            new_importance = config.apply_decay(importance, days_since_access)

            # Apply access boost
            new_importance = config.apply_access_boost(new_importance, access_count)

            # Clamp
            new_importance = max(config.min_importance, min(config.max_importance, new_importance))

            if abs(new_importance - importance) > 0.001:
                self._db.conn.execute(
                    "UPDATE memories SET importance = ? WHERE id = ?",
                    (round(new_importance, 4), mem_id),
                )
                updated += 1

        self._db.conn.commit()
        logger.info("Recomputed importance for %d memories", updated)
        return updated

    def get_graph_stats(self) -> dict[str, Any]:
        """Get comprehensive knowledge graph statistics."""
        from arriadne.visualization import get_graph_stats
        return get_graph_stats(self)

    def export_dot(self, path: str) -> dict[str, Any]:
        """Export knowledge graph as DOT/Graphviz format."""
        from arriadne.visualization import export_dot
        return export_dot(self, path)

    def export_mermaid(self, path: str) -> dict[str, Any]:
        """Export knowledge graph as Mermaid diagram."""
        from arriadne.visualization import export_mermaid
        return export_mermaid(self, path)

    def export_json_graph(self, path: str) -> dict[str, Any]:
        """Export knowledge graph as D3.js-compatible JSON."""
        from arriadne.visualization import export_json_graph
        return export_json_graph(self, path)

    def export_json(self, path: str) -> dict[str, Any]:
        """Export all memories to JSON."""
        from arriadne.migration import export_json
        return export_json(self, path)

    def import_json(self, path: str) -> dict[str, Any]:
        """Import memories from Ariadne JSON."""
        from arriadne.migration import import_json
        return import_json(self, path)

    def import_from_text(self, path: str, category: str = "semantic") -> dict[str, Any]:
        """Import memories from plain text file."""
        from arriadne.migration import import_from_text
        return import_from_text(self, path, category=category)

    def import_from_markdown(self, path: str) -> dict[str, Any]:
        """Import memories from markdown file."""
        from arriadne.migration import import_from_markdown
        return import_from_markdown(self, path)

    def export_markdown(self, path: str) -> dict[str, Any]:
        """Export memories as human-readable markdown."""
        from arriadne.migration import export_markdown
        return export_markdown(self, path)

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
