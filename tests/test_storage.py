"""Tests for AriadneDB storage layer."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from arriadne.config import AriadneConfig
from arriadne.storage import AriadneDB, _fts_escape, _hash_content, _jaccard_similarity


@pytest.fixture
def db() -> AriadneDB:
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        config = AriadneConfig(
            db_path=db_path,
            embedding_dim=8,  # Small for fast tests
            ivf_threshold=5,  # Low threshold for testing upgrade
        )
        with AriadneDB(config) as database:
            yield database
    finally:
        # Cleanup
        for suffix in ["", "-wal", "-shm", ".faiss"]:
            p = Path(db_path + suffix)
            if p.exists():
                p.unlink()


@pytest.fixture
def db_with_vectors() -> AriadneDB:
    """Create a database pre-populated with vectors."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        config = AriadneConfig(db_path=db_path, embedding_dim=8)
        with AriadneDB(config) as database:
            for i in range(5):
                emb = np.random.randn(8).astype(np.float32)
                database.add_memory(
                    content=f"Test memory {i} about topic {i % 3}",
                    embedding=emb,
                    memory_type="semantic",
                    importance=0.3 + i * 0.15,
                )
            yield database
    finally:
        for suffix in ["", "-wal", "-shm", ".faiss"]:
            p = Path(db_path + suffix)
            if p.exists():
                p.unlink()


class TestHashAndEscape:
    """Tests for utility functions."""

    def test_hash_content(self) -> None:
        h = _hash_content("hello world")
        assert len(h) == 64  # SHA-256 hex
        assert _hash_content("hello world") == h
        assert _hash_content("hello world!") != h

    def test_fts_escape_simple(self) -> None:
        result = _fts_escape("hello world")
        assert result == '"hello" OR "world"'

    def test_fts_escape_special_chars(self) -> None:
        result = _fts_escape('test "quotes" and *stars*')
        assert '"test"' in result
        assert '"quotes"' in result
        assert '"stars"' in result

    def test_fts_escape_empty(self) -> None:
        result = _fts_escape("")
        assert result == '""'

    def test_jaccard_identical(self) -> None:
        assert _jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self) -> None:
        assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_partial(self) -> None:
        result = _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        # Intersection: {"b", "c"} = 2, Union: {"a", "b", "c", "d"} = 4, Jaccard = 0.5
        assert abs(result - 0.5) < 1e-6

    def test_jaccard_empty(self) -> None:
        assert _jaccard_similarity(set(), set()) == 0.0


class TestMemoryCRUD:
    """Tests for memory create, read, update, delete."""

    def test_add_memory(self, db: AriadneDB) -> None:
        result = db.add_memory("Hello world", memory_type="semantic")
        assert result["status"] == "created"
        assert result["memory_id"] > 0

    def test_add_duplicate(self, db: AriadneDB) -> None:
        r1 = db.add_memory("Hello world")
        r2 = db.add_memory("Hello world")
        assert r1["status"] == "created"
        assert r2["status"] == "duplicate"
        assert r1["memory_id"] == r2["memory_id"]

    def test_duplicate_hash_is_scoped_by_namespace(self, db: AriadneDB) -> None:
        """Identical content in different namespaces is not a duplicate."""
        r1 = db.add_memory("Shared wording", namespace="project-a")
        r2 = db.add_memory("Shared wording", namespace="project-b")

        assert r1["status"] == "created"
        assert r2["status"] == "created"
        assert r1["memory_id"] != r2["memory_id"]

    def test_get_memory(self, db: AriadneDB) -> None:
        result = db.add_memory("Test content", importance=0.8)
        # get_memory is a pure read now — it does NOT bump access_count.
        memory = db.get_memory(result["memory_id"])
        assert memory is not None
        assert memory["content"] == "Test content"
        assert memory["importance"] == 0.8
        assert memory["namespace"] == "default"
        assert memory["scope"] == "session"
        assert memory["access_count"] == 0  # pure read, no mutation

        # Access is recorded explicitly via touch.
        db.touch_memory(result["memory_id"])
        memory = db.get_memory(result["memory_id"])
        assert memory is not None
        assert memory["access_count"] == 1

    def test_add_memory_stores_namespace_scope_and_ids(self, db: AriadneDB) -> None:
        result = db.add_memory(
            "Namespaced content",
            namespace="org/user/project",
            scope="project",
            user_id="user-1",
            agent_id="agent-1",
            session_id="session-1",
            project_id="project-1",
        )

        memory = db.get_memory(result["memory_id"])

        assert memory is not None
        assert memory["namespace"] == "org/user/project"
        assert memory["scope"] == "project"
        assert memory["user_id"] == "user-1"
        assert memory["agent_id"] == "agent-1"
        assert memory["session_id"] == "session-1"
        assert memory["project_id"] == "project-1"

    def test_fts_search_filters_by_namespace(self, db: AriadneDB) -> None:
        db.add_memory("deploy secret only in project alpha", namespace="alpha")
        db.add_memory("deploy secret only in project beta", namespace="beta")

        alpha = db.fts_search("deploy secret", k=10, namespace="alpha")
        beta = db.fts_search("deploy secret", k=10, namespace="beta")

        assert [m["namespace"] for m in alpha] == ["alpha"]
        assert [m["namespace"] for m in beta] == ["beta"]
        assert alpha[0]["id"] != beta[0]["id"]

    def test_get_nonexistent(self, db: AriadneDB) -> None:
        assert db.get_memory(99999) is None

    def test_update_memory(self, db: AriadneDB) -> None:
        result = db.add_memory("Original content", importance=0.3)
        updated = db.update_memory(result["memory_id"], content="Updated", importance=0.9)
        assert updated is True
        memory = db.get_memory(result["memory_id"])
        assert memory is not None
        assert memory["content"] == "Updated"
        assert memory["importance"] == 0.9

    def test_soft_delete(self, db: AriadneDB) -> None:
        result = db.add_memory("To be deleted")
        deleted = db.delete_memory(result["memory_id"], hard=False)
        assert deleted is True
        db.get_memory(result["memory_id"])
        # Soft deleted memories may still be returned by get_memory
        # but should not appear in search results

    def test_hard_delete(self, db: AriadneDB) -> None:
        result = db.add_memory("To be hard deleted")
        deleted = db.delete_memory(result["memory_id"], hard=True)
        assert deleted is True
        assert db.get_memory(result["memory_id"]) is None

    def test_delete_nonexistent(self, db: AriadneDB) -> None:
        assert db.delete_memory(99999) is False

    def test_add_with_entities(self, db: AriadneDB) -> None:
        result = db.add_memory(
            "Paris is in France",
            entities=["Paris", "France"],
        )
        assert result["status"] == "created"
        # Verify entities were created
        cursor = db.conn.execute("SELECT COUNT(*) FROM entities")
        assert cursor.fetchone()[0] == 2

    def test_add_with_metadata(self, db: AriadneDB) -> None:
        result = db.add_memory(
            "Test with metadata",
            metadata={"source": "test", "score": 42},
        )
        memory = db.get_memory(result["memory_id"])
        assert memory is not None
        assert memory["metadata"]["source"] == "test"
        assert memory["metadata"]["score"] == 42


class TestVectorSearch:
    """Tests for FAISS vector search."""

    def test_vector_search(self, db_with_vectors: AriadneDB) -> None:
        query = np.ones(8, dtype=np.float32)
        results = db_with_vectors.vector_search(query, k=3)
        assert len(results) <= 3
        assert len(results) > 0
        for r in results:
            assert "score" in r
            assert r["search_type"] == "vector"

    def test_vector_search_empty(self, db: AriadneDB) -> None:
        query = np.ones(8, dtype=np.float32)
        results = db.vector_search(query, k=5)
        assert results == []

    def test_vector_search_respects_k(self, db_with_vectors: AriadneDB) -> None:
        query = np.ones(8, dtype=np.float32)
        results = db_with_vectors.vector_search(query, k=2)
        assert len(results) <= 2


class TestFTSSearch:
    """Tests for full-text search."""

    def test_fts_search(self, db: AriadneDB) -> None:
        db.add_memory("The quick brown fox")
        db.add_memory("A lazy dog sleeps")
        db.add_memory("The brown dog runs")

        results = db.fts_search("brown", k=10)
        assert len(results) >= 1
        for r in results:
            assert "brown" in r["content"].lower()

    def test_fts_search_empty(self, db: AriadneDB) -> None:
        results = db.fts_search("nonexistent", k=10)
        assert results == []

    def test_fts_search_special_chars(self, db: AriadneDB) -> None:
        db.add_memory("Test with special *chars* and \"quotes\"")
        # Should not crash
        results = db.fts_search("special chars", k=10)
        assert isinstance(results, list)


class TestHybridSearch:
    """Tests for hybrid search with RRF."""

    def test_hybrid_search(self, db: AriadneDB) -> None:
        db.add_memory("Python programming language", memory_type="semantic")
        db.add_memory("JavaScript web development", memory_type="semantic")
        db.add_memory("Python data science", memory_type="semantic")

        # Search with both text and embedding
        emb = np.random.randn(8).astype(np.float32)
        results = db.hybrid_search("Python", embedding=emb, k=5)
        assert len(results) > 0
        # Python memories should rank higher
        assert any("Python" in r["content"] for r in results)

    def test_hybrid_search_fts_only(self, db: AriadneDB) -> None:
        db.add_memory("Machine learning algorithms")
        db.add_memory("Deep neural networks")

        results = db.hybrid_search("machine learning", k=5)
        assert len(results) > 0


class TestGraphOperations:
    """Tests for knowledge graph operations."""

    def test_add_edge(self, db: AriadneDB) -> None:
        db.add_edge("Python", "programming", "is_a")
        cursor = db.conn.execute("SELECT COUNT(*) FROM edges")
        assert cursor.fetchone()[0] == 1

    def test_traverse_graph(self, db: AriadneDB) -> None:
        db.add_edge("Python", "programming", "is_a")
        db.add_edge("programming", "computer_science", "is_a")
        db.add_edge("Python", "data_science", "used_for")

        result = db.traverse_graph("Python", hops=2)
        assert "Python" in result["nodes"]
        assert len(result["nodes"]) > 1
        assert len(result["edges"]) > 0

    def test_traverse_graph_no_entity(self, db: AriadneDB) -> None:
        result = db.traverse_graph("nonexistent", hops=1)
        assert result["nodes"] == ["nonexistent"]
        assert result["edges"] == []

    def test_traverse_graph_with_edge_type(self, db: AriadneDB) -> None:
        db.add_edge("A", "B", "related")
        db.add_edge("A", "C", "unrelated")

        result = db.traverse_graph("A", hops=1, edge_type="related")
        assert "B" in result["nodes"]


class TestLifecycle:
    """Tests for retention scoring, priority scoring, eviction."""

    def test_retention_strength(self, db: AriadneDB) -> None:
        now = time.time()
        memory = {
            "importance": 1.0,
            "accessed_at": now,  # Just accessed
            "access_count": 1,
            "created_at": now,
            "retention_strength": 1.0,
        }
        score = db.compute_retention_strength(memory)
        # Recently accessed should have high retention
        assert score > 0.5

    def test_retention_strength_old(self, db: AriadneDB) -> None:
        old_time = time.time() - 86400 * 30  # 30 days ago
        memory = {
            "importance": 0.5,
            "accessed_at": old_time,
            "access_count": 1,
            "created_at": old_time,
            "retention_strength": 1.0,
        }
        score = db.compute_retention_strength(memory)
        # Old memory should have low retention
        assert score < 0.5

    def test_priority_score(self, db: AriadneDB) -> None:
        now = time.time()
        high_priority = {
            "importance": 1.0,
            "created_at": now,
            "accessed_at": now,
            "access_count": 50,
            "retention_strength": 1.0,
        }
        low_priority = {
            "importance": 0.1,
            "created_at": now - 86400 * 30,
            "accessed_at": now - 86400 * 30,
            "access_count": 0,
            "retention_strength": 0.0,
        }
        high_score = db.compute_priority_score(high_priority)
        low_score = db.compute_priority_score(low_priority)
        assert high_score > low_score

    def test_eviction(self, db: AriadneDB) -> None:
        # Add some memories
        for i in range(10):
            importance = i / 10.0
            db.add_memory(
                f"Memory {i}",
                importance=importance,
            )

        # Without a capacity configured, eviction is a no-op: Ariadne never
        # destroys memories implicitly (the old behavior silently soft-deleted
        # 10% of the store on every maintenance pass).
        assert db.evict() == 0
        cursor = db.conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
        assert cursor.fetchone()[0] == 10

        # Over an explicit capacity, eviction removes the overflow — capped by
        # eviction_budget (10% of the store per run) — lowest priority first.
        evicted = db.evict(max_memories=8)
        assert evicted == 1
        # Repeated runs walk the store down to the capacity.
        while db.evict(max_memories=8):
            pass
        cursor = db.conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
        assert cursor.fetchone()[0] == 8
        # The two lowest-importance memories (0.0 and 0.1) are the ones gone.
        cursor = db.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE is_deleted = 0 AND importance < 0.15"
        )
        assert cursor.fetchone()[0] == 0

        # At/under capacity again: nothing more is evicted.
        assert db.evict(max_memories=8) == 0
        assert db.evict(max_memories=20) == 0

    def test_eviction_config_capacity(self, db: AriadneDB) -> None:
        """config.max_memories drives eviction when no explicit arg is passed."""
        db._config.max_memories = 6
        for i in range(10):
            db.add_memory(f"Config capacity memory {i}", importance=i / 10.0)
        # budget caps a single run at 10% of the store (1 memory here)
        evicted = db.evict()
        assert evicted == 1
        # Repeated runs walk the store back down to the capacity.
        while db.evict():
            pass
        cursor = db.conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
        assert cursor.fetchone()[0] == 6

    def test_consolidation(self, db: AriadneDB) -> None:
        # Add similar memories
        db.add_memory("Python is a programming language")
        db.add_memory("Python is a programming language for data science")
        db.add_memory("Python is a programming language for web")

        # These share enough words to potentially consolidate
        result = db.consolidate()
        # Just check it doesn't crash
        assert isinstance(result, int)


class TestContextManager:
    """Tests for context manager support."""

    def test_context_manager(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            config = AriadneConfig(db_path=db_path, embedding_dim=8)
            with AriadneDB(config) as db:
                db.add_memory("Context test")
                cursor = db.conn.execute("SELECT COUNT(*) FROM memories")
                assert cursor.fetchone()[0] == 1
        finally:
            for suffix in ["", "-wal", "-shm", ".faiss"]:
                p = Path(db_path + suffix)
                if p.exists():
                    p.unlink()


class TestStats:
    """Tests for the stats method."""

    def test_stats_empty(self, db: AriadneDB) -> None:
        stats = db.stats()
        assert stats["active_memories"] == 0
        assert stats["total_entities"] == 0
        assert stats["total_edges"] == 0
        assert stats["faiss_vectors"] == 0

    def test_stats_populated(self, db: AriadneDB) -> None:
        db.add_memory("Test 1", importance=0.5)
        db.add_memory("Test 2", importance=0.8, entities=["Entity1"])
        db.add_edge("Entity1", "Entity2")

        stats = db.stats()
        assert stats["active_memories"] == 2
        assert stats["total_entities"] >= 2
        assert stats["total_edges"] == 1
        assert stats["faiss_vectors"] == 0  # No embeddings added


class TestFAISSIndex:
    """Tests for FAISS index operations."""

    def test_faiss_auto_upgrade(self, db: AriadneDB) -> None:
        """Test that FAISS index upgrades from FlatIP to IVFFlat."""
        # ivf_threshold is 5 in the fixture
        for i in range(8):  # Exceed threshold
            emb = np.random.randn(8).astype(np.float32)
            db.add_memory(f"Vector {i}", embedding=emb)

        # After adding enough vectors, should be IVFFlat (wrapped in IndexIDMap2)
        import faiss
        assert isinstance(db._faiss_index, faiss.IndexIDMap2)
        assert isinstance(faiss.downcast_index(db._faiss_index.index), faiss.IndexIVFFlat)

    def test_faiss_search_after_upgrade(self, db: AriadneDB) -> None:
        """Test search works after index upgrade."""
        embeddings = []
        for i in range(8):
            emb = np.random.randn(8).astype(np.float32)
            embeddings.append(emb)
            db.add_memory(f"Vector {i}", embedding=emb)

        # Search should still work
        results = db.vector_search(embeddings[0], k=3)
        assert len(results) > 0
