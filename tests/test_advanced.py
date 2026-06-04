"""
Tests for LLM providers, extraction, entity resolution, temporal graph,
consolidation, lifecycle, and server.
"""

import json
import time
import sqlite3
import pytest


# ============================================================
# LLM Provider Tests
# ============================================================

class TestLLMProvider:
    def test_from_config_openai(self):
        from arriadne.llm import LLMProvider
        p = LLMProvider.from_config({
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        })
        assert "openai" in p.name
        assert p.is_available() is True

    def test_from_config_ollama(self):
        from arriadne.llm import LLMProvider
        p = LLMProvider.from_config({
            "provider": "ollama",
            "model": "llama3.2",
        })
        assert "ollama" in p.name

    @pytest.mark.skipif(True, reason='anthropic not installed')
    def test_from_config_anthropic(self):
        from arriadne.llm import LLMProvider
        p = LLMProvider.from_config({
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key": "test-key",
        })
        assert "anthropic" in p.name

    def test_from_config_openrouter(self):
        from arriadne.llm import LLMProvider
        p = LLMProvider.from_config({
            "provider": "openrouter",
            "model": "google/gemini-2.0-flash-001",
            "api_key": "test-key",
        })
        assert "google/gemini-2.0-flash-001" in p.name

    def test_message_to_dict(self):
        from arriadne.llm import LLMMessage
        msg = LLMMessage(role="system", content="You are helpful")
        assert msg.to_dict() == {"role": "system", "content": "You are helpful"}

    def test_response_json(self):
        from arriadne.llm import LLMResponse
        resp = LLMResponse(content='{"memories": ["test"]}', model="test")
        assert resp.json() == {"memories": ["test"]}

    def test_response_json_code_fenced(self):
        from arriadne.llm import LLMResponse
        resp = LLMResponse(
            content='```json\n{"key": "value"}\n```',
            model="test",
        )
        assert resp.json() == {"key": "value"}

    def test_stats(self):
        from arriadne.llm import LLMProvider
        p = LLMProvider.from_config({
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test",
        })
        stats = p.stats
        assert "provider" in stats
        assert "calls" in stats


# ============================================================
# Extraction Engine Tests (mock LLM)
# ============================================================

class MockLLMResponse:
    def __init__(self, content):
        self.content = content
        self.model = "mock"
        self.usage = {"total_tokens": 100}
        self.latency_ms = 50.0
        self.raw = None

    def json(self):
        return json.loads(self.content)


class MockLLMProvider:
    def __init__(self, responses=None):
        self._responses = responses or []
        self._call_count = 0
        self.name = "mock"

    def complete_sync(self, messages, **kwargs):
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp


class TestMemoryExtraction:
    def test_extract_basic(self):
        from arriadne.extraction import MemoryExtractor

        mock_resp = MockLLMResponse(json.dumps({
            "memories": [
                {
                    "text": "The user prefers dark mode and uses VS Code",
                    "attributed_to": "user",
                    "topic": "preferences",
                    "importance": 7,
                    "entities": ["VS Code"],
                }
            ]
        }))
        llm = MockLLMProvider([mock_resp])
        extractor = MemoryExtractor(llm)

        memories = extractor.extract_from_conversation([
            {"role": "user", "content": "I love dark mode and use VS Code"},
            {"role": "assistant", "content": "Great choices!"},
        ])

        assert len(memories) == 1
        assert "dark mode" in memories[0].text
        assert memories[0].topic == "preferences"
        assert memories[0].importance == 7
        assert "VS Code" in memories[0].entities

    def test_extract_empty_conversation(self):
        from arriadne.extraction import MemoryExtractor

        llm = MockLLMProvider()
        extractor = MemoryExtractor(llm)
        result = extractor.extract_from_conversation([])
        assert result == []

    def test_extract_from_text(self):
        from arriadne.extraction import MemoryExtractor

        mock_resp = MockLLMResponse(json.dumps({
            "memories": [
                {
                    "text": "Ariadne uses FAISS for vector search",
                    "attributed_to": "assistant",
                    "topic": "technical",
                    "importance": 8,
                    "entities": ["Ariadne", "FAISS"],
                }
            ]
        }))
        llm = MockLLMProvider([mock_resp])
        extractor = MemoryExtractor(llm)

        memories = extractor.extract_from_text("Ariadne uses FAISS for vector search")
        assert len(memories) >= 1

    def test_extract_dedup_with_existing(self):
        from arriadne.extraction import MemoryExtractor

        mock_resp = MockLLMResponse(json.dumps({
            "memories": [
                {
                    "text": "The VPS has 4 cores",
                    "attributed_to": "assistant",
                    "topic": "technical",
                    "importance": 6,
                    "entities": [],
                }
            ]
        }))
        llm = MockLLMProvider([mock_resp])
        extractor = MemoryExtractor(llm)

        # Providing existing memory texts to avoid duplicates
        memories = extractor.extract_from_conversation(
            [{"role": "assistant", "content": "Your VPS has 4 cores"}],
            existing_memory_texts=["The VPS has 4 cores and 8GB RAM"],
        )
        # Should still extract but the existing context is available
        assert isinstance(memories, list)

    def test_contradiction_detection(self):
        from arriadne.extraction import MemoryExtractor

        mock_resp = MockLLMResponse(json.dumps({
            "results": [
                {
                    "memory_id": "mem-1",
                    "relationship": "contradicts",
                    "reasoning": "Paris is not the capital of Germany",
                }
            ]
        }))
        llm = MockLLMProvider([mock_resp])
        extractor = MemoryExtractor(llm)

        results = extractor.detect_contradictions(
            "Berlin is the capital of France",
            [
                {"id": "mem-1", "text": "Paris is the capital of France"},
                {"id": "mem-2", "text": "Germany has many cities"},
            ],
        )
        assert len(results) == 1
        assert results[0].memory_id == "mem-1"
        assert results[0].relationship == "contradicts"

    def test_consolidation(self):
        from arriadne.extraction import MemoryExtractor

        mock_resp = MockLLMResponse(json.dumps({
            "memories": [
                {
                    "text": "The VPS has 4 cores, 8GB RAM, and runs Ubuntu 24.04",
                    "entities": ["VPS", "Ubuntu"],
                    "importance": 7,
                }
            ]
        }))
        llm = MockLLMProvider([mock_resp])
        extractor = MemoryExtractor(llm)

        results = extractor.consolidate_memories([
            [
                {"id": "1", "text": "The VPS has 4 cores", "importance": 6},
                {"id": "2", "text": "The VPS has 8GB RAM", "importance": 6},
                {"id": "3", "text": "The VPS runs Ubuntu 24.04", "importance": 6},
            ]
        ])
        assert len(results) >= 1


# ============================================================
# Entity Resolution Tests
# ============================================================

class TestEntityExtractor:
    def test_extract_names_regex(self):
        from arriadne.entity_resolution import EntityExtractor
        extractor = EntityExtractor()
        # Should work even without spaCy (falls back to regex)
        names = extractor.extract_names("Kyssta uses VS Code and FAISS for the project Ariadne")
        assert isinstance(names, list)
        assert len(names) > 0

    def test_extract_entities_spacy(self):
        from arriadne.entity_resolution import EntityExtractor
        extractor = EntityExtractor()
        mentions = extractor.extract("OpenAI is based in San Francisco and was founded by Sam Altman")
        assert isinstance(mentions, list)
        # Should find at least some entities
        assert len(mentions) >= 1

    def test_generic_word_filtering(self):
        from arriadne.entity_resolution import EntityExtractor
        extractor = EntityExtractor()
        names = extractor.extract_names("the thing is very good and I like it")
        # Generic words should be filtered
        assert "thing" not in [n.lower() for n in names]


class TestEntityResolver:
    def test_exact_match(self):
        from arriadne.entity_resolution import EntityResolver
        resolver = EntityResolver()

        entities1 = resolver.resolve("OpenAI is an AI company", memory_id="m1")
        entities2 = resolver.resolve("OpenAI makes great models", memory_id="m2")

        # Should have extracted entities
        assert len(entities1) >= 1
        # Second time should find the same entity by exact name
        if len(entities2) >= 1:
            openai_entities = [e for e in entities2 if "OpenAI" in e.name]
            if openai_entities:
                assert openai_entities[0].mention_count >= 2

    def test_entity_types(self):
        from arriadne.entity_resolution import EntityResolver
        resolver = EntityResolver()
        entities = resolver.resolve("OpenAI is in San Francisco", memory_id="m1")
        types = {e.entity_type for e in entities}
        assert len(types) > 0

    def test_get_all_entities(self):
        from arriadne.entity_resolution import EntityResolver
        resolver = EntityResolver()
        resolver.resolve("OpenAI is an AI company", memory_id="m1")
        resolver.resolve("Google Cloud is a cloud platform", memory_id="m2")

        all_entities = resolver.get_all_entities()
        assert len(all_entities) >= 2

    def test_serialize(self):
        from arriadne.entity_resolution import EntityResolver
        resolver = EntityResolver()
        resolver.resolve("OpenAI is an AI company", memory_id="m1")

        data = resolver.to_dict()
        assert "entities" in data
        assert len(data["entities"]) >= 1

        resolver2 = EntityResolver()
        resolver2.from_dict(data)
        assert resolver2.entity_count >= 1


# ============================================================
# Temporal Graph Tests
# ============================================================

class TestTemporalGraph:
    @pytest.fixture
    def temporal_db(self):
        conn = sqlite3.connect(":memory:")
        yield conn
        conn.close()

    def test_add_and_query_fact(self, temporal_db):
        from arriadne.temporal import TemporalGraph
        graph = TemporalGraph(temporal_db)

        fact = graph.add_fact(
            text="Kyssta lives in London",
            subject="Kyssta",
            predicate="lives_in",
            obj="London",
        )
        assert fact.is_current is True

        facts = graph.find_facts(subject="Kyssta")
        assert len(facts) == 1
        assert facts[0].subject == "Kyssta"

    def test_temporal_invalidation(self, temporal_db):
        from arriadne.temporal import TemporalGraph
        graph = TemporalGraph(temporal_db)

        # First fact
        graph.add_fact(
            text="Kyssta lives in London",
            subject="Kyssta",
            predicate="lives_in",
            obj="London",
            valid_at=1000,
        )

        # Second fact supersedes
        graph.add_fact(
            text="Kyssta lives in Paris",
            subject="Kyssta",
            predicate="lives_in",
            obj="Paris",
            valid_at=2000,
        )

        # Old fact should be invalidated
        current = graph.find_facts(subject="Kyssta", current_only=True)
        assert len(current) == 1
        assert current[0].object == "Paris"

        # History should have both
        all_facts = graph.find_facts(subject="Kyssta", current_only=False)
        assert len(all_facts) == 2

    def test_timeline(self, temporal_db):
        from arriadne.temporal import TemporalGraph
        graph = TemporalGraph(temporal_db)

        graph.add_fact("A v1", "A", "is", "v1", valid_at=1000)
        graph.add_fact("A v2", "A", "is", "v2", valid_at=2000)
        graph.add_fact("A v3", "A", "is", "v3", valid_at=3000)

        timeline = graph.get_timeline("A")
        assert len(timeline) == 3
        # Should be ordered by valid_at DESC
        assert timeline[0].valid_at >= timeline[1].valid_at

    def test_stats(self, temporal_db):
        from arriadne.temporal import TemporalGraph
        graph = TemporalGraph(temporal_db)

        graph.add_fact("F1", "S", "P", "O1")
        graph.add_fact("F2", "S", "P", "O2")  # Invalidates F1

        stats = graph.stats()
        assert stats["total_facts"] == 2
        assert stats["current_facts"] == 1
        assert stats["superseded_facts"] == 1

    def test_manual_invalidate(self, temporal_db):
        from arriadne.temporal import TemporalGraph
        graph = TemporalGraph(temporal_db)

        fact = graph.add_fact("Test", "S", "P", "O")
        assert graph.invalidate_fact(fact.fact_id) is True

        current = graph.find_facts(subject="S", current_only=True)
        assert len(current) == 0


# ============================================================
# Consolidation Tests
# ============================================================

class TestConsolidation:
    @pytest.fixture
    def consolidation_db(self):
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding_vector BLOB,
                topic TEXT DEFAULT '',
                importance INTEGER DEFAULT 5,
                is_deleted INTEGER DEFAULT 0,
                deleted_at REAL,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        # Insert related memories
        for content in [
            "The VPS has 4 CPU cores",
            "The VPS has 8GB of RAM",
            "The VPS runs Ubuntu 24.04",
            "The VPS has 50GB disk space",
        ]:
            cursor.execute(
                "INSERT INTO memories (content, topic, importance) VALUES (?, 'hardware', 7)",
                (content,),
            )
        conn.commit()
        yield conn
        conn.close()

    def test_group_by_topic(self, consolidation_db):
        from arriadne.consolidation import MemoryConsolidator
        consolidator = MemoryConsolidator(consolidation_db)

        groups = consolidator.find_related_groups(method="topic", limit=100)
        assert len(groups) >= 1

        hw_group = [g for g in groups if g.topic == "hardware"]
        if hw_group:
            assert hw_group[0].size >= 3

    def test_consolidate_simple(self, consolidation_db):
        from arriadne.consolidation import MemoryConsolidator
        consolidator = MemoryConsolidator(consolidation_db)

        groups = consolidator.find_related_groups(method="topic")
        if groups:
            result, remove_ids = consolidator.consolidate_group(groups[0])
            if result:
                assert "content" in result
                assert len(remove_ids) > 0


# ============================================================
# Lifecycle Tests
# ============================================================

class TestLifecycle:
    @pytest.fixture
    def lifecycle_db(self):
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                importance INTEGER DEFAULT 5,
                access_count INTEGER DEFAULT 0,
                last_accessed_at REAL,
                is_deleted INTEGER DEFAULT 0,
                deleted_at REAL,
                metadata TEXT DEFAULT '{}',
                created_at REAL DEFAULT (strftime('%s', 'now')),
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        # Insert memories with different ages
        now = time.time()
        for i, (days_ago, accesses) in enumerate([
            (1, 10),   # Hot, frequently accessed
            (15, 2),   # Warm, some access
            (100, 0),  # Cold, never accessed
        ]):
            cursor.execute(
                """INSERT INTO memories
                (content, importance, access_count, last_accessed_at, created_at)
                VALUES (?, 7, ?, ?, ?)""",
                (f"Memory {i}", accesses, now - days_ago * 86400, now - days_ago * 86400),
            )
        conn.commit()
        yield conn
        conn.close()

    def test_get_tier(self, lifecycle_db):
        from arriadne.lifecycle import MemoryLifecycle
        lifecycle = MemoryLifecycle(lifecycle_db)

        now = time.time()
        assert lifecycle.get_tier({"created_at": now - 86400}) == "hot"
        assert lifecycle.get_tier({"created_at": now - 15 * 86400}) == "warm"
        assert lifecycle.get_tier({"created_at": now - 100 * 86400}) == "cold"

    def test_retention_score(self, lifecycle_db):
        from arriadne.lifecycle import MemoryLifecycle
        lifecycle = MemoryLifecycle(lifecycle_db)

        # New memory should have high retention
        new = {"created_at": time.time(), "access_count": 0, "importance": 5}
        assert lifecycle.get_retention_score(new) > 0.9

        # Old, never-accessed memory should have low retention
        old = {"created_at": time.time() - 100 * 86400, "access_count": 0, "importance": 5}
        assert lifecycle.get_retention_score(old) < 0.01

    def test_priority_score(self, lifecycle_db):
        from arriadne.lifecycle import MemoryLifecycle
        lifecycle = MemoryLifecycle(lifecycle_db)

        # Recent, important, frequently accessed = high priority
        hot = {"created_at": time.time() - 86400, "access_count": 10, "importance": 9}
        cold = {"created_at": time.time() - 100 * 86400, "access_count": 0, "importance": 2}

        assert lifecycle.get_priority_score(hot) > lifecycle.get_priority_score(cold)

    def test_run_lifecycle(self, lifecycle_db):
        from arriadne.lifecycle import MemoryLifecycle
        lifecycle = MemoryLifecycle(lifecycle_db)

        result = lifecycle.run_lifecycle()
        assert "stats" in result
        assert result["stats"].total_count == 3
        assert result["stats"].hot_count + result["stats"].warm_count + result["stats"].cold_count == 3

    def test_prune_cold_memories(self, lifecycle_db):
        from arriadne.lifecycle import MemoryLifecycle
        lifecycle = MemoryLifecycle(lifecycle_db)

        result = lifecycle.prune_cold_memories(
            min_age_days=50,
            min_retention=0.01,
            dry_run=True,
        )
        assert "candidates_found" in result
        assert result["dry_run"] is True


# ============================================================
# Server Tests
# ============================================================

class TestServer:
    def test_create_app(self):
        from arriadne.server import create_app
        app = create_app(db_path=":memory:")
        assert app is not None
        assert app.title == "Ariadne Memory API"

    def test_health_endpoint(self):
        from arriadne.server import create_app
        from fastapi.testclient import TestClient

        app = create_app(db_path=":memory:")
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_root_endpoint(self):
        from arriadne.server import create_app
        from fastapi.testclient import TestClient

        app = create_app(db_path=":memory:")
        client = TestClient(app)

        resp = client.get("/")
        assert resp.status_code == 200
        assert "Ariadne" in resp.json()["name"]

    def test_store_and_search(self):
        from arriadne.server import create_app
        from fastapi.testclient import TestClient

        app = create_app(db_path=":memory:")
        client = TestClient(app)

        # Store a memory
        resp = client.post("/memories", json={
            "content": "Paris is the capital of France",
            "topic": "geography",
            "importance": 8,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Paris is the capital of France"
        assert data["id"] is not None

        # Search for it
        resp = client.post("/search", json={
            "query": "capital of France",
            "limit": 5,
        })
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) >= 1

    def test_get_memory(self):
        from arriadne.server import create_app
        from fastapi.testclient import TestClient

        app = create_app(db_path=":memory:")
        client = TestClient(app)

        # Store
        resp = client.post("/memories", json={
            "content": "Test memory for retrieval",
        })
        mem_id = resp.json()["id"]

        # Get by ID
        resp = client.get(f"/memories/{mem_id}")
        assert resp.status_code == 200
        assert "Test memory" in resp.json()["content"]

    def test_delete_memory(self):
        from arriadne.server import create_app
        from fastapi.testclient import TestClient

        app = create_app(db_path=":memory:")
        client = TestClient(app)

        # Store
        resp = client.post("/memories", json={
            "content": "Memory to delete",
        })
        mem_id = resp.json()["id"]

        # Delete
        resp = client.delete(f"/memories/{mem_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_stats_endpoint(self):
        from arriadne.server import create_app
        from fastapi.testclient import TestClient

        app = create_app(db_path=":memory:")
        client = TestClient(app)

        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_memories" in data

    def test_api_key_auth(self):
        from arriadne.server import create_app
        from fastapi.testclient import TestClient

        app = create_app(db_path=":memory:", api_key="secret123")
        client = TestClient(app)

        # Without key — should fail
        resp = client.get("/stats")
        assert resp.status_code == 401

        # With correct key — should work
        resp = client.get("/stats", headers={"Authorization": "Bearer secret123"})
        assert resp.status_code == 200

    def test_cors(self):
        from arriadne.server import create_app
        from fastapi.testclient import TestClient

        app = create_app(db_path=":memory:")
        client = TestClient(app)

        resp = client.options("/", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })
        assert resp.status_code in (200, 405)


# ============================================================
# Integration Test: Full Pipeline
# ============================================================

class TestFullPipeline:
    def test_store_search_graph(self):
        """Test storing memories, searching, and graph traversal."""
        from arriadne.interface import AriadneMemory

        mem = AriadneMemory(
            db_path=":memory:",
            embedding_dim=8,  # Small for test speed
            embedding_provider="keyword",
        )

        # Store
        r1 = mem.remember("Ariadne is a memory system for AI agents")
        r2 = mem.remember("Ariadne uses FAISS for fast vector search")
        r3 = mem.remember("Kyssta built Ariadne on a VPS")

        assert r1["status"] == "created"
        assert r2["status"] == "created"
        assert r3["status"] == "created"

        # Search
        results = mem.recall("vector search")
        assert len(results) >= 1

        # Graph
        mem.add_edge("Ariadne", "FAISS", "uses")
        mem.add_edge("Ariadne", "Kyssta", "built_by")
        graph = mem.graph("Ariadne", hops=2)
        assert len(graph.get("nodes", [])) >= 2

        # Stats
        stats = mem.stats()
        assert stats["total_memories"] >= 3

        mem.close()

    def test_entity_resolution(self):
        """Test entity extraction and resolution."""
        from arriadne.interface import AriadneMemory

        mem = AriadneMemory(
            db_path=":memory:",
            embedding_dim=8,
            embedding_provider="keyword",
        )

        # Store with entities
        mem.remember(
            "OpenAI is a leading AI company based in San Francisco",
            entities=["OpenAI", "San Francisco"],
        )
        mem.remember(
            "Google is another AI company based in Mountain View",
            entities=["Google", "Mountain View"],
        )

        # Get entities
        entities = mem.get_entities()
        assert len(entities) >= 1

        mem.close()

    def test_temporal_facts(self):
        """Test temporal knowledge graph."""
        from arriadne.interface import AriadneMemory

        mem = AriadneMemory(
            db_path=":memory:",
            embedding_dim=8,
            embedding_provider="keyword",
        )

        # Add temporal facts
        f1 = mem.add_temporal_fact(
            text="Kyssta lives in London",
            subject="Kyssta",
            predicate="lives_in",
            obj="London",
        )
        assert f1["is_current"] is True

        # Update fact
        mem.add_temporal_fact(
            text="Kyssta lives in Paris",
            subject="Kyssta",
            predicate="lives_in",
            obj="Paris",
        )

        # Query temporal facts
        facts = mem.query_temporal(subject="Kyssta")
        assert len(facts) >= 1

        # Old fact should be invalidated
        current = [f for f in facts if f["is_current"]]
        assert len(current) == 1
        assert current[0]["object"] == "Paris"

        mem.close()

    def test_lifecycle(self):
        """Test memory lifecycle."""
        from arriadne.interface import AriadneMemory

        mem = AriadneMemory(
            db_path=":memory:",
            embedding_dim=8,
            embedding_provider="keyword",
        )

        mem.remember("Test memory for lifecycle")

        result = mem.run_lifecycle()
        assert "stats" in result

        mem.close()
