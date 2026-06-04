"""
Tests for new Ariadne features:
- Community Detection
- Enhanced NLI Contradiction Detection
- Memory Importance Scoring
- Server API v2 endpoints
- Client library
"""

import os
import time
import pytest


# === Community Detection Tests ===

class TestCommunityDetector:
    """Tests for community detection module."""

    @pytest.fixture
    def detector(self, tmp_path):
        """Create a CommunityDetector with test data."""
        from arriadne.storage import AriadneDB
        from arriadne.config import AriadneConfig
        from arriadne.community import CommunityDetector

        db_path = str(tmp_path / "test_community.db")
        config = AriadneConfig(db_path=db_path)
        db = AriadneDB(config)
        db.open()

        # Add memories with entities
        db.add_memory("Python is a programming language", memory_type="semantic", importance=0.8)
        db.add_memory("JavaScript is used for web development", memory_type="semantic", importance=0.7)
        db.add_memory("React is a JavaScript framework", memory_type="semantic", importance=0.6)
        db.add_memory("Node.js runs JavaScript on servers", memory_type="semantic", importance=0.65)
        db.add_memory("Docker containers are lightweight", memory_type="semantic", importance=0.7)
        db.add_memory("Kubernetes orchestrates containers", memory_type="semantic", importance=0.75)

        # Add edges
        db.add_edge("Python", "JavaScript", "related")
        db.add_edge("JavaScript", "React", "used_by")
        db.add_edge("JavaScript", "Node.js", "runs_on")
        db.add_edge("Docker", "Kubernetes", "orchestrated_by")
        db.add_edge("React", "Node.js", "related")

        detector = CommunityDetector(db.conn)
        yield detector, db

        db.close()
        os.unlink(db_path)

    def test_detect_communities(self, detector):
        """Test basic community detection."""
        det, db = detector
        communities = det.detect_communities()
        assert len(communities) > 0
        for c in communities:
            assert c.id >= 0
            assert c.name
            assert c.size >= 2
            assert c.modularity >= 0

    def test_get_community(self, detector):
        """Test getting a specific community."""
        det, db = detector
        communities = det.detect_communities()
        if communities:
            community = det.get_community(communities[0].id)
            assert community is not None
            assert community.id == communities[0].id

    def test_get_communities_list(self, detector):
        """Test listing all communities."""
        det, db = detector
        det.detect_communities()
        communities = det.get_communities()
        assert isinstance(communities, list)

    def test_community_metrics(self, detector):
        """Test community metrics computation."""
        det, db = detector
        det.detect_communities()
        metrics = det.metrics()
        assert metrics.num_communities >= 0
        assert 0 <= metrics.coverage <= 1.0

    def test_empty_graph(self, tmp_path):
        """Test community detection on empty graph."""
        from arriadne.storage import AriadneDB
        from arriadne.config import AriadneConfig
        from arriadne.community import CommunityDetector

        db_path = str(tmp_path / "empty.db")
        config = AriadneConfig(db_path=db_path)
        db = AriadneDB(config)
        db.open()

        detector = CommunityDetector(db.conn)
        communities = detector.detect_communities()
        assert len(communities) == 0

        metrics = detector.metrics()
        assert metrics.num_communities == 0

        db.close()
        os.unlink(db_path)

    def test_community_to_dict(self, detector):
        """Test community serialization."""
        det, db = detector
        communities = det.detect_communities()
        if communities:
            d = communities[0].to_dict()
            assert "id" in d
            assert "name" in d
            assert "entity_count" in d
            assert "memory_count" in d
            assert "modularity" in d

    def test_get_community_summary(self, detector):
        """Test community summary generation."""
        det, db = detector
        communities = det.detect_communities()
        if communities:
            summary = det.get_community_summary(communities[0].id)
            assert isinstance(summary, str)


# === NLI Contradiction Detection Tests ===

class TestEnhancedContradictionDetector:
    """Tests for enhanced NLI contradiction detection."""

    @pytest.fixture
    def detector(self):
        """Create an EnhancedContradictionDetector."""
        from arriadne.nli import EnhancedContradictionDetector
        return EnhancedContradictionDetector()

    def test_detect_contradiction_regex(self, detector):
        """Test contradiction detection via regex."""
        result = detector.detect(
            "Paris is the capital of France",
            "Paris is not the capital of France",
            max_tier=1,
        )
        assert result.label == "contradiction"
        assert result.confidence > 0.5
        assert result.method == "regex"

    def test_detect_paraphrase(self, detector):
        """Test paraphrase detection."""
        result = detector.detect(
            "The cat sat on the mat",
            "A cat was sitting on a mat",
            max_tier=1,
        )
        # Without similarity, should be neutral
        assert result.label in ("neutral", "contradiction")

    def test_detect_neutral(self, detector):
        """Test neutral detection."""
        result = detector.detect(
            "Paris is the capital of France",
            "Dogs are wonderful pets",
            max_tier=1,
        )
        assert result.label == "neutral"

    def test_detect_batch(self, detector):
        """Test batch contradiction detection."""
        report = detector.detect_batch(
            "Paris is the capital of France",
            [
                {"id": "1", "text": "Paris is not the capital of France"},
                {"id": "2", "text": "Berlin is the capital of Germany"},
                {"id": "3", "text": "Paris is the capital of France"},
            ],
            max_tier=1,
        )
        assert report.new_text == "Paris is the capital of France"
        assert len(report.contradictions) >= 1
        assert report.latency_ms >= 0

    def test_combine_signals(self, detector):
        """Test signal combination."""
        from arriadne.nli import NLIResult
        r1 = NLIResult(label="contradiction", confidence=0.7, method="regex")
        r2 = NLIResult(label="neutral", confidence=0.3, method="similarity")
        combined = detector._combine_signals(r1, r2)
        assert combined.label in ("contradiction", "neutral")
        assert combined.method == "combined"


# === Memory Importance Scoring Tests ===

class TestMemoryImportanceScorer:
    """Tests for memory importance scoring."""

    @pytest.fixture
    def scorer(self, tmp_path):
        """Create a MemoryImportanceScorer with test data."""
        from arriadne.storage import AriadneDB
        from arriadne.config import AriadneConfig
        from arriadne.scoring import MemoryImportanceScorer

        db_path = str(tmp_path / "test_scoring.db")
        config = AriadneConfig(db_path=db_path)
        db = AriadneDB(config)
        db.open()

        # Add diverse memories
        db.add_memory(
            "The VPS at 51.75.73.169 runs Ubuntu 24.04 with 4 cores and 8GB RAM",
            memory_type="technical", importance=0.9,
        )
        db.add_memory(
            "User prefers dark mode interfaces",
            memory_type="preference", importance=0.6,
        )
        db.add_memory(
            "Python is a programming language used for data science and machine learning applications",
            memory_type="semantic", importance=0.7,
        )
        db.add_edge("Ubuntu", "Linux", "based_on")
        db.add_edge("Python", "Data Science", "used_for")

        scorer = MemoryImportanceScorer(db.conn)
        yield scorer, db

        db.close()
        os.unlink(db_path)

    def test_score_memory(self, scorer):
        """Test scoring a single memory."""
        sc, db = scorer
        score = sc.score_memory(1)
        assert score.composite >= 0.0
        assert score.composite <= 1.0
        assert score.information_density >= 0.0
        assert score.recency >= 0.0
        assert score.access_frequency >= 0.0

    def test_score_memories_batch(self, scorer):
        """Test batch scoring."""
        sc, db = scorer
        scores = sc.score_memories([1, 2, 3])
        assert len(scores) == 3
        for mid, score in scores.items():
            assert score.composite >= 0.0
            assert score.composite <= 1.0

    def test_rank_memories(self, scorer):
        """Test memory ranking."""
        sc, db = scorer
        ranked = sc.rank_memories([1, 2, 3], top_k=2)
        assert len(ranked) == 2
        # Should be sorted by composite score (descending)
        assert ranked[0][1].composite >= ranked[1][1].composite

    def test_information_density(self, scorer):
        """Test information density computation."""
        sc, db = scorer
        # High density: specific, unique terms
        high = sc._compute_information_density(
            "The VPS at 51.75.73.169 runs Ubuntu 24.04 with 4 cores and 8GB RAM"
        )
        # Low density: generic words
        low = sc._compute_information_density(
            "the the the the the the the the the the"
        )
        assert high > low

    def test_recency_score(self, scorer):
        """Test recency scoring."""
        sc, db = scorer
        # Recent memory
        recent = sc._compute_recency({
            "accessed_at": time.time(),
            "access_count": 5,
            "importance": 0.8,
            "created_at": time.time(),
        })
        # Old memory
        old = sc._compute_recency({
            "accessed_at": time.time() - 86400 * 30,
            "access_count": 0,
            "importance": 0.5,
            "created_at": time.time() - 86400 * 30,
        })
        assert recent > old

    def test_access_frequency(self, scorer):
        """Test access frequency scoring."""
        sc, db = scorer
        zero = sc._compute_access_frequency({"access_count": 0})
        five = sc._compute_access_frequency({"access_count": 5})
        fifty = sc._compute_access_frequency({"access_count": 50})
        assert zero < five < fifty

    def test_get_top_memories(self, scorer):
        """Test getting top memories."""
        sc, db = scorer
        top = sc.get_top_memories(limit=2)
        assert len(top) == 2
        for item in top:
            assert "id" in item
            assert "score" in item

    def test_score_to_dict(self, scorer):
        """Test score serialization."""
        sc, db = scorer
        score = sc.score_memory(1)
        d = score.to_dict()
        assert "composite" in d
        assert "information_density" in d
        assert "recency" in d


# === Client Library Tests ===

class TestAriadneClient:
    """Tests for the client library."""

    def test_client_init(self):
        """Test client initialization."""
        from arriadne.client import AriadneClient
        client = AriadneClient("http://localhost:8899")
        assert client._base_url == "http://localhost:8899"

    def test_client_with_api_key(self):
        """Test client with API key."""
        from arriadne.client import AriadneClient
        client = AriadneClient("http://localhost:8899", api_key="test-key")
        assert client._api_key == "test-key"
        assert "Bearer test-key" in client._headers.get("Authorization", "")

    def test_client_context_manager(self):
        """Test client as context manager."""
        from arriadne.client import AriadneClient
        with AriadneClient("http://localhost:8899") as client:
            assert client._base_url == "http://localhost:8899"


# === Server API v2 Tests ===

class TestServerAPI:
    """Tests for the expanded server API."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a test client."""
        from arriadne.server import create_app
        from fastapi.testclient import TestClient

        db_path = str(tmp_path / "test_server.db")
        app = create_app(db_path=db_path)
        return TestClient(app), db_path

    def test_root(self, client):
        """Test root endpoint."""
        tc, db_path = client
        response = tc.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] in ("2.0.0", "2.1.0", "3.0.0")
        assert "features" in data

    def test_health(self, client):
        """Test health endpoint."""
        tc, db_path = client
        response = tc.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness(self, client):
        """Test readiness endpoint."""
        tc, db_path = client
        response = tc.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_stats(self, client):
        """Test stats endpoint."""
        tc, db_path = client
        response = tc.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_memories" in data
        assert "communities" in data

    def test_metrics(self, client):
        """Test metrics endpoint."""
        tc, db_path = client
        response = tc.get("/metrics?format=json")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "uptime_seconds" in data

    def test_store_and_get_memory(self, client):
        """Test memory CRUD."""
        tc, db_path = client

        # Store
        response = tc.post("/memories", json={
            "content": "Test memory content",
            "topic": "test",
            "importance": 7,
        })
        assert response.status_code == 200
        data = response.json()
        memory_id = data["id"]

        # Get
        response = tc.get(f"/memories/{memory_id}")
        assert response.status_code == 200
        assert response.json()["content"] == "Test memory content"

        # Delete
        response = tc.delete(f"/memories/{memory_id}")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

    def test_search(self, client):
        """Test search endpoint."""
        tc, db_path = client

        # Store some memories first
        tc.post("/memories", json={"content": "Python is great for data science"})
        tc.post("/memories", json={"content": "JavaScript is used for web apps"})

        # Search
        response = tc.post("/search", json={
            "query": "Python data science",
            "limit": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "latency_ms" in data

    def test_communities_endpoint(self, client):
        """Test community endpoints."""
        tc, db_path = client

        # Get communities (should be empty initially)
        response = tc.get("/communities")
        assert response.status_code == 200

        # Detect communities
        response = tc.post("/communities/detect")
        assert response.status_code == 200

    def test_nli_endpoint(self, client):
        """Test NLI detection endpoint."""
        tc, db_path = client
        response = tc.post("/nli/detect", json={
            "text_a": "Paris is the capital of France",
            "text_b": "Paris is not the capital of France",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "contradiction"

    def test_import_export(self, client):
        """Test import and export endpoints."""
        tc, db_path = client

        # Import
        response = tc.post("/import", json=[
            {"content": "Memory 1", "topic": "test"},
            {"content": "Memory 2", "topic": "test"},
        ])
        assert response.status_code == 200
        assert response.json()["imported"] == 2

        # Export
        response = tc.get("/export")
        assert response.status_code == 200
        assert response.json()["count"] >= 2

    def test_lifecycle_endpoint(self, client):
        """Test lifecycle endpoint."""
        tc, db_path = client
        response = tc.get("/lifecycle")
        assert response.status_code == 200

    def test_graph_entities(self, client):
        """Test graph entity endpoints."""
        tc, db_path = client
        response = tc.get("/graph/entities")
        assert response.status_code == 200
        assert "entities" in response.json()

    def test_response_timing_header(self, client):
        """Test that response timing header is present."""
        tc, db_path = client
        response = tc.get("/health")
        assert "X-Response-Time" in response.headers

    def test_ranked_memories(self, client):
        """Test ranked memories endpoint."""
        tc, db_path = client
        # Store some memories
        tc.post("/memories", json={"content": "Important memory about Python"})
        tc.post("/memories", json={"content": "Another memory about JavaScript"})

        response = tc.get("/memories/ranked?limit=5")
        assert response.status_code == 200
        assert "memories" in response.json()
