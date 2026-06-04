"""
Tests for Ariadne Observability, Enhanced Server API, and Client Local Mode.

Covers:
- Prometheus metrics rendering
- ObservabilityCollector tracking
- RequestLogger structured logging
- PrometheusHistogram percentile calculations
- Enhanced server endpoints (batch search, temporal, entity CRUD, evict)
- Client local mode
- Client batch operations
- Metrics endpoint (both JSON and Prometheus format)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import sqlite3
import pytest
from unittest.mock import MagicMock, patch


# ============================================================
# Observability Module Tests
# ============================================================

class TestPrometheusHistogram:
    """Tests for PrometheusHistogram."""

    def test_observe_and_render(self):
        from arriadne.observability import PrometheusHistogram
        h = PrometheusHistogram("test_duration_seconds", "Test histogram")
        h.observe(0.01)
        h.observe(0.05)
        h.observe(0.1)

        rendered = h.render()
        assert "# HELP test_duration_seconds Test histogram" in rendered
        assert "# TYPE test_duration_seconds histogram" in rendered
        assert "test_duration_seconds_count 3" in rendered
        assert "test_duration_seconds_sum" in rendered

    def test_percentiles(self):
        from arriadne.observability import PrometheusHistogram
        h = PrometheusHistogram("test_hist", "Test", buckets=[0.1, 0.5, 1.0, 5.0])

        # Add 100 observations
        for i in range(100):
            h.observe(i * 0.05)  # 0 to 4.95

        assert h.avg > 0
        assert h.p50 > 0
        assert h.p95 > 0
        assert h.p99 > 0
        assert h._count == 100

    def test_empty_histogram(self):
        from arriadne.observability import PrometheusHistogram
        h = PrometheusHistogram("empty", "Empty")
        assert h.avg == 0.0
        assert h.p50 == 0.0
        assert h._count == 0

    def test_custom_buckets(self):
        from arriadne.observability import PrometheusHistogram
        h = PrometheusHistogram("custom", "Custom", buckets=[1, 2, 3])
        h.observe(0.5)
        h.observe(1.5)
        h.observe(2.5)
        rendered = h.render()
        assert 'custom_bucket{le="1"} 1' in rendered
        assert 'custom_bucket{le="2"} 2' in rendered
        assert 'custom_bucket{le="3"} 3' in rendered
        assert 'custom_bucket{le="+Inf"} 3' in rendered


class TestPrometheusCounter:
    """Tests for PrometheusCounter."""

    def test_inc_and_render(self):
        from arriadne.observability import PrometheusCounter
        c = PrometheusCounter("test_counter", "Test counter")
        c.inc()
        c.inc(5)
        rendered = c.render()
        assert "test_counter 6.0" in rendered

    def test_with_labels(self):
        from arriadne.observability import PrometheusCounter
        c = PrometheusCounter("test_counter", "Test")
        c.inc(labels={"method": "GET"})
        c.inc(labels={"method": "POST"})
        c.inc(labels={"method": "GET"})
        rendered = c.render()
        assert 'method="GET"' in rendered
        assert 'method="POST"' in rendered


class TestPrometheusGauge:
    """Tests for PrometheusGauge."""

    def test_set_and_render(self):
        from arriadne.observability import PrometheusGauge
        g = PrometheusGauge("test_gauge", "Test gauge")
        g.set(42.0)
        rendered = g.render()
        assert "test_gauge 42.0" in rendered

    def test_with_labels(self):
        from arriadne.observability import PrometheusGauge
        g = PrometheusGauge("test_gauge", "Test")
        g.set(10.0, labels={"type": "semantic"})
        g.set(20.0, labels={"type": "episodic"})
        rendered = g.render()
        assert 'type="semantic"' in rendered
        assert 'type="episodic"' in rendered


class TestObservabilityCollector:
    """Tests for ObservabilityCollector."""

    def test_record_request(self):
        from arriadne.observability import ObservabilityCollector
        obs = ObservabilityCollector()
        obs.record_request("GET /health", 5.0)
        obs.record_request("POST /search", 10.0, "error")

        assert obs.request_count == 2
        assert obs.error_count == 1
        assert obs.total_latency_ms == 15.0

    def test_record_search(self):
        from arriadne.observability import ObservabilityCollector
        obs = ObservabilityCollector()
        obs.record_search(15.0, 5)
        obs.record_search(10.0, 3)

        assert obs.search_metrics["total_searches"] == 2
        assert obs.search_metrics["total_results_returned"] == 8

    def test_render_prometheus(self):
        from arriadne.observability import ObservabilityCollector
        obs = ObservabilityCollector()
        obs.record_request("GET /health", 5.0)

        prom = obs.render_prometheus()
        assert "arriadne_requests_total" in prom
        assert "arriadne_request_duration_seconds" in prom
        assert "arriadne_uptime_seconds" in prom
        assert "arriadne_sqlite_db_size_bytes" in prom
        assert "# HELP" in prom
        assert "# TYPE" in prom

    def test_to_dict(self):
        from arriadne.observability import ObservabilityCollector
        obs = ObservabilityCollector()
        obs.record_request("GET /test", 3.0)

        d = obs.to_dict()
        assert "uptime_seconds" in d
        assert d["total_requests"] == 1
        assert d["total_errors"] == 0
        assert "search_metrics" in d
        assert "sqlite_metrics" in d
        assert "endpoint_stats" in d

    def test_update_sqlite_metrics(self):
        from arriadne.observability import ObservabilityCollector
        obs = ObservabilityCollector()

        # Use a file-based SQLite database (in-memory returns 0 for page stats)
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = sqlite3.connect(db_path)
            # Create a table so the DB has real pages
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            conn.commit()
            obs.update_sqlite_metrics(conn)
            assert obs.sqlite_metrics.journal_mode in ("delete", "wal", "memory")
            # File-based DBs have real page sizes
            assert obs.sqlite_metrics.page_size >= 512
            assert obs.sqlite_metrics.page_count >= 1
            assert obs.sqlite_metrics.db_size_bytes > 0
            conn.close()
        finally:
            os.unlink(db_path)

    def test_endpoint_stats_tracking(self):
        from arriadne.observability import ObservabilityCollector
        obs = ObservabilityCollector()
        obs.record_request("GET /memories", 5.0)
        obs.record_request("GET /memories", 3.0)
        obs.record_request("GET /memories", 8.0, "error")

        stats = obs.endpoint_stats["GET /memories"]
        assert stats["count"] == 3
        assert stats["errors"] == 1
        assert stats["latency_ms"] == 16.0

    def test_singleton_get_collector(self):
        from arriadne.observability import get_collector
        c1 = get_collector()
        c2 = get_collector()
        assert c1 is c2


class TestRequestLogger:
    """Tests for RequestLogger."""

    def test_log_request(self, caplog):
        import logging
        from arriadne.observability import RequestLogger
        logger = RequestLogger()

        with caplog.at_level(logging.INFO, logger="arriadne.request"):
            logger.log_request("GET", "/health", 200, 5.0, client_ip="127.0.0.1")

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "GET /health -> 200" in record.message

    def test_log_error_request(self, caplog):
        import logging
        from arriadne.observability import RequestLogger
        logger = RequestLogger()

        with caplog.at_level(logging.ERROR, logger="arriadne.request"):
            logger.log_request("POST", "/search", 500, 100.0, error="Internal error")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "ERROR"


# ============================================================
# Server API Tests (Enhanced)
# ============================================================

class TestServerEnhancedAPI:
    """Tests for the enhanced server API endpoints."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a test client."""
        from arriadne.server import create_app
        from fastapi.testclient import TestClient

        db_path = str(tmp_path / "test_enhanced.db")
        app = create_app(db_path=db_path)
        return TestClient(app), db_path

    def test_metrics_prometheus_format(self, client):
        """Test Prometheus text format metrics endpoint."""
        tc, _ = client
        response = tc.get("/metrics?format=prometheus")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        text = response.text
        assert "arriadne_requests_total" in text
        assert "# HELP" in text
        assert "# TYPE" in text

    def test_metrics_json_format(self, client):
        """Test JSON format metrics endpoint (backward compatible)."""
        tc, _ = client
        response = tc.get("/metrics?format=json")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "uptime_seconds" in data
        assert "sqlite_metrics" in data

    def test_metrics_increments(self, client):
        """Test that metrics increment after requests."""
        tc, _ = client
        # Make some requests
        tc.get("/health")
        tc.get("/stats")

        response = tc.get("/metrics?format=json")
        data = response.json()
        assert data["total_requests"] >= 2

    def test_batch_search(self, client):
        """Test batch search endpoint."""
        tc, _ = client
        # Store some memories
        tc.post("/memories", json={"content": "Python is great for data science"})
        tc.post("/memories", json={"content": "JavaScript is used for web apps"})

        # Batch search
        response = tc.post("/batch/search", json={
            "queries": ["Python data science", "JavaScript web"],
            "limit": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "queries" in data
        assert len(data["queries"]) == 2
        assert "total_results" in data
        assert "latency_ms" in data

    def test_temporal_facts_crud(self, client):
        """Test temporal fact CRUD."""
        tc, _ = client

        # Add a fact
        response = tc.post("/temporal/facts", json={
            "text": "Alice lives in New York",
            "subject": "Alice",
            "predicate": "lives_in",
            "obj": "New York",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["subject"] == "Alice"
        assert data["object"] == "New York"

        # Query facts
        response = tc.get("/temporal/facts?subject=Alice")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1

        # Timeline
        response = tc.get("/temporal/timeline/Alice")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1

    def test_temporal_invalidation(self, client):
        """Test temporal fact invalidation."""
        tc, _ = client

        # Add a fact
        response = tc.post("/temporal/facts", json={
            "text": "Bob works at Google",
            "subject": "Bob",
            "predicate": "works_at",
            "obj": "Google",
        })
        fact_id = response.json()["fact_id"]

        # Invalidate
        response = tc.post(f"/temporal/invalidate/{fact_id}")
        assert response.status_code == 200
        assert response.json()["invalidated"] is True

    def test_temporal_stats(self, client):
        """Test temporal stats endpoint."""
        tc, _ = client
        response = tc.get("/temporal/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_facts" in data

    def test_entity_create_and_delete(self, client):
        """Test entity CRUD operations."""
        tc, _ = client

        # Create entity
        response = tc.post("/graph/entities", json={
            "name": "TestEntity",
            "entity_type": "concept",
        })
        assert response.status_code == 200
        assert response.json()["created"] is True

        # List entities
        response = tc.get("/graph/entities")
        assert response.status_code == 200
        names = [e["name"] for e in response.json()["entities"]]
        assert "TestEntity" in names

        # Delete entity
        response = tc.delete("/graph/entities/TestEntity")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

    def test_lifecycle_evict(self, client):
        """Test lifecycle eviction endpoint."""
        tc, _ = client
        # Add some memories
        for i in range(5):
            tc.post("/memories", json={"content": f"Memory {i}", "importance": i})

        response = tc.post("/lifecycle/evict?target_count=10")
        assert response.status_code == 200
        assert "evicted" in response.json()

    def test_sse_stream_search(self, client):
        """Test SSE streaming search."""
        tc, _ = client
        tc.post("/memories", json={"content": "Python is amazing for machine learning"})

        # Test streaming endpoint exists and returns proper headers
        with tc.stream("GET", "/search/stream?query=Python&limit=5") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

    def test_response_headers(self, client):
        """Test that response headers include timing and request ID."""
        tc, _ = client
        response = tc.get("/health")
        assert "X-Response-Time" in response.headers
        assert "X-Request-Id" in response.headers

    def test_api_key_auth(self, client):
        """Test API key authentication."""
        from arriadne.server import create_app
        from fastapi.testclient import TestClient

        # Create app with API key
        app = create_app(db_path="/tmp/test_auth.db", api_key="secret123")
        tc = TestClient(app)

        # Without API key should fail
        response = tc.get("/stats")
        assert response.status_code == 401

        # With correct key should work
        response = tc.get("/stats", headers={"Authorization": "Bearer secret123"})
        assert response.status_code == 200

        # With wrong key should fail
        response = tc.get("/stats", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401

        # Cleanup
        for suffix in ["", "-wal", "-shm", ".faiss"]:
            p = f"/tmp/test_auth.db{suffix}"
            if os.path.exists(p):
                os.unlink(p)

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        tc, _ = client
        response = tc.options("/health", headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        })
        assert response.status_code == 200
        # FastAPI CORS middleware adds these headers
        assert "access-control-allow-origin" in response.headers or \
               response.headers.get("Access-Control-Allow-Origin") == "*"


# ============================================================
# Client Tests (Local Mode)
# ============================================================

class TestAriadneClientLocal:
    """Tests for AriadneClient in local mode."""

    def test_local_mode_init(self, tmp_path):
        """Test client initialization in local mode."""
        from arriadne.client import AriadneClient
        db_path = str(tmp_path / "local_test.db")
        client = AriadneClient(local_db=db_path)
        assert client.is_local is True
        assert client.is_remote is False
        client.close()

    def test_local_remember(self, tmp_path):
        """Test storing memory in local mode."""
        from arriadne.client import AriadneClient
        db_path = str(tmp_path / "local_remember.db")
        with AriadneClient(local_db=db_path) as client:
            result = client.remember("Test memory", topic="test", importance=7)
            assert result["status"] == "created"

    def test_local_search(self, tmp_path):
        """Test searching in local mode."""
        from arriadne.client import AriadneClient
        db_path = str(tmp_path / "local_search.db")
        with AriadneClient(local_db=db_path) as client:
            client.remember("Python is great for data science")
            client.remember("JavaScript is used for web apps")

            result = client.search("Python data science")
            assert "results" in result
            assert "count" in result
            assert "latency_ms" in result

    def test_local_recall(self, tmp_path):
        """Test recall (convenience method) in local mode."""
        from arriadne.client import AriadneClient
        db_path = str(tmp_path / "local_recall.db")
        with AriadneClient(local_db=db_path) as client:
            client.remember("Docker containers are lightweight")
            results = client.recall("Docker containers")
            assert isinstance(results, list)

    def test_local_health(self, tmp_path):
        """Test health check in local mode."""
        from arriadne.client import AriadneClient
        db_path = str(tmp_path / "local_health.db")
        with AriadneClient(local_db=db_path) as client:
            health = client.health()
            assert health["status"] == "healthy"

    def test_local_stats(self, tmp_path):
        """Test stats in local mode."""
        from arriadne.client import AriadneClient
        db_path = str(tmp_path / "local_stats.db")
        with AriadneClient(local_db=db_path) as client:
            client.remember("Test")
            stats = client.stats()
            assert isinstance(stats, dict)

    def test_local_batch_search(self, tmp_path):
        """Test batch search in local mode."""
        from arriadne.client import AriadneClient
        db_path = str(tmp_path / "local_batch.db")
        with AriadneClient(local_db=db_path) as client:
            client.remember("Python data science")
            client.remember("JavaScript web apps")

            result = client.batch_search(["Python", "JavaScript"], limit=5)
            assert "queries" in result
            assert len(result["queries"]) == 2
            assert "total_results" in result

    def test_local_import_export(self, tmp_path):
        """Test import/export in local mode."""
        from arriadne.client import AriadneClient
        db_path = str(tmp_path / "local_import.db")
        with AriadneClient(local_db=db_path) as client:
            # Import
            result = client.import_memories([
                {"content": "Memory 1", "topic": "test"},
                {"content": "Memory 2", "topic": "test"},
            ])
            assert result["imported"] == 2

            # Export
            result = client.export_memories()
            assert result["count"] >= 2

    def test_local_metrics(self, tmp_path):
        """Test metrics in local mode returns error."""
        from arriadne.client import AriadneClient
        db_path = str(tmp_path / "local_metrics.db")
        with AriadneClient(local_db=db_path) as client:
            result = client.metrics()
            assert "error" in result

    def test_context_manager(self, tmp_path):
        """Test client as context manager."""
        from arriadne.client import AriadneClient
        db_path = str(tmp_path / "local_ctx.db")
        with AriadneClient(local_db=db_path) as client:
            client.remember("Context test")
            assert client._local_mem is not None
        assert client._local_mem is None

    def test_local_search_stream(self, tmp_path):
        """Test SSE streaming in local mode."""
        from arriadne.client import AriadneClient
        db_path = str(tmp_path / "local_stream.db")
        with AriadneClient(local_db=db_path) as client:
            client.remember("Streamable content")
            results = list(client.search_stream("Streamable"))
            assert len(results) >= 1
            assert "result" in results[0]
            assert "done" in results[0]


class TestAriadneClientRemote:
    """Tests for AriadneClient remote mode (basic)."""

    def test_client_init(self):
        """Test client initialization."""
        from arriadne.client import AriadneClient
        client = AriadneClient("http://localhost:8899")
        assert client._base_url == "http://localhost:8899"
        assert client.is_remote is True

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


class TestAriadneClientAsync:
    """Tests for async client initialization."""

    def test_async_client_init(self):
        """Test async client initialization."""
        from arriadne.client import AriadneClientAsync
        client = AriadneClientAsync("http://localhost:8899")
        assert client._base_url == "http://localhost:8899"

    def test_async_client_with_api_key(self):
        """Test async client with API key."""
        from arriadne.client import AriadneClientAsync
        client = AriadneClientAsync("http://localhost:8899", api_key="key123")
        assert client._api_key == "key123"
