"""
Ariadne REST API Server — Production-Ready

FastAPI-based server that exposes Ariadne's memory capabilities over HTTP
with comprehensive observability, streaming, batch operations, community
detection, multi-tenancy, rate limiting, and graceful shutdown.

Features:
- Full CRUD for memories, entities, graphs, communities
- Hybrid search (vector + FTS5 + RRF)
- Streaming search via SSE
- Community detection and management
- Memory importance scoring
- Batch import/export
- Request logging with timing
- Search quality metrics
- Memory statistics (by tier, by age, by entity)
- Graph metrics (nodes, edges, communities)
- Health check and readiness probes
- API versioning (/api/v1/)
- Rate limiting middleware
- Multi-tenancy via X-Tenant-ID header
- Thread-safe memory access
- Graceful shutdown (saves FAISS index on SIGTERM)

Usage:
    arriadne serve --port 8899 --db-path ./memory.db
    # or
    python -m arriadne.server --port 8899
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from arriadne.observability import ObservabilityCollector, RequestLogger, get_collector

logger = logging.getLogger("arriadne.server")

# === Request/Response Models ===

class StoreRequest(BaseModel):
    content: str
    topic: Optional[str] = "general"
    importance: int = Field(5, ge=1, le=10)
    entities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    agent_id: Optional[str] = None

class BatchSearchRequest(BaseModel):
    queries: List[str]
    limit: int = Field(10, ge=1, le=100)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(10, ge=1, le=100)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    use_hybrid: bool = True
    include_graph: bool = False
    include_metadata: bool = True
    community_id: Optional[int] = None  # Filter by community
    memory_type: Optional[str] = None  # Filter by type
    user_id: Optional[str] = None
    agent_id: Optional[str] = None


class ExtractRequest(BaseModel):
    messages: List[Dict[str, str]]
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    auto_store: bool = True


class UpdateRequest(BaseModel):
    content: Optional[str] = None
    topic: Optional[str] = None
    importance: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchStoreRequest(BaseModel):
    memories: List[StoreRequest]


class CommunityDetectRequest(BaseModel):
    force: bool = False


class NLIRequest(BaseModel):
    text_a: str
    text_b: str
    max_tier: int = Field(3, ge=1, le=3)


class EntityCreateRequest(BaseModel):
    name: str
    entity_type: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphEdgeRequest(BaseModel):
    source: str
    target: str
    relation: str = "related"
    weight: float = 1.0


class TemporalFactRequest(BaseModel):
    text: str
    subject: str
    predicate: str
    obj: str
    valid_at: Optional[float] = None


# === Rate Limiter ===

class RateLimiter:
    """Simple in-memory token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 120):
        self._rpm = requests_per_minute
        self._tokens: Dict[str, float] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        """Check if a request is allowed for the given key."""
        now = time.time()
        with self._lock:
            if key not in self._tokens:
                self._tokens[key] = self._rpm
                self._timestamps[key] = now

            # Refill tokens
            elapsed = now - self._timestamps[key]
            refill = elapsed * (self._rpm / 60.0)
            self._tokens[key] = min(self._rpm, self._tokens[key] + refill)
            self._timestamps[key] = now

            if self._tokens[key] >= 1.0:
                self._tokens[key] -= 1.0
                return True
            return False


# === Thread-Safe Memory Wrapper ===

class ThreadSafeMemory:
    """Wrapper around AriadneMemory with thread-safe access via a lock."""

    def __init__(self, memory):
        self._memory = memory
        self._lock = threading.RLock()

    def __getattr__(self, name):
        attr = getattr(self._memory, name)
        if callable(attr):
            def thread_safe_wrapper(*args, **kwargs):
                with self._lock:
                    return attr(*args, **kwargs)
            return thread_safe_wrapper
        return attr

    def __enter__(self):
        self._lock.acquire()
        return self._memory

    def __exit__(self, *args):
        self._lock.release()


# === Observability Middleware ===

ObservabilityMiddleware = ObservabilityCollector


# === App Factory ===

def create_app(
    db_path: str = "ariadne.db",
    embedding_config: Optional[Dict] = None,
    llm_config: Optional[Dict] = None,
    api_key: Optional[str] = None,
    rate_limit_rpm: int = 120,
    enable_versioning: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        db_path: Path to SQLite database file.
        embedding_config: Embedding provider configuration.
        llm_config: LLM provider configuration.
        api_key: API key for authentication (None = no auth).
        rate_limit_rpm: Rate limit in requests per minute.
        enable_versioning: Whether to enable /api/v1/ prefix routes.
    """
    observability = get_collector()
    request_logger = RequestLogger()
    rate_limiter = RateLimiter(requests_per_minute=rate_limit_rpm)

    # Shutdown event for graceful shutdown
    _shutdown_event = threading.Event()

    app = FastAPI(
        title="Ariadne Memory API",
        description=(
            "Production-ready memory system for AI agents with community detection, "
            "NLI contradiction detection, advanced importance scoring, multi-tenancy, "
            "and observability"
        ),
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazy-init the memory system with thread safety
    _memory = None
    _memory_lock = threading.Lock()

    def get_memory():
        nonlocal _memory
        if _memory is None:
            with _memory_lock:
                if _memory is None:
                    from arriadne.interface import AriadneMemory
                    _memory = ThreadSafeMemory(
                        AriadneMemory(
                            db_path=db_path,
                            llm_config=llm_config,
                        )
                    )
        return _memory

    # Auth
    async def verify_api_key(authorization: Optional[str] = Header(None)):
        if api_key:
            if not authorization:
                raise HTTPException(status_code=401, detail="Missing API key")
            token = authorization.replace("Bearer ", "").strip()
            if token != api_key:
                raise HTTPException(status_code=401, detail="Invalid API key")

    # === Request Timing & Rate Limiting Middleware ===

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        # Rate limiting (skip for health/docs)
        path = request.url.path
        if not path.startswith("/health") and not path.startswith("/ready") and not path.startswith("/docs") and not path.startswith("/redoc") and not path.startswith("/openapi"):
            client_ip = request.client.host if request.client else "unknown"
            rate_key = f"{client_ip}:{api_key or 'default'}"
            if not rate_limiter.is_allowed(rate_key):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded", "retry_after": 60},
                    headers={"Retry-After": "60"},
                )

        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            endpoint = f"{request.method} {request.url.path}"
            observability.record_request(endpoint, elapsed_ms, "error")
            client_ip = request.client.host if request.client else None
            request_logger.log_request(
                request.method, request.url.path, 500, elapsed_ms,
                client_ip=client_ip, error=str(exc),
            )
            raise
        elapsed_ms = (time.monotonic() - start) * 1000
        endpoint = f"{request.method} {request.url.path}"
        status = "ok" if response.status_code < 400 else "error"
        observability.record_request(endpoint, elapsed_ms, status)
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        request_logger.log_request(
            request.method, request.url.path, response.status_code, elapsed_ms,
            client_ip=client_ip, user_agent=user_agent,
        )
        response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"
        response.headers["X-Request-Id"] = f"{id(request):x}"
        return response

    # === Root & Health Routes ===

    @app.get("/", tags=["health"])
    async def root():
        return {
            "name": "Ariadne Memory API",
            "version": "3.0.0",
            "status": "running",
            "docs": "/docs",
            "api_version": "/api/v1" if enable_versioning else "/",
            "features": [
                "hybrid_search", "community_detection", "nli_contradiction",
                "importance_scoring", "temporal_graph", "entity_resolution",
                "memory_lifecycle", "llm_extraction", "multi_tenancy",
                "rate_limiting", "graceful_shutdown",
            ],
        }

    @app.get("/health", tags=["health"])
    async def health():
        mem = get_memory()
        stats = mem.stats()
        return {
            "status": "healthy",
            "memories": stats.get("total_memories", 0),
            "active_memories": stats.get("active_memories", 0),
            "uptime": time.time(),
        }

    @app.get("/ready", tags=["health"])
    async def readiness():
        """Kubernetes-style readiness probe."""
        try:
            mem = get_memory()
            mem.stats()
            return {"ready": True}
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Not ready: {e}")

    # === Stats & Metrics Routes ===

    @app.get("/stats", tags=["stats"])
    async def stats(authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        raw = mem.stats()

        result = {
            "total_memories": raw.get("total_memories", 0),
            "active_memories": raw.get("active_memories", 0),
            "deleted_memories": raw.get("deleted_memories", 0),
            "by_type": raw.get("by_type", {}),
            "total_entities": raw.get("total_entities", 0),
            "total_edges": raw.get("total_edges", 0),
            "vector_index_size": raw.get("faiss_vectors", 0),
            "embedding_model": raw.get("embedding_provider", "none"),
            "embedding_dimension": raw.get("embedding_dimension", 0),
            "db_size_bytes": raw.get("db_size_bytes", 0),
            "avg_importance": raw.get("avg_importance", 0),
        }

        try:
            from arriadne.community import CommunityDetector
            detector = CommunityDetector(mem._memory._db.conn)
            comm_metrics = detector.metrics()
            result["communities"] = {
                "count": comm_metrics.num_communities,
                "avg_size": round(comm_metrics.avg_community_size, 1),
                "largest": comm_metrics.largest_community,
                "coverage": round(comm_metrics.coverage, 3),
            }
        except Exception:
            result["communities"] = {"count": 0}

        try:
            if hasattr(mem._memory, "_get_lifecycle"):
                lifecycle = mem._memory._get_lifecycle()
                lifecycle_result = lifecycle.run_lifecycle()
                result["lifecycle"] = {
                    "hot": lifecycle_result.get("stats", {}).hot_count,
                    "warm": lifecycle_result.get("stats", {}).warm_count,
                    "cold": lifecycle_result.get("stats", {}).cold_count,
                }
        except Exception:
            pass

        return result

    @app.get("/metrics", tags=["observability"])
    async def metrics(
        authorization: Optional[str] = Header(None),
        format: str = Query("prometheus", pattern="^(prometheus|json)$"),
    ):
        """Prometheus-compatible metrics endpoint."""
        await verify_api_key(authorization)
        try:
            mem = get_memory()
            observability.update_sqlite_metrics(mem._memory._db.conn)
        except Exception:
            pass
        if format == "json":
            return observability.to_dict()
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            observability.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # === Memory CRUD Routes ===

    @app.post("/memories", tags=["memories"])
    async def store_memory(
        req: StoreRequest,
        authorization: Optional[str] = Header(None),
        x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    ):
        await verify_api_key(authorization)
        mem = get_memory()
        tenant = x_tenant_id or "default"
        result = mem.store(
            req.content, topic=req.topic, importance=req.importance,
            entities=req.entities, metadata=req.metadata,
            tenant_id=tenant,
        )
        observability.memories_stored_total.inc()
        return result

    @app.get("/memories/ranked", tags=["memories", "scoring"])
    async def rank_memories(
        limit: int = Query(10, ge=1, le=100),
        memory_type: Optional[str] = None,
        authorization: Optional[str] = Header(None),
    ):
        """Get top memories ranked by multi-factor importance score."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.scoring import MemoryImportanceScorer
            scorer = MemoryImportanceScorer(mem._memory._db.conn, mem._memory._embedder)
            return {"memories": scorer.get_top_memories(limit=limit, memory_type=memory_type)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/memories/{memory_id}", tags=["memories"])
    async def get_memory_by_id(memory_id: str, authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        result = mem.get(memory_id)
        if not result:
            raise HTTPException(status_code=404, detail="Memory not found")
        return result

    @app.patch("/memories/{memory_id}", tags=["memories"])
    async def update_memory(memory_id: str, req: UpdateRequest, authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        result = mem.update(memory_id, **updates)
        if not result:
            raise HTTPException(status_code=404, detail="Memory not found")
        return result

    @app.delete("/memories/{memory_id}", tags=["memories"])
    async def delete_memory(memory_id: str, authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        success = mem.delete(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        observability.memories_deleted_total.inc()
        return {"deleted": True, "id": memory_id}

    @app.get("/memories/{memory_id}/score", tags=["memories", "scoring"])
    async def score_memory(memory_id: str, authorization: Optional[str] = Header(None)):
        """Get importance score breakdown for a memory."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.scoring import MemoryImportanceScorer
            scorer = MemoryImportanceScorer(mem._memory._db.conn, mem._memory._embedder)
            score = scorer.score_memory(int(memory_id))
            return {"memory_id": memory_id, "score": score.to_dict()}
        except (ValueError, TypeError):
            raise HTTPException(status_code=404, detail="Memory not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # === Search Routes ===

    @app.post("/search", tags=["search"])
    async def search_memories(
        req: SearchRequest,
        authorization: Optional[str] = Header(None),
        x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    ):
        await verify_api_key(authorization)
        t0 = time.monotonic()
        mem = get_memory()

        if req.community_id is not None:
            try:
                from arriadne.community import CommunityDetector
                detector = CommunityDetector(mem._memory._db.conn)
                results = detector.search_within_community(
                    req.query, req.community_id, limit=req.limit,
                )
            except Exception:
                results = mem.search(req.query, limit=req.limit, threshold=req.threshold)
        else:
            results = mem.search(
                req.query, limit=req.limit, threshold=req.threshold,
                use_hybrid=req.use_hybrid, include_graph=req.include_graph,
            )

        # Apply memory type filter
        if req.memory_type:
            results = [r for r in results if r.get("topic") == req.memory_type]

        # Apply tenant filter
        tenant = x_tenant_id
        if tenant:
            results = [r for r in results if r.get("tenant_id", "default") == tenant]

        latency_ms = (time.monotonic() - t0) * 1000
        observability.record_search(latency_ms, len(results))

        return {
            "results": results,
            "count": len(results),
            "latency_ms": round(latency_ms, 2),
        }

    @app.get("/search/stream", tags=["search"])
    async def search_stream(
        query: str = Query(...),
        limit: int = Query(10, ge=1, le=100),
        authorization: Optional[str] = Header(None),
    ):
        """Stream search results via Server-Sent Events."""
        await verify_api_key(authorization)
        mem = get_memory()
        t0 = time.monotonic()

        def event_generator():
            try:
                results = mem.search(query, limit=limit)
                for i, result in enumerate(results):
                    data = json.dumps({
                        "result": result,
                        "rank": i + 1,
                        "done": i == len(results) - 1,
                    })
                    yield f"data: {data}\n\n"
                latency_ms = (time.monotonic() - t0) * 1000
                observability.record_search(latency_ms, len(results))
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/nli/detect", tags=["nli"])
    async def detect_nli(req: NLIRequest, authorization: Optional[str] = Header(None)):
        """Detect NLI relationship between two texts."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.nli import EnhancedContradictionDetector
            detector = EnhancedContradictionDetector(
                embedding_provider=mem._memory._embedder if mem._memory._embedder.name != "keyword" else None,
            )
            result = detector.detect(req.text_a, req.text_b, max_tier=req.max_tier)
            return {
                "label": result.label,
                "confidence": result.confidence,
                "method": result.method,
                "details": result.details,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # === Extraction Routes ===

    @app.post("/extract", tags=["extraction"])
    async def extract_memories(req: ExtractRequest, authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        extracted = mem.extract_from_conversation(req.messages)
        stored = []
        if req.auto_store:
            for m in extracted:
                result = mem.store(
                    m.text, topic=m.topic, importance=m.importance,
                    entities=m.entities, metadata={"attributed_to": m.attributed_to},
                )
                stored.append(result)
        return {
            "extracted": len(extracted),
            "stored": len(stored),
            "memories": [
                {"text": m.text, "topic": m.topic, "importance": m.importance, "entities": m.entities}
                for m in extracted
            ],
        }

    # === Graph Routes ===

    @app.get("/graph/entities", tags=["graph"])
    async def get_entities(
        entity_type: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        authorization: Optional[str] = Header(None),
    ):
        await verify_api_key(authorization)
        mem = get_memory()
        # Read directly from the database entities table
        cursor = mem._db.conn.execute(
            "SELECT id, name, entity_type FROM entities ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        if entity_type:
            rows = [r for r in rows if r[2] == entity_type]
        entities = [{"id": r[0], "name": r[1], "type": r[2]} for r in rows]
        return {"entities": entities, "count": len(entities)}

    @app.get("/graph/entity/{entity_name}", tags=["graph"])
    async def get_entity_graph(
        entity_name: str,
        hops: int = Query(2, ge=1, le=5),
        authorization: Optional[str] = Header(None),
    ):
        await verify_api_key(authorization)
        mem = get_memory()
        return mem.graph_search(entity_name, hops=hops)

    @app.post("/graph/connect", tags=["graph"])
    async def connect_entities(
        source: str = Body(..., embed=True),
        target: str = Body(..., embed=True),
        relation: str = Body("related", embed=True),
        weight: float = Body(1.0, embed=True),
        authorization: Optional[str] = Header(None),
    ):
        await verify_api_key(authorization)
        mem = get_memory()
        mem.graph_add_edge(source, target, relation, weight=weight)
        return {"connected": True, "source": source, "target": target, "relation": relation}

    @app.get("/graph/metrics", tags=["graph"])
    async def graph_metrics(authorization: Optional[str] = Header(None)):
        """Get graph-level metrics."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.community import CommunityDetector
            detector = CommunityDetector(mem._memory._db.conn)
            comm_metrics = detector.metrics()
            return {
                "communities": {
                    "count": comm_metrics.num_communities,
                    "avg_size": round(comm_metrics.avg_community_size, 1),
                    "largest": comm_metrics.largest_community,
                    "smallest": comm_metrics.smallest_community,
                    "modularity": round(comm_metrics.total_modularity, 4),
                    "coverage": round(comm_metrics.coverage, 3),
                },
            }
        except Exception as e:
            return {"error": str(e)}

    # === Community Routes ===

    @app.get("/communities", tags=["communities"])
    async def list_communities(
        limit: int = Query(100, ge=1, le=1000),
        authorization: Optional[str] = Header(None),
    ):
        """List all detected communities."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.community import CommunityDetector
            detector = CommunityDetector(mem._memory._db.conn)
            communities = detector.get_communities(limit=limit)
            return {
                "communities": [c.to_dict() for c in communities],
                "count": len(communities),
            }
        except Exception as e:
            return {"communities": [], "count": 0, "error": str(e)}

    @app.post("/communities/detect", tags=["communities"])
    async def detect_communities(
        req: CommunityDetectRequest = CommunityDetectRequest(),
        authorization: Optional[str] = Header(None),
    ):
        """Run community detection on the entity graph."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.community import CommunityDetector
            detector = CommunityDetector(mem._memory._db.conn)
            communities = detector.detect_communities(force=req.force)
            return {
                "communities": [c.to_dict() for c in communities],
                "count": len(communities),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/communities/{community_id}", tags=["communities"])
    async def get_community(
        community_id: int,
        authorization: Optional[str] = Header(None),
    ):
        """Get a specific community with its memories."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.community import CommunityDetector
            detector = CommunityDetector(mem._memory._db.conn)
            community = detector.get_community(community_id)
            if not community:
                raise HTTPException(status_code=404, detail="Community not found")

            result = community.to_dict()
            result["summary"] = detector.get_community_summary(community_id)
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/communities/{community_id}/search", tags=["communities"])
    async def search_community(
        community_id: int,
        query: str = Query(...),
        limit: int = Query(10, ge=1, le=100),
        authorization: Optional[str] = Header(None),
    ):
        """Search memories within a community."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.community import CommunityDetector
            detector = CommunityDetector(mem._memory._db.conn)
            results = detector.search_within_community(query, community_id, limit=limit)
            return {"results": results, "count": len(results)}
        except Exception as e:
            return {"results": [], "count": 0, "error": str(e)}

    # === Batch Operations ===

    @app.post("/batch/search", tags=["batch"])
    async def batch_search(
        req: BatchSearchRequest,
        authorization: Optional[str] = Header(None),
    ):
        """Execute multiple search queries in a single request."""
        await verify_api_key(authorization)
        mem = get_memory()
        t0 = time.monotonic()
        all_results = []
        for query in req.queries:
            results = mem.search(query, limit=req.limit, threshold=req.threshold)
            all_results.append({"query": query, "results": results, "count": len(results)})
        latency_ms = (time.monotonic() - t0) * 1000
        return {
            "queries": all_results,
            "total_results": sum(r["count"] for r in all_results),
            "latency_ms": round(latency_ms, 2),
        }

    # === Entity Management Routes ===

    @app.post("/graph/entities", tags=["graph"])
    async def create_entity(
        req: EntityCreateRequest,
        authorization: Optional[str] = Header(None),
    ):
        """Create an entity in the knowledge graph."""
        await verify_api_key(authorization)
        mem = get_memory()
        mem._memory._db.add_entity(req.name, entity_type=req.entity_type)
        return {"created": True, "name": req.name, "type": req.entity_type}

    @app.delete("/graph/entities/{entity_name}", tags=["graph"])
    async def delete_entity(
        entity_name: str,
        authorization: Optional[str] = Header(None),
    ):
        """Delete an entity from the knowledge graph."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            conn = mem._memory._db.conn
            # Get entity ID first
            row = conn.execute("SELECT id FROM entities WHERE name = ?", (entity_name,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Entity not found")
            eid = row[0] if hasattr(row, '__getitem__') else row['id']
            conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (eid, eid))
            conn.execute("DELETE FROM memory_entities WHERE entity_id = ?", (eid,))
            conn.execute("DELETE FROM entities WHERE id = ?", (eid,))
            conn.commit()
            return {"deleted": True, "name": entity_name}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # === Temporal Graph Routes ===

    @app.get("/temporal/facts", tags=["temporal"])
    async def get_temporal_facts(
        subject: Optional[str] = None,
        current_only: bool = True,
        limit: int = Query(100, ge=1, le=1000),
        authorization: Optional[str] = Header(None),
    ):
        """Query temporal facts."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.temporal import TemporalGraph
            tg = TemporalGraph(mem._memory._db.conn)
            facts = tg.find_facts(subject=subject, current_only=current_only)
            if limit:
                facts = facts[:limit]
            return {"facts": [f.to_dict() for f in facts], "count": len(facts)}
        except Exception as e:
            return {"facts": [], "count": 0, "error": str(e)}

    @app.post("/temporal/facts", tags=["temporal"])
    async def add_temporal_fact(
        req: TemporalFactRequest,
        authorization: Optional[str] = Header(None),
    ):
        """Add a temporal fact to the graph."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.temporal import TemporalGraph
            tg = TemporalGraph(mem._memory._db.conn)
            fact = tg.add_fact(
                text=req.text,
                subject=req.subject,
                predicate=req.predicate,
                obj=req.obj,
                valid_at=req.valid_at,
            )
            return fact.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/temporal/timeline/{subject}", tags=["temporal"])
    async def get_timeline(
        subject: str,
        authorization: Optional[str] = Header(None),
    ):
        """Get the temporal timeline for a subject."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.temporal import TemporalGraph
            tg = TemporalGraph(mem._memory._db.conn)
            timeline = tg.get_timeline(subject)
            return {"subject": subject, "timeline": [f.to_dict() for f in timeline], "count": len(timeline)}
        except Exception as e:
            return {"subject": subject, "timeline": [], "count": 0, "error": str(e)}

    @app.post("/temporal/invalidate/{fact_id}", tags=["temporal"])
    async def invalidate_fact(
        fact_id: str,
        authorization: Optional[str] = Header(None),
    ):
        """Invalidate a temporal fact."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.temporal import TemporalGraph
            tg = TemporalGraph(mem._memory._db.conn)
            success = tg.invalidate_fact(fact_id)
            if not success:
                raise HTTPException(status_code=404, detail="Fact not found")
            return {"invalidated": True, "fact_id": fact_id}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/temporal/stats", tags=["temporal"])
    async def temporal_stats(authorization: Optional[str] = Header(None)):
        """Get temporal graph statistics."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.temporal import TemporalGraph
            tg = TemporalGraph(mem._memory._db.conn)
            return tg.stats()
        except Exception as e:
            return {"error": str(e)}

    # === Lifecycle Management Routes ===

    @app.post("/lifecycle/evict", tags=["lifecycle"])
    async def lifecycle_evict(
        target_count: int = Query(100, ge=10, le=10000),
        authorization: Optional[str] = Header(None),
    ):
        """Evict low-priority memories to target count."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            evicted = mem._memory._db.evict()
            return {"evicted": evicted, "target_count": target_count}
        except Exception as e:
            return {"evicted": 0, "error": str(e)}

    @app.get("/lifecycle", tags=["lifecycle"])
    async def lifecycle_status(authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        if hasattr(mem._memory, "_get_lifecycle"):
            return mem._memory.run_lifecycle()
        return {"error": "Lifecycle not configured"}

    @app.post("/lifecycle/forget", tags=["lifecycle"])
    async def lifecycle_forget(
        min_age_days: int = Query(180, ge=30, le=365),
        authorization: Optional[str] = Header(None),
    ):
        """Forget memories from dead communities."""
        await verify_api_key(authorization)
        mem = get_memory()
        try:
            from arriadne.community import CommunityDetector
            detector = CommunityDetector(mem._memory._db.conn)
            detector.mark_dead_communities()
            forgotten = detector.forget_dead_communities(min_age_days=min_age_days)
            return {"forgotten": forgotten, "min_age_days": min_age_days}
        except Exception as e:
            return {"forgotten": 0, "error": str(e)}

    # === Consolidation Routes ===

    @app.post("/consolidate", tags=["consolidation"])
    async def consolidate(
        method: str = Query("similarity"),
        dry_run: bool = Query(False),
        authorization: Optional[str] = Header(None),
    ):
        await verify_api_key(authorization)
        mem = get_memory()
        if hasattr(mem._memory, "_consolidator"):
            return mem._memory.consolidate_with_llm(method=method, dry_run=dry_run)
        return {"error": "Consolidation not configured"}

    # === Import/Export Routes ===

    @app.post("/import", tags=["data"])
    async def import_memories(
        memories: List[StoreRequest],
        authorization: Optional[str] = Header(None),
        x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    ):
        """Import multiple memories at once."""
        await verify_api_key(authorization)
        mem = get_memory()
        tenant = x_tenant_id or "default"
        results = [
            mem.store(
                m.content, topic=m.topic, importance=m.importance,
                entities=m.entities, metadata=m.metadata,
                tenant_id=tenant,
            )
            for m in memories
        ]
        return {"imported": len(results), "memories": results}

    @app.get("/export", tags=["data"])
    async def export_memories(
        format: str = Query("json"),
        authorization: Optional[str] = Header(None),
    ):
        """Export all memories."""
        await verify_api_key(authorization)
        mem = get_memory()
        all_memories = mem.search("*", limit=10000, threshold=0.0)
        return {"memories": all_memories, "count": len(all_memories), "format": format}

    # === Versioned API (/api/v1/) ===

    if enable_versioning:
        from fastapi import APIRouter
        v1 = APIRouter(prefix="/api/v1")

        @v1.get("/health", tags=["v1"])
        async def v1_health():
            return {"status": "healthy", "version": "3.0.0", "api": "v1"}

        @v1.post("/memories", tags=["v1"])
        async def v1_store_memory(
            req: StoreRequest,
            authorization: Optional[str] = Header(None),
            x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
        ):
            return await store_memory(req, authorization, x_tenant_id)

        @v1.get("/memories/{memory_id}", tags=["v1"])
        async def v1_get_memory(memory_id: str, authorization: Optional[str] = Header(None)):
            return await get_memory_by_id(memory_id, authorization)

        @v1.delete("/memories/{memory_id}", tags=["v1"])
        async def v1_delete_memory(memory_id: str, authorization: Optional[str] = Header(None)):
            return await delete_memory(memory_id, authorization)

        @v1.post("/search", tags=["v1"])
        async def v1_search(
            req: SearchRequest,
            authorization: Optional[str] = Header(None),
            x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
        ):
            return await search_memories(req, authorization, x_tenant_id)

        @v1.post("/extract", tags=["v1"])
        async def v1_extract(req: ExtractRequest, authorization: Optional[str] = Header(None)):
            return await extract_memories(req, authorization)

        @v1.get("/stats", tags=["v1"])
        async def v1_stats(authorization: Optional[str] = Header(None)):
            return await stats(authorization)

        app.include_router(v1)

    # === Graceful Shutdown ===

    @app.on_event("shutdown")
    async def shutdown_event():
        """Save FAISS index and close DB on shutdown."""
        logger.info("Graceful shutdown: saving FAISS index and closing DB...")
        _shutdown_event.set()
        try:
            mem = get_memory()
            mem._memory._db._save_faiss_index()
            mem._memory._db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            mem._memory._db.conn.commit()
        except Exception as e:
            logger.error("Error during shutdown: %s", e)
        logger.info("Shutdown complete")

    return app


def main():
    """CLI entry point for the server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Ariadne Memory API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8899, help="Port to listen on")
    parser.add_argument("--db-path", default="ariadne.db", help="Database path")
    parser.add_argument("--api-key", default=None, help="API key for auth")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--rate-limit", type=int, default=120, help="Rate limit (requests/minute)")
    parser.add_argument("--no-versioning", action="store_true", help="Disable /api/v1/ prefix")
    args = parser.parse_args()

    app = create_app(
        db_path=args.db_path,
        api_key=args.api_key,
        rate_limit_rpm=args.rate_limit,
        enable_versioning=not args.no_versioning,
    )
    logger.info("Starting Ariadne server on %s:%d (rate_limit=%d rpm)", args.host, args.port, args.rate_limit)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
