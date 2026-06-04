"""
Ariadne REST API Server

FastAPI-based server that exposes Ariadne's memory capabilities over HTTP.

Usage:
    arriadne-server --port 8899 --db-path ./memory.db
    # or
    python -m arriadne.server --port 8899
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("arriadne.server")


# === Request/Response Models (module-level for Pydantic v2) ===

class StoreRequest(BaseModel):
    content: str
    topic: Optional[str] = "general"
    importance: int = Field(5, ge=1, le=10)
    entities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    agent_id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(10, ge=1, le=100)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    use_hybrid: bool = True
    include_graph: bool = False
    include_metadata: bool = True
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


# === App Factory ===

def create_app(
    db_path: str = "ariadne.db",
    embedding_config: Optional[Dict] = None,
    llm_config: Optional[Dict] = None,
    api_key: Optional[str] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Ariadne Memory API",
        description="Fast local memory system for AI agents",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazy-init the memory system
    _memory = None

    def get_memory():
        nonlocal _memory
        if _memory is None:
            from arriadne.interface import AriadneMemory
            _memory = AriadneMemory(
                db_path=db_path,
                llm_config=llm_config,
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

    # === Routes ===

    @app.get("/", tags=["health"])
    async def root():
        return {"name": "Ariadne Memory API", "version": "1.0.0", "status": "running", "docs": "/docs"}

    @app.get("/health", tags=["health"])
    async def health():
        mem = get_memory()
        stats = mem.stats()
        return {"status": "healthy", "memories": stats.get("total_memories", 0), "uptime": time.time()}

    @app.get("/stats", tags=["stats"])
    async def stats(authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        raw = mem.stats()
        return {
            "total_memories": raw.get("total_memories", 0),
            "active_memories": raw.get("active_memories", 0),
            "graph_nodes": raw.get("graph_nodes", 0),
            "graph_edges": raw.get("graph_edges", 0),
            "vector_index_size": raw.get("vector_index_size", 0),
            "embedding_model": raw.get("embedding_provider", "none"),
            "avg_latency_ms": raw.get("avg_latency_ms", 0),
        }

    @app.post("/memories", tags=["memories"])
    async def store_memory(req: StoreRequest, authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        result = mem.store(
            req.content, topic=req.topic, importance=req.importance,
            entities=req.entities, metadata=req.metadata,
        )
        return result

    @app.post("/search", tags=["search"])
    async def search_memories(req: SearchRequest, authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        results = mem.search(
            req.query, limit=req.limit, threshold=req.threshold,
            use_hybrid=req.use_hybrid, include_graph=req.include_graph,
        )
        return {"results": results, "count": len(results)}

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
        return {"extracted": len(extracted), "stored": len(stored), "memories": [
            {"text": m.text, "topic": m.topic, "importance": m.importance, "entities": m.entities}
            for m in extracted
        ]}

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
        return {"deleted": True, "id": memory_id}

    @app.get("/graph/entities", tags=["graph"])
    async def get_entities(entity_type: Optional[str] = None, limit: int = 100, authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        entities = mem.get_entities(entity_type=entity_type, limit=limit)
        return {"entities": entities, "count": len(entities)}

    @app.get("/graph/entity/{entity_name}", tags=["graph"])
    async def get_entity_graph(entity_name: str, hops: int = 2, authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        return mem.graph_search(entity_name, hops=hops)

    @app.post("/graph/connect", tags=["graph"])
    async def connect_entities(
        source: str = Body(..., embed=True),
        target: str = Body(..., embed=True),
        relation: str = Body(..., embed=True),
        weight: float = Body(1.0, embed=True),
        authorization: Optional[str] = Header(None),
    ):
        await verify_api_key(authorization)
        mem = get_memory()
        mem.graph_add_edge(source, target, relation, weight=weight)
        return {"connected": True, "source": source, "target": target, "relation": relation}

    @app.get("/lifecycle", tags=["lifecycle"])
    async def lifecycle_status(authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        if hasattr(mem, "_lifecycle"):
            return mem.run_lifecycle()
        return {"error": "Lifecycle not configured"}

    @app.post("/consolidate", tags=["consolidation"])
    async def consolidate(method: str = "similarity", dry_run: bool = False, authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        if hasattr(mem, "_consolidator"):
            return mem.consolidate_with_llm(method=method, dry_run=dry_run)
        return {"error": "Consolidation not configured"}

    @app.post("/import", tags=["data"])
    async def import_memories(memories: List[StoreRequest], authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        results = [mem.store(m.content, topic=m.topic, importance=m.importance, entities=m.entities, metadata=m.metadata) for m in memories]
        return {"imported": len(results), "memories": results}

    @app.get("/export", tags=["data"])
    async def export_memories(format: str = "json", authorization: Optional[str] = Header(None)):
        await verify_api_key(authorization)
        mem = get_memory()
        all_memories = mem.search("*", limit=10000, threshold=0.0)
        return {"memories": all_memories, "count": len(all_memories), "format": format}

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
    args = parser.parse_args()

    app = create_app(db_path=args.db_path, api_key=args.api_key)
    logger.info(f"Starting Ariadne server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
