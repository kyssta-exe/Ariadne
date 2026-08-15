"""Ariadne Dashboard — FastAPI server exposing memory system stats and a single-file SPA.

Usage:
    from arriadne.dashboard.server import create_app
    uvicorn.run(create_app(db_path="arriadne.db"), host="127.0.0.1", port=8765)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from arriadne.config import AriadneConfig
from arriadne.interface import AriadneMemory

# ---------------------------------------------------------------------------
# Lazy FastAPI import so the module can be imported without fastapi installed
# (the CLI catches ImportError and tells the user to install extras).
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi import UploadFile
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "FastAPI is required for the dashboard.  "
        "Install it with:  pip install 'arriadne[dashboard]'"
    ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"


def _jsonable(obj: Any) -> Any:
    """Convert numpy/Path/bytes types to JSON-safe primitives.

    Also sanitizes strings that contain invalid UTF-8 sequences, which
    can crash Pydantic's JSON serializer.
    """
    import numpy as _np

    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        # BLOBs (embeddings etc.) — decode safely, fall back to hex
        try:
            return obj.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return obj.hex()
    if isinstance(obj, str):
        # Strip any invalid UTF-8 surrogates that SQLite may have stored
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict."""
    if row is None:
        return {}
    return dict(row)


def _parse_range(range_str: str) -> float:
    """Convert a human range like '7d' / '30d' / '1h' to seconds."""
    range_str = range_str.strip().lower()
    if range_str.endswith("d"):
        return float(range_str[:-1]) * 86400
    if range_str.endswith("h"):
        return float(range_str[:-1]) * 3600
    if range_str.endswith("m"):
        return float(range_str[:-1]) * 60
    return float(range_str)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(db_path: str | Path = "arriadne.db") -> FastAPI:
    """Create and return the FastAPI application wired to an AriadneMemory instance.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Configured FastAPI instance ready to serve.
    """
    config = AriadneConfig(db_path=str(db_path))
    mem = AriadneMemory(config=config)

    app = FastAPI(title="Ariadne Dashboard", version="0.1.0")

    # Keep the local dashboard local; wildcard origins with credentials are
    # invalid in browsers and unsafe when the server is exposed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8765", "http://localhost:8765"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # API routes
    # -----------------------------------------------------------------------

    @app.get("/api/stats")
    def api_stats() -> Any:
        """Return comprehensive memory system statistics."""
        return _jsonable(mem.stats())

    @app.get("/api/memories")
    def api_memories_list(
        page: int = Query(1, ge=1),
        per_page: int = Query(20, ge=1, le=200),
        type: str = Query(""),
        search: str = Query(""),
        sort: str = Query("created_at"),
        order: str = Query("desc"),
    ) -> Any:
        """Paginated memory list with optional type filter and FTS search."""
        db = mem._db
        assert db.conn is not None
        conn = db.conn

        allowed_sorts = {"created_at", "importance", "access_count", "updated_at"}
        if sort not in allowed_sorts:
            sort = "created_at"
        order_dir = "DESC" if order.lower() == "desc" else "ASC"

        offset = (page - 1) * per_page

        # Build query
        where_parts: list[str] = ["m.is_deleted = 0"]
        params: list[Any] = []

        if type:
            where_parts.append("m.memory_type = ?")
            params.append(type)

        where_sql = " AND ".join(where_parts)

        # Count total
        count_sql = f"SELECT COUNT(*) FROM memories m WHERE {where_sql}"
        total = conn.execute(count_sql, params).fetchone()[0]

        # Fetch page — if search is given, use FTS to narrow results
        if search.strip():
            # Use FTS for search filtering
            from arriadne.storage import _fts_escape

            fts_q = _fts_escape(search)
            data_sql = f"""
                SELECT m.id, m.content, m.memory_type, m.importance,
                       m.created_at, m.updated_at, m.accessed_at,
                       m.access_count, m.is_deleted, m.metadata
                FROM memories m
                JOIN memories_fts fts ON fts.rowid = m.id
                WHERE memories_fts MATCH ? AND {where_sql}
                ORDER BY m.{sort} {order_dir}
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(data_sql, [fts_q] + params + [per_page, offset]).fetchall()
            # Recount with search filter
            total_sql = f"""
                SELECT COUNT(*) FROM memories m
                JOIN memories_fts fts ON fts.rowid = m.id
                WHERE memories_fts MATCH ? AND {where_sql}
            """
            total = conn.execute(total_sql, [fts_q] + params).fetchone()[0]
        else:
            data_sql = f"""
                SELECT m.id, m.content, m.memory_type, m.importance,
                       m.created_at, m.updated_at, m.accessed_at,
                       m.access_count, m.is_deleted, m.metadata
                FROM memories m
                WHERE {where_sql}
                ORDER BY m.{sort} {order_dir}
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(data_sql, params + [per_page, offset]).fetchall()

        memories = []
        for r in rows:
            d = dict(r)
            # Parse metadata JSON
            if d.get("metadata"):
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass
            memories.append(d)

        return {
            "memories": _jsonable(memories),
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": max(1, -(-total // per_page)),  # ceil div
        }

    @app.get("/api/memories/{memory_id}")
    def api_memories_detail(memory_id: int) -> Any:
        """Single memory detail with entity associations."""
        db = mem._db
        conn = db.conn
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Memory not found")
        d = dict(row)
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        # Fetch associated entities
        entities_rows = conn.execute(
            """SELECT e.id, e.name, e.entity_type
               FROM entities e
               JOIN memory_entities me ON me.entity_id = e.id
               WHERE me.memory_id = ?""",
            (memory_id,),
        ).fetchall()
        d["entities"] = [_row_to_dict(e) for e in entities_rows]
        return _jsonable(d)

    @app.post("/api/memories")
    async def api_memories_create(request: Request) -> Any:
        """Add a new memory."""
        body = await _safe_json(request)
        content = body.get("content", "")
        if not content.strip():
            raise HTTPException(status_code=400, detail="content is required")
        result = mem.remember(
            content=content,
            memory_type=body.get("type", "semantic"),
            importance=float(body.get("importance", 0.5)),
            entities=body.get("entities"),
            metadata=body.get("metadata"),
        )
        if result["status"] == "duplicate":
            return JSONResponse(
                status_code=409,
                content={"error": "duplicate", "duplicate_of": result.get("duplicate_of")},
            )
        return _jsonable(result)

    @app.put("/api/memories/{memory_id}")
    async def api_memories_update(memory_id: int, request: Request) -> Any:
        """Update an existing memory."""
        body = await _safe_json(request)
        ok = mem.update(
            memory_id,
            content=body.get("content"),
            importance=body.get("importance"),
            metadata=body.get("metadata"),
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Memory not found or unchanged")
        return {"ok": True}

    @app.delete("/api/memories/{memory_id}")
    def api_memories_delete(memory_id: int) -> Any:
        """Soft-delete a memory."""
        ok = mem.forget(memory_id, hard=False)
        if not ok:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"ok": True}

    @app.get("/api/search")
    def api_search(
        q: str = Query(""),
        k: int = Query(10, ge=1, le=100),
        type: str = Query(""),
    ) -> Any:
        """Search memories via AriadneMemory.recall()."""
        results = mem.recall(
            query=q,
            k=k,
            type_filter=type if type else None,
        )
        return {"results": _jsonable(results)}

    @app.get("/api/graph")
    def api_graph(
        entity: str = Query(""),
        hops: int = Query(1, ge=1, le=5),
    ) -> Any:
        """Graph traversal from an entity."""
        if not entity.strip():
            raise HTTPException(status_code=400, detail="entity query param is required")
        result = mem.graph(entity=entity, hops=hops)
        return _jsonable(result)

    @app.get("/api/graph/all")
    def api_graph_all() -> Any:
        """Return all entities and edges for full graph visualization."""
        db = mem._db
        conn = db.conn

        entities_rows = conn.execute("SELECT id, name, entity_type FROM entities").fetchall()
        edges_rows = conn.execute(
            """SELECT e.id, s.name AS source, t.name AS target,
                      e.edge_type, e.weight
               FROM edges e
               JOIN entities s ON s.id = e.source_id
               JOIN entities t ON t.id = e.target_id"""
        ).fetchall()

        nodes = []
        for r in entities_rows:
            d = dict(r)
            d["label"] = d["name"]
            nodes.append(d)

        edges = []
        for r in edges_rows:
            edges.append(dict(r))

        return {"nodes": _jsonable(nodes), "edges": _jsonable(edges)}

    @app.get("/api/activity")
    def api_activity(range: str = Query("7d")) -> Any:
        """Memories ingested per day for activity chart."""
        db = mem._db
        conn = db.conn
        seconds = _parse_range(range)
        cutoff = time.time() - seconds
        rows = conn.execute(
            """SELECT date(created_at, 'unixepoch') as day, COUNT(*) as cnt
               FROM memories
               WHERE is_deleted = 0 AND created_at >= ?
               GROUP BY day
               ORDER BY day""",
            (cutoff,),
        ).fetchall()
        # Top accessed
        top_rows = conn.execute(
            """SELECT id, content, access_count, memory_type, importance
               FROM memories
               WHERE is_deleted = 0
               ORDER BY access_count DESC
               LIMIT 10"""
        ).fetchall()
        # Recent feed
        recent_rows = conn.execute(
            """SELECT id, content, memory_type, created_at
               FROM memories
               WHERE is_deleted = 0
               ORDER BY created_at DESC
               LIMIT 15"""
        ).fetchall()

        return {
            "timeline": [_jsonable(dict(r)) for r in rows],
            "top_accessed": [_jsonable(dict(r)) for r in top_rows],
            "recent": [_jsonable(dict(r)) for r in recent_rows],
        }

    @app.get("/api/config")
    def api_config() -> Any:
        """Return current config as JSON."""
        from dataclasses import asdict

        return _jsonable(asdict(config))

    @app.get("/api/backup")
    def api_backup() -> FileResponse:
        """Download the database file as a backup.

        Performs a WAL checkpoint first to ensure all pending writes
        are flushed to the main database file.
        """
        db = mem._db
        conn = db.conn
        # WAL checkpoint: flush WAL into the main .db file
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db_file = Path(db._config.db_path)
        if not db_file.exists():
            raise HTTPException(status_code=500, detail="Database file not found")
        return FileResponse(
            path=str(db_file),
            media_type="application/octet-stream",
            filename=db_file.name,
        )

    @app.post("/api/restore")
    async def api_restore(file: UploadFile) -> dict[str, Any]:
        """Restore the database from an uploaded .db file.

        Steps:
        1. Create a safety backup of the current database.
        2. Save the uploaded file over the current database path.
        3. Reinitialize the memory system.
        """
        nonlocal mem  # rebinding the closure variable on reinit below
        db = mem._db
        db_path = Path(db._config.db_path)

        # --- safety backup ---------------------------------------------------
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safety_backup = backup_dir / f"{db_path.stem}_backup_{ts}.db"
        if db_path.exists():
            shutil.copy2(str(db_path), str(safety_backup))

        # --- close the live connection BEFORE overwriting the files ----------
        # (Writing over a database that still has an open WAL connection, or
        # leaving stale -wal/-shm files beside the new file, corrupts it.)
        try:
            mem.close()
        except Exception:
            pass

        # --- replace DB with uploaded file -----------------------------------
        try:
            contents = await file.read()
            for suffix in ("-wal", "-shm"):
                stale = Path(str(db_path) + suffix)
                if stale.exists():
                    stale.unlink()
            with open(str(db_path), "wb") as f:
                f.write(contents)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to save uploaded file: {exc}"
            ) from exc

        # --- reinitialize memory system --------------------------------------
        try:
            # Re-create AriadneMemory with the same config (points to the new file)
            mem = AriadneMemory(config=config)
        except Exception as exc:
            # If reinit fails, try to restore from safety backup
            if safety_backup.exists():
                shutil.copy2(str(safety_backup), str(db_path))
                mem = AriadneMemory(config=config)
            raise HTTPException(
                status_code=500,
                detail=f"Restore succeeded but reinitialization failed: {exc}; "
                f"safety backup saved at {safety_backup}",
            ) from exc

        return {
            "status": "ok",
            "safety_backup": str(safety_backup),
            "db_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
        }

    @app.post("/api/consolidate")
    def api_consolidate() -> Any:
        """Run memory consolidation."""
        count = mem.consolidate()
        return {"consolidated_groups": count}

    @app.post("/api/maintenance")
    def api_maintenance() -> Any:
        """Run full maintenance cycle."""
        result = mem.maintenance()
        return _jsonable(result)

    @app.get("/api/composition")
    def api_composition() -> Any:
        """Memory composition breakdown by type with counts and percentages."""
        db = mem._db
        conn = db.conn
        rows = conn.execute(
            """SELECT memory_type, COUNT(*) as cnt
               FROM memories WHERE is_deleted = 0
               GROUP BY memory_type ORDER BY cnt DESC"""
        ).fetchall()
        total = sum(dict(r)["cnt"] for r in rows) or 1
        return {
            "types": [
                {
                    "type": dict(r)["memory_type"],
                    "count": dict(r)["cnt"],
                    "pct": round(dict(r)["cnt"] / total * 100, 1),
                }
                for r in rows
            ],
            "total": total,
        }

    @app.get("/api/link-types")
    def api_link_types() -> Any:
        """Edge/link type breakdown with counts and percentages."""
        db = mem._db
        conn = db.conn
        rows = conn.execute(
            """SELECT edge_type, COUNT(*) as cnt
               FROM edges
               GROUP BY edge_type ORDER BY cnt DESC"""
        ).fetchall()
        total = sum(dict(r)["cnt"] for r in rows) or 1
        return {
            "types": [
                {
                    "type": dict(r)["edge_type"],
                    "count": dict(r)["cnt"],
                    "pct": round(dict(r)["cnt"] / total * 100, 1),
                }
                for r in rows
            ],
            "total": total,
        }

    @app.get("/api/recent")
    def api_recent(limit: int = Query(10, ge=1, le=50)) -> Any:
        """Recent memory ingestion events."""
        db = mem._db
        conn = db.conn
        rows = conn.execute(
            """SELECT id, content, memory_type, importance, created_at, tags
               FROM memories WHERE is_deleted = 0
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return {"memories": [_jsonable(dict(r)) for r in rows]}

    @app.get("/api/memory-graph")
    def api_memory_graph() -> Any:
        """Return memories as nodes with edges from shared entities and memory_links."""
        db = mem._db
        conn = db.conn

        # Fetch all active memories
        mem_rows = conn.execute(
            """SELECT id, content, memory_type, importance, created_at, tags
               FROM memories WHERE is_deleted = 0 ORDER BY id"""
        ).fetchall()

        nodes = []
        for r in mem_rows:
            d = dict(r)
            content = d["content"] or ""
            label = content[:30] + ("..." if len(content) > 30 else "")
            nodes.append(
                {
                    "id": d["id"],
                    "label": label,
                    "content": content,
                    "memory_type": d["memory_type"],
                    "importance": d["importance"],
                    "tags": json.loads(d["tags"])
                    if isinstance(d.get("tags"), str) and d["tags"]
                    else [],
                }
            )

        # Build edges from memory_links table
        link_rows = conn.execute(
            """SELECT source_id, target_id, link_type, strength
               FROM memory_links"""
        ).fetchall()

        edges = []
        seen = set()
        for r in link_rows:
            d = dict(r)
            key = (d["source_id"], d["target_id"])
            if key not in seen:
                seen.add(key)
                edges.append(
                    {
                        "source": d["source_id"],
                        "target": d["target_id"],
                        "relation": d["link_type"],
                        "weight": d["strength"],
                    }
                )

        # Build edges from shared entities
        entity_mem = conn.execute(
            """SELECT me.memory_id, e.name
               FROM memory_entities me
               JOIN entities e ON e.id = me.entity_id"""
        ).fetchall()

        # Group memories by entity
        entity_to_mems: dict[str, list[int]] = {}
        for r in entity_mem:
            d = dict(r)
            ent = d["name"]
            mid = d["memory_id"]
            entity_to_mems.setdefault(ent, []).append(mid)

        # Connect memories that share entities
        for ent, mem_ids in entity_to_mems.items():
            for i in range(len(mem_ids)):
                for j in range(i + 1, len(mem_ids)):
                    key = (min(mem_ids[i], mem_ids[j]), max(mem_ids[i], mem_ids[j]))
                    if key not in seen:
                        seen.add(key)
                        edges.append(
                            {
                                "source": key[0],
                                "target": key[1],
                                "relation": f"shared:{ent}",
                                "weight": 0.5,
                            }
                        )

        # If no edges from memory_entities or memory_links, connect memories
        # that share the same memory_type (co-category edges)
        if not edges:
            type_to_mems: dict[str, list[int]] = {}
            for n in nodes:
                mt = n.get("memory_type", "general")
                type_to_mems.setdefault(mt, []).append(n["id"])

            for mt, mem_ids in type_to_mems.items():
                # Connect each pair within the same type (limit to avoid cliques)
                for i in range(len(mem_ids)):
                    for j in range(i + 1, min(i + 5, len(mem_ids))):
                        key = (min(mem_ids[i], mem_ids[j]), max(mem_ids[i], mem_ids[j]))
                        if key not in seen:
                            seen.add(key)
                            edges.append(
                                {
                                    "source": key[0],
                                    "target": key[1],
                                    "relation": f"type:{mt}",
                                    "weight": 0.3,
                                }
                            )

        return {"nodes": _jsonable(nodes), "edges": _jsonable(edges)}

    @app.get("/api/health-report")
    def api_health_report() -> Any:
        """Return a comprehensive health and deduplication report."""
        db = mem._db
        conn = db.conn
        now = time.time()

        total_memories = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE is_deleted = 0"
        ).fetchone()[0]

        avg_importance = (
            conn.execute("SELECT AVG(importance) FROM memories WHERE is_deleted = 0").fetchone()[0]
            or 0.0
        )

        oldest_memory = conn.execute(
            "SELECT MIN(created_at) FROM memories WHERE is_deleted = 0"
        ).fetchone()[0]
        oldest_age_days = round((now - oldest_memory) / 86400, 1) if oldest_memory else 0

        # AriadneMemory keeps a per-namespace MinHash Deduplicator; the dashboard
        # surface mirrors MemoryStore.stats()'s aggregate of dedup sizes. The old
        # code referenced mem._dedup (which doesn't exist), crashing this route.
        dedup_index_size = sum(d.size for d in getattr(mem, "_dedup_by_namespace", {}).values())

        db_path = Path(db._config.db_path)
        db_size_kb = round(db_path.stat().st_size / 1024, 1) if db_path.exists() else 0

        decay_candidates = conn.execute(
            """SELECT COUNT(*) FROM memories 
               WHERE is_deleted = 0 AND created_at < ? AND importance < 0.3""",
            (now - 2592000,),
        ).fetchone()[0]

        return {
            "total_memories": total_memories,
            "avg_importance": round(avg_importance, 3),
            "oldest_memory_days": oldest_age_days,
            "db_size_kb": db_size_kb,
            "dedup_index_size": dedup_index_size,
            "decay_candidates": decay_candidates,
            "status": "healthy" if total_memories > 0 else "empty",
        }

    @app.get("/api/graph/neighbors")
    def api_graph_neighbors(
        memory_id: int = Query(..., description="Starting memory ID"),
        depth: int = Query(2, ge=1, le=5, description="Max traversal depth"),
    ) -> Any:
        """Return depth-limited graph neighborhood using recursive CTE."""
        db = mem._db
        conn = db.conn

        base_mem = conn.execute(
            "SELECT id, content, memory_type FROM memories WHERE id = ? AND is_deleted = 0",
            (memory_id,),
        ).fetchone()

        if not base_mem:
            raise HTTPException(status_code=404, detail="Memory not found")

        cte_query = """
            WITH RECURSIVE neighbors(id, depth) AS (
                SELECT ?, 0
                UNION
                SELECT
                    CASE
                        WHEN ml.source_id = n.id THEN ml.target_id
                        ELSE ml.source_id
                    END,
                    n.depth + 1
                FROM memory_links ml
                JOIN neighbors n ON (ml.source_id = n.id OR ml.target_id = n.id)
                WHERE n.depth < ?
            )
            SELECT DISTINCT id FROM neighbors
        """
        neighbor_ids = [row[0] for row in conn.execute(cte_query, (memory_id, depth)).fetchall()]

        if not neighbor_ids:
            return {"center_id": memory_id, "depth": depth, "nodes": [], "edges": []}

        placeholders = ",".join("?" * len(neighbor_ids))
        nodes = []
        for row in conn.execute(
            f"SELECT id, content, memory_type, importance FROM memories WHERE id IN ({placeholders}) AND is_deleted = 0",
            tuple(neighbor_ids),
        ).fetchall():
            d = dict(row)
            content = d["content"] or ""
            nodes.append(
                {
                    "id": d["id"],
                    "label": content[:40] + ("..." if len(content) > 40 else ""),
                    "memory_type": d["memory_type"],
                    "importance": d["importance"],
                }
            )

        edges = []
        for row in conn.execute(
            f"""SELECT source_id, target_id, link_type, strength 
                FROM memory_links 
                WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})""",
            tuple(neighbor_ids) * 2,
        ).fetchall():
            d = dict(row)
            edges.append(
                {
                    "source": d["source_id"],
                    "target": d["target_id"],
                    "relation": d["link_type"],
                    "weight": d["strength"],
                }
            )

        return {
            "center_id": memory_id,
            "depth": depth,
            "nodes": _jsonable(nodes),
            "edges": _jsonable(edges),
        }

    # -----------------------------------------------------------------------
    # Static file serving + SPA fallback
    # -----------------------------------------------------------------------

    # Mount static assets at /dashboard/static/ (matches HTML references)
    if _STATIC_DIR.exists():
        app.mount("/dashboard/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/api/export")
    def api_export() -> Any:
        """Export all memories, entities, and links as JSON."""
        return _jsonable(mem.export_json())

    @app.post("/api/import")
    async def api_import(request: Request) -> Any:
        """Import from previously exported JSON."""
        data = await _safe_json(request)
        if not data or "memories" not in data:
            return {"error": "Invalid import data", "count": 0}
        count = mem.import_json(data)
        return {"count": count, "status": "ok"}

    @app.get("/api/tags")
    def api_tags() -> Any:
        """Return all unique tags across all active memories."""
        db = mem._db
        conn = db.conn
        rows = conn.execute(
            "SELECT DISTINCT tags FROM memories WHERE is_deleted = 0 AND tags != '[]'"
        ).fetchall()
        all_tags = set()
        for row in rows:
            try:
                tag_list = json.loads(row[0])
                all_tags.update(tag_list)
            except (json.JSONDecodeError, TypeError):
                pass
        return {"tags": sorted(all_tags)}

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/dashboard/", response_class=HTMLResponse, include_in_schema=False)
    def serve_dashboard() -> HTMLResponse:
        """Serve the dashboard SPA shell."""
        index = _STATIC_DIR / "index.html"
        if index.exists():
            return HTMLResponse(content=index.read_text(encoding="utf-8"))
        return HTMLResponse(
            content="<h1>Ariadne Dashboard</h1><p>Frontend not built.</p>", status_code=200
        )

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str) -> HTMLResponse:
        """Serve index.html for all non-API routes (SPA fallback)."""
        index = _STATIC_DIR / "index.html"
        if index.exists():
            return HTMLResponse(content=index.read_text(encoding="utf-8"))
        return HTMLResponse(
            content="<h1>Ariadne Dashboard</h1><p>Frontend not built.</p>", status_code=200
        )

    return app


async def _safe_json(request: Request) -> dict[str, Any]:
    """Parse JSON body, return empty dict on failure.

    Starlette only exposes the request body asynchronously (``await
    request.body()``); the previous sync implementation read the unset
    ``_body`` attribute and always returned ``{}``, silently breaking every
    POST/PUT endpoint.
    """
    try:
        body_bytes = await request.body()
        parsed = json.loads(body_bytes) if body_bytes else {}
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# CLI entry point for `python -m arriadne.dashboard.server`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ariadne Dashboard Server")
    parser.add_argument("--db-path", default="arriadne.db", help="Database path")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    args = parser.parse_args()

    app = create_app(db_path=args.db_path)

    try:
        import uvicorn
    except ImportError:
        sys.exit(
            "uvicorn is required to run the dashboard server.  "
            "Install it with:  pip install 'arriadne[dashboard]'"
        )

    print(f"Ariadne Dashboard running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


# ---------------------------------------------------------------------------
# Module-level app for `uvicorn arriadne.dashboard.server:app`
#
# Lazy ASGI wrapper: the real app (and its SQLite database) is only created on
# the first request, so merely importing this module — e.g. `from
# arriadne.dashboard.server import create_app` — no longer creates an
# `arriadne.db` file in the current working directory as a side effect.
# ---------------------------------------------------------------------------


class _LazyApp:
    def __init__(self) -> None:
        self._app: FastAPI | None = None

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if self._app is None:
            self._app = create_app(db_path=os.environ.get("ARIADNE_DB_PATH", "arriadne.db"))
        await self._app(scope, receive, send)


app = _LazyApp()
