"""
Export/Import Migration Tools

Provides functions for exporting and importing memories between
Ariadne and other systems (ChromaDB, Mem0, plain text, markdown).
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("arriadne.migration")


def export_json(
    memory: Any,
    path: str | Path,
    format: str = "ariadne",
) -> Dict[str, Any]:
    """
    Export all memories to a JSON file.

    Args:
        memory: AriadneMemory instance.
        path: Output file path.
        format: Export format ("ariadne" or "compact").

    Returns:
        Dict with export statistics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cursor = memory._db.conn.execute(
        """SELECT id, content, content_hash, memory_type, importance,
                  created_at, updated_at, accessed_at, access_count,
                  retention_strength, metadata, tenant_id, category
           FROM memories WHERE is_deleted = 0"""
    )
    memories = []
    for row in cursor.fetchall():
        mem = {
            "id": row[0],
            "content": row[1],
            "memory_type": row[3],
            "category": row[12] or "semantic",
            "importance": row[4],
            "created_at": row[5],
            "updated_at": row[6],
            "accessed_at": row[7],
            "access_count": row[8],
            "retention_strength": row[9],
        }
        if row[10]:
            try:
                mem["metadata"] = json.loads(row[10])
            except Exception:
                mem["metadata"] = {}
        else:
            mem["metadata"] = {}
        mem["tenant_id"] = row[11] or "default"

        # Get entities
        entity_cursor = memory._db.conn.execute(
            """SELECT e.name FROM entities e
               JOIN memory_entities me ON me.entity_id = e.id
               WHERE me.memory_id = ?""",
            (row[0],),
        )
        mem["entities"] = [e[0] for e in entity_cursor.fetchall()]
        memories.append(mem)

    # Get edges
    edge_cursor = memory._db.conn.execute(
        """SELECT s.name, e.edge_type, t.name, e.weight
           FROM edges e
           JOIN entities s ON s.id = e.source_id
           JOIN entities t ON t.id = e.target_id"""
    )
    edges = [
        {"source": r[0], "type": r[1], "target": r[2], "weight": r[3]}
        for r in edge_cursor.fetchall()
    ]

    export_data = {
        "format": "ariadne",
        "version": "1.0",
        "exported_at": time.time(),
        "count": len(memories),
        "memories": memories,
        "edges": edges,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    logger.info("Exported %d memories to %s", len(memories), path)
    return {"exported": len(memories), "edges": len(edges), "path": str(path)}


def import_json(
    memory: Any,
    path: str | Path,
    dedup: bool = True,
) -> Dict[str, Any]:
    """
    Import memories from Ariadne JSON export.

    Args:
        memory: AriadneMemory instance.
        path: Input file path.
        dedup: Whether to check for duplicates.

    Returns:
        Dict with import statistics.
    """
    path = Path(path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    memories = data.get("memories", [])
    edges = data.get("edges", [])

    imported = 0
    skipped = 0
    errors = 0

    for mem in memories:
        try:
            content = mem.get("content", "")
            if not content:
                skipped += 1
                continue

            result = memory.remember(
                content=content,
                memory_type=mem.get("memory_type", "semantic"),
                category=mem.get("category", "semantic"),
                importance=mem.get("importance", 0.5),
                entities=mem.get("entities"),
                metadata=mem.get("metadata"),
                auto_embed=True,
            )

            if result.get("status") == "created":
                imported += 1
            elif result.get("status") == "duplicate":
                skipped += 1
            else:
                errors += 1
        except Exception as e:
            logger.warning("Failed to import memory: %s", e)
            errors += 1

    # Import edges
    edges_imported = 0
    for edge in edges:
        try:
            memory.add_edge(
                source=edge["source"],
                target=edge["target"],
                edge_type=edge.get("type", "related"),
                weight=edge.get("weight", 1.0),
            )
            edges_imported += 1
        except Exception as e:
            logger.warning("Failed to import edge: %s", e)

    logger.info(
        "Imported %d memories (%d skipped, %d errors, %d edges)",
        imported, skipped, errors, edges_imported,
    )
    return {
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
        "edges_imported": edges_imported,
    }


def import_from_chromadb(
    memory: Any,
    collection_path: str | Path,
    collection_name: Optional[str] = None,
    dedup: bool = True,
) -> Dict[str, Any]:
    """
    Import memories from a ChromaDB persistent collection.

    Args:
        memory: AriadneMemory instance.
        collection_path: Path to ChromaDB persistent directory.
        collection_name: Name of collection (None = first collection found).
        dedup: Whether to check for duplicates.

    Returns:
        Dict with import statistics.
    """
    try:
        import chromadb
    except ImportError:
        return {"error": "chromadb not installed. Install with: pip install chromadb"}

    try:
        client = chromadb.PersistentClient(path=str(collection_path))
        collections = client.list_collections()

        if not collections:
            return {"error": "No collections found in ChromaDB"}

        if collection_name:
            collection = client.get_collection(collection_name)
        else:
            collection = collections[0]

        results = collection.get(include=["documents", "metadatas"])

        imported = 0
        skipped = 0

        for doc, meta in zip(results["documents"], results["metadatas"]):
            if not doc:
                skipped += 1
                continue

            metadata = meta or {}
            result = memory.remember(
                content=doc,
                memory_type="semantic",
                importance=metadata.get("importance", 0.5),
                metadata=metadata,
                auto_embed=True,
            )

            if result.get("status") == "created":
                imported += 1
            else:
                skipped += 1

        return {"imported": imported, "skipped": skipped, "source": "chromadb"}

    except Exception as e:
        return {"error": str(e)}


def import_from_mem0(
    memory: Any,
    path: str | Path,
    dedup: bool = True,
) -> Dict[str, Any]:
    """
    Import memories from a Mem0 JSON export.

    Args:
        memory: AriadneMemory instance.
        path: Path to Mem0 export file.
        dedup: Whether to check for duplicates.

    Returns:
        Dict with import statistics.
    """
    path = Path(path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Mem0 format: list of memory objects with 'memory' field
    if isinstance(data, dict):
        memories_list = data.get("memories", data.get("results", []))
    elif isinstance(data, list):
        memories_list = data
    else:
        return {"error": "Unexpected Mem0 format"}

    imported = 0
    skipped = 0

    for item in memories_list:
        if isinstance(item, str):
            content = item
            meta = {}
        elif isinstance(item, dict):
            content = item.get("memory", item.get("text", item.get("content", "")))
            meta = {k: v for k, v in item.items() if k not in ("memory", "text", "content")}
        else:
            continue

        if not content:
            skipped += 1
            continue

        result = memory.remember(
            content=content,
            memory_type="semantic",
            importance=meta.get("importance", 0.5),
            metadata=meta,
            auto_embed=True,
        )

        if result.get("status") == "created":
            imported += 1
        else:
            skipped += 1

    return {"imported": imported, "skipped": skipped, "source": "mem0"}


def import_from_text(
    memory: Any,
    path: str | Path,
    category: str = "semantic",
    dedup: bool = True,
) -> Dict[str, Any]:
    """
    Import memories from a plain text file (one memory per paragraph).

    Args:
        memory: AriadneMemory instance.
        path: Path to text file.
        category: Memory category for imported memories.
        dedup: Whether to check for duplicates.

    Returns:
        Dict with import statistics.
    """
    path = Path(path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    content = path.read_text(encoding="utf-8")

    # Split by double newlines (paragraphs)
    paragraphs = re.split(r"\n\s*\n", content)
    paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 10]

    imported = 0
    skipped = 0

    for para in paragraphs:
        result = memory.remember(
            content=para,
            memory_type="semantic",
            category=category,
            importance=0.5,
            auto_embed=True,
        )

        if result.get("status") == "created":
            imported += 1
        else:
            skipped += 1

    return {"imported": imported, "skipped": skipped, "source": "text"}


def import_from_markdown(
    memory: Any,
    path: str | Path,
    dedup: bool = True,
) -> Dict[str, Any]:
    """
    Import memories from a markdown file with headers as entities.

    Format:
        # Entity Name
        - Memory content paragraph 1
        - Memory content paragraph 2

        ## Sub Entity
        - Another memory

    Args:
        memory: AriadneMemory instance.
        path: Path to markdown file.
        dedup: Whether to check for duplicates.

    Returns:
        Dict with import statistics.
    """
    path = Path(path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")

    imported = 0
    skipped = 0
    current_entity = None
    current_paragraph = []

    def flush_paragraph():
        nonlocal imported, skipped
        text = "\n".join(current_paragraph).strip()
        if text and len(text) > 10:
            entities = [current_entity] if current_entity else None
            result = memory.remember(
                content=text,
                memory_type="semantic",
                importance=0.6,
                entities=entities,
                auto_embed=True,
            )
            if result.get("status") == "created":
                imported += 1
            else:
                skipped += 1

    for line in lines:
        # Check for headers
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            # Flush any pending paragraph
            flush_paragraph()
            current_paragraph = []
            current_entity = header_match.group(2).strip()
            continue

        # Check for list items
        list_match = re.match(r"^[-*+]\s+(.+)$", line)
        if list_match:
            flush_paragraph()
            current_paragraph = [list_match.group(1)]
            continue

        # Empty line = paragraph separator
        if not line.strip():
            flush_paragraph()
            current_paragraph = []
            continue

        # Continuation of current paragraph
        current_paragraph.append(line)

    # Flush final paragraph
    flush_paragraph()

    return {"imported": imported, "skipped": skipped, "source": "markdown"}


def export_markdown(
    memory: Any,
    path: str | Path,
) -> Dict[str, Any]:
    """
    Export memories as human-readable markdown.

    Groups memories by entity, with headers for each entity.

    Args:
        memory: AriadneMemory instance.
        path: Output file path.

    Returns:
        Dict with export statistics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get all memories
    cursor = memory._db.conn.execute(
        """SELECT id, content, memory_type, importance, created_at, category
           FROM memories WHERE is_deleted = 0
           ORDER BY created_at DESC"""
    )
    memories = []
    entity_map = {}

    for row in cursor.fetchall():
        mem = {
            "id": row[0],
            "content": row[1],
            "memory_type": row[2],
            "importance": row[3],
            "created_at": row[4],
            "category": row[5] or "semantic",
        }
        memories.append(mem)

        # Get entities for this memory
        entity_cursor = memory._db.conn.execute(
            """SELECT e.name FROM entities e
               JOIN memory_entities me ON me.entity_id = e.id
               WHERE me.memory_id = ?""",
            (row[0],),
        )
        entities = [e[0] for e in entity_cursor.fetchall()]
        for entity in entities:
            if entity not in entity_map:
                entity_map[entity] = []
            entity_map[entity].append(mem)

    # Generate markdown
    lines = [
        "# Ariadne Memory Export",
        f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total memories: {len(memories)}",
        "",
    ]

    # Memories with entities, grouped by entity
    written_ids = set()
    for entity, entity_memories in sorted(entity_map.items()):
        lines.append(f"## {entity}")
        lines.append("")
        for mem in entity_memories:
            lines.append(f"- [{mem['category']}] {mem['content']}")
            written_ids.add(mem["id"])
        lines.append("")

    # Memories without entities
    orphans = [m for m in memories if m["id"] not in written_ids]
    if orphans:
        lines.append("## Uncategorized")
        lines.append("")
        for mem in orphans:
            lines.append(f"- [{mem['category']}] {mem['content']}")
        lines.append("")

    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")

    logger.info("Exported %d memories to markdown at %s", len(memories), path)
    return {"exported": len(memories), "entities": len(entity_map), "path": str(path)}
