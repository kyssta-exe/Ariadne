"""
Graph Visualization

Provides multiple export formats for visualizing the knowledge graph:
- DOT/Graphviz format
- Mermaid diagrams
- D3.js-compatible JSON
- Graph statistics (centrality, connected components, degree distribution)
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("arriadne.visualization")


def export_dot(
    memory: Any,
    path: str | Path,
    directed: bool = True,
) -> Dict[str, Any]:
    """
    Export the knowledge graph as a DOT/Graphviz file.

    Args:
        memory: AriadneMemory instance.
        path: Output file path.
        directed: Whether to create a directed graph.

    Returns:
        Dict with export statistics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = memory._db.conn

    # Get all entities
    cursor = conn.execute(
        "SELECT id, name, entity_type FROM entities ORDER BY name"
    )
    entities = {row[0]: {"name": row[1], "type": row[2]} for row in cursor.fetchall()}

    # Get all edges
    cursor = conn.execute(
        """SELECT e.source_id, e.target_id, e.edge_type, e.weight
           FROM edges e"""
    )
    edges = cursor.fetchall()

    # Get entity -> memory count for sizing
    cursor = conn.execute(
        """SELECT me.entity_id, COUNT(*) as cnt
           FROM memory_entities me
           JOIN memories m ON m.id = me.memory_id
           WHERE m.is_deleted = 0
           GROUP BY me.entity_id"""
    )
    entity_memory_counts = {row[0]: row[1] for row in cursor.fetchall()}

    # Generate DOT
    graph_type = "digraph" if directed else "graph"
    arrow = " -> " if directed else " -- "
    lines = [
        f"{graph_type} AriadneGraph {{",
        '    rankdir=LR;',
        '    node [shape=ellipse, style=filled, fillcolor=lightblue];',
        '    edge [color=gray50];',
        "",
    ]

    # Add nodes
    for eid, entity in entities.items():
        count = entity_memory_counts.get(eid, 0)
        label = entity["name"].replace('"', '\\"')
        # Scale font size by memory count
        fontsize = max(10, min(24, 10 + count))
        lines.append(
            f'    "{label}" [label="{label}\\n({count} memories)", '
            f'fontsize={fontsize}];'
        )

    lines.append("")

    # Add edges
    for source_id, target_id, edge_type, weight in edges:
        if source_id in entities and target_id in entities:
            src = entities[source_id]["name"].replace('"', '\\"')
            tgt = entities[target_id]["name"].replace('"', '\\"')
            penwidth = max(0.5, min(4.0, weight * 2))
            lines.append(
                f'    "{src}"{arrow}"{tgt}" '
                f'[label="{edge_type}", penwidth={penwidth:.1f}];'
            )

    lines.append("}")

    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")

    stats = {
        "nodes": len(entities),
        "edges": len(edges),
        "path": str(path),
    }
    logger.info("Exported DOT graph: %d nodes, %d edges", stats["nodes"], stats["edges"])
    return stats


def export_mermaid(
    memory: Any,
    path: str | Path,
) -> Dict[str, Any]:
    """
    Export the knowledge graph as a Mermaid diagram.

    Args:
        memory: AriadneMemory instance.
        path: Output file path.

    Returns:
        Dict with export statistics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = memory._db.conn

    # Get all entities
    cursor = conn.execute(
        "SELECT id, name FROM entities ORDER BY name"
    )
    entities = {row[0]: row[1] for row in cursor.fetchall()}

    # Get all edges
    cursor = conn.execute(
        """SELECT e.source_id, e.target_id, e.edge_type
           FROM edges e"""
    )
    edges = cursor.fetchall()

    # Generate Mermaid
    lines = ["graph LR"]

    # Create a node ID map (Mermaid needs safe IDs)
    node_ids = {}
    for i, (eid, name) in enumerate(entities.items()):
        safe_id = f"N{i}"
        safe_name = name.replace('"', "'").replace("[", "(").replace("]", ")")
        node_ids[eid] = safe_id
        lines.append(f"    {safe_id}[\"{safe_name}\"]")

    lines.append("")

    # Add edges
    edge_labels = defaultdict(int)
    for source_id, target_id, edge_type in edges:
        if source_id in node_ids and target_id in node_ids:
            src = node_ids[source_id]
            tgt = node_ids[target_id]
            edge_key = (src, tgt)
            edge_labels[edge_key] += 1
            if edge_labels[edge_key] == 1:
                lines.append(f"    {src} -->|{edge_type}| {tgt}")

    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")

    stats = {
        "nodes": len(entities),
        "edges": len(edges),
        "path": str(path),
    }
    logger.info("Exported Mermaid graph: %d nodes, %d edges", stats["nodes"], stats["edges"])
    return stats


def export_json_graph(
    memory: Any,
    path: str | Path,
) -> Dict[str, Any]:
    """
    Export the knowledge graph as D3.js-compatible JSON.

    Format: { "nodes": [...], "links": [...] }

    Args:
        memory: AriadneMemory instance.
        path: Output file path.

    Returns:
        Dict with export statistics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = memory._db.conn

    # Get entities
    cursor = conn.execute(
        "SELECT id, name, entity_type FROM entities ORDER BY name"
    )
    entities = []
    entity_id_to_idx = {}
    for i, row in enumerate(cursor.fetchall()):
        entity_id_to_idx[row[0]] = i
        entities.append({
            "id": i,
            "name": row[1],
            "type": row[2],
        })

    # Get memory counts per entity
    cursor = conn.execute(
        """SELECT me.entity_id, COUNT(*) as cnt
           FROM memory_entities me
           JOIN memories m ON m.id = me.memory_id
           WHERE m.is_deleted = 0
           GROUP BY me.entity_id"""
    )
    for row in cursor.fetchall():
        if row[0] in entity_id_to_idx:
            entities[entity_id_to_idx[row[0]]]["memory_count"] = row[1]

    # Get edges
    cursor = conn.execute(
        """SELECT e.source_id, e.target_id, e.edge_type, e.weight
           FROM edges e"""
    )
    links = []
    for source_id, target_id, edge_type, weight in cursor.fetchall():
        if source_id in entity_id_to_idx and target_id in entity_id_to_idx:
            links.append({
                "source": entity_id_to_idx[source_id],
                "target": entity_id_to_idx[target_id],
                "type": edge_type,
                "weight": weight,
            })

    graph_data = {
        "nodes": entities,
        "links": links,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    stats = {
        "nodes": len(entities),
        "links": len(links),
        "path": str(path),
    }
    logger.info("Exported JSON graph: %d nodes, %d links", stats["nodes"], stats["links"])
    return stats


def get_graph_stats(memory: Any) -> Dict[str, Any]:
    """
    Compute comprehensive graph statistics.

    Returns:
        Dict with nodes, edges, degree distribution, connected components,
        centrality metrics, and density.
    """
    conn = memory._db.conn

    # Node count
    cursor = conn.execute("SELECT COUNT(*) FROM entities")
    num_nodes = cursor.fetchone()[0]

    # Edge count
    cursor = conn.execute("SELECT COUNT(*) FROM edges")
    num_edges = cursor.fetchone()[0]

    # Memory count
    cursor = conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
    num_memories = cursor.fetchone()[0]

    if num_nodes == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "memories": num_memories,
            "density": 0,
            "degree_distribution": {},
            "connected_components": 0,
            "largest_component_size": 0,
            "avg_degree": 0,
            "max_degree": 0,
            "centrality": {},
        }

    # Get all edges as adjacency list
    cursor = conn.execute(
        "SELECT source_id, target_id FROM edges"
    )
    adj = defaultdict(set)
    for source_id, target_id in cursor.fetchall():
        adj[source_id].add(target_id)
        adj[target_id].add(source_id)

    # Degree distribution
    degrees = Counter()
    for node_id in range(1, num_nodes + 1):
        deg = len(adj.get(node_id, set()))
        degrees[deg] += 1

    # Connected components (BFS)
    visited = set()
    components = []
    for start in range(1, num_nodes + 1):
        if start in visited:
            continue
        component = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in component:
                continue
            component.add(node)
            visited.add(node)
            for neighbor in adj.get(node, set()):
                if neighbor not in component:
                    queue.append(neighbor)
        components.append(component)

    largest_component = max(components, key=len) if components else set()

    # Approximate centrality (degree centrality)
    max_degree = max((len(adj.get(n, set())) for n in range(1, num_nodes + 1)), default=0)
    centrality = {}
    for node_id in range(1, num_nodes + 1):
        deg = len(adj.get(node_id, set()))
        centrality[node_id] = deg / max(max_degree, 1)

    # Top 5 central nodes
    cursor = conn.execute("SELECT id, name FROM entities")
    id_to_name = {row[0]: row[1] for row in cursor.fetchall()}
    top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    top_central_named = [
        {"name": id_to_name.get(nid, str(nid)), "centrality": round(c, 4)}
        for nid, c in top_central
    ]

    # Graph density (actual edges / possible edges)
    max_possible = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 1
    density = num_edges / max_possible if max_possible > 0 else 0

    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

    return {
        "nodes": num_nodes,
        "edges": num_edges,
        "memories": num_memories,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 2),
        "max_degree": max_degree,
        "connected_components": len(components),
        "largest_component_size": len(largest_component),
        "degree_distribution": {str(k): v for k, v in sorted(degrees.items())},
        "centrality_top5": top_central_named,
    }
