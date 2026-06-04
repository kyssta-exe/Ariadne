"""
Community Detection for Knowledge Graph Memories

Graphiti/Zep-inspired community detection that clusters related memories
and entities into topical communities. Enables:
- Topic-aware search (search within a community)
- Community summaries (LLM-generated summary of a cluster)
- Topic-based forgetting (forget entire dead topics)
- Graph metrics (modularity, diameter, centrality)

Algorithm: Louvain-style greedy modularity optimization on the
entity co-occurrence graph, running directly in SQLite without
external graph libraries.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("arriadne.community")


@dataclass
class Community:
    """A detected community of related entities/memories."""
    id: int
    name: str  # Auto-generated from most central entity
    entity_ids: List[int] = field(default_factory=list)
    memory_ids: List[int] = field(default_factory=list)
    size: int = 0
    modularity: float = 0.0
    avg_internal_edge_weight: float = 0.0
    created_at: float = 0.0
    summary: str = ""  # LLM-generated summary
    is_dead: bool = False  # True if all memories expired

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_count": len(self.entity_ids),
            "memory_count": len(self.memory_ids),
            "size": self.size,
            "modularity": round(self.modularity, 4),
            "is_dead": self.is_dead,
            "summary": self.summary,
        }


@dataclass
class CommunityMetrics:
    """Graph-level community metrics."""
    num_communities: int = 0
    total_modularity: float = 0.0
    avg_community_size: float = 0.0
    largest_community: int = 0
    smallest_community: int = 0
    coverage: float = 0.0  # fraction of entities in any community
    avg_internal_density: float = 0.0


class CommunityDetector:
    """
    Detect communities in the entity co-occurrence graph.

    Uses a simplified Louvain-style algorithm:
    1. Build entity co-occurrence graph from memory-entity links
    2. Initialize each entity in its own community
    3. Greedily move entities to maximize modularity
    4. Repeat until convergence

    Community detection runs in SQL using the existing edge table,
    making it zero-dependency (no networkx/igraph needed).
    """

    def __init__(
        self,
        db_conn: Any,
        resolution: float = 1.0,
        min_community_size: int = 2,
        max_iterations: int = 10,
        min_modularity_gain: float = 0.001,
    ):
        self._conn = db_conn
        self._resolution = resolution
        self._min_community_size = min_community_size
        self._max_iterations = max_iterations
        self._min_modularity_gain = min_modularity_gain
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create community tables if they don't exist."""
        cursor = self._conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS communities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                modularity REAL DEFAULT 0.0,
                avg_internal_edge_weight REAL DEFAULT 0.0,
                is_dead INTEGER DEFAULT 0,
                summary TEXT DEFAULT '',
                created_at REAL DEFAULT (strftime('%s', 'now')),
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            );

            CREATE TABLE IF NOT EXISTS entity_communities (
                entity_id INTEGER NOT NULL,
                community_id INTEGER NOT NULL,
                membership_strength REAL DEFAULT 1.0,
                PRIMARY KEY (entity_id, community_id),
                FOREIGN KEY (entity_id) REFERENCES entities(id),
                FOREIGN KEY (community_id) REFERENCES communities(id)
            );

            CREATE TABLE IF NOT EXISTS memory_communities (
                memory_id INTEGER NOT NULL,
                community_id INTEGER NOT NULL,
                PRIMARY KEY (memory_id, community_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id),
                FOREIGN KEY (community_id) REFERENCES communities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_ec_community
                ON entity_communities(community_id);
            CREATE INDEX IF NOT EXISTS idx_mc_community
                ON memory_communities(community_id);
        """)
        self._conn.commit()

    def detect_communities(self, force: bool = False) -> List[Community]:
        """
        Run community detection on the entity graph.

        Args:
            force: If True, re-detect even if recently run.

        Returns:
            List of detected communities.
        """
        t0 = time.monotonic()

        # Build co-occurrence graph from edges
        edges, weights, node_degrees = self._build_cooccurrence_graph()
        if not edges:
            logger.info("No edges found for community detection")
            return []

        total_weight = sum(weights.values())
        if total_weight == 0:
            return []

        # Initialize: each node in its own community
        communities: Dict[int, int] = {n: n for n in node_degrees}
        community_nodes: Dict[int, Set[int]] = {n: {n} for n in node_degrees}

        # Louvain iterations
        best_modularity = 0.0
        for iteration in range(self._max_iterations):
            improved = False
            moved = 0

            for node in list(communities.keys()):
                if node not in communities:
                    continue

                current_comm = communities[node]
                best_comm = current_comm
                best_gain = 0.0

                # Get neighbors
                neighbors = self._get_neighbors(node, edges)
                neighbor_comms: Set[int] = set()
                for neighbor in neighbors:
                    if neighbor in communities:
                        neighbor_comms.add(communities[neighbor])

                for neighbor_comm in neighbor_comms:
                    if neighbor_comm == current_comm:
                        continue

                    gain = self._modularity_gain(
                        node, current_comm, neighbor_comm,
                        communities, community_nodes, edges, weights,
                        node_degrees, total_weight,
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_comm = neighbor_comm

                if best_comm != current_comm and best_gain > self._min_modularity_gain:
                    # Move node
                    community_nodes[current_comm].discard(node)
                    if not community_nodes[current_comm]:
                        del community_nodes[current_comm]
                    communities[node] = best_comm
                    community_nodes[best_comm].add(node)
                    improved = True
                    moved += 1

            # Compute modularity
            modularity = self._compute_modularity(
                communities, edges, weights, node_degrees, total_weight,
            )

            logger.debug(
                "Community iteration %d: modularity=%.4f, moved=%d",
                iteration, modularity, moved,
            )

            if not improved or (iteration > 0 and modularity - best_modularity < self._min_modularity_gain):
                break

            best_modularity = max(best_modularity, modularity)

        # Filter small communities and build result
        result = []
        comm_id = 0
        for comm_nodes in community_nodes.values():
            if len(comm_nodes) < self._min_community_size:
                continue

            # Find the most central entity (highest degree)
            central_entity_id = max(comm_nodes, key=lambda n: node_degrees.get(n, 0))

            # Get entity names
            entity_names = self._get_entity_names(list(comm_nodes))
            name = entity_names.get(central_entity_id, f"Community_{comm_id}")

            # Find memories linked to these entities
            memory_ids = self._find_community_memories(list(comm_nodes))

            # Compute internal density
            internal_edges = 0
            max_possible = len(comm_nodes) * (len(comm_nodes) - 1) / 2
            for u in comm_nodes:
                for v in comm_nodes:
                    if u < v and (u, v) in edges:
                        internal_edges += 1
            density = internal_edges / max_possible if max_possible > 0 else 0.0

            community = Community(
                id=comm_id,
                name=name,
                entity_ids=list(comm_nodes),
                memory_ids=memory_ids,
                size=len(comm_nodes),
                modularity=best_modularity,
                avg_internal_edge_weight=density,
                created_at=time.time(),
            )
            result.append(community)

            # Store in database
            self._store_community(community, comm_nodes)
            comm_id += 1

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "Detected %d communities in %.1fms (modularity=%.4f)",
            len(result), elapsed, best_modularity,
        )

        return result

    def get_community(self, community_id: int) -> Optional[Community]:
        """Get a community by ID."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT id, name, modularity, avg_internal_edge_weight, is_dead, summary, created_at "
            "FROM communities WHERE id = ?",
            (community_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        # Get entity IDs
        cursor.execute(
            "SELECT entity_id FROM entity_communities WHERE community_id = ?",
            (community_id,),
        )
        entity_ids = [r[0] for r in cursor.fetchall()]

        # Get memory IDs
        cursor.execute(
            "SELECT memory_id FROM memory_communities WHERE community_id = ?",
            (community_id,),
        )
        memory_ids = [r[0] for r in cursor.fetchall()]

        return Community(
            id=row[0], name=row[1], entity_ids=entity_ids, memory_ids=memory_ids,
            size=len(entity_ids), modularity=row[2], avg_internal_edge_weight=row[3],
            is_dead=bool(row[4]), summary=row[5] or "", created_at=row[6],
        )

    def get_communities(self, limit: int = 100) -> List[Community]:
        """Get all communities."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT id FROM communities ORDER BY modularity DESC LIMIT ?", (limit,))
        result: List[Community] = []
        for r in cursor.fetchall():
            c = self.get_community(r[0])
            if c is not None:
                result.append(c)
        return result

    def get_community_for_entity(self, entity_name: str) -> Optional[Community]:
        """Get the community an entity belongs to."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT id FROM entities WHERE name = ?", (entity_name,))
        row = cursor.fetchone()
        if not row:
            return None

        cursor.execute(
            "SELECT community_id FROM entity_communities WHERE entity_id = ?",
            (row[0],),
        )
        comm_row = cursor.fetchone()
        if not comm_row:
            return None

        return self.get_community(comm_row[0])

    def search_within_community(
        self, query: str, community_id: int, limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search memories within a specific community."""
        community = self.get_community(community_id)
        if not community or not community.memory_ids:
            return []

        placeholders = ",".join("?" * len(community.memory_ids))
        cursor = self._conn.cursor()
        cursor.execute(
            f"""SELECT id, content, memory_type, importance, created_at, metadata
                FROM memories
                WHERE id IN ({placeholders}) AND is_deleted = 0
                ORDER BY importance DESC LIMIT ?""",
            community.memory_ids + [limit],
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_community_summary(self, community_id: int) -> str:
        """Get or generate a summary of a community's content."""
        community = self.get_community(community_id)
        if not community:
            return ""

        # Fetch memory content for this community
        if not community.memory_ids:
            return ""

        placeholders = ",".join("?" * min(20, len(community.memory_ids)))
        cursor = self._conn.cursor()
        cursor.execute(
            f"""SELECT content FROM memories
                WHERE id IN ({placeholders}) AND is_deleted = 0
                ORDER BY importance DESC LIMIT 20""",
            community.memory_ids[:20],
        )
        contents = [r[0] for r in cursor.fetchall()]

        # Simple extractive summary (top sentences by information density)
        sentences = []
        for content in contents:
            for s in content.split(". "):
                s = s.strip()
                if len(s) > 10:
                    sentences.append(s)

        # Rank by information density (unique words / total words)
        scored = []
        seen_words: Set[str] = set()
        for s in sentences:
            words = set(s.lower().split())
            unique = words - seen_words
            density = len(unique) / max(1, len(words))
            scored.append((density, s))
            seen_words.update(words)

        scored.sort(reverse=True)
        summary = ". ".join(s for _, s in scored[:5]) + "."
        return summary

    def mark_dead_communities(self, threshold_days: int = 90) -> int:
        """Mark communities as dead if all their memories are old."""
        cutoff = time.time() - (threshold_days * 86400)
        cursor = self._conn.cursor()

        cursor.execute("""
            SELECT c.id FROM communities c
            WHERE c.is_dead = 0
            AND NOT EXISTS (
                SELECT 1 FROM memory_communities mc
                JOIN memories m ON m.id = mc.memory_id
                WHERE mc.community_id = c.id
                AND m.created_at > ?
                AND m.is_deleted = 0
            )
        """, (cutoff,))

        dead_ids = [r[0] for r in cursor.fetchall()]
        for cid in dead_ids:
            cursor.execute(
                "UPDATE communities SET is_dead = 1, updated_at = ? WHERE id = ?",
                (time.time(), cid),
            )

        self._conn.commit()
        if dead_ids:
            logger.info("Marked %d communities as dead", len(dead_ids))
        return len(dead_ids)

    def forget_dead_communities(self, min_age_days: int = 180) -> int:
        """Soft-delete memories in dead communities older than threshold."""
        cutoff = time.time() - (min_age_days * 86400)
        cursor = self._conn.cursor()

        cursor.execute("""
            SELECT mc.memory_id FROM memory_communities mc
            JOIN communities c ON c.id = mc.community_id
            JOIN memories m ON m.id = mc.memory_id
            WHERE c.is_dead = 1
            AND m.created_at < ?
            AND m.is_deleted = 0
        """, (cutoff,))

        memory_ids = [r[0] for r in cursor.fetchall()]
        if not memory_ids:
            return 0

        now = time.time()
        for mid in memory_ids:
            cursor.execute(
                "UPDATE memories SET is_deleted = 1, deleted_at = ? WHERE id = ?",
                (now, mid),
            )

        self._conn.commit()
        logger.info("Forgot %d memories from dead communities", len(memory_ids))
        return len(memory_ids)

    def metrics(self) -> CommunityMetrics:
        """Compute graph-level community metrics."""
        cursor = self._conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM communities")
        num_communities = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(modularity) FROM communities")
        avg_mod = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT COUNT(*) FROM entity_communities")
        total_in_communities = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM entities")
        total_entities = cursor.fetchone()[0]

        cursor.execute(
            "SELECT community_id, COUNT(*) as size FROM entity_communities "
            "GROUP BY community_id ORDER BY size"
        )
        sizes = [r[1] for r in cursor.fetchall()]

        return CommunityMetrics(
            num_communities=num_communities,
            total_modularity=avg_mod,
            avg_community_size=sum(sizes) / len(sizes) if sizes else 0.0,
            largest_community=max(sizes) if sizes else 0,
            smallest_community=min(sizes) if sizes else 0,
            coverage=total_in_communities / total_entities if total_entities > 0 else 0.0,
        )

    # === Internal Methods ===

    def _build_cooccurrence_graph(self) -> Tuple[Dict, Dict, Dict]:
        """Build entity co-occurrence graph from memory-entity links."""
        cursor = self._conn.cursor()

        # Get memory-entity pairs
        cursor.execute("""
            SELECT me1.entity_id, me2.entity_id, COUNT(*) as co_count
            FROM memory_entities me1
            JOIN memory_entities me2 ON me1.memory_id = me2.memory_id
            AND me1.entity_id < me2.entity_id
            GROUP BY me1.entity_id, me2.entity_id
        """)

        edges: Dict[Tuple[int, int], float] = {}
        node_degrees: Dict[int, float] = defaultdict(float)

        for row in cursor.fetchall():
            u, v, weight = row[0], row[1], row[2]
            edges[(u, v)] = weight
            node_degrees[u] += weight
            node_degrees[v] += weight

        # Also add graph edges
        cursor.execute("""
            SELECT source_id, target_id, weight FROM edges
        """)
        for row in cursor.fetchall():
            u, v, weight = row[0], row[1], row[2]
            key = (min(u, v), max(u, v))
            edges[key] = edges.get(key, 0) + weight
            node_degrees[u] += weight
            node_degrees[v] += weight

        return edges, dict(edges), dict(node_degrees)

    def _get_neighbors(self, node: int, edges: Dict) -> List[int]:
        """Get neighbors of a node in the graph."""
        neighbors = []
        for (u, v) in edges:
            if u == node:
                neighbors.append(v)
            elif v == node:
                neighbors.append(u)
        return neighbors

    def _modularity_gain(
        self,
        node: int,
        current_comm: int,
        target_comm: int,
        communities: Dict[int, int],
        community_nodes: Dict[int, Set[int]],
        edges: Dict[Tuple[int, int], float],
        weights: Dict[Tuple[int, int], float],
        node_degrees: Dict[int, float],
        total_weight: float,
    ) -> float:
        """Compute modularity gain from moving node to target community."""
        if total_weight == 0:
            return 0.0

        # Sum of weights from node to target community
        ki_in_target = 0.0
        ki_in_current = 0.0
        ki_total = node_degrees.get(node, 0)

        for neighbor in community_nodes.get(target_comm, set()):
            key = (min(node, neighbor), max(node, neighbor))
            ki_in_target += weights.get(key, 0)

        for neighbor in community_nodes.get(current_comm, set()):
            if neighbor == node:
                continue
            key = (min(node, neighbor), max(node, neighbor))
            ki_in_current += weights.get(key, 0)

        # Sum of degrees in target and current communities
        sigma_target = sum(node_degrees.get(n, 0) for n in community_nodes.get(target_comm, set()))
        sigma_current = sum(node_degrees.get(n, 0) for n in community_nodes.get(current_comm, set()))

        # Modularity gain formula (Louvain)
        gain = (
            (ki_in_target - ki_in_current) / total_weight
            - self._resolution * ki_total * (sigma_target - sigma_current + ki_total) / (2 * total_weight * total_weight)
        )

        return gain

    def _compute_modularity(
        self,
        communities: Dict[int, int],
        edges: Dict[Tuple[int, int], float],
        weights: Dict[Tuple[int, int], float],
        node_degrees: Dict[int, float],
        total_weight: float,
    ) -> float:
        """Compute modularity of the current partition."""
        if total_weight == 0:
            return 0.0

        Q = 0.0
        for (u, v), w in edges.items():
            cu = communities.get(u)
            cv = communities.get(v)
            if cu is not None and cu == cv:
                du = node_degrees.get(u, 0)
                dv = node_degrees.get(v, 0)
                Q += w - (du * dv) / (2 * total_weight)

        return Q / (2 * total_weight) if total_weight > 0 else 0.0

    def _get_entity_names(self, entity_ids: List[int]) -> Dict[int, str]:
        """Get entity names for a list of IDs."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" * len(entity_ids))
        cursor = self._conn.cursor()
        cursor.execute(
            f"SELECT id, name FROM entities WHERE id IN ({placeholders})",
            entity_ids,
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def _find_community_memories(self, entity_ids: List[int]) -> List[int]:
        """Find all memories linked to entities in a community."""
        if not entity_ids:
            return []
        placeholders = ",".join("?" * len(entity_ids))
        cursor = self._conn.cursor()
        cursor.execute(
            f"""SELECT DISTINCT memory_id FROM memory_entities
                WHERE entity_id IN ({placeholders})""",
            entity_ids,
        )
        return [row[0] for row in cursor.fetchall()]

    def _store_community(
        self, community: Community, entity_ids: Set[int],
    ) -> None:
        """Store a community and its memberships in the database."""
        cursor = self._conn.cursor()

        # Insert or replace community
        cursor.execute("""
            INSERT OR REPLACE INTO communities
            (id, name, modularity, avg_internal_edge_weight, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            community.id, community.name, community.modularity,
            community.avg_internal_edge_weight, community.created_at, time.time(),
        ))

        # Clear old memberships
        cursor.execute("DELETE FROM entity_communities WHERE community_id = ?", (community.id,))
        cursor.execute("DELETE FROM memory_communities WHERE community_id = ?", (community.id,))

        # Store entity memberships
        for eid in entity_ids:
            cursor.execute(
                "INSERT OR IGNORE INTO entity_communities (entity_id, community_id) VALUES (?, ?)",
                (eid, community.id),
            )

        # Store memory memberships
        for mid in community.memory_ids:
            cursor.execute(
                "INSERT OR IGNORE INTO memory_communities (memory_id, community_id) VALUES (?, ?)",
                (mid, community.id),
            )

        self._conn.commit()
