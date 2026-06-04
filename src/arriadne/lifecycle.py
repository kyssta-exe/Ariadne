"""
Three-Tier Memory Lifecycle

Inspired by MemGPT/Letta's memory architecture:
- Tier 1 (Hot): Core memories — always in context, high priority
- Tier 2 (Warm): Active memories — searchable, recent, medium priority
- Tier 3 (Cold): Archived memories — older, lower priority, still searchable

Automatic lifecycle management:
- Hot memories age into Warm based on access patterns
- Warm memories age into Cold based on Ebbinghaus forgetting curve
- Cold memories can be permanently pruned if never accessed
- Promotions happen on access (cold -> warm -> hot)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("arriadne.lifecycle")


# Default lifecycle thresholds (in seconds)
DEFAULT_HOT_THRESHOLD = 7 * 24 * 3600  # 7 days — memories younger than this are "hot"
DEFAULT_WARM_THRESHOLD = 30 * 24 * 3600  # 30 days — memories younger than this are "warm"
DEFAULT_COLD_THRESHOLD = 90 * 24 * 3600  # 90 days — memories older than this are candidates for pruning

# Access-based promotion thresholds
DEFAULT_PROMOTION_ACCESS_COUNT = 5  # Access this many times to promote from cold
DEFAULT_PROMOTION_RECENCY = 24 * 3600  # Accessed within this window to promote


@dataclass
class LifecycleStats:
    """Statistics about the memory lifecycle."""

    hot_count: int = 0
    warm_count: int = 0
    cold_count: int = 0
    total_count: int = 0
    avg_access_count: float = 0.0
    memories_due_for_demotion: int = 0
    memories_due_for_pruning: int = 0


class MemoryLifecycle:
    """
    Manages the lifecycle of memories across hot/warm/cold tiers.

    Memories flow through tiers based on:
    - Age (time since creation)
    - Access frequency (how often they're retrieved)
    - Retention score (Ebbinghaus forgetting curve)
    - Importance (manual or LLM-assigned priority)
    """

    def __init__(
        self,
        db_conn: Any,
        hot_threshold: float = DEFAULT_HOT_THRESHOLD,
        warm_threshold: float = DEFAULT_WARM_THRESHOLD,
        cold_threshold: float = DEFAULT_COLD_THRESHOLD,
        promotion_access_count: int = DEFAULT_PROMOTION_ACCESS_COUNT,
        promotion_recency: float = DEFAULT_PROMOTION_RECENCY,
        importance_weight: float = 0.3,
    ):
        self._conn = db_conn
        self._hot_threshold = hot_threshold
        self._warm_threshold = warm_threshold
        self._cold_threshold = cold_threshold
        self._promotion_access_count = promotion_access_count
        self._promotion_recency = promotion_recency
        self._importance_weight = importance_weight

    def get_tier(self, memory: Dict[str, Any]) -> str:
        """Determine the current tier for a memory."""
        now = time.time()
        created = memory.get("created_at", now)
        age = now - created

        if age < self._hot_threshold:
            return "hot"
        elif age < self._warm_threshold:
            return "warm"
        else:
            return "cold"

    def record_access(self, memory_id: str) -> None:
        """Record that a memory was accessed (for promotion logic)."""
        cursor = self._conn.cursor()
        cursor.execute(
            """UPDATE memories
            SET access_count = COALESCE(access_count, 0) + 1,
                last_accessed_at = ?
            WHERE id = ?""",
            (time.time(), memory_id),
        )
        self._conn.commit()

    def check_and_promote(self, memory_id: str) -> Optional[str]:
        """
        Check if a memory should be promoted to a higher tier.

        Returns the new tier if promoted, None otherwise.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        memory = self._row_to_dict(row)
        current_tier = self.get_tier(memory)
        access_count = memory.get("access_count", 0) or 0
        last_accessed = memory.get("last_accessed_at") or 0
        now = time.time()

        new_tier = current_tier

        # Promotion: cold -> warm (if recently accessed enough)
        if current_tier == "cold":
            if (
                access_count >= self._promotion_access_count
                and (now - last_accessed) < self._promotion_recency
            ):
                new_tier = "warm"
                logger.info(f"Promoted memory {memory_id}: cold -> warm")

        # Promotion: warm -> hot (if very frequently accessed)
        if current_tier == "warm":
            if (
                access_count >= self._promotion_access_count * 2
                and (now - last_accessed) < self._promotion_recency // 2
            ):
                new_tier = "hot"
                logger.info(f"Promoted memory {memory_id}: warm -> hot")

        return new_tier if new_tier != current_tier else None

    def run_lifecycle(self) -> Dict[str, Any]:
        """
        Run the full lifecycle process:
        1. Demote hot memories that are old enough
        2. Demote warm memories that are old enough
        3. Identify cold memories due for pruning
        4. Promote recently-accessed cold/warm memories
        """
        t0 = time.monotonic()
        now = time.time()

        cursor = self._conn.cursor()

        # Step 1: Demote hot -> warm (older than hot_threshold)
        cursor.execute(
            """UPDATE memories SET metadata = json_set(
                COALESCE(metadata, '{}'), '$.tier', 'warm'
            ) WHERE deleted_at IS NULL
            AND (metadata IS NULL OR json_extract(metadata, '$.tier') IS NULL OR json_extract(metadata, '$.tier') = 'hot')
            AND created_at < ?""",
            (now - self._hot_threshold,),
        )
        demoted_to_warm = cursor.rowcount

        # Step 2: Demote warm -> cold (older than warm_threshold)
        cursor.execute(
            """UPDATE memories SET metadata = json_set(
                COALESCE(metadata, '{}'), '$.tier', 'cold'
            ) WHERE deleted_at IS NULL
            AND json_extract(metadata, '$.tier') = 'warm'
            AND created_at < ?""",
            (now - self._warm_threshold,),
        )
        demoted_to_cold = cursor.rowcount

        # Step 3: Count tiers
        cursor.execute(
            """SELECT
                SUM(CASE WHEN json_extract(metadata, '$.tier') = 'hot'
                    OR (json_extract(metadata, '$.tier') IS NULL AND created_at >= ?) THEN 1 ELSE 0 END),
                SUM(CASE WHEN json_extract(metadata, '$.tier') = 'warm'
                    OR (json_extract(metadata, '$.tier') IS NULL AND created_at >= ? AND created_at < ?) THEN 1 ELSE 0 END),
                SUM(CASE WHEN json_extract(metadata, '$.tier') = 'cold'
                    OR (json_extract(metadata, '$.tier') IS NULL AND created_at < ?) THEN 1 ELSE 0 END),
                COUNT(*)
            FROM memories WHERE deleted_at IS NULL""",
            (
                now - self._hot_threshold,
                now - self._warm_threshold,
                now - self._hot_threshold,
                now - self._warm_threshold,
            ),
        )
        tier_counts = cursor.fetchone()
        hot_count = tier_counts[0] or 0
        warm_count = tier_counts[1] or 0
        cold_count = tier_counts[2] or 0
        total = tier_counts[3] or 0

        # Step 4: Memories due for pruning (cold + old + never accessed)
        cold_prune_age = now - self._cold_threshold
        cursor.execute(
            """SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL
            AND created_at < ?
            AND (access_count IS NULL OR access_count = 0)""",
            (cold_prune_age,),
        )
        due_for_pruning = cursor.fetchone()[0]

        # Step 5: Calculate Ebbinghaus retention for each tier
        avg_retention_hot = self._avg_retention_in_tier("hot")
        avg_retention_warm = self._avg_retention_in_tier("warm")
        avg_retention_cold = self._avg_retention_in_tier("cold")

        self._conn.commit()

        latency = (time.monotonic() - t0) * 1000

        stats = LifecycleStats(
            hot_count=hot_count,
            warm_count=warm_count,
            cold_count=cold_count,
            total_count=total,
            memories_due_for_demotion=demoted_to_warm + demoted_to_cold,
            memories_due_for_pruning=due_for_pruning,
        )

        return {
            "stats": stats,
            "demoted_to_warm": demoted_to_warm,
            "demoted_to_cold": demoted_to_cold,
            "due_for_pruning": due_for_pruning,
            "avg_retention": {
                "hot": round(avg_retention_hot, 3),
                "warm": round(avg_retention_warm, 3),
                "cold": round(avg_retention_cold, 3),
            },
            "latency_ms": round(latency, 1),
        }

    def get_retention_score(self, memory: Dict[str, Any]) -> float:
        """
        Calculate Ebbinghaus forgetting curve retention score.

        Returns 0.0 (forgotten) to 1.0 (perfectly retained).
        """
        access_count = memory.get("access_count", 0) or 0
        importance = memory.get("importance", 5) or 5
        created = memory.get("created_at", time.time())

        # Stability factor: grows with each access (10 accesses = 40x more stable)
        stability = 1.0 + (access_count * 4.0)

        # Time since creation in hours
        age_hours = (time.time() - created) / 3600.0
        if age_hours <= 0:
            return 1.0

        # Ebbinghaus: R = e^(-t/S)
        retention = math.exp(-age_hours / stability)

        # Importance boost: higher importance memories decay slower
        importance_factor = 1.0 + (importance - 5) * self._importance_weight * 0.1
        retention *= importance_factor

        return max(0.0, min(1.0, retention))

    def get_priority_score(self, memory: Dict[str, Any]) -> float:
        """
        Combined priority score combining recency, importance, and retention.

        Inspired by Generative Agents' weighted retrieval.
        """
        now = time.time()
        created = memory.get("created_at", now)
        access_count = memory.get("access_count", 0) or 0
        importance = memory.get("importance", 5) or 5

        # Recency score (exponential decay, half-life 7 days)
        age_days = (now - created) / 86400.0
        recency = math.exp(-0.693 * age_days / 7.0)  # 0.693 = ln(2)

        # Importance score (normalized to 0-1)
        importance_score = importance / 10.0

        # Access score (logarithmic, saturates at 20)
        access_score = math.log1p(access_count) / math.log1p(20)

        # Weighted combination
        priority = (
            0.4 * recency
            + 0.3 * importance_score
            + 0.2 * access_score
            + 0.1 * self.get_retention_score(memory)
        )

        return round(priority, 4)

    def prune_cold_memories(
        self,
        min_age_days: int = 90,
        min_retention: float = 0.01,
        limit: int = 100,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Identify and optionally prune cold, forgotten memories.

        Only prunes memories that:
        1. Are older than min_age_days
        2. Have retention score below min_retention
        3. Have never been accessed (or very rarely)
        """
        now = time.time()
        cutoff = now - (min_age_days * 86400)

        cursor = self._conn.cursor()
        cursor.execute(
            """SELECT id, content, created_at, access_count, importance
            FROM memories WHERE deleted_at IS NULL AND created_at < ?
            ORDER BY created_at ASC LIMIT ?""",
            (cutoff, limit),
        )
        rows = cursor.fetchall()

        candidates = []
        for row in rows:
            memory = {
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "access_count": row[3] or 0,
                "importance": row[4] or 5,
            }
            retention = self.get_retention_score(memory)
            if retention < min_retention:
                candidates.append({
                    "id": memory["id"],
                    "content": memory["content"][:50],
                    "retention": round(retention, 4),
                    "age_days": round((now - memory["created_at"]) / 86400),
                    "access_count": memory["access_count"],
                })

        pruned = 0
        if not dry_run and candidates:
            for c in candidates:
                cursor.execute(
                    "UPDATE memories SET deleted_at = ? WHERE id = ?",
                    (now, c["id"]),
                )
            self._conn.commit()
            pruned = len(candidates)

        return {
            "candidates_found": len(candidates),
            "pruned": pruned,
            "dry_run": dry_run,
            "candidates": candidates[:20],  # Return first 20 for review
        }

    def _avg_retention_in_tier(self, tier: str) -> float:
        """Calculate average retention score for memories in a tier."""
        now = time.time()
        if tier == "hot":
            cutoff = now - self._hot_threshold
        elif tier == "warm":
            cutoff = now - self._warm_threshold
        else:
            cutoff = now - self._cold_threshold

        cursor = self._conn.cursor()
        cursor.execute(
            """SELECT created_at, access_count, importance
            FROM memories WHERE deleted_at IS NULL AND created_at >= ?""",
            (cutoff,),
        )
        rows = cursor.fetchall()

        if not rows:
            return 0.0

        total = 0.0
        for row in rows:
            memory = {
                "created_at": row[0],
                "access_count": row[1] or 0,
                "importance": row[2] or 5,
            }
            total += self.get_retention_score(memory)

        return total / len(rows)

    def _row_to_dict(self, row: tuple) -> Dict[str, Any]:
        """Convert a database row to a dict."""
        return {
            "id": row[0],
            "content": row[1] if len(row) > 1 else "",
            "created_at": row[2] if len(row) > 2 else time.time(),
            "access_count": row[3] if len(row) > 3 else 0,
            "importance": row[4] if len(row) > 4 else 5,
            "metadata": row[5] if len(row) > 5 else None,
        }
