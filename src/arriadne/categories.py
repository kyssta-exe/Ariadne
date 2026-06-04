"""
Memory Categories & Types

Supports four memory categories with different lifecycle behaviors:
- episodic: events/experiences — higher initial importance, decays faster
- semantic: facts/knowledge — stable importance, slow decay
- procedural: how-to knowledge — never decays (if used frequently)
- working: temporary context — fast decay, auto-pruned

Each category defines default importance, decay rate, and lifecycle parameters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("arriadne.categories")


@dataclass
class CategoryConfig:
    """Configuration for a memory category."""
    name: str
    default_importance: float = 0.5
    decay_rate: float = 0.01  # Importance loss per day without access
    min_importance: float = 0.0  # Floor importance
    max_importance: float = 1.0  # Ceiling importance
    access_boost: float = 0.05  # Importance gain per access
    auto_prune: bool = False  # Whether to auto-prune old memories
    prune_after_days: int = 7  # Days before auto-pruning (if auto_prune=True)
    description: str = ""

    def apply_decay(self, importance: float, days_since_access: float) -> float:
        """Apply time-based decay to importance."""
        new_importance = importance - (self.decay_rate * days_since_access)
        return max(self.min_importance, min(self.max_importance, new_importance))

    def apply_access_boost(self, importance: float, access_count: int) -> float:
        """Apply access-based importance boost (diminishing returns)."""
        import math
        boost = self.access_boost * math.log1p(access_count) / math.log1p(10)
        return min(self.max_importance, importance + boost)


# Default category configurations
DEFAULT_CATEGORIES: Dict[str, CategoryConfig] = {
    "episodic": CategoryConfig(
        name="episodic",
        default_importance=0.7,
        decay_rate=0.03,
        min_importance=0.05,
        access_boost=0.08,
        auto_prune=True,
        prune_after_days=90,
        description="Events, experiences, and temporal memories",
    ),
    "semantic": CategoryConfig(
        name="semantic",
        default_importance=0.5,
        decay_rate=0.005,
        min_importance=0.1,
        access_boost=0.03,
        auto_prune=False,
        description="Facts, knowledge, and general information",
    ),
    "procedural": CategoryConfig(
        name="procedural",
        default_importance=0.6,
        decay_rate=0.0,  # Never decays
        min_importance=0.2,
        access_boost=0.02,
        auto_prune=False,
        description="How-to knowledge, procedures, and skills",
    ),
    "working": CategoryConfig(
        name="working",
        default_importance=0.3,
        decay_rate=0.1,
        min_importance=0.0,
        access_boost=0.01,
        auto_prune=True,
        prune_after_days=7,
        description="Temporary context, session-specific information",
    ),
}


class MemoryCategoryManager:
    """
    Manages memory categories and their lifecycle behaviors.

    Provides:
    - Category validation and defaults
    - Category-aware importance calculations
    - Category statistics
    - Lifecycle adjustments based on category
    """

    def __init__(self, custom_categories: Optional[Dict[str, CategoryConfig]] = None):
        """Initialize with default or custom categories."""
        self._categories = dict(DEFAULT_CATEGORIES)
        if custom_categories:
            self._categories.update(custom_categories)

    def get_config(self, category: str) -> CategoryConfig:
        """Get configuration for a category, falling back to 'semantic'."""
        if category not in self._categories:
            logger.warning("Unknown category '%s', defaulting to 'semantic'", category)
            return self._categories["semantic"]
        return self._categories[category]

    def get_default_importance(self, category: str) -> float:
        """Get the default importance for a category."""
        return self.get_config(category).default_importance

    def apply_category_decay(
        self,
        importance: float,
        category: str,
        days_since_access: float,
    ) -> float:
        """Apply category-specific decay to a memory's importance."""
        config = self.get_config(category)
        return config.apply_decay(importance, days_since_access)

    def apply_category_access_boost(
        self,
        importance: float,
        category: str,
        access_count: int,
    ) -> float:
        """Apply category-specific access boost."""
        config = self.get_config(category)
        return config.apply_access_boost(importance, access_count)

    def should_auto_prune(self, category: str) -> bool:
        """Check if a category should be auto-pruned."""
        return self.get_config(category).auto_prune

    def get_prune_threshold_days(self, category: str) -> int:
        """Get auto-prune threshold in days for a category."""
        return self.get_config(category).prune_after_days

    def get_all_categories(self) -> List[str]:
        """Return list of all category names."""
        return list(self._categories.keys())

    def get_category_stats(self, db_conn: Any) -> Dict[str, Any]:
        """
        Get statistics for each category from the database.

        Returns:
            Dict mapping category names to their stats (count, avg_importance, etc.)
        """
        cursor = db_conn.cursor()

        # Get counts per category
        cursor.execute(
            """SELECT category, COUNT(*), AVG(importance), AVG(access_count),
                      MIN(created_at), MAX(created_at)
               FROM memories WHERE is_deleted = 0
               GROUP BY category"""
        )
        stats = {}
        for row in cursor.fetchall():
            cat = row[0] or "semantic"
            stats[cat] = {
                "count": row[1],
                "avg_importance": round(float(row[2] or 0), 4),
                "avg_access_count": round(float(row[3] or 0), 2),
                "oldest_created": row[4],
                "newest_created": row[5],
            }

        # Add categories with zero count
        for cat in self._categories:
            if cat not in stats:
                stats[cat] = {
                    "count": 0,
                    "avg_importance": 0,
                    "avg_access_count": 0,
                    "oldest_created": None,
                    "newest_created": None,
                }

        return stats

    def validate_category(self, category: str) -> str:
        """Validate and normalize a category name. Returns normalized name."""
        if not category:
            return "semantic"
        normalized = category.lower().strip()
        if normalized in self._categories:
            return normalized
        # Fuzzy match
        for cat in self._categories:
            if cat.startswith(normalized[:3]):
                return cat
        logger.warning("Unknown category '%s', defaulting to 'semantic'", category)
        return "semantic"
