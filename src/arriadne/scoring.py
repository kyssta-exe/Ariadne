"""
Advanced Memory Importance Scoring

Multi-factor memory importance scoring inspired by modern memory systems:

1. **Information Density** — Unique information vs redundant content
2. **Recency (Ebbinghaus)** — Time-based decay with stability theory
3. **Access Frequency** — How often the memory is recalled
4. **Entity Centrality** — How central the memory's entities are in the graph
5. **LLM Importance** — When available, LLM-assigned importance rating
6. **Contradiction Freshness** — Newer contradictory memories score higher

The composite score combines these factors with configurable weights.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


logger = logging.getLogger("arriadne.scoring")


@dataclass
class ImportanceScore:
    """Breakdown of a memory's importance score."""
    composite: float = 0.0  # Final combined score (0.0 - 1.0)
    information_density: float = 0.0  # Unique info content
    recency: float = 0.0  # Time-based decay
    access_frequency: float = 0.0  # How often accessed
    entity_centrality: float = 0.0  # Graph centrality of entities
    llm_importance: float = 0.0  # LLM-assigned rating
    contradiction_freshness: float = 0.0  # Freshness of contradictory info

    def to_dict(self) -> Dict[str, float]:
        return {
            "composite": round(self.composite, 4),
            "information_density": round(self.information_density, 4),
            "recency": round(self.recency, 4),
            "access_frequency": round(self.access_frequency, 4),
            "entity_centrality": round(self.entity_centrality, 4),
            "llm_importance": round(self.llm_importance, 4),
            "contradiction_freshness": round(self.contradiction_freshness, 4),
        }


@dataclass
class ScoringConfig:
    """Configuration for importance scoring weights."""
    weight_information_density: float = 0.25
    weight_recency: float = 0.25
    weight_access_frequency: float = 0.20
    weight_entity_centrality: float = 0.15
    weight_llm_importance: float = 0.10
    weight_contradiction_freshness: float = 0.05

    # Ebbinghaus parameters
    half_life_hours: float = 24.0  # Default 24-hour half-life
    stability_factor: float = 4.0  # How much each access extends memory

    # Information density
    min_unique_ratio: float = 0.1  # Minimum unique word ratio
    stop_words: Set[str] = field(default_factory=lambda: {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "don", "now", "and", "but", "or", "if", "while", "that", "this",
        "it", "its", "i", "me", "my", "we", "our", "you", "your", "he",
        "him", "his", "she", "her", "they", "them", "their", "what", "which",
        "who", "whom",
    })


class MemoryImportanceScorer:
    """
    Compute multi-factor importance scores for memories.

    The scorer analyzes each memory across multiple dimensions and produces
    a composite importance score that can be used for:
    - Ranking search results
    - Deciding which memories to evict
    - Prioritizing which memories to consolidate
    - Determining which memories to surface in context windows
    """

    def __init__(
        self,
        db_conn: Any,
        embedding_provider: Any = None,
        config: Optional[ScoringConfig] = None,
    ):
        self._conn = db_conn
        self._embedder = embedding_provider
        self._config = config or ScoringConfig()
        self._entity_centrality_cache: Dict[str, float] = {}
        self._stop_words = self._config.stop_words

    def score_memory(self, memory_id: int) -> ImportanceScore:
        """
        Compute the full importance score for a single memory.

        Args:
            memory_id: The memory's database ID.

        Returns:
            ImportanceScore with all component scores.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """SELECT id, content, memory_type, importance, created_at,
                      updated_at, accessed_at, access_count, metadata
               FROM memories WHERE id = ? AND is_deleted = 0""",
            (memory_id,),
        )
        row = cursor.fetchone()
        if not row:
            return ImportanceScore()

        memory = {
            "id": row[0], "content": row[1], "memory_type": row[2],
            "importance": row[3], "created_at": row[4], "updated_at": row[5],
            "accessed_at": row[6], "access_count": row[7] or 0,
            "metadata": row[8],
        }

        return self._score_from_dict(memory)

    def score_memories(self, memory_ids: List[int]) -> Dict[int, ImportanceScore]:
        """Score multiple memories at once (batch optimized)."""
        if not memory_ids:
            return {}

        placeholders = ",".join("?" * len(memory_ids))
        cursor = self._conn.cursor()
        cursor.execute(
            f"""SELECT id, content, memory_type, importance, created_at,
                       updated_at, accessed_at, access_count, metadata
                FROM memories WHERE id IN ({placeholders}) AND is_deleted = 0""",
            memory_ids,
        )

        results = {}
        for row in cursor.fetchall():
            memory = {
                "id": row[0], "content": row[1], "memory_type": row[2],
                "importance": row[3], "created_at": row[4], "updated_at": row[5],
                "accessed_at": row[6], "access_count": row[7] or 0,
                "metadata": row[8],
            }
            results[row[0]] = self._score_from_dict(memory)

        return results

    def _score_from_dict(self, memory: Dict[str, Any]) -> ImportanceScore:
        """Compute importance score from a memory dict."""
        score = ImportanceScore()

        # Component scores
        score.information_density = self._compute_information_density(
            memory.get("content", "")
        )
        score.recency = self._compute_recency(memory)
        score.access_frequency = self._compute_access_frequency(memory)
        score.entity_centrality = self._compute_entity_centrality(memory)
        score.llm_importance = self._compute_llm_importance(memory)
        score.contradiction_freshness = self._compute_contradiction_freshness(memory)

        # Weighted composite
        cfg = self._config
        score.composite = (
            cfg.weight_information_density * score.information_density
            + cfg.weight_recency * score.recency
            + cfg.weight_access_frequency * score.access_frequency
            + cfg.weight_entity_centrality * score.entity_centrality
            + cfg.weight_llm_importance * score.llm_importance
            + cfg.weight_contradiction_freshness * score.contradiction_freshness
        )

        # Clamp to [0, 1]
        score.composite = max(0.0, min(1.0, score.composite))

        return score

    def _compute_information_density(self, content: str) -> float:
        """
        Compute information density of a memory.

        Measures how unique/informative the content is compared to
        generic text. High density = many unique, specific terms.
        Low density = common words, low information content.
        """
        if not content:
            return 0.0

        words = re.findall(r"\w+", content.lower())
        if not words:
            return 0.0

        # Count unique words (excluding stop words)
        content_words = [w for w in words if w not in self._stop_words and len(w) > 2]
        if not content_words:
            return 0.1  # Some credit for having content

        unique_words = set(content_words)
        unique_ratio = len(unique_words) / len(content_words)

        # Information-theoretic measure: entropy of word distribution
        word_counts = Counter(content_words)
        total = len(content_words)
        entropy = 0.0
        for count in word_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize entropy (max entropy = log2(unique_words))
        max_entropy = math.log2(max(1, len(unique_words)))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Bonus for specific entities (proper nouns, numbers, technical terms)
        specificity_bonus = 0.0
        proper_nouns = len(re.findall(r"\b[A-Z][a-z]+\b", content))
        numbers = len(re.findall(r"\b\d+\.?\d*\b", content))
        technical = len(re.findall(r"\b(?:pip|install|import|def|class|sudo|apt|docker|git)\b", content, re.IGNORECASE))

        specificity_bonus = min(0.2, (proper_nouns * 0.02 + numbers * 0.02 + technical * 0.03))

        # Combine: 60% unique ratio + 30% normalized entropy + 10% specificity
        density = 0.6 * unique_ratio + 0.3 * normalized_entropy + 0.1 * specificity_bonus
        return min(1.0, max(0.0, density))

    def _compute_recency(self, memory: Dict[str, Any]) -> float:
        """
        Compute recency score using Ebbinghaus forgetting curve with stability theory.

        R = e^(-t/S) where:
        - t = time since last access (hours)
        - S = stability = half_life * (1 + access_count * stability_factor)

        Each access increases stability, making the memory decay slower.
        """
        now = time.time()
        accessed_at = memory.get("accessed_at", memory.get("created_at", now))
        access_count = memory.get("access_count", 0) or 0
        importance = memory.get("importance", 0.5) or 0.5

        # Time since last access in hours
        hours_since_access = (now - accessed_at) / 3600.0
        if hours_since_access <= 0:
            return 1.0

        # Stability grows with each access (power law)
        stability = self._config.half_life_hours * (1 + access_count * self._config.stability_factor)

        # Ebbinghaus: R = e^(-t/S)
        retention = math.exp(-hours_since_access / stability)

        # Importance modulates decay rate
        importance_factor = 0.8 + 0.4 * importance  # 0.8 to 1.2 range
        retention *= importance_factor

        return max(0.0, min(1.0, retention))

    def _compute_access_frequency(self, memory: Dict[str, Any]) -> float:
        """
        Compute access frequency score.

        Uses logarithmic scaling: log(1 + access_count) / log(1 + max_expected)
        This gives diminishing returns for very frequent accesses.
        """
        access_count = memory.get("access_count", 0) or 0
        max_expected = 50  # Cap at this many accesses for normalization

        if access_count == 0:
            return 0.0

        score = math.log1p(access_count) / math.log1p(max_expected)
        return min(1.0, score)

    def _compute_entity_centrality(self, memory: Dict[str, Any]) -> float:
        """
        Compute entity centrality score.

        Memories whose entities are central (high degree) in the
        knowledge graph score higher. This surfaces memories about
        important/conected topics.
        """
        # Get entities linked to this memory
        cursor = self._conn.cursor()
        cursor.execute(
            """SELECT e.name FROM entities e
               JOIN memory_entities me ON me.entity_id = e.id
               WHERE me.memory_id = ?""",
            (memory["id"],),
        )
        entities = [row[0] for row in cursor.fetchall()]

        if not entities:
            return 0.3  # Default for memories without entities

        # Compute centrality for each entity (degree centrality)
        centrality_scores = []
        for entity_name in entities:
            if entity_name in self._entity_centrality_cache:
                centrality_scores.append(self._entity_centrality_cache[entity_name])
                continue

            # Get degree (number of connections)
            cursor.execute(
                """SELECT COUNT(*) FROM edges e
                   JOIN entities en ON en.id = e.source_id OR en.id = e.target_id
                   WHERE en.name = ?""",
                (entity_name,),
            )
            degree = cursor.fetchone()[0] or 0

            # Normalize by max degree in graph
            cursor.execute("SELECT MAX(degree) FROM (SELECT COUNT(*) as degree FROM edges GROUP BY source_id)")
            max_degree_row = cursor.fetchone()
            max_degree = max_degree_row[0] if max_degree_row and max_degree_row[0] else 1

            centrality = degree / max(max_degree, 1)
            self._entity_centrality_cache[entity_name] = centrality
            centrality_scores.append(centrality)

        # Average centrality of all entities
        return sum(centrality_scores) / len(centrality_scores) if centrality_scores else 0.3

    def _compute_llm_importance(self, memory: Dict[str, Any]) -> float:
        """
        Extract LLM-assigned importance if available.

        Checks metadata for 'llm_importance' field set by extraction engine.
        """
        metadata = memory.get("metadata")
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except Exception:
                return 0.5  # Default

        if metadata and isinstance(metadata, dict):
            llm_imp = metadata.get("llm_importance")
            if llm_imp is not None:
                # Normalize 1-10 scale to 0.0-1.0
                return max(0.0, min(1.0, (float(llm_imp) - 1) / 9.0))

        # Fall back to the importance field (0.0-1.0)
        importance = memory.get("importance", 0.5) or 0.5
        return float(importance)

    def _compute_contradiction_freshness(self, memory: Dict[str, Any]) -> float:
        """
        Compute contradiction freshness score.

        Memories that contradict older memories should score higher,
        as they represent updated/corrected information.
        """
        content = memory.get("content", "")
        if not content:
            return 0.5

        # Check for negation patterns that suggest contradiction
        negation_patterns = [
            r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bneither\b", r"\bnor\b",
            r"\bwithout\b", r"\bcannot\b", r"\bcan't\b", r"\bwon't\b",
            r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b", r"\bisn't\b",
            r"\bare\b\s*\bnot\b", r"\bwasn't\b", r"\bweren't\b",
            r"\bhasn't\b", r"\bhaven't\b", r"\bhadn't\b",
            r"\bactually\b", r"\bin\s+fact\b", r"\bcorrection\b",
            r"\bupdate\b", r"\bchanged\b", r"\bmoved\b",
        ]

        has_negation = any(re.search(p, content, re.IGNORECASE) for p in negation_patterns)

        if has_negation:
            # Check if there's an older, non-negated version
            cursor = self._conn.cursor()
            cursor.execute(
                """SELECT COUNT(*) FROM memories
                   WHERE is_deleted = 0
                   AND created_at < ?
                   AND content NOT LIKE '%not%'
                   AND content NOT LIKE '%never%'
                   AND content NOT LIKE '%no %'""",
                (memory.get("created_at", time.time()),),
            )
            older_count = cursor.fetchone()[0] or 0

            if older_count > 0:
                return 0.8  # High freshness - this is an update

        return 0.5  # Default neutral freshness

    def rank_memories(
        self,
        memory_ids: List[int],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, ImportanceScore]]:
        """
        Rank memories by composite importance score.

        Args:
            memory_ids: List of memory IDs to rank.
            top_k: If specified, return only top K results.

        Returns:
            List of (memory_id, ImportanceScore) tuples, sorted descending.
        """
        scores = self.score_memories(memory_ids)
        ranked = sorted(scores.items(), key=lambda x: x[1].composite, reverse=True)

        if top_k:
            ranked = ranked[:top_k]

        return ranked

    def get_top_memories(
        self,
        limit: int = 10,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get the most important memories across the entire store."""
        cursor = self._conn.cursor()
        query = "SELECT id FROM memories WHERE is_deleted = 0"
        params: list = []
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        query += " ORDER BY created_at DESC LIMIT 500"  # Sample recent

        cursor.execute(query, params)
        ids = [row[0] for row in cursor.fetchall()]

        ranked = self.rank_memories(ids, top_k=limit)
        return [{"id": mid, "score": score.to_dict()} for mid, score in ranked]

    def invalidate_caches(self) -> None:
        """Clear all scoring caches."""
        self._entity_centrality_cache.clear()
