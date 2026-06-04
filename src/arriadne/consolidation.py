"""
Memory Consolidation Engine

Merges related memories into fewer, richer ones.
Inspired by MemGPT/Letta's summarization system.

Features:
- Cluster related memories by embedding similarity
- LLM-powered merging (preserves all unique facts)
- Scheduled consolidation (automatic or manual)
- Quality scoring for consolidated results
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("arriadne.consolidation")


@dataclass
class ConsolidationGroup:
    """A group of related memories to be consolidated."""

    group_id: int
    memories: List[Dict[str, Any]]
    topic: str = ""
    avg_similarity: float = 0.0

    @property
    def size(self) -> int:
        return len(self.memories)


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""

    groups_processed: int
    memories_before: int
    memories_after: int
    consolidated_groups: List[ConsolidationGroup]
    latency_ms: float = 0.0

    @property
    def compression_ratio(self) -> float:
        if self.memories_before == 0:
            return 1.0
        return self.memories_after / self.memories_before


class MemoryConsolidator:
    """
    Consolidates related memories into fewer, richer ones.

    Three modes:
    1. Similarity-based: Cluster by embedding similarity, merge clusters
    2. Topic-based: Group by topic/category, merge per topic
    3. Temporal: Group by time proximity, merge recent memories
    """

    def __init__(
        self,
        db_conn: Any,
        embedding_provider: Optional[Any] = None,
        llm_provider: Optional[Any] = None,
        similarity_threshold: float = 0.75,
        min_group_size: int = 2,
        max_group_size: int = 10,
    ):
        self._conn = db_conn
        self._embeddings = embedding_provider
        self._llm = llm_provider
        self._similarity_threshold = similarity_threshold
        self._min_group_size = min_group_size
        self._max_group_size = max_group_size

    def find_related_groups(
        self,
        method: str = "similarity",
        limit: int = 50,
    ) -> List[ConsolidationGroup]:
        """
        Find groups of related memories that can be consolidated.

        Methods:
        - "similarity": Group by embedding cosine similarity
        - "topic": Group by topic/category
        - "temporal": Group by creation time proximity
        """
        if method == "topic":
            return self._group_by_topic(limit)
        elif method == "temporal":
            return self._group_by_time(limit)
        else:
            return self._group_by_similarity(limit)

    def consolidate_group(
        self,
        group: ConsolidationGroup,
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Consolidate a group of related memories.

        Returns:
            (consolidated_memory_or_None, list_of_original_memory_ids_to_remove)
        """
        if group.size < self._min_group_size:
            return None, []

        memories = group.memories
        memory_texts = [m.get("content", "") for m in memories]

        if self._llm:
            return self._consolidate_with_llm(memories, memory_texts)
        else:
            return self._consolidate_simple(memories, memory_texts)

    def consolidate_all(
        self,
        method: str = "similarity",
        dry_run: bool = False,
    ) -> ConsolidationResult:
        """Run consolidation across all memories."""
        t0 = time.monotonic()

        groups = self.find_related_groups(method)
        consolidated = []
        total_before = 0
        total_after = 0

        for group in groups:
            total_before += group.size
            result, remove_ids = self.consolidate_group(group)

            if result:
                total_after += 1
                consolidated.append(group)

                if not dry_run:
                    self._apply_consolidation(result, remove_ids, group)

        latency = (time.monotonic() - t0) * 1000

        return ConsolidationResult(
            groups_processed=len(groups),
            memories_before=total_before,
            memories_after=total_after,
            consolidated_groups=consolidated,
            latency_ms=latency,
        )

    def _group_by_similarity(self, limit: int) -> List[ConsolidationGroup]:
        """Group memories by embedding similarity."""
        if not self._embeddings:
            logger.warning("No embedding provider for similarity grouping")
            return []

        cursor = self._conn.cursor()
        cursor.execute(
            """SELECT id, content, embedding_vector, topic, importance
            FROM memories WHERE deleted_at IS NULL
            ORDER BY created_at DESC LIMIT ?""",
            (limit * 5,),  # Get more than we need
        )
        rows = cursor.fetchall()

        if len(rows) < self._min_group_size:
            return []

        # Build embedding matrix
        import numpy as np

        ids = []
        texts = []
        vectors = []
        metadata = []

        for row in rows:
            vec_bytes = row[2]
            if vec_bytes is None:
                continue
            try:
                vec = np.frombuffer(vec_bytes, dtype=np.float32)
                if vec.shape[0] == 0:
                    continue
                ids.append(row[0])
                texts.append(row[1])
                vectors.append(vec)
                metadata.append({"topic": row[3] or "", "importance": row[4] or 5})
            except Exception:
                continue

        if len(vectors) < self._min_group_size:
            return []

        matrix = np.array(vectors, dtype=np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        matrix_norm = matrix / norms

        # Compute similarity matrix
        sim_matrix = matrix_norm @ matrix_norm.T

        # Greedy clustering
        visited = set()
        groups = []
        group_id = 0

        for i in range(len(ids)):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            for j in range(i + 1, len(ids)):
                if j in visited:
                    continue
                if sim_matrix[i][j] >= self._similarity_threshold:
                    if len(cluster) < self._max_group_size:
                        cluster.append(j)
                        visited.add(j)

            if len(cluster) >= self._min_group_size:
                group_memories = []
                for idx in cluster:
                    group_memories.append({
                        "id": ids[idx],
                        "content": texts[idx],
                        "importance": metadata[idx].get("importance", 5),
                        "topic": metadata[idx].get("topic", ""),
                    })

                avg_sim = float(np.mean([
                    sim_matrix[cluster[0]][k] for k in cluster[1:]
                ])) if len(cluster) > 1 else 1.0

                groups.append(ConsolidationGroup(
                    group_id=group_id,
                    memories=group_memories,
                    avg_similarity=avg_sim,
                ))
                group_id += 1

        return groups

    def _group_by_topic(self, limit: int) -> List[ConsolidationGroup]:
        """Group memories by topic/category."""
        cursor = self._conn.cursor()
        cursor.execute(
            """SELECT id, content, topic, importance
            FROM memories WHERE deleted_at IS NULL AND topic IS NOT NULL AND topic != ''
            ORDER BY created_at DESC LIMIT ?""",
            (limit * 5,),
        )
        rows = cursor.fetchall()

        by_topic = defaultdict(list)
        for row in rows:
            by_topic[row[2]].append({
                "id": row[0],
                "content": row[1],
                "importance": row[3] or 5,
                "topic": row[2],
            })

        groups = []
        group_id = 0
        for topic, memories in by_topic.items():
            if len(memories) >= self._min_group_size:
                groups.append(ConsolidationGroup(
                    group_id=group_id,
                    memories=memories[:self._max_group_size],
                    topic=topic,
                ))
                group_id += 1

        return groups

    def _group_by_time(self, limit: int) -> List[ConsolidationGroup]:
        """Group memories by temporal proximity (within 1 hour)."""
        cursor = self._conn.cursor()
        cursor.execute(
            """SELECT id, content, created_at, topic, importance
            FROM memories WHERE deleted_at IS NULL
            ORDER BY created_at DESC LIMIT ?""",
            (limit * 5,),
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        # Sort by time
        sorted_rows = sorted(rows, key=lambda r: r[2] or 0, reverse=True)

        groups = []
        group_id = 0
        current_group = []

        for i, row in enumerate(sorted_rows):
            current_group.append({
                "id": row[0],
                "content": row[1],
                "importance": row[4] or 5,
                "topic": row[3] or "",
            })

            # Check if next memory is within 1 hour
            if i + 1 < len(sorted_rows):
                time_diff = (row[2] or 0) - (sorted_rows[i + 1][2] or 0)
                if time_diff > 3600:  # More than 1 hour
                    if len(current_group) >= self._min_group_size:
                        groups.append(ConsolidationGroup(
                            group_id=group_id,
                            memories=current_group[:self._max_group_size],
                        ))
                        group_id += 1
                    current_group = []

        # Final group
        if len(current_group) >= self._min_group_size:
            groups.append(ConsolidationGroup(
                group_id=group_id,
                memories=current_group[:self._max_group_size],
            ))

        return groups

    def _consolidate_with_llm(
        self,
        memories: List[Dict[str, Any]],
        memory_texts: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Consolidate using LLM."""
        from arriadne.llm import LLMMessage

        consolidation_prompt = """You are a memory consolidation engine. Merge these related memories into fewer, richer ones.

RULES:
1. Preserve ALL unique facts — never lose information
2. Remove redundancy and merge duplicates
3. Keep specific details: names, dates, numbers, technical terms
4. Use the most recent information when there's a conflict
5. Output 1-3 consolidated memories
6. Each should be 20-100 words, self-contained
7. Assign importance 1-10 and list key entities

Memories to consolidate:
"""
        for i, text in enumerate(memory_texts):
            consolidation_prompt += f"\n[{i}] {text}"

        consolidation_prompt += '\n\nReturn JSON: {"memories": [{"text": "...", "entities": [...], "importance": N}]}'

        try:
            response = self._llm.complete_sync(
                [
                    LLMMessage(
                        "system",
                        "You are a memory consolidation engine. Merge related memories, "
                        "preserving all unique facts. Output JSON.",
                    ),
                    LLMMessage("user", consolidation_prompt),
                ],
                temperature=0.2,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )

            data = response.json()
            if isinstance(data, dict):
                consolidated = data.get("memories", data.get("memory", []))
            elif isinstance(data, list):
                consolidated = data
            else:
                return None, []

            if not consolidated:
                return None, []

            # Use first consolidated memory as the primary
            primary = consolidated[0]
            remove_ids = [m.get("id", "") for m in memories if m.get("id")]

            return {
                "content": primary.get("text", ""),
                "entities": primary.get("entities", []),
                "importance": primary.get("importance", 7),
                "topic": memories[0].get("topic", ""),
                "metadata": {
                    "consolidated_from": remove_ids,
                    "consolidated_count": len(remove_ids),
                    "additional_consolidated": consolidated[1:] if len(consolidated) > 1 else [],
                },
            }, remove_ids

        except Exception as e:
            logger.error(f"LLM consolidation failed: {e}")
            return None, []

    def _consolidate_simple(
        self,
        memories: List[Dict[str, Any]],
        memory_texts: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Simple consolidation without LLM — concatenate and deduplicate."""
        # Take the longest text as primary (likely most informative)

        # Simple deduplication of sentences
        sentences = set()
        for text in memory_texts:
            for sentence in text.split(". "):
                sentence = sentence.strip()
                if sentence and len(sentence) > 5:
                    sentences.add(sentence)

        consolidated_text = ". ".join(sorted(sentences))

        # Use highest importance
        max_importance = max(m.get("importance", 5) for m in memories)
        topics = list(set(m.get("topic", "") for m in memories if m.get("topic")))

        remove_ids = [m.get("id", "") for m in memories if m.get("id")]

        return {
            "content": consolidated_text,
            "entities": [],
            "importance": max_importance,
            "topic": topics[0] if topics else "",
            "metadata": {
                "consolidated_from": remove_ids,
                "consolidated_count": len(remove_ids),
            },
        }, remove_ids

    def _apply_consolidation(
        self,
        consolidated: Dict[str, Any],
        remove_ids: List[str],
        group: ConsolidationGroup,
    ) -> None:
        """Apply consolidation results to the database."""
        cursor = self._conn.cursor()

        # Insert consolidated memory
        import hashlib
        content = consolidated.get("content", "")
        mem_hash = hashlib.md5(content.lower().encode()).hexdigest()[:16]

        cursor.execute(
            """INSERT INTO memories
            (content, content_hash, memory_type, importance, embedding,
             created_at, updated_at, accessed_at, access_count,
             retention_strength, is_deleted, metadata)
            VALUES (?, ?, 'semantic', ?, NULL, ?, ?, ?, 0, 1.0, 0, ?)""",
            (
                content,
                mem_hash,
                consolidated.get("importance", 0.7),
                time.time(),
                time.time(),
                time.time(),
                str(consolidated.get("metadata", {})),
            ),
        )

        # Soft-delete originals (don't hard delete — preserve history)
        now = time.time()
        for mid in remove_ids:
            if mid:
                cursor.execute(
                    "UPDATE memories SET deleted_at = ? WHERE id = ?",
                    (now, mid),
                )

        self._conn.commit()
        logger.info(
            "Consolidated group %s: %d memories -> 1",
            group.group_id, group.size,
        )

    # ─── Advanced Consolidation Features ──────────────────────

    def consolidate_by_topic(
        self,
        min_cluster_size: int = 3,
        method: str = "similarity",
    ) -> ConsolidationResult:
        """
        Consolidate memories grouped by topic similarity.

        Clusters memories by embedding similarity and merges each cluster.

        Args:
            min_cluster_size: Minimum memories in a cluster to consolidate.
            method: Clustering method ("similarity", "temporal").

        Returns:
            ConsolidationResult with summary of operations.
        """
        t0 = time.monotonic()

        if method == "similarity" and self._embeddings:
            groups = self._group_by_similarity(limit=500)
        elif method == "temporal":
            groups = self._group_by_time(limit=500)
        else:
            groups = self._group_by_topic(limit=500)

        # Filter by min_cluster_size
        groups = [g for g in groups if g.size >= min_cluster_size]

        total_before = 0
        total_after = 0
        consolidated = []

        for group in groups:
            total_before += group.size
            result, remove_ids = self.consolidate_group(group)
            if result:
                total_after += 1
                consolidated.append(group)
                self._apply_consolidation(result, remove_ids, group)

        latency = (time.monotonic() - t0) * 1000

        return ConsolidationResult(
            groups_processed=len(groups),
            memories_before=total_before,
            memories_after=total_after,
            consolidated_groups=consolidated,
            latency_ms=latency,
        )

    def get_consolidation_suggestions(
        self,
        min_cluster_size: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Suggest memory clusters that should be consolidated.

        Returns clusters of related memories without actually merging them.

        Args:
            min_cluster_size: Minimum cluster size to suggest.

        Returns:
            List of suggestion dicts with cluster info.
        """
        groups = self.find_related_groups(method="similarity", limit=200)
        suggestions = []

        for group in groups:
            if group.size >= min_cluster_size:
                # Detect contradictions within the group
                contradictions = self._detect_group_contradictions(group)

                suggestions.append({
                    "group_id": group.group_id,
                    "size": group.size,
                    "topic": group.topic,
                    "avg_similarity": round(group.avg_similarity, 4),
                    "memories": [
                        {"id": m.get("id"), "content": m.get("content", "")[:100]}
                        for m in group.memories[:5]
                    ],
                    "has_contradictions": len(contradictions) > 0,
                    "contradictions": contradictions,
                    "suggested_action": "merge" if len(contradictions) == 0 else "review_contradictions",
                })

        return suggestions

    def _detect_group_contradictions(
        self,
        group: ConsolidationGroup,
    ) -> List[Dict[str, Any]]:
        """
        Detect contradictions within a consolidation group.

        Uses simple pattern matching for negation and factual conflict detection.
        """
        contradictions = []
        negation_patterns = [
            r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bcan't\b",
            r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b", r"\bisn't\b",
        ]

        for i, mem_a in enumerate(group.memories):
            content_a = mem_a.get("content", "")
            for j, mem_b in enumerate(group.memories):
                if j <= i:
                    continue
                content_b = mem_b.get("content", "")

                # Check if one contains negation of the other
                has_neg_a = any(re.search(p, content_a, re.IGNORECASE) for p in negation_patterns)
                has_neg_b = any(re.search(p, content_b, re.IGNORECASE) for p in negation_patterns)

                if has_neg_a != has_neg_b:
                    # One has negation, one doesn't — potential contradiction
                    # Check word overlap
                    words_a = set(content_a.lower().split())
                    words_b = set(content_b.lower().split())
                    overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)

                    if overlap > 0.3:
                        contradictions.append({
                            "memory_a_id": mem_a.get("id"),
                            "memory_b_id": mem_b.get("id"),
                            "memory_a_preview": content_a[:100],
                            "memory_b_preview": content_b[:100],
                            "overlap": round(overlap, 3),
                        })

        return contradictions

    def generate_cluster_summary(
        self,
        group: ConsolidationGroup,
    ) -> str:
        """
        Generate a text summary for a consolidation group.

        Simple extractive summary: takes the most informative sentences.
        """
        if not group.memories:
            return ""

        sentences = []
        for mem in group.memories:
            content = mem.get("content", "")
            for sentence in content.split(". "):
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:
                    sentences.append(sentence)

        if not sentences:
            return group.memories[0].get("content", "")

        # Score sentences by uniqueness (simple TF-IDF-like)
        word_freq: Dict[str, int] = {}
        for sent in sentences:
            for word in sent.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        scored = []
        for sent in sentences:
            words = sent.lower().split()
            score = sum(1.0 / word_freq.get(w, 1) for w in words) / max(len(words), 1)
            scored.append((score, sent))

        # Take top 3 sentences
        scored.sort(reverse=True)
        top_sentences = [sent for _, sent in scored[:3]]

        return ". ".join(top_sentences) + "."
