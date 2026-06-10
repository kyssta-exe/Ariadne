"""Deduplication and contradiction detection for Ariadne memory system.

Provides MinHash LSH deduplication and negation-based contradiction detection.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

# Common negation patterns for contradiction detection
_NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bno\b",
    r"\bnever\b",
    r"\bneither\b",
    r"\bnor\b",
    r"\bwithout\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bwon't\b",
    r"\bwouldn't\b",
    r"\bshouldn't\b",
    r"\bdon't\b",
    r"\bdoesn't\b",
    r"\bdidn't\b",
    r"\bisn't\b",
    r"\bare\bnot\b",
    r"\bwasn't\b",
    r"\bweren't\b",
    r"\bhasn't\b",
    r"\bhaven't\b",
    r"\bhadn't\b",
]

# Patterns for extracting factual claims - using sentence-level matching
# Each pattern extracts (subject, predicate, full_match) from simple sentences
_FACT_PATTERNS = [
    # "X is Y" or "X are Y" - match up to period or end
    re.compile(r"(\b\w+(?:\s+\w+)?)\s+(?:is|are|was|were)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    # "X has Y" or "X have Y"
    re.compile(r"(\b\w+(?:\s+\w+)?)\s+(?:has|have|had)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    # "X can Y" or "X could Y"
    re.compile(r"(\b\w+(?:\s+\w+)?)\s+(?:can|could|may|might)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    # "X does Y" or "X did Y"
    re.compile(r"(\b\w+(?:\s+\w+)?)\s+(?:does|did)\s+(.+?)(?:\.|$)", re.IGNORECASE),
]


class Deduplicator:
    """MinHash LSH-based deduplication for text content.

    Uses locality-sensitive hashing to efficiently find near-duplicate
    content based on word-level shingles.

    Args:
        threshold: Similarity threshold for duplicate detection (0.0-1.0).
        num_perm: Number of MinHash permutations (higher = more accurate).
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._threshold = threshold
        self._num_perm = num_perm
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._minhashes: dict[str, MinHash] = {}
        self._contents: dict[str, str] = {}
        self._next_id = 0
        logger.debug("Initialized Deduplicator (threshold=%.2f, num_perm=%d)", threshold, num_perm)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words."""
        return re.findall(r"\w+", text.lower())

    def _create_minhash(self, text: str) -> MinHash:
        """Create a MinHash for the given text."""
        m = MinHash(num_perm=self._num_perm)
        tokens = self._tokenize(text)
        # Use 2-word shingles for better accuracy
        for i in range(len(tokens) - 1):
            shingle = f"{tokens[i]} {tokens[i+1]}"
            m.update(shingle.encode("utf-8"))
        # Also add single words
        for token in tokens:
            m.update(token.encode("utf-8"))
        return m

    def add(self, content: str, doc_id: str | None = None) -> str:
        """Add content to the deduplication index.

        Args:
            content: Text content to index.
            doc_id: Optional document ID. If None, auto-generates one.

        Returns:
            The document ID.
        """
        if doc_id is None:
            doc_id = f"doc_{self._next_id}"
            self._next_id += 1

        if doc_id in self._minhashes:
            # Remove old entry first
            self.remove(doc_id)

        m = self._create_minhash(content)
        try:
            self._lsh.insert(doc_id, m)
        except ValueError:
            # Document already exists, remove and re-insert
            self._lsh.remove(doc_id)
            self._lsh.insert(doc_id, m)

        self._minhashes[doc_id] = m
        self._contents[doc_id] = content
        logger.debug("Added document %s to dedup index", doc_id)
        return doc_id

    def remove(self, doc_id: str) -> bool:
        """Remove a document from the deduplication index.

        Args:
            doc_id: Document ID to remove.

        Returns:
            True if removed, False if not found.
        """
        if doc_id not in self._minhashes:
            return False

        try:
            self._lsh.remove(doc_id)
            del self._minhashes[doc_id]
            del self._contents[doc_id]
            logger.debug("Removed document %s from dedup index", doc_id)
            return True
        except Exception as e:
            logger.warning("Error removing document %s: %s", doc_id, e)
            return False

    def is_duplicate(self, content: str) -> bool:
        """Check if content is a near-duplicate of any indexed content.

        Args:
            content: Text content to check.

        Returns:
            True if a duplicate exists above the threshold.
        """
        m = self._create_minhash(content)
        results = self._lsh.query(m)
        return len(results) > 0

    def find_duplicates(self, content: str) -> list[dict[str, Any]]:
        """Find all near-duplicates of the given content.

        Args:
            content: Text content to find duplicates for.

        Returns:
            List of dicts with 'id', 'content', and 'similarity' for each duplicate.
        """
        m = self._create_minhash(content)
        results = self._lsh.query(m)

        duplicates = []
        for doc_id in results:
            if doc_id in self._minhashes:
                similarity = m.jaccard(self._minhashes[doc_id])
                duplicates.append({
                    "id": doc_id,
                    "content": self._contents.get(doc_id, ""),
                    "similarity": round(similarity, 4),
                })

        duplicates.sort(key=lambda x: x["similarity"], reverse=True)
        return duplicates

    def find_related(self, content: str, limit: int = 10) -> list[dict[str, Any]]:
        """Find content related (but not necessarily duplicate) to the query.

        Uses a lower threshold to find loosely related content.

        Args:
            content: Text content to find related items for.
            limit: Maximum results to return.

        Returns:
            List of related content dicts.
        """
        m = self._create_minhash(content)
        # Use all indexed documents and compute similarity
        related = []
        for doc_id, stored_m in self._minhashes.items():
            similarity = m.jaccard(stored_m)
            if similarity > 0.0:
                related.append({
                    "id": doc_id,
                    "content": self._contents.get(doc_id, ""),
                    "similarity": round(similarity, 4),
                })

        related.sort(key=lambda x: x["similarity"], reverse=True)
        return related[:limit]

    @property
    def size(self) -> int:
        """Return the number of indexed documents."""
        return len(self._minhashes)


class ContradictionDetector:
    """Detects contradictions between text statements using pattern matching.

    Extracts factual claims and checks for negation patterns that indicate
    contradictory statements.
    """

    def __init__(self) -> None:
        self._negation_re = re.compile(
            "|".join(_NEGATION_PATTERNS), re.IGNORECASE
        )
        # Split on conjunctions and sentence boundaries to get individual clauses
        self._clause_splitter = re.compile(r"\s+(?:and|but|yet|however|,)\s+|\.|\;")
        # Simple fact patterns for individual clauses
        self._simple_fact_re = re.compile(
            r"^(\w+(?:\s+\w+)?)\s+(?:is|are|was|were|has|have|had|can|could|may|might|does|did)\s+(.+)$",
            re.IGNORECASE,
        )

    def _split_clauses(self, text: str) -> list[str]:
        """Split text into individual clauses."""
        clauses = self._clause_splitter.split(text)
        return [c.strip() for c in clauses if c.strip()]

    def extract_facts(self, text: str) -> list[dict[str, str]]:
        """Extract factual claims from text.

        Splits text into clauses and extracts simple facts from each.

        Args:
            text: Input text.

        Returns:
            List of fact dicts with 'subject', 'predicate', and 'negated' fields.
        """
        facts = []
        clauses = self._split_clauses(text)

        for clause in clauses:
            match = self._simple_fact_re.match(clause)
            if match:
                subject = match.group(1).strip().lower()
                predicate = match.group(2).strip().lower()
                is_negated = bool(self._negation_re.search(clause))
                facts.append({
                    "subject": subject,
                    "predicate": predicate,
                    "negated": is_negated,
                    "original": clause.strip(),
                })

        return facts

    def detect_contradictions(
        self, text_a: str, text_b: str
    ) -> list[dict[str, Any]]:
        """Detect contradictions between two text statements.

        Args:
            text_a: First text statement.
            text_b: Second text statement.

        Returns:
            List of contradiction dicts with details about each conflict.
        """
        facts_a = self.extract_facts(text_a)
        facts_b = self.extract_facts(text_b)

        contradictions = []

        for fa in facts_a:
            for fb in facts_b:
                # Normalize predicates by removing negation words
                norm_pred_a = self._normalize_predicate(fa["predicate"])
                norm_pred_b = self._normalize_predicate(fb["predicate"])

                # Same subject, same normalized predicate, different negation
                if (
                    fa["subject"] == fb["subject"]
                    and norm_pred_a == norm_pred_b
                    and fa["negated"] != fb["negated"]
                ):
                    contradictions.append({
                        "subject": fa["subject"],
                        "predicate": fa["predicate"],
                        "statement_a": fa["original"],
                        "statement_b": fb["original"],
                        "negated_in_a": fa["negated"],
                        "negated_in_b": fb["negated"],
                    })

        return contradictions

    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize a predicate by removing negation words.

        Args:
            predicate: The predicate string.

        Returns:
            Normalized predicate without negation words.
        """
        # Remove common negation words
        normalized = predicate
        for word in ["not", "no", "never", "neither", "nor", "without",
                      "cannot", "can't", "won't", "wouldn't", "shouldn't",
                      "don't", "doesn't", "didn't", "isn't", "aren't",
                      "wasn't", "weren't", "hasn't", "haven't", "hadn't"]:
            # Remove the word and surrounding whitespace
            normalized = re.sub(rf"\b{re.escape(word)}\b\s*", "", normalized, flags=re.IGNORECASE)
        return normalized.strip()

    def is_contradictory(self, text_a: str, text_b: str) -> bool:
        """Quick check if two texts contain contradictions.

        Args:
            text_a: First text statement.
            text_b: Second text statement.

        Returns:
            True if contradictions are found.
        """
        return len(self.detect_contradictions(text_a, text_b)) > 0
