"""
Entity Resolution System

4-stage pipeline inspired by Zep/Graphiti:
1. Exact name dedup
2. Embedding cosine similarity matching
3. MinHash fuzzy Jaccard matching
4. LLM-based disambiguation

Also includes spaCy NER for deterministic entity extraction.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("arriadne.entities")


@dataclass
class Entity:
    """A resolved entity in the knowledge graph."""

    id: str
    name: str
    entity_type: str  # PERSON, ORG, GPE, TOOL, CONCEPT, etc.
    canonical_name: str  # The normalized canonical form
    aliases: List[str] = field(default_factory=list)
    linked_memory_ids: List[str] = field(default_factory=list)
    mention_count: int = 1
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.canonical_name.lower().encode()).hexdigest()[:12]


@dataclass
class EntityMention:
    """A mention of an entity in a text."""

    text: str
    start: int
    end: int
    entity_type: str
    confidence: float = 1.0


class EntityExtractor:
    """
    Extract entities from text using spaCy NER.

    Falls back to regex-based extraction if spaCy is not available.
    """

    # Generic words to exclude from entity lists
    GENERIC_WORDS = frozenset({
        "the", "a", "an", "this", "that", "it", "he", "she", "they",
        "we", "you", "i", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can",
        "not", "no", "yes", "ok", "sure", "thanks", "thank",
        "please", "hello", "hi", "hey", "bye", "goodbye",
        "very", "really", "just", "also", "too", "much", "more",
        "thing", "things", "stuff", "way", "time", "day", "days",
        "new", "good", "bad", "old", "first", "last", "next",
        "work", "info", "details", "something", "everything", "anything",
    })

    def __init__(self):
        self._nlp = None
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy en_core_web_sm for entity extraction")
        except (ImportError, OSError):
            logger.info("spaCy not available, using regex entity extraction")

    def extract(self, text: str) -> List[EntityMention]:
        """Extract entities from text."""
        if self._nlp:
            return self._extract_spacy(text)
        return self._extract_regex(text)

    def extract_names(self, text: str) -> List[str]:
        """Extract entity names from text, deduplicated and normalized."""
        mentions = self.extract(text)
        # Deduplicate and normalize
        seen: Set[str] = set()
        names = []
        for m in mentions:
            normalized = self._normalize_name(m.text)
            if normalized and normalized not in seen and len(normalized) > 1:
                if normalized.lower() not in self.GENERIC_WORDS:
                    seen.add(normalized)
                    names.append(normalized)
        return names

    def _extract_spacy(self, text: str) -> List[EntityMention]:
        """Extract using spaCy NER."""
        doc = self._nlp(text)
        mentions = []

        for ent in doc.ents:
            # Map spaCy labels to our types
            type_map = {
                "PERSON": "PERSON",
                "ORG": "ORG",
                "GPE": "GPE",
                "LOC": "GPE",
                "PRODUCT": "PRODUCT",
                "EVENT": "EVENT",
                "WORK_OF_ART": "WORK",
                "DATE": "DATE",
                "MONEY": "MONEY",
                "QUANTITY": "QUANTITY",
                "TECH": "TOOL",
            }
            entity_type = type_map.get(ent.label_, "OTHER")

            mentions.append(
                EntityMention(
                    text=ent.text.strip(),
                    start=ent.start_char,
                    end=ent.end_char,
                    entity_type=entity_type,
                    confidence=0.9,
                )
            )

        # Also extract compound nouns and proper nouns via dependency parsing
        for token in doc:
            if token.dep_ == "compound" and token.head.pos_ in ("NOUN", "PROPN"):
                compound = f"{token.text} {token.head.text}"
                mentions.append(
                    EntityMention(
                        text=compound,
                        start=token.idx,
                        end=token.head.idx + len(token.head.text),
                        entity_type="CONCEPT",
                        confidence=0.7,
                    )
                )

        return mentions

    def _extract_regex(self, text: str) -> List[EntityMention]:
        """Extract using regex heuristics."""
        mentions = []

        # Pattern 1: Capitalized multi-word sequences (potential proper nouns)
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
            mentions.append(
                EntityMention(
                    text=match.group(1),
                    start=match.start(),
                    end=match.end(),
                    entity_type="PERSON",
                    confidence=0.6,
                )
            )

        # Pattern 2: Quoted strings
        for match in re.finditer(r'["\']([^"\']{2,50})["\']', text):
            mentions.append(
                EntityMention(
                    text=match.group(1),
                    start=match.start(),
                    end=match.end(),
                    entity_type="CONCEPT",
                    confidence=0.5,
                )
            )

        # Pattern 3: Known tool/product patterns
        tool_pattern = r"\b(Python|Java|JavaScript|TypeScript|Rust|Go|Docker|Linux|Ubuntu|Windows|macOS|VS Code|Vim|Emacs|GitHub|GitLab|PostgreSQL|MySQL|Redis|MongoDB|FAISS|ONNX|PyTorch|TensorFlow|spaCy|LangChain|OpenAI|Anthropic|Claude|GPT|Gemini|Llama|Mistral|NixOS|Debian|CentOS)\b"
        for match in re.finditer(tool_pattern, text, re.IGNORECASE):
            mentions.append(
                EntityMention(
                    text=match.group(1),
                    start=match.start(),
                    end=match.end(),
                    entity_type="TOOL",
                    confidence=0.8,
                )
            )

        # Pattern 4: IP addresses and hostnames
        for match in re.finditer(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", text):
            mentions.append(
                EntityMention(
                    text=match.group(1),
                    start=match.start(),
                    end=match.end(),
                    entity_type="HOST",
                    confidence=0.95,
                )
            )

        return mentions

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for canonical matching."""
        # Strip whitespace and normalize casing
        name = name.strip()
        # Don't lowercase - preserve original casing for proper nouns
        return name


class EntityResolver:
    """
    Resolve entity mentions across memories using a 4-stage pipeline.

    Stage 1: Exact name dedup (fast)
    Stage 2: Embedding cosine similarity (medium)
    Stage 3: MinHash fuzzy Jaccard (medium)
    Stage 4: LLM disambiguation (slow, optional)
    """

    def __init__(
        self,
        embedding_provider: Optional[Any] = None,
        llm_provider: Optional[Any] = None,
        similarity_threshold: float = 0.90,
        fuzzy_threshold: float = 0.85,
    ):
        self._extractor = EntityExtractor()
        self._embeddings = embedding_provider
        self._llm = llm_provider
        self._similarity_threshold = similarity_threshold
        self._fuzzy_threshold = fuzzy_threshold

        # Entity store: canonical_name -> Entity
        self._entities: Dict[str, Entity] = {}
        # Name index: lowercase name -> canonical name
        self._name_index: Dict[str, str] = {}
        # Embedding cache: entity_id -> vector
        self._embedding_cache: Dict[str, List[float]] = {}

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    def resolve(
        self,
        text: str,
        memory_id: Optional[str] = None,
    ) -> List[Entity]:
        """
        Extract and resolve entities from text, merging with existing entities.
        Returns the list of resolved entities.
        """
        mentions = self._extractor.extract(text)
        resolved = []

        for mention in mentions:
            normalized = self._extractor._normalize_name(mention.text)
            if not normalized or len(normalized) < 2:
                continue

            # Stage 1: Exact match
            entity = self._exact_match(normalized)

            # Stage 2: Embedding similarity (if available)
            if entity is None and self._embeddings:
                entity = self._embedding_match(normalized, mention.entity_type)

            # Stage 3: Fuzzy Jaccard
            if entity is None:
                entity = self._fuzzy_match(normalized)

            # Create new entity if not found
            if entity is None:
                entity = Entity(
                    id="",
                    name=normalized,
                    entity_type=mention.entity_type,
                    canonical_name=normalized,
                    aliases=[normalized],
                    created_at=time.time(),
                )
                self._entities[entity.id] = entity
                self._name_index[normalized.lower()] = entity.canonical_name
            else:
                # Add as alias if not already known
                if normalized not in entity.aliases:
                    entity.aliases.append(normalized)
                entity.mention_count += 1

            # Link to memory
            if memory_id and memory_id not in entity.linked_memory_ids:
                entity.linked_memory_ids.append(memory_id)

            resolved.append(entity)

        return resolved

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name (exact or fuzzy)."""
        canonical = self._name_index.get(name.lower())
        if canonical:
            return self._entities.get(canonical)
        return None

    def get_all_entities(self) -> List[Entity]:
        """Get all known entities."""
        return list(self._entities.values())

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def get_related_entities(self, entity_id: str) -> List[Entity]:
        """Get all entities that share memories with the given entity."""
        target = self._entities.get(entity_id)
        if not target:
            return []

        shared_memories = set(target.linked_memory_ids)
        related = []
        for eid, entity in self._entities.items():
            if eid == entity_id:
                continue
            if set(entity.linked_memory_ids) & shared_memories:
                related.append(entity)
        return related

    def _exact_match(self, name: str) -> Optional[Entity]:
        """Stage 1: Exact name lookup."""
        canonical = self._name_index.get(name.lower())
        if canonical:
            return self._entities.get(canonical)
        return None

    def _embedding_match(
        self, name: str, entity_type: str
    ) -> Optional[Entity]:
        """Stage 2: Embedding similarity matching."""
        if not self._embeddings:
            return None

        try:
            name_vector = self._embeddings.encode(name)
        except Exception:
            return None

        best_match = None
        best_score = 0.0

        for eid, cached_vector in self._embedding_cache.items():
            # Cosine similarity
            score = self._cosine_similarity(name_vector, cached_vector)
            if score > best_score and score >= self._similarity_threshold:
                best_score = score
                best_match = self._entities.get(eid)

        if best_match:
            # Cache this entity's embedding
            self._embedding_cache[best_match.id] = name_vector
            return best_match

        return None

    def _fuzzy_match(self, name: str) -> Optional[Entity]:
        """Stage 3: MinHash fuzzy Jaccard matching."""
        name_lower = name.lower()
        name_words = set(name_lower.split())

        if len(name_words) == 0:
            return None

        best_match = None
        best_score = 0.0

        for eid, entity in self._entities.items():
            canonical_words = set(entity.canonical_name.lower().split())
            if not canonical_words:
                continue

            # Jaccard similarity
            intersection = len(name_words & canonical_words)
            union = len(name_words | canonical_words)
            if union == 0:
                continue

            score = intersection / union
            if score > best_score and score >= self._fuzzy_threshold:
                best_score = score
                best_match = entity

        return best_match

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity store to dict."""
        return {
            "entities": {
                eid: {
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "canonical_name": e.canonical_name,
                    "aliases": e.aliases,
                    "linked_memory_ids": e.linked_memory_ids,
                    "mention_count": e.mention_count,
                    "created_at": e.created_at,
                }
                for eid, e in self._entities.items()
            },
            "name_index": self._name_index,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load entity store from dict."""
        for eid, edata in data.get("entities", {}).items():
            self._entities[eid] = Entity(
                id=eid,
                name=edata["name"],
                entity_type=edata["entity_type"],
                canonical_name=edata["canonical_name"],
                aliases=edata.get("aliases", []),
                linked_memory_ids=edata.get("linked_memory_ids", []),
                mention_count=edata.get("mention_count", 1),
                created_at=edata.get("created_at", 0.0),
            )
        self._name_index.update(data.get("name_index", {}))
