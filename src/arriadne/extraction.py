"""
LLM-Powered Memory Extraction Engine

Extracts structured memories from conversations using LLM intelligence.
Inspired by Mem0 V3 additive extraction but works locally with any LLM.

Key features:
- Extracts from both user and assistant messages
- Temporal grounding (converts relative dates to absolute)
- Multi-topic extraction (no first-topic dominance)
- Self-contained facts (readable without context)
- Quality scoring and filtering
- Entity extraction and linking
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("arriadne.extraction")


EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction engine for an AI assistant's long-term memory system.

Your task: Extract factual statements from conversations that would be valuable for the AI to remember long-term.

RULES:
1. Extract facts from BOTH the user AND assistant messages
2. Each memory must be self-contained — readable without conversation context
3. Convert relative time references to absolute dates (use "Observation Date" provided)
4. Preserve specific details: proper nouns, titles, quantities, dates, technical terms
5. Each memory should be 15-80 words
6. Focus on durable facts, not transient requests
7. Multi-topic extraction — extract from ALL topics discussed, not just the first
8. Preserve the speaker attribution (who said what)

GOOD examples:
- "Kyssta prefers dark mode interfaces and uses VS Code as their primary editor"
- "The VPS at 51.75.73.169 runs Ubuntu 24.04 with 4 cores and 8GB RAM"
- "Ariadne uses FAISS for vector search, achieving 0.83ms latency on 10K memories"
- "The user asked to remove HTTP basic auth from the Hermes dashboard on May 30, 2026"

BAD examples:
- "The user said hello" (too transient)
- "Tell me about Paris" (not a fact, just a request)
- "I think maybe the server could potentially..." (not a concrete fact)

OUTPUT: JSON array of memory objects with these fields:
- text: The factual statement (15-80 words)
- attributed_to: "user" or "assistant"
- topic: Category (e.g., "preferences", "technical", "project", "personal", "work")
- importance: 1-10 scale (10 = critical to remember, 1 = nice to have)
- entities: List of key entities mentioned (names, places, tools, etc.)
"""

CONTRADICTION_SYSTEM_PROMPT = """You are a contradiction detection engine for an AI memory system.

Given a NEW memory and a list of EXISTING memories, determine:
1. Does the new memory CONTRADICT any existing memory?
2. Does the new memory UPDATE/SUPERSEDE any existing memory?
3. Does the new memory DUPLICATE any existing memory?
4. Is the new memory entirely NEW?

For each relationship found, output:
- memory_id: The ID of the related existing memory
- relationship: "contradicts" | "updates" | "duplicates" | "related"
- reasoning: Brief explanation

If no relationships found, return an empty array.

IMPORTANT: "contradicts" means the facts are incompatible (e.g., "X is red" vs "X is blue").
"updates" means the new info supersedes the old (e.g., "X moved to NYC" when we knew "X lives in LA").
"duplicates" means essentially the same information.
"""

CONSOLIDATION_SYSTEM_PROMPT = """You are a memory consolidation engine. Given a group of related memories, merge them into fewer, richer memories.

RULES:
1. Preserve ALL unique facts — never lose information
2. Remove redundancy
3. Merge related facts into richer statements
4. Keep entity attributions
5. Maintain temporal awareness (use most recent info when there's a progression)
6. Output 1-3 consolidated memories per group
7. Each consolidated memory should be 20-100 words

OUTPUT: JSON array of consolidated memory objects with "text", "entities", "importance" fields.
"""


@dataclass
class ExtractedMemory:
    """A memory extracted from conversation by the LLM."""

    text: str
    attributed_to: str = "user"
    topic: str = "general"
    importance: int = 5
    entities: List[str] = field(default_factory=list)
    hash: str = ""
    confidence: float = 1.0
    source_messages: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.md5(self.text.strip().lower().encode()).hexdigest()[:16]


@dataclass
class ContradictionResult:
    """Result of contradiction detection between new and existing memories."""

    memory_id: str
    relationship: str  # "contradicts", "updates", "duplicates", "related"
    reasoning: str
    confidence: float = 1.0


class MemoryExtractor:
    """
    LLM-powered memory extraction from conversations.

    Usage:
        from arriadne.llm import LLMProvider
        from arriadne.extraction import MemoryExtractor

        llm = LLMProvider.auto_detect()
        extractor = MemoryExtractor(llm)

        memories = extractor.extract_from_conversation([
            {"role": "user", "content": "I'm planning to deploy Ariadne on my VPS"},
            {"role": "assistant", "content": "Great! The VPS has 4 cores and 8GB RAM..."},
        ])
    """

    def __init__(
        self,
        llm_provider: Any,
        min_importance: int = 3,
        max_importance: int = 10,
        max_memories_per_extraction: int = 20,
    ):
        self._llm = llm_provider
        self._min_importance = min_importance
        self._max_importance = max_importance
        self._max_memories = max_memories_per_extraction
        self._extraction_count = 0
        self._total_latency_ms = 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "extractions": self._extraction_count,
            "avg_latency_ms": round(
                self._total_latency_ms / max(1, self._extraction_count), 1
            ),
        }

    def extract_from_conversation(
        self,
        messages: List[Dict[str, str]],
        observation_date: Optional[str] = None,
        existing_memory_texts: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> List[ExtractedMemory]:
        """
        Extract memories from a conversation.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            observation_date: Current date for temporal grounding
            existing_memory_texts: Recently extracted memories (for dedup)
            session_id: Optional session identifier
        """
        if not messages:
            return []

        # Format conversation for the LLM
        conversation_text = self._format_conversation(messages)

        # Build the extraction prompt
        user_prompt = f"Extract memories from this conversation:\n\n{conversation_text}"

        if observation_date:
            user_prompt += f"\n\nObservation Date: {observation_date}"

        if existing_memory_texts:
            recent = existing_memory_texts[-10:]  # Last 10 for context
            user_prompt += (
                "\n\nRecently extracted memories (avoid duplicates):\n"
                + "\n".join(f"- {m}" for m in recent)
            )

        user_prompt += (
            "\n\nReturn a JSON array of memory objects. "
            'Each object must have: "text", "attributed_to" ("user" or "assistant"), '
            '"topic", "importance" (1-10), "entities" (array of strings). '
            "Return ONLY the JSON array, no other text."
        )

        # Call LLM
        from arriadne.llm import LLMMessage

        t0 = time.monotonic()
        try:
            response = self._llm.complete_sync(
                [
                    LLMMessage("system", EXTRACTION_SYSTEM_PROMPT),
                    LLMMessage("user", user_prompt),
                ],
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            latency = (time.monotonic() - t0) * 1000
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

        self._extraction_count += 1
        self._total_latency_ms += latency

        # Parse response
        try:
            data = response.json()
            if isinstance(data, dict):
                raw_memories = data.get("memories", data.get("memory", []))
            elif isinstance(data, list):
                raw_memories = data
            else:
                raw_memories = []
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response.content[:500]}")
            return []

        # Convert to ExtractedMemory objects
        memories = []
        for raw in raw_memories[: self._max_memories]:
            try:
                text = raw.get("text", "").strip()
                if not text or len(text) < 10:
                    continue

                importance = int(raw.get("importance", 5))
                importance = max(self._min_importance, min(self._max_importance, importance))

                mem = ExtractedMemory(
                    text=text,
                    attributed_to=raw.get("attributed_to", "user"),
                    topic=raw.get("topic", "general"),
                    importance=importance,
                    entities=raw.get("entities", []),
                    source_messages=[m.get("content", "") for m in messages[-3:]],
                )
                memories.append(mem)
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Skipping malformed memory: {e}")
                continue

        logger.info(
            f"Extracted {len(memories)} memories from {len(messages)} messages "
            f"in {latency:.0f}ms"
        )
        return memories

    def extract_from_text(
        self,
        text: str,
        attributed_to: str = "user",
        observation_date: Optional[str] = None,
    ) -> List[ExtractedMemory]:
        """Extract memories from a single text block."""
        messages = [{"role": attributed_to, "content": text}]
        return self.extract_from_conversation(
            messages, observation_date=observation_date
        )

    def detect_contradictions(
        self,
        new_memory: str,
        existing_memories: List[Dict[str, Any]],
    ) -> List[ContradictionResult]:
        """
        Detect contradictions between a new memory and existing memories.

        Args:
            new_memory: The new memory text
            existing_memories: List of {"id": str, "text": str} dicts
        """
        if not existing_memories:
            return []

        # Format existing memories for the prompt
        memory_list = "\n".join(
            f"[ID: {m['id']}] {m['text']}" for m in existing_memories[:50]
        )

        user_prompt = f"""NEW memory: "{new_memory}"

EXISTING memories:
{memory_list}

Analyze the new memory against each existing memory. For each relationship found, return:
- memory_id: The ID from above
- relationship: "contradicts" | "updates" | "duplicates" | "related"
- reasoning: One sentence explanation

Return a JSON array. If no relationships, return []."""

        from arriadne.llm import LLMMessage

        t0 = time.monotonic()
        try:
            response = self._llm.complete_sync(
                [
                    LLMMessage("system", CONTRADICTION_SYSTEM_PROMPT),
                    LLMMessage("user", user_prompt),
                ],
                temperature=0.0,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"Contradiction detection failed: {e}")
            return []

        latency = (time.monotonic() - t0) * 1000
        logger.debug(f"Contradiction detection: {latency:.0f}ms")

        # Parse results
        try:
            data = response.json()
            if isinstance(data, dict):
                results_raw = data.get("results", data.get("relationships", []))
            elif isinstance(data, list):
                results_raw = data
            else:
                results_raw = []
        except (json.JSONDecodeError, ValueError):
            return []

        results = []
        for r in results_raw:
            try:
                results.append(
                    ContradictionResult(
                        memory_id=r["memory_id"],
                        relationship=r["relationship"],
                        reasoning=r.get("reasoning", ""),
                    )
                )
            except (KeyError, TypeError):
                continue

        return results

    def consolidate_memories(
        self,
        memory_groups: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Consolidate groups of related memories into fewer, richer ones.

        Args:
            memory_groups: List of groups, each group is a list of
                          {"id": str, "text": str, "importance": int} dicts
        """
        all_consolidated = []

        for group in memory_groups:
            if len(group) <= 1:
                # Single memory, no consolidation needed
                all_consolidated.append(group[0])
                continue

            memory_list = "\n".join(
                f"[{m.get('id', '?')}] {m['text']} (importance: {m.get('importance', 5)})"
                for m in group
            )

            user_prompt = f"""Consolidate these related memories into fewer, richer ones:

{memory_list}

Return a JSON array of consolidated memories with "text", "entities" (array), and "importance" (1-10) fields."""

            from arriadne.llm import LLMMessage

            try:
                response = self._llm.complete_sync(
                    [
                        LLMMessage("system", CONSOLIDATION_SYSTEM_PROMPT),
                        LLMMessage("user", user_prompt),
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
                    consolidated = group  # Fallback to original

                all_consolidated.extend(consolidated)
            except Exception as e:
                logger.warning(f"Consolidation failed for group: {e}")
                all_consolidated.extend(group)

        return all_consolidated

    def _format_conversation(
        self, messages: List[Dict[str, str]], max_chars: int = 8000
    ) -> str:
        """Format conversation for LLM consumption."""
        parts = []
        total = 0

        # Take the most recent messages (within char limit)
        for msg in reversed(messages):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "").strip()
            if not content:
                continue

            formatted = f"[{role}]: {content}"
            if total + len(formatted) > max_chars:
                break
            parts.append(formatted)
            total += len(formatted)

        parts.reverse()
        return "\n\n".join(parts)
