"""Conversation memory and agent tools for Ariadne.

Provides:
- ConversationTracker: Extract facts and entities from conversations
- AgentTools: OpenAI function calling compatible tool definitions
- ContextManager: Manage context windows for LLM conversations
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from arriadne.storage import AriadneDB

logger = logging.getLogger(__name__)

# Common entity patterns
_ENTITY_PATTERNS = [
    # Proper nouns (capitalized words)
    re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"),
    # Quoted strings
    re.compile(r'"([^"]+)"'),
    # File paths
    re.compile(r"(/[\w./-]+)"),
    # URLs
    re.compile(r"(https?://\S+)"),
    # IP addresses
    re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"),
]

# Fact extraction patterns
_FACT_PATTERNS = [
    # "X is Y" / "X are Y"
    re.compile(
        r"\b(\w+(?:\s+\w+){0,3})\s+(?:is|are|was|were)\s+(.{10,200}?)(?:\.|$)",
        re.IGNORECASE,
    ),
    # "X has Y" / "X have Y"
    re.compile(
        r"\b(\w+(?:\s+\w+){0,3})\s+(?:has|have|had)\s+(.{10,200}?)(?:\.|$)",
        re.IGNORECASE,
    ),
    # "X uses Y" / "X uses Y for Z"
    re.compile(
        r"\b(\w+(?:\s+\w+){0,3})\s+(?:uses?|runs?|needs?|requires?|supports?)\s+(.{10,200}?)(?:\.|$)",
        re.IGNORECASE,
    ),
    # "X runs on Y"
    re.compile(
        r"\b(\w+(?:\s+\w+){0,3})\s+(?:runs?\s+on|deployed?\s+(?:to|on)|hosted?\s+on)\s+(.{10,200}?)(?:\.|$)",
        re.IGNORECASE,
    ),
]

# Instruction patterns (user preferences, commands, etc.)
_INSTRUCTION_PATTERNS = [
    # "always X" / "never X"
    re.compile(r"\b(always|never|don't|do not|must|must not|should|should not)\s+(.{10,200}?)(?:\.|$)", re.IGNORECASE),
    # "I want" / "I need" / "I prefer"
    re.compile(r"\b(?:I\s+)?(want|need|prefer|like|dislike|hate)\s+(.{10,200}?)(?:\.|$)", re.IGNORECASE),
    # "remember that" / "note that"
    re.compile(r"\b(?:remember|note|keep in mind|don't forget)\s+(?:that\s+)?(.{10,200}?)(?:\.|$)", re.IGNORECASE),
]


class ConversationTracker:
    """Tracks conversations and extracts structured memories.

    Extracts facts, entities, and instructions from conversation turns,
    then stores them as memories with entity links in the knowledge graph.
    """

    def __init__(self, db: AriadneDB) -> None:
        self._db = db
        self._turn_count = 0

    def sync_turn(
        self,
        role: str,
        content: str,
        extract_facts: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a conversation turn and extract memories.

        Args:
            role: Speaker role ("user", "assistant", "system").
            content: Message content.
            extract_facts: Whether to extract facts and entities.
            metadata: Optional metadata to attach.

        Returns:
            Dict with extracted facts, entities, and stored memory IDs.
        """
        self._turn_count += 1
        result: dict[str, Any] = {
            "turn": self._turn_count,
            "role": role,
            "facts": [],
            "entities": [],
            "memory_ids": [],
        }

        # Always store the raw turn as an episodic memory
        turn_meta = {
            "role": role,
            "turn": self._turn_count,
            "type": "conversation_turn",
            **(metadata or {}),
        }

        # Determine importance based on role and content
        importance = self._estimate_importance(role, content)

        # Store the turn
        emb = None
        turn_result = self._db.add_memory(
            content=content,
            embedding=emb,
            memory_type="episodic",
            importance=importance,
            metadata=turn_meta,
        )
        if turn_result["status"] == "created":
            result["memory_ids"].append(turn_result["memory_id"])

        if not extract_facts:
            return result

        # Extract facts
        facts = self._extract_facts(content)
        for fact in facts:
            fact_result = self._db.add_memory(
                content=fact["text"],
                memory_type="semantic",
                importance=min(1.0, importance + 0.1),
                entities=fact.get("entities"),
                metadata={
                    "source": "conversation_extraction",
                    "role": role,
                    "turn": self._turn_count,
                    "subject": fact.get("subject", ""),
                },
            )
            if fact_result["status"] == "created":
                result["memory_ids"].append(fact_result["memory_id"])
                result["facts"].append(fact)

        # Extract and link entities
        entities = self._extract_entities(content)
        result["entities"] = entities

        # Link entities to each other if found in the same turn
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                self._db.add_edge(e1, e2, "co-occur", weight=0.5)

        # Link entities to the turn memory
        if result["memory_ids"]:
            turn_id = result["memory_ids"][0]
            for entity in entities:
                self._db.add_edge(
                    f"memory:{turn_id}", entity, "mentions", weight=1.0,
                )

        logger.info(
            "Synced turn %d (%s): %d facts, %d entities, %d memories",
            self._turn_count, role, len(result["facts"]),
            len(entities), len(result["memory_ids"]),
        )

        return result

    def _estimate_importance(self, role: str, content: str) -> float:
        """Estimate the importance of a conversation turn."""
        importance = 0.5

        # User turns slightly more important (they drive the conversation)
        if role == "user":
            importance += 0.1

        # Instructions are high importance
        for pattern in _INSTRUCTION_PATTERNS:
            if pattern.search(content):
                importance += 0.2
                break

        # Questions are moderate importance
        if "?" in content:
            importance += 0.05

        # Longer content slightly more important (more information)
        if len(content) > 200:
            importance += 0.05
        if len(content) > 500:
            importance += 0.05

        # Technical content (contains code/commands)
        if any(kw in content.lower() for kw in ["pip install", "import ", "def ", "class ", "sudo", "apt "]):
            importance += 0.1

        return min(1.0, importance)

    def _extract_facts(self, text: str) -> list[dict[str, Any]]:
        """Extract factual claims from text."""
        facts = []
        seen = set()

        for pattern in _FACT_PATTERNS:
            for match in pattern.finditer(text):
                subject = match.group(1).strip()
                predicate = match.group(2).strip()

                # Skip very short or very generic subjects
                if len(subject) < 2 or len(predicate) < 5:
                    continue

                # Deduplicate
                key = f"{subject.lower()}|{predicate[:50].lower()}"
                if key in seen:
                    continue
                seen.add(key)

                # Extract entities from the fact
                entities = self._extract_entities(f"{subject} {predicate}")

                facts.append({
                    "subject": subject,
                    "predicate": predicate,
                    "text": f"{subject} {predicate}".strip(),
                    "entities": entities if entities else None,
                })

        return facts

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text."""
        entities = set()

        for pattern in _ENTITY_PATTERNS:
            for match in pattern.finditer(text):
                entity = match.group(1).strip()
                # Filter out common false positives
                if (
                    len(entity) >= 2
                    and entity.lower() not in {
                        "the", "this", "that", "with", "from", "have",
                        "been", "were", "they", "their", "what", "when",
                        "where", "which", "about", "would", "could",
                        "should", "there", "then", "than", "some",
                        "each", "very", "also", "just", "only",
                    }
                ):
                    entities.add(entity)

        return sorted(entities)

    def get_context(
        self,
        query: str,
        max_turns: int = 10,
        max_tokens_estimate: int = 2000,
    ) -> list[dict[str, Any]]:
        """Get relevant conversation context for an LLM prompt.

        Searches for relevant past turns and returns them formatted
        for inclusion in a conversation prompt.

        Args:
            query: Current query/topic to find context for.
            max_turns: Maximum number of turns to return.
            max_tokens_estimate: Rough token budget (chars / 4).

        Returns:
            List of context dicts with role, content, and relevance.
        """
        results = self._db.fts_search(query, k=max_turns * 2)

        context = []
        token_budget = max_tokens_estimate

        for mem in results:
            meta = mem.get("metadata") or {}
            if meta.get("type") != "conversation_turn":
                continue

            content = mem["content"]
            chars_needed = len(content)
            tokens_needed = chars_needed // 4

            if tokens_needed > token_budget:
                # Truncate to fit
                content = content[: token_budget * 4] + "..."
                token_budget = 0
            else:
                token_budget -= tokens_needed

            context.append({
                "role": meta.get("role", "unknown"),
                "content": content,
                "turn": meta.get("turn", 0),
                "relevance": mem.get("score", 0),
            })

            if token_budget <= 0:
                break

        # Sort by turn number (chronological)
        context.sort(key=lambda x: x["turn"])

        return context


class AgentTools:
    """OpenAI function calling compatible tool definitions for Ariadne.

    Provides tool schemas that can be used with OpenAI, Anthropic,
    or any other LLM that supports function calling.
    """

    TOOL_DEFINITIONS = [
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": "Store a memory. Use this to save important facts, preferences, instructions, or observations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to remember",
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance score (0.0-1.0). Default 0.5.",
                            "default": 0.5,
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["semantic", "episodic", "procedural", "preference"],
                            "description": "Type of memory",
                            "default": "semantic",
                        },
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Named entities to associate with this memory",
                        },
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Search memories for relevant information. Use this to find facts, past conversations, or stored knowledge.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results (default 5)",
                            "default": 5,
                        },
                        "memory_type": {
                            "type": "string",
                            "description": "Filter by memory type",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall_graph",
                "description": "Traverse the knowledge graph from an entity. Use this to find related entities and their connections.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "Starting entity name",
                        },
                        "hops": {
                            "type": "integer",
                            "description": "Maximum traversal depth (default 2)",
                            "default": 2,
                        },
                    },
                    "required": ["entity"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "link_entities",
                "description": "Create a relationship between two entities in the knowledge graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source entity",
                        },
                        "target": {
                            "type": "string",
                            "description": "Target entity",
                        },
                        "relationship": {
                            "type": "string",
                            "description": "Relationship type (e.g., 'uses', 'depends_on', 'related')",
                            "default": "related",
                        },
                    },
                    "required": ["source", "target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "forget",
                "description": "Delete a memory by ID. Use this when information is outdated or incorrect.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "integer",
                            "description": "ID of the memory to forget",
                        },
                    },
                    "required": ["memory_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_stats",
                "description": "Get statistics about the memory system. Use this to understand what's stored.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ]

    @staticmethod
    def get_tools() -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool definitions."""
        return AgentTools.TOOL_DEFINITIONS

    @staticmethod
    def get_tool_schemas_for_prompt() -> str:
        """Return tool definitions as a string for LLM prompts."""
        return json.dumps(AgentTools.TOOL_DEFINITIONS, indent=2)
